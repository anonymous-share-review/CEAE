import argparse
import json
import traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os
import re
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
from utils.level1_tools import GeoState
import utils.level1_tools as l1t
from datetime import datetime
import copy

# 初始化 GPU 环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def load_before_geojson_for_plan(
    plan_path: Path,
    plan_suffix: str = "_plan_level_1.json",
) -> Optional[Dict[str, Any]]:
    """
    plan: city/bbox/model_id/{stem}_plan_level_1.json
    geo:  city/bbox/{stem}.geojson

    Strategy:
    1) Look in plan_path.parent.parent (bbox dir)
    2) Prefer exact match: {stem}.geojson
    3) Ignore label/after/pred/etc
    4) Ambiguous => error
    """

    # city/bbox/
    bbox_dir = plan_path.parent.parent
    if not bbox_dir.is_dir():
        return None

    stem = plan_path.name.replace(plan_suffix, "")
    exact = bbox_dir / f"{stem}.geojson"

    # ---- 1) exact match (strongly preferred) ----
    if exact.exists():
        with open(exact, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- 2) fallback scan (still strict) ----
    geojson_files: List[Path] = list(bbox_dir.glob("*.geojson"))
    if not geojson_files:
        return None

    def is_derived(p: Path) -> bool:
        s = p.stem.lower()
        bad_keys = [
            "_label",
            "_after",
            "_pred",
            "_tool",
            "_exec",
        ]
        return any(k in s for k in bad_keys)

    candidates = [p for p in geojson_files if not is_derived(p)]

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        raise RuntimeError(
            "[Ambiguous original geojson]\n"
            f"plan = {plan_path}\n"
            f"bbox_dir = {bbox_dir}\n"
            f"candidates = {[p.name for p in candidates]}"
        )

    with open(candidates[0], "r", encoding="utf-8") as f:
        return json.load(f)

def build_tool_registry() -> Dict[str, Callable]:
    """
    name -> function(state, **args) registry
    """
    tool_names = [
        # Read / Locate
        "get_feature_by_id",
        "get_feature_geometry",
        "get_feature_properties",
        # Road reference
        "find_nearest_feature",
        "project_point_to_line",
        "get_line_tangent_at",
        # Move
        "move_point_by_distance_direction",
        "move_point_along_bearing",
        "move_point_in_local_frame",
        "apply_move_with_constraints",
        # Constraint / Repair
        "compute_current_offset_to_line",
        "snap_point_to_nearest_line_offset",
    ]
    reg = {}
    for n in tool_names:
        if hasattr(l1t, n):
            reg[n] = getattr(l1t, n)
    return reg

def execute_tool_calls_on_state(
    state_fc: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable],
) -> Dict[str, Any]:
    """
    最简执行器：
    - 按 tool_calls 顺序执行
    - 每步记录 name/args/ok/result_or_error
    - 返回 trace + mutated state_fc
    """
    state = GeoState(state_fc)
    trace = []
    for step_idx, call in enumerate(tool_calls):
        name = call.get("name")
        args = call.get("args", {})
        if name not in tool_registry:
            trace.append({
                "step": step_idx,
                "name": name,
                "args": args,
                "ok": False,
                "error": f"TOOL_NOT_ALLOWED_OR_NOT_FOUND: {name}",
            })
            # 最简策略：遇到非法 tool 直接停止
            break

        fn = tool_registry[name]
        try:
            out = fn(state, **args)
            trace.append({
                "step": step_idx,
                "name": name,
                "args": args,
                "ok": True,
                "output": out,
            })
        except Exception as e:
            trace.append({
                "step": step_idx,
                "name": name,
                "args": args,
                "ok": False,
                "error": repr(e),
            })
            # 最简策略：失败就停止
            break

    # 把 state.fc（可能被 move 工具修改过）返回
    return {"trace": trace, "state_fc": state.fc}

GEOEXECUTOR_SYSTEM_PROMPT = r"""
You are the GeoExecutor of an urban geospatial editing system.

ROLE & RESPONSIBILITY
---------------------
You are tool-aware and execution-oriented.

Your sole responsibility is to COMPILE an intent-level plan.json
into an ORDERED sequence of deterministic tool calls.

IMPORTANT:
- Tools do NOT support variable references.
- The ONLY valid feature handle in this system is the digits-only target_id string itself.
- The output MUST be a COMPLETE and VALID JSON array.
- All '{' must be closed by '}', and all '[' must be closed by ']'.
- Every tool-call object MUST be fully closed before the array is closed.
- If the JSON structure would be invalid or incomplete, output [] instead.

FINAL OUTPUT CHECK (HARD)
- Ensure the output is valid JSON parseable by json.loads.
- Ensure '{' count equals '}' count and '[' count equals ']' count.
- If invalid, rewrite the entire JSON array correctly.

You do NOT:
- validate results,
- run checks,
- decide pass/fail,
- explain reasoning,
- output comments,
- invent tools,
- reorder subtasks.

You MUST:
- strictly follow the semantic order implied by plan.subtasks,
- only use tool names from the allowlist,
- strictly follow tool signatures,
- output ONLY a JSON array of tool calls.

IMPORTANT LIMITATION
--------------------
- If plan.parsed_from_instruction.target_id is null, output [] (empty array) immediately.
- If plan.parsed_from_instruction.target_id is not a digits-only string, output [] immediately.

INPUTS YOU WILL RECEIVE
-----------------------
1) plan.json (intent-level), including:
   - parsed_from_instruction.target_id
   - parsed_from_instruction.operation
   - parsed_from_instruction.direction (optional)
   - parsed_from_instruction.distance_m (optional)
   - constraints
   - subtasks (ordered)

OUTPUT FORMAT (STRICT)
----------------------
Return ONLY a JSON array:

[
  {"name": "<tool_name>", "args": { ... }},
  ...
]

- The first non-space character MUST be '['
- The last non-space character MUST be ']'
- No prose, no markdown, no prefixes like "Here is ..."
- The output MUST be directly parseable by json.loads


TOOL ARGUMENT CONVENTIONS (HARD)
--------------------------------
- All tools in this system operate on INTERNAL feature handles.
- A feature handle is a digits-only string corresponding to properties.id.
- Tool arguments MUST NOT contain any type prefixes such as "node/", "way/", or "relation/".
- Tool arguments MUST NOT contain "/" or any non-numeric characters.
- If an ID cannot be represented as a digits-only string, do NOT call any tool.

TOOL ALLOWLIST (EXACT NAMES ONLY)
---------------------------------
(A) Read / Locate
- get_feature_by_id(id:str)
- get_feature_geometry(feature:str)
- get_feature_properties(feature:str)

(B) Road Reference (OPTIONAL)
- find_nearest_feature(from_feature:str, filter:dict|null, max_dist_m:float)
- project_point_to_line(point:str, line:str)
- get_line_tangent_at(line:str, t_norm:float, planar_epsg:int|null)

(C) Move Point (STATE-MUTATING)
- move_point_by_distance_direction(point:str, direction:str, distance_m:float)
- move_point_along_bearing(point:str, bearing_deg:float, distance_m:float)
- move_point_in_local_frame(point:str, ref_line:str, axis:str, signed_distance_m:float)
- apply_move_with_constraints(point:str, move:dict, constraints:dict)

EXECUTION COMPILATION RULES (HARD)
----------------------------------

Subtask Order (MUST FOLLOW)
---------------------------
- You MUST respect the semantic order of plan.subtasks.
- Tool calls for earlier subtasks MUST appear before later ones.
- Do NOT interleave tool calls from different subtasks.

ID NORMALIZATION (HARD RULE)
----------------------------
- plan.parsed_from_instruction.target_id MUST already be a digits-only string (no "node/", "way/", or "relation/").
- Use target_id exactly as provided in plan.json.
- NEVER add any type prefixes.
- NEVER pass IDs with prefixes to tools.
- If you cannot extract a numeric target_id from the instruction, output [] immediately.

Rule 1: Target Resolution (S1)
------------------------------
- Call get_feature_by_id(plan.parsed_from_instruction.target_id).

Rule 2: Geometry Access Before Mutation
---------------------------------------
- Before any STATE-MUTATING move tool,
  you MUST call get_feature_geometry(plan.parsed_from_instruction.target_id).

Rule 3: Parallel-to-Road Constraint
-----------------------------------
If constraints include a requirement equivalent to "parallel to road":
- Prefer using apply_move_with_constraints with:
  move = {"direction": <direction or null>, "distance_m": <distance_m>}
  constraints = {"parallel_to_road": true, "max_road_dist_m": 100.0}
- Do NOT call validator tools.

Rule 4: Unconstrained Move
--------------------------
If no road-parallel constraint exists:
- If direction is one of the 8 compass directions: use move_point_by_distance_direction
- Else if bearing is explicitly available: use move_point_along_bearing
- No-op is NOT allowed.

Rule 5: Minimal Sufficiency
---------------------------
- Do NOT add extra tool calls unless required by constraints.
- The sequence should be minimal but sufficient.

REMEMBER
--------
Output ONLY the ordered JSON array of tool calls.
""".strip()


def load_model_and_tokenizer(model_id: str, device: str):
    print(f"[Init] Loading model: {model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map=device
    )
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[Error] Failed to read {path}: {e}")
        return None


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_raw(path: Path, raw_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_text or "")


def generate_text(
    model,
    tok,
    messages,
    max_new_tokens: int = 800,
    do_sample: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    try:
        enc = tok.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)
    except TypeError:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        eos_token_id=tok.eos_token_id,
        pad_token_id=pad_id,
        do_sample=bool(do_sample),
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=float(temperature), top_p=float(top_p)))

    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)

    out = tok.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return out.strip()


def _extract_json_array_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
    """
    Expect a JSON array:
      [
        {"name": "...", "args": {...}},
        ...
      ]

    We DO NOT try fancy "middle patching".
    We just extract the first [...] block if any.
    """
    if not text:
        return None

    # pick the first JSON array block
    m = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
    except Exception:
        return None

    # basic shape check
    if not isinstance(obj, list):
        return None
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            return None
        if "name" not in item or "args" not in item:
            return None
    return obj

def build_geoexecutor_messages(plan_json: Dict[str, Any]) -> list:
    return [
        {"role": "system", "content": GEOEXECUTOR_SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps({"plan": plan_json}, ensure_ascii=False)},
    ]

def generate_tool_calls_from_plan(
    plan_json: Dict[str, Any],
    model,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    messages = build_geoexecutor_messages(plan_json)
    raw = generate_text(
        model=model,
        tok=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    tool_calls = _extract_json_array_from_text(raw)

    if tool_calls is None:
        # quick bracket diagnostics
        lb = raw.count("[")
        rb = raw.count("]")
        lc = raw.count("{")
        rc = raw.count("}")
        print(f"[ParseFail] brackets: [={lb}, ]={rb}, {{={lc}, }}={rc}")
        print("[ParseFail] raw_tail:", repr(raw[-200:]))

    return tool_calls, raw

    return tool_calls, raw


def iter_plan_paths(data_root: Path, suffix: str = "_plan_level_1.json"):
    for p in data_root.rglob(f"*{suffix}"):
        yield p

def is_exec_success(exec_trace: list, tool_calls: list) -> bool:
    return (
        isinstance(exec_trace, list)
        and len(exec_trace) == len(tool_calls)
        and all(step.get("ok", False) for step in exec_trace)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data_1_2")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--plan_suffix", type=str, default="_plan_level_1.json")
    parser.add_argument("--execute", action="store_true", default=True)
    parser.add_argument("--max_retries", type=int, default=2)

    # 可选：限制 raw 保存长度，避免 bundle 太大（0 表示不截断）
    parser.add_argument("--raw_keep_chars", type=int, default=0)

    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_id, args.device)
    tool_registry = build_tool_registry()

    root_path = Path(args.data_root)
    plan_files = list(iter_plan_paths(root_path, suffix=args.plan_suffix))
    print(f"[Info] Found {len(plan_files)} plan files")

    total_count = 0
    fail_count = 0
    start_time = time.time()

    def _maybe_truncate_raw(raw: str) -> str:
        if raw is None:
            return ""
        if args.raw_keep_chars and args.raw_keep_chars > 0:
            return raw[: args.raw_keep_chars]
        return raw

    for idx, plan_path in enumerate(plan_files, start=1):
        total_count += 1

        try:
            stem = plan_path.name.replace(args.plan_suffix, "")


            out_dir = plan_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Sample {idx:04d}] path={out_dir}")
            plan_json = load_json(plan_path)
            if plan_json is None:
                fail_count += 1
                continue

            parsed = plan_json.get("parsed_from_instruction", {})
            target_id = parsed.get("target_id")

            # ---------- Case 1: target_id missing ----------
            if not target_id:
                save_json(
                    out_dir / f"{stem}_fail_null_target_id.json",
                    {
                        "plan_path": str(plan_path),
                        "reason": "target_id missing or null",
                        "parsed_from_instruction": parsed,
                    },
                )
                fail_count += 1
                continue

            # ---------- Load initial geojson if execution enabled ----------
            before_fc = None
            if args.execute:
                before_fc = load_before_geojson_for_plan(plan_path)
                if before_fc is None:
                    fail_count += 1
                    continue

            sample_success = False

            # ✅ 聚合所有 attempt 的信息到一个文件
            attempts_bundle = {
                "plan_path": str(plan_path),
                "stem": stem,
                "max_retries": args.max_retries,
                "execute": bool(args.execute),
                "attempts": [],  # 每次 attempt 的记录都 append 进来
                "final": {
                    "success": False,
                    "success_attempt": None,
                },
            }

            # ---------- Retry loop ----------
            for attempt in range(args.max_retries + 1):
                tool_calls, raw = generate_tool_calls_from_plan(
                    plan_json=plan_json,
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

                att_rec = {
                    "attempt": attempt,
                    "geoexec_ok": tool_calls is not None,
                    "raw": _maybe_truncate_raw(raw),
                    "tool_calls": tool_calls,   # 直接保存解析后的 tool_calls（None 或 list）
                    "exec_ok": None,
                    "exec_trace": None,
                }

                if tool_calls is None:
                    print(f"  ├─ GeoExec attempt {attempt}: FAILED ✘")
                    attempts_bundle["attempts"].append(att_rec)
                    continue

                print(f"  ├─ GeoExec attempt {attempt}: tool_calls generated ✔")

                # 只保存一份最终（最近一次）的 tool_calls
                save_json(out_dir / f"{stem}_tool_calls_level_1.json", tool_calls)

                # ---------- No execution mode ----------
                if not args.execute:
                    att_rec["exec_ok"] = True
                    attempts_bundle["attempts"].append(att_rec)
                    sample_success = True
                    attempts_bundle["final"]["success"] = True
                    attempts_bundle["final"]["success_attempt"] = attempt
                    break

                exec_result = execute_tool_calls_on_state(
                    state_fc=copy.deepcopy(before_fc),
                    tool_calls=tool_calls,
                    tool_registry=tool_registry,
                )

                att_rec["exec_trace"] = exec_result["trace"]
                ok = is_exec_success(exec_result["trace"], tool_calls)
                att_rec["exec_ok"] = bool(ok)

                attempts_bundle["attempts"].append(att_rec)

                if ok:
                    print(f"  ├─ Exec attempt {attempt}: SUCCESS ✔")
                    sample_success = True
                    attempts_bundle["final"]["success"] = True
                    attempts_bundle["final"]["success_attempt"] = attempt

                    # 成功时才落 after
                    save_json(out_dir / f"{stem}_after.geojson", exec_result["state_fc"])
                    break
                else:
                    print(f"  ├─ Exec attempt {attempt}: FAILED ✘")
                    print(
                        f"  [Debug] exec_failed: trace_len={len(exec_result['trace'])}, "
                        f"tool_calls_len={len(tool_calls)}, "
                        f"last_step={exec_result['trace'][-1] if exec_result['trace'] else None}"
                    )
                    continue
            # 无论成功/失败，都落一个 bundle（一个文件包含全部 attempt 记录）
            save_json(out_dir / f"{stem}_attempts_bundle.json", attempts_bundle)

            # ---------- Final decision ----------
            if not sample_success:
                fail_count += 1
                save_json(
                    out_dir / f"{stem}_final_fail.json",
                    {
                        "plan_path": str(plan_path),
                        "max_retries": args.max_retries,
                        "bundle_file": str(out_dir / f"{stem}_attempts_bundle.json"),
                    },
                )
                print(f"  └─ FINAL: FAILED after {args.max_retries + 1} retries ✘")

            if idx % 50 == 0:
                print(f"[Progress] {idx}/{len(plan_files)}")

        except Exception as e:
            print(f"[Error] {plan_path}: {e}")
            traceback.print_exc()
            fail_count += 1

    results_dir = Path(os.path.dirname(__file__)) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_subdir = results_dir / sanitize_model_id(args.model_id) / "Level_1"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level1_geoexecutor_running_time.txt"

    elapsed = time.time() - start_time
    failure_rate = fail_count / total_count if total_count > 0 else 0.0

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-1 GeoExecutor Summary\n"
            f"----------------------------\n"
            f"Model: {args.model_id}\n"
            f"Data root: {str(root_path)}\n"
            f"Plan suffix: {args.plan_suffix}\n"
            f"Total plan files: {total_count}\n"
            f"Failed samples: {fail_count}\n"
            f"Failure rate: {failure_rate:.4f}\n"
            f"Max retries: {args.max_retries}\n"
            f"Execution enabled: {args.execute}\n"
            f"Total running time (seconds): {elapsed:.2f}\n"
            f"Total running time (minutes): {elapsed / 60.0:.2f}\n"
        )

    print(f"[Info] GeoExecutor runtime summary saved to {summary_file}")



if __name__ == "__main__":
    main()