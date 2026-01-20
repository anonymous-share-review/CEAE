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
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def load_before_geojson_for_prompt(prompt_path: Path, prompt_suffix: str = "_prompt_level_1.json") -> Optional[
    Dict[str, Any]]:
    """
    prompt: city/bbox/{stem}_prompt_level_1.json
    geo:    city/bbox/{stem}.geojson

    Strategy:
    1) bbox_dir = prompt_path.parent.parent?  -> 这里 prompt_path.parent 就是 bbox_dir
       所以 bbox_dir = prompt_path.parent
    2) Prefer exact match: {stem}.geojson
    3) Ignore label/after/pred/etc
    4) Ambiguous => error
    """
    bbox_dir = prompt_path.parent
    if not bbox_dir.is_dir():
        return None

    stem = prompt_path.name.replace(prompt_suffix, "")
    exact = bbox_dir / f"{stem}.geojson"

    # ---- 1) exact match ----
    if exact.exists():
        with open(exact, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- 2) fallback scan (still strict) ----
    geojson_files: List[Path] = list(bbox_dir.glob("*.geojson"))
    if not geojson_files:
        return None

    def is_derived(p: Path) -> bool:
        s = p.stem.lower()
        bad_keys = ["_label", "_after", "_pred", "_tool", "_exec", "_plan", "_prompt"]
        return any(k in s for k in bad_keys)

    candidates = [p for p in geojson_files if not is_derived(p)]

    if len(candidates) == 0:
        return None

    if len(candidates) > 1:
        raise RuntimeError(
            "[Ambiguous original geojson]\n"
            f"prompt = {prompt_path}\n"
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
            # 你的工具函数签名是 fn(state: GeoState, ...)
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


def read_prompt_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt", "")
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return ""


SYSTEM_PROMPT = r"""
You are a One-Shot Tool Executor of an urban geospatial editing system.

ROLE & RESPONSIBILITY
--------------
You are tool-aware and execution-oriented.

Your sole responsibility is to DIRECTLY convert a natural-language editing instruction
into an ORDERED sequence of deterministic tool calls.

You do NOT:
- validate results,
- run checks,
- decide pass/fail,
- explain reasoning,
- output comments,
- invent tools.

You MUST:
- only use tool names from the allowlist,
- strictly follow tool signatures,
- output ONLY a JSON array of tool calls.

INPUTS YOU WILL RECEIVE
-----------------------

1) A natural-language editing instruction, which may contain:
   - target_id (e.g., node/..., way/..., relation/...)
   - operation
   - direction (optional)
   - distance_m (optional)
   - constraints (optional)

OUTPUT FORMAT (STRICT)
----------------------
Return ONLY a JSON array:

[
  {"name": "<tool_name>", "args": { ... }},
  ...
]

- The first non-space character MUST be '['
- The last non-space character MUST be ']'
- No prose, no markdown, no explanations
- The output MUST be directly parseable by json.loads

TOOL ARGUMENT CONVENTIONS (HARD)
--------------------------------
- All tools in this system operate on INTERNAL feature handles.
- A feature handle is a digits-only string corresponding to properties.id.
- Tool arguments MUST NOT contain any type prefixes such as "node/", "way/", or "relation/".
- Tool arguments MUST NOT contain "/" or any non-numeric characters.
- If an ID cannot be represented as a digits-only string, do NOT call any tool.

TOOL ALLOWLIST (EXACT NAMES ONLY)
--------------------------------
(A) Read / Locate
- get_feature_by_id(id:str)
- get_feature_geometry(feature:str)
- get_feature_properties(feature:str)

(B) Road Reference (OPTIONAL, if constraints include a requirement equivalent to "parallel to road": Do NOT call find_nearest_feature / project_point_to_line / get_line_tangent_at / move_point_along_bearing, use apply_move_with_constraints as the ONLY move operation.)
- find_nearest_feature(from_feature:str, filter:null, max_dist_m:float)
- project_point_to_line(point:str, line:str)
- get_line_tangent_at(line:str, t_norm:float, planar_epsg:null)

(C) Move Point (STATE-MUTATING)
- move_point_by_distance_direction(point:str, direction:str, distance_m:float)
- move_point_along_bearing(point:str, bearing_deg:float, distance_m:float)
- move_point_in_local_frame(point:str, ref_line:str, axis:str, signed_distance_m:float)
- apply_move_with_constraints(point:str, move:dict, constraints:dict)


ID NORMALIZATION (HARD RULE)
----------------------------
- The instruction may include a target_id with optional prefixes like "node/", "way/", or "relation/".
- You MUST extract the target_id from the instruction and normalize it BEFORE calling any tool:
  1) Remove any prefix such as "node/", "way/", or "relation/".
  2) Keep ONLY the numeric digits as a string.

- If you cannot extract a numeric target_id from the instruction, output [] immediately.


Rule 1: Target Resolution (S1)
------------------------------
- Call get_feature_by_id using the normalized numeric target_id extracted from the instruction.

Rule 2: Geometry Access Before Mutation
---------------------------------------
- Before any STATE-MUTATING move tool, call get_feature_geometry using the normalized numeric target_id.

Rule 3: Parallel-to-Road Constraint
-----------------------------------
If constraints include a requirement equivalent to "parallel to road":
- MUST use apply_move_with_constraints as the ONLY move operation.
- Do NOT call find_nearest_feature / project_point_to_line / get_line_tangent_at / move_point_along_bearing.
- Use:
  move = {"direction": <direction or null>, "distance_m": <distance_m>}
  constraints = {"parallel_to_road": true, "max_road_dist_m": 100.0}


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


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_raw(path: Path, raw_text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(raw_text or "")



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


def build_oneshot_messages(prompt_text: str) -> list:
    """
    Baseline: ONLY feed the natural-language instruction (prompt_text) to the LLM.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def generate_text_via_openai(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int = 800,
    service_tier: str = "flex",  # "flex" / "default" / "auto"
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns: (raw_text, usage_dict)
    """
    resp = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=int(max_output_tokens),
        service_tier=service_tier,  # flex/default/auto
        prompt_cache_key="cityedit_l1_baseline",
    )
    raw = getattr(resp, "output_text", "") or ""
    usage_obj = getattr(resp, "usage", None)

    if usage_obj is None:
        usage = {}
    elif isinstance(usage_obj, dict):
        usage = usage_obj
    elif hasattr(usage_obj, "model_dump"):  # pydantic v2 / OpenAI SDK 常见
        usage = usage_obj.model_dump()
    elif hasattr(usage_obj, "dict"):  # pydantic v1
        usage = usage_obj.dict()
    else:
        # 最后兜底：转字符串，至少不崩
        usage = {"raw_usage": str(usage_obj)}
    return raw.strip(), usage


def generate_tool_calls_from_instruction(
    instruction_text: str,
    client: OpenAI,
    api_model: str,
    max_output_tokens: int,
    service_tier: str,
) -> Tuple[Optional[List[Dict[str, Any]]], str, Dict[str, Any]]:
    raw, usage = generate_text_via_openai(
        client,
        model=api_model,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=instruction_text,
        max_output_tokens=max_output_tokens,
        service_tier=service_tier,
    )
    tool_calls = _extract_json_array_from_text(raw)
    return tool_calls, raw, usage

def iter_prompt_paths(data_root: Path, suffix: str = "_prompt_level_1.json"):
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
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execute", action="store_true", default=True, help="Execute tool_calls once (one-shot).")
    parser.add_argument("--raw_keep_chars", type=int, default=0)
    parser.add_argument("--api_model", type=str, default="gpt-5-mini")
    parser.add_argument("--service_tier", type=str, default="flex", choices=["flex", "default", "auto"])
    parser.add_argument("--timeout_s", type=float, default=900.0)  # flex 建议加大 :contentReference[oaicite:4]{index=4}
    parser.add_argument("--max_output_tokens", type=int, default=800)

    args = parser.parse_args()
    client = OpenAI(timeout=float(args.timeout_s))

    tool_registry = build_tool_registry()

    root_path = Path(args.data_root)
    suffix = "_prompt_level_1.json"
    prompt_files = list(root_path.rglob(f"*{suffix}"))
    print(f"[Info] Found {len(prompt_files)} prompt files matching '*{suffix}'")

    global_start_time = time.time()
    processed_count = 0
    successful_count = 0  # tool-calls parse+exec success（若 execute=True）
    total_count = 0
    fail_count = 0

    def _maybe_truncate_raw(raw: str) -> str:
        if raw is None:
            return ""
        if args.raw_keep_chars and args.raw_keep_chars > 0:
            return raw[: args.raw_keep_chars]
        return raw

    for idx, p_path in enumerate(prompt_files, start=1):
        stem_name = p_path.name.replace(suffix, "")
        total_count += 1

        try:
            prompt_text = read_prompt_file(p_path)

            bbox_dir = p_path.parent
            model_tag = sanitize_model_id(args.api_model)

            out_dir = bbox_dir / "baseline" / model_tag
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Sample {idx:04d}] out_dir={out_dir}")

            tool_calls, raw_output, usage = generate_tool_calls_from_instruction(
                instruction_text=prompt_text,
                client=client,
                api_model=args.api_model,
                max_output_tokens=args.max_output_tokens,
                service_tier=args.service_tier,
            )

            save_json(out_dir / f"{stem_name}_usage_level_1.json", usage)

            if tool_calls is None:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_parse_tool_calls.json",
                    {
                        "prompt_path": str(p_path),
                        "reason": "failed to parse JSON array tool_calls from LLM output",
                        "raw": _maybe_truncate_raw(raw_output),
                    },
                )
                processed_count += 1
                print(f"  └─ RESULT: PARSE_FAIL ✘ ({stem_name})")
                continue

            # save tool_calls
            save_json(out_dir / f"{stem_name}_tool_calls_level_1.json", tool_calls)
            print(f"  ├─ tool_calls generated ✔ (n_calls={len(tool_calls)})")
            # ---- load before geojson ----
            before_fc = load_before_geojson_for_prompt(p_path, prompt_suffix=suffix)

            # ---- execute once ----
            exec_result = execute_tool_calls_on_state(
                state_fc=copy.deepcopy(before_fc),
                tool_calls=tool_calls,
                tool_registry=tool_registry,
            )

            save_json(out_dir / f"{stem_name}_exec_trace.json", exec_result["trace"])

            if is_exec_success(exec_result["trace"], tool_calls):
                successful_count += 1
                save_json(out_dir / f"{stem_name}_after.geojson", exec_result["state_fc"])
                print(f"  └─ RESULT: EXEC_SUCCESS ✔ ({stem_name})")
            else:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_exec.json",
                    {
                        "prompt_path": str(p_path),
                        "reason": "execution failed (one-shot, no retry)",
                        "exec_trace": exec_result["trace"],
                        "tool_calls": tool_calls,
                    },
                )
                print(f"  └─ RESULT: EXEC_FAIL ✘ ({stem_name})")

            processed_count += 1
            if idx % 50 == 0:
                print(f"[Info] Progress: {idx}/{len(prompt_files)}")

        except Exception as e:
            fail_count += 1
            processed_count += 1
            print(f"[Fail] Error processing {stem_name}: {e}")
            traceback.print_exc()

    total_time_sec = time.time() - global_start_time
    success_rate = successful_count / total_count if total_count > 0 else 0.0
    failure_rate = fail_count / total_count if total_count > 0 else 0.0

    print(f"[Done] Processed {processed_count} tasks.")

    results_dir = Path(os.path.dirname(__file__)) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_subdir = results_dir / "baseline" / model_tag / "Level_1"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level1_baseline_running_time.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-1 One-shot Baseline Summary (Prompt-only)\n"
            f"----------------------------------------------\n"
            f"Model: {args.api_model}\n"
            f"Data root: {str(root_path)}\n"
            f"Prompt suffix: {suffix}\n"
            f"Total tasks: {total_count}\n"
            f"Successful tasks: {successful_count}\n"
            f"Success rate: {success_rate:.4f}\n"
            f"Failed tasks: {fail_count}\n"
            f"Failure rate: {failure_rate:.4f}\n"
            f"Execution enabled: {args.execute}\n"
            f"Total running time (seconds): {total_time_sec:.2f}\n"
        )

    print(f"[Info] Baseline runtime summary saved to {summary_file}")


if __name__ == "__main__":
    # DEFAULT_MODEL_IDS = [
    #     # "meta-llama/Meta-Llama-3-8B-Instruct",
    #     "meta-llama/Llama-3.1-8B-Instruct",
    #     "Qwen/Qwen2.5-3B-Instruct",
    #     "Qwen/Qwen2.5-7B-Instruct",
    #     "Qwen/Qwen3-8B",
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    #     "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    #     "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # ]
    main()