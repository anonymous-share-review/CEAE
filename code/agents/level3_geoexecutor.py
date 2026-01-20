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
from datetime import datetime
import copy

from utils.level2_tools import GeoState
import utils.level3_tools as l3t

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def load_model_and_tokenizer(model_id: str, device: str):
    print(f"[Init] Loading model: {model_id} ...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map=device
    )
    model.eval()

    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    return tok, model


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


def read_geojson(path: Path) -> Optional[Dict[str, Any]]:
    return load_json(path)


def save_geojson(path: Path, fc: Dict[str, Any]) -> None:
    save_json(path, fc)

def load_before_geojson_for_plan(plan_path: Path) -> Optional[Dict[str, Any]]:
    plan_path = Path(plan_path)

    if not plan_path.name.endswith("_plan.json"):
        raise RuntimeError(
            f"[Invalid plan filename] expected '*_plan.json' but got: {plan_path.name}"
        )

    bbox_dir = plan_path.parent.parent
    if not bbox_dir.is_dir():
        raise RuntimeError(
            f"[Invalid plan path] cannot locate bbox dir = plan_path.parent.parent\n"
            f"plan_path = {plan_path}\n"
            f"bbox_dir  = {bbox_dir}"
        )

    stem = plan_path.name[: -len("_plan.json")]  # remove suffix
    geo_path = bbox_dir / f"{stem}.geojson"

    if not geo_path.exists():
        raise RuntimeError(
            f"[Missing before geojson]\n"
            f"plan_path = {plan_path}\n"
            f"expected = {geo_path}"
        )

    with open(geo_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tool_registry() -> Dict[str, Callable]:
    """
    Level-3 tool registry: name -> function(state, **args)
    """
    tool_names = [
        # S1
        "l3_collect_green_ids",
        "l3_prepare_local_projection",
        "l3_compute_green_area_stats",
        # S2
        "l3_set_green_area_target",
        "l3_build_forbidden_union_m",
        "l3_plan_budget_allocation",
        # S3
        "l3_execute_expansion_loop",
        "l3_refresh_green_area_stats",
    ]

    reg: Dict[str, Callable] = {}
    for n in tool_names:
        if hasattr(l3t, n):
            reg[n] = getattr(l3t, n)
    return reg


def execute_tool_calls_on_state(
    state_fc: Dict[str, Any],
    tool_calls: List[Dict[str, Any]],
    tool_registry: Dict[str, Callable],
) -> Dict[str, Any]:
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
            break

    return {"trace": trace, "state_fc": state.fc}


def is_exec_success(exec_trace: list, tool_calls: list) -> bool:
    return (
        isinstance(exec_trace, list)
        and len(exec_trace) == len(tool_calls)
        and all(step.get("ok", False) for step in exec_trace)
    )


GEOEXECUTOR_SYSTEM_PROMPT = r"""
You are the GeoExecutor of an urban geospatial editing system.
ROLE & RESPONSIBILITY
---------------------
You are tool-aware and execution-oriented.

Your sole responsibility is to COMPILE an intent-level plan.json
into an ORDERED sequence of deterministic tool calls.

IMPORTANT:
- Tools do NOT support variable references.
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

HARD FAIL CONDITIONS (MUST OUTPUT [])
------------------------------------
Output [] immediately if ANY of the following is true:
- plan.parsed.target_ratio is null/empty
- plan.parsed.target_ratio is non-numeric or <= 0
- plan.parsed.target_ratio is > 0.95
- plan.subtasks is missing or not an array or empty

INPUTS YOU WILL RECEIVE
-----------------------
1) plan.json (intent-level), including:
   - plan.parsed.target_ratio            
   - plan.parsed.tolerance               
   - plan.constraints.hard               
   - plan.constraints.soft               
   - plan.subtasks                       


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


LEVEL-3 TOOLING MODEL (CRITICAL)
--------------------------------
- You MUST call tools in the required dependency order.
- Level-3 operates on the in-memory GeoState; no feature id is required.
- The goal is to increase TOTAL green area by ratio (A_target = A0 * (1 + ratio)) under constraints.



TOOL ALLOWLIST (EXACT NAMES ONLY)
---------------------------------
(A) Collect / Setup
- l3_collect_green_ids()
- l3_prepare_local_projection()

(B) Stats / Target
- l3_compute_green_area_stats()
- l3_set_green_area_target(ratio: float)

(C) Constraints / Planning
- l3_build_forbidden_union_m()
- l3_plan_budget_allocation()


(D) Execute / Refresh
- l3_execute_expansion_loop(tol: float)
- l3_refresh_green_area_stats()

PARAMETER CONVENTIONS
----------------------
- ratio = plan.parsed.target_ratio (float)
- tol   = plan.parsed.tolerance if present else 0.02

REQUIRED CALL SEQUENCE (STRICT)
--------------------------------
You MUST produce tool calls that follow this exact logical order:

Rule 1: Green Target Collection (S1)
--------------------------------------------------------
- You MUST start by collecting the editable green targets from the current GeoState.
- Call l3_collect_green_ids() first.
- Then call l3_prepare_local_projection() to ensure all subsequent area operations are in meters.
- Then call l3_compute_green_area_stats() to establish the baseline green area A0.
Required order (no deviation):
1) l3_collect_green_ids()
2) l3_prepare_local_projection()
3) l3_compute_green_area_stats()


Rule 2: Target Setting Before Any Expansion (S2)
----------------------------------------------
- You MUST set the area-growth target before building forbidden zones or allocating budgets.
- Call l3_set_green_area_target(ratio=plan.parsed.target_ratio) exactly once.
- The ratio MUST be taken from plan.parsed.target_ratio (decimal, e.g., 0.30). Never infer a new ratio.

Required call:
4) l3_set_green_area_target(ratio=<plan.parsed.target_ratio>)


Rule 3: Forbidden Zone Construction (S2)
-------------------------------------------------
- You MUST build forbidden zones before planning or executing any expansion steps.

Required call:
5) l3_build_forbidden_union_m()


Rule 4: Budget Allocation Planning (S2)
----------------------------------------
- You MUST plan the expansion budget AFTER forbidden zones are built.

Required call:
6) l3_plan_budget_allocation()




Rule 5: Execute Expansion Loop and Refresh Stats (S3)
----------------------------------
- You MUST execute the expansion loop AFTER budget allocation.
- After execution, you MUST refresh stats to expose the final A1 (post-edit green area).


Required order:
7) l3_execute_expansion_loop()
8) l3_refresh_green_area_stats()


Rule 6: Minimal Sufficiency
----------------------------------
- Do NOT add extra tool calls unless required by the rules above.
- Do NOT introduce any spacing-enforcement tool calls; soft spacing is not implemented here.

REMEMBER
--------
Output ONLY the ordered JSON array of tool calls.
""".strip()

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
    Expect:
      [
        {"name": "...", "args": {...}},
        ...
      ]
    Extract first [...] block.
    """
    if not text:
        return None

    m = re.search(r"\[\s*\{.*\}\s*\]", text, flags=re.DOTALL)
    if not m:
        return None

    candidate = m.group(0)
    try:
        obj = json.loads(candidate)
    except Exception:
        return None

    if not isinstance(obj, list):
        return None
    for item in obj:
        if not isinstance(item, dict):
            return None
        if "name" not in item:
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
        lb = raw.count("[")
        rb = raw.count("]")
        lc = raw.count("{")
        rc = raw.count("}")
        print(f"[ParseFail] brackets: [={lb}, ]={rb}, {{={lc}, }}={rc}")
        print("[ParseFail] raw_tail:", repr(raw[-200:]))

    return tool_calls, raw


def iter_plan_paths(data_root: Path, suffix: str = "_plan.json"):
    for p in data_root.rglob(f"*{suffix}"):
        yield p



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--plan_suffix", type=str, default="_plan.json")
    parser.add_argument("--execute", action="store_true", default=True)
    parser.add_argument("--max_retries", type=int, default=2)

    parser.add_argument("--raw_keep_chars", type=int, default=0)

    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_id, args.device)
    tool_registry = build_tool_registry()

    root_path = Path(args.data_root)
    plan_files = list(iter_plan_paths(root_path, suffix=args.plan_suffix))
    print(f"[Info] Found {len(plan_files)} plan files (Level-3) under: {root_path}")

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

            # ---- minimal required field checks (leave strict logic to tools/runtime) ----
            parsed = plan_json.get("parsed", {}) or plan_json.get("parsed_from_instruction", {}) or {}
            target_ratio = parsed.get("target_ratio", None)

            if target_ratio is None:
                save_json(
                    out_dir / f"{stem}_fail_null_target_ratio_level_3.json",
                    {
                        "plan_path": str(plan_path),
                        "reason": "target_ratio missing or null in plan.parsed.target_ratio",
                        "parsed": parsed,
                    },
                )
                fail_count += 1
                continue

            # Load initial geojson
            before_fc = None
            if args.execute:
                before_fc = load_before_geojson_for_plan(plan_path)
                if before_fc is None:
                    fail_count += 1
                    continue

            sample_success = False

            attempts_bundle = {
                "plan_path": str(plan_path),
                "stem": stem,
                "max_retries": args.max_retries,
                "execute": bool(args.execute),
                "attempts": [],
                "final": {"success": False, "success_attempt": None},
            }

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
                    "tool_calls": tool_calls,
                    "exec_ok": None,
                    "exec_trace": None,
                }

                if tool_calls is None:
                    print(f"  ├─ GeoExec attempt {attempt}: FAILED ✘")
                    attempts_bundle["attempts"].append(att_rec)
                    continue

                print(f"  ├─ GeoExec attempt {attempt}: tool_calls generated ✔")

                # save latest tool_calls next to plan
                save_json(out_dir / f"{stem}_tool_calls_level_3.json", tool_calls)

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

                    # success => save after geojson at SAME DIR as plan
                    save_geojson(out_dir / f"{stem}_after_level_3.geojson", exec_result["state_fc"])
                    break
                else:
                    print(f"  ├─ Exec attempt {attempt}: FAILED ✘")
                    print(
                        f"  [Debug] exec_failed: trace_len={len(exec_result['trace'])}, "
                        f"tool_calls_len={len(tool_calls)}, "
                        f"last_step={exec_result['trace'][-1] if exec_result['trace'] else None}"
                    )
                    continue

            # always save bundle
            save_json(out_dir / f"{stem}_attempts_bundle_level_3.json", attempts_bundle)

            if not sample_success:
                fail_count += 1
                save_json(
                    out_dir / f"{stem}_final_fail_level_3.json",
                    {
                        "plan_path": str(plan_path),
                        "max_retries": args.max_retries,
                        "bundle_file": str(out_dir / f"{stem}_attempts_bundle_level_3.json"),
                    },
                )
                print(f"  └─ FINAL: FAILED after {args.max_retries + 1} retries ✘")

            if idx % 50 == 0:
                print(f"[Progress] {idx}/{len(plan_files)}")

        except Exception as e:
            print(f"[Error] {plan_path}: {e}")
            traceback.print_exc()
            fail_count += 1

    # results summary
    results_dir = Path(os.path.dirname(__file__)) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_subdir = results_dir / sanitize_model_id(args.model_id) / "Level_3"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level3_geoexecutor_running_time.txt"

    elapsed = time.time() - start_time
    failure_rate = fail_count / total_count if total_count > 0 else 0.0

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-3 GeoExecutor Summary\n"
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