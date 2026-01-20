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
import copy

from utils.level2_tools import GeoState
import utils.level3_tools as l3t


from openai import OpenAI
os.environ['OPENAI_API_KEY'] = ""
# 初始化 GPU 环境
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)



def load_before_geojson_for_prompt(
    prompt_path: Path,
    prompt_suffix: str = "_prompt_level_3.json",
) -> Optional[Dict[str, Any]]:
    """
    prompt: city/bbox/{stem}_prompt_level_3.json
    geo:    city/bbox/{stem}.geojson

    Strategy:
    1) bbox_dir = prompt_path.parent
    2) Prefer exact match: {stem}.geojson
    3) Ignore derived: label/after/pred/tool/exec/plan/prompt
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
        bad_keys = ["_label", "_after", "_pred", "_tool", "_exec", "_plan", "_prompt", "_geoexec"]
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
    """
    One-shot baseline executor:
    - 顺序执行 tool_calls
    - trace 记录 name/args/ok/output_or_error
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


def read_prompt_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt", "")
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return ""


LEVEL3_OBS_BASELINE_SYSTEM_PROMPT = r"""
You are a One-Shot Tool Executor of an urban geospatial editing system.

ROLE & RESPONSIBILITY
---------------------
You are tool-aware and execution-oriented.

Your sole responsibility is to DIRECTLY convert a natural-language editing instruction
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
- invent tools.

You MUST:
- only use tool names from the allowlist,
- strictly follow tool signatures,
- output ONLY a JSON array of tool calls.

HARD FAIL CONDITIONS (MUST OUTPUT [])
------------------------------------
Output [] immediately if ANY of the following is true:
- You cannot extract a numeric target_ratio from the instruction
- target_ratio is null/empty/non-numeric or <= 0
- target_ratio is > 0.95

INPUTS YOU WILL RECEIVE
-----------------------
1) A natural-language editing instruction, which may contain:
   - target_ratio ()
   - tolerance ()
   - constraints ()

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
- Level-3 operates on the in-memory GeoState; no feature id is required.
- The goal is to increase TOTAL green area by ratio under constraints.

TOOL ALLOWLIST (EXACT NAMES ONLY)
---------------------------------
(A) Collect / Setup
- l3_collect_green_ids()
- l3_prepare_local_projection()

(B) Stats / Target
- l3_compute_green_area_stats()
- l3_set_green_area_target(ratio: float)

(C) Constraints / Planning
- l3_build_forbidden_union_m(include_non_green_polygons: bool, include_green_others: bool, polygon_setback_m: float)
- l3_plan_budget_allocation(policy: str, max_steps: int, min_step_area_m2: float)

(D) Execute / Refresh
- l3_execute_expansion_loop(max_total_steps: int, tol: float)
- l3_refresh_green_area_stats()

PARAMETER CONVENTIONS
----------------------
- ratio MUST be extracted from the instruction and converted to decimal:
  - If instruction says "30%", use ratio = 0.30
  - If instruction says "0.30", use ratio = 0.30
- tol = extracted tolerance if present

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
- Call l3_set_green_area_target(ratio=<extracted_ratio>) exactly once.
- The ratio MUST be extracted from the instruction (decimal, e.g., 0.30). Never infer a new ratio.

Required call:
4) l3_set_green_area_target(ratio=<extracted_ratio>)


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


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)

def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)




def _extract_json_array_from_text(text: str) -> Optional[List[Dict[str, Any]]]:
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
        if "name" not in item or "args" not in item:
            return None
    return obj


def build_oneshot_messages(prompt_text: str) -> list:
    return [
        {"role": "system", "content": LEVEL3_OBS_BASELINE_SYSTEM_PROMPT},
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
        prompt_cache_key="cityedit_l3_baseline",
    )
    raw = getattr(resp, "output_text", "") or ""
    usage_obj = getattr(resp, "usage", None)

    if usage_obj is None:
        usage = {}
    elif isinstance(usage_obj, dict):
        usage = usage_obj
    elif hasattr(usage_obj, "model_dump"):
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

def is_exec_success(exec_trace: list, tool_calls: list) -> bool:
    return (
        isinstance(exec_trace, list)
        and isinstance(tool_calls, list)
        and len(exec_trace) == len(tool_calls)
        and all(step.get("ok", False) for step in exec_trace)
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--execute", action="store_true", default=True, help="Execute tool_calls once (one-shot).")
    parser.add_argument("--raw_keep_chars", type=int, default=0)
    parser.add_argument("--api_model", type=str, default="gpt-5-mini")
    parser.add_argument("--service_tier", type=str, default="flex", choices=["flex", "default", "auto"])
    parser.add_argument("--timeout_s", type=float, default=900.0)
    parser.add_argument("--max_output_tokens", type=int, default=800)
    parser.add_argument("--prompt_suffix", type=str, default="_prompt_level_3.json")

    args = parser.parse_args()
    client = OpenAI(timeout=float(args.timeout_s))

    tool_registry = build_tool_registry()

    root_path = Path(args.data_root)
    suffix = args.prompt_suffix
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

            save_json(out_dir / f"{stem_name}_usage_level_3.json", usage)

            if tool_calls is None:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_parse_tool_calls_level_3.json",
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
            save_json(out_dir / f"{stem_name}_tool_calls_level_3.json", tool_calls)
            print(f"  ├─ tool_calls generated ✔ (n_calls={len(tool_calls)})")
            # ---- load before geojson ----
            before_fc = load_before_geojson_for_prompt(p_path, prompt_suffix=suffix)

            # ---- execute once ----
            exec_result = execute_tool_calls_on_state(
                state_fc=copy.deepcopy(before_fc),
                tool_calls=tool_calls,
                tool_registry=tool_registry,
            )

            save_json(out_dir / f"{stem_name}_exec_trace_level_3.json", exec_result["trace"])

            if is_exec_success(exec_result["trace"], tool_calls):
                successful_count += 1
                save_json(out_dir / f"{stem_name}_after_level_3.geojson", exec_result["state_fc"])
                print(f"  └─ RESULT: EXEC_SUCCESS ✔ ({stem_name})")
            else:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_exec_level_3.json",
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

    model_subdir = results_dir / "baseline" / model_tag / "Level_3"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level3_baseline_running_time.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-3 One-shot Baseline Summary (Prompt-only)\n"
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
    main()