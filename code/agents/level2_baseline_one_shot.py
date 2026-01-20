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

# IMPORTANT: Level2 uses Level2 GeoState (with scratch)
from utils.level2_tools import GeoState
import utils.level2_tools as l2t
os.environ['OPENAI_API_KEY'] = ""
# 初始化 GPU 环境
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


def load_before_geojson_for_prompt(
    prompt_path: Path,
    prompt_suffix: str = "_prompt_level_2.json",
) -> Optional[Dict[str, Any]]:
    """
    prompt: city/bbox/{stem}_prompt_level_2.json
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
    Level-2 tool registry: name -> function(state, **args)
    """
    tool_names = [
        "get_feature_by_id",
        "get_feature_geometry",
        "get_linestring_stats",
        "select_linestring_endpoint",
        "set_length_change_m",      # 你 prompt 里使用的是这个
        "apply_linestring_extend",
        "apply_linestring_shorten",
        "commit_linestring_geometry",
    ]
    reg = {}
    for n in tool_names:
        if hasattr(l2t, n):
            reg[n] = getattr(l2t, n)
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


SYSTEM_PROMPT = r"""
You are a One-Shot Tool Executor of an urban geospatial editing system.

ROLE & RESPONSIBILITY
--------------
You are tool-aware and execution-oriented.

Your sole responsibility is to DIRECTLY convert a natural-language editing instruction
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

You MUST:
- only use tool names from the allowlist,
- strictly follow tool signatures,
- output ONLY a JSON array of tool calls.

HARD FAIL CONDITIONS (MUST OUTPUT [])
------------------------------------
Output [] immediately if ANY of the following is true:
- You cannot extract a numeric target_id from the instruction
- operation is null/empty or unsupported
- distance_m is null/empty/non-numeric or <= 0

INPUTS YOU WILL RECEIVE
-----------------------
1) A natural-language editing instruction, which may contain:
   - target_id (e.g., node/..., way/..., relation/...)
   - operation (extend / shorten)
   - distance_m (REQUIRED, meters)
   - endpoint (optional; if absent, choose "tail" by default)
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

LEVEL-2 TOOLING MODEL (CRITICAL)
--------------------------------
- You MUST call tools in the required dependency order.
- You MUST call commit_linestring_geometry(line) as the final step to apply edits.

TOOL ALLOWLIST (EXACT NAMES ONLY)
---------------------------------
(A) Read / Locate 
- get_feature_by_id(id:str)
- get_feature_geometry(feature:str)

(B) Linestring Context / Stats (SCRATCH-POPULATING)
- get_linestring_stats(line:str)

(C) Endpoint Selection (SCRATCH-POPULATING)
- select_linestring_endpoint(line:str, endpoint:str)    # endpoint in {"head","tail"}

(D) Delta / Change Amount (SCRATCH-POPULATING)
- set_length_change_m(delta_m:float)

(E) Apply Edit on Scratch Working Coords (SCRATCH-MUTATING ONLY)
- apply_linestring_extend()
- apply_linestring_shorten()

(F) Commit (THE ONLY GEOJSON-MUTATING STEP)
- commit_linestring_geometry(line:str)

ID NORMALIZATION (HARD RULE)
----------------------------
- You MUST extract the target_id from the instruction and normalize it BEFORE calling any tool:
  1) Remove any prefix such as "node/", "way/", or "relation/".
  2) Keep ONLY the numeric digits as a string.
- Use the normalized numeric target_id for ALL tool calls.
- NEVER add any type prefixes.
- NEVER pass IDs with prefixes to tools.

Rule 1: Target Resolution (S1) 
--------------------------------------------------------
- Call get_feature_by_id using the normalized numeric target_id extracted from the instruction.

Rule 2: Geometry Access Before Edit 
----------------------------------------------
- You MAY call get_feature_geometry using the normalized numeric target_id for traceability.
- However, get_linestring_stats(line) is REQUIRED before any scratch-based edit.

Rule 3: Required Scratch-Building Sequence 
-------------------------------------------------
Before applying extend/shorten, you MUST call in this order:
1) get_linestring_stats(line=target_id)
2) select_linestring_endpoint(line=target_id, endpoint="head" or "tail")
3) set_length_change_m(delta_m=distance_m)
4) apply_linestring_extend() OR apply_linestring_shorten()
5) commit_linestring_geometry(line=target_id)

Rule 4: Operation-to-Tool Mapping 
----------------------------------------
- If operation indicates "extend" / "expand" / "lengthen":
  use apply_linestring_extend()
- If operation indicates "shorten" / "trim" / "reduce length":
  use apply_linestring_shorten()
- If operation is unknown or unsupported: output [].

Rule 5: Minimal Sufficiency 
----------------------------------
- Do NOT add extra tool calls unless required by the rules above.
- apply_linestring_extend and apply_linestring_shorten MUST be called with empty args: {}.

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


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


def generate_tool_calls_from_instruction(
    instruction_text: str,
    model,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    messages = build_oneshot_messages(instruction_text)
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
    return tool_calls, raw


def is_exec_success(exec_trace: list, tool_calls: list) -> bool:
    return (
        isinstance(exec_trace, list)
        and isinstance(tool_calls, list)
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

    # baseline: one-shot, optional execute
    parser.add_argument("--execute", action="store_true", default=True, help="Execute tool_calls once (one-shot).")

    parser.add_argument("--raw_keep_chars", type=int, default=0)

    # Level2 prompt suffix
    parser.add_argument("--prompt_suffix", type=str, default="_prompt_level_2.json")

    args = parser.parse_args()

    tokenizer, model = load_model_and_tokenizer(args.model_id, args.device)
    tool_registry = build_tool_registry()

    root_path = Path(args.data_root)
    suffix = args.prompt_suffix
    prompt_files = list(root_path.rglob(f"*{suffix}"))
    print(f"[Info] Found {len(prompt_files)} prompt files matching '*{suffix}'")

    global_start_time = time.time()
    processed_count = 0
    successful_count = 0
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
            out_dir = bbox_dir / "baseline" / sanitize_model_id(args.model_id)
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"[Sample {idx:04d}] out_dir={out_dir}")

            tool_calls, raw_output = generate_tool_calls_from_instruction(
                instruction_text=prompt_text,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            if tool_calls is None:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_parse_tool_calls_level_2.json",
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
            save_json(out_dir / f"{stem_name}_tool_calls_level_2.json", tool_calls)
            print(f"  ├─ tool_calls generated ✔ (n_calls={len(tool_calls)})")

            # ---- load before geojson ----
            before_fc = load_before_geojson_for_prompt(p_path, prompt_suffix=suffix)
            if before_fc is None:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_missing_before_geojson_level_2.json",
                    {
                        "prompt_path": str(p_path),
                        "reason": "cannot locate original {stem}.geojson for this prompt",
                    },
                )
                processed_count += 1
                print(f"  └─ RESULT: MISSING_BEFORE_GEOJSON ✘ ({stem_name})")
                continue

            # ---- execute once ----
            exec_result = execute_tool_calls_on_state(
                state_fc=copy.deepcopy(before_fc),
                tool_calls=tool_calls,
                tool_registry=tool_registry,
            )

            save_json(out_dir / f"{stem_name}_exec_trace_level_2.json", exec_result["trace"])

            if is_exec_success(exec_result["trace"], tool_calls):
                successful_count += 1
                save_json(out_dir / f"{stem_name}_after_level_2.geojson", exec_result["state_fc"])
                print(f"  └─ RESULT: EXEC_SUCCESS ✔ ({stem_name})")
            else:
                fail_count += 1
                save_json(
                    out_dir / f"{stem_name}_fail_exec_level_2.json",
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

    # results summary
    results_dir = Path(os.path.dirname(__file__)) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_subdir = results_dir / "baseline" / sanitize_model_id(args.model_id) / "Level_2"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level2_baseline_running_time.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-2 One-shot Baseline Summary (Prompt-only)\n"
            f"----------------------------------------------\n"
            f"Model: {args.model_id}\n"
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
