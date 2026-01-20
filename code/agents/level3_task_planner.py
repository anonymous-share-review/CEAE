from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# =========================
# 0) Path bootstrap
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


# =========================
# 1) Model Loading
# =========================
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


# =========================
# 2) IO helpers
# =========================
def read_prompt_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("prompt", "")
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return ""


def sanitize_model_id(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


def save_plan_json(output_file: Path, plan_json: Dict[str, Any]):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(plan_json, f, indent=2, ensure_ascii=False)


# =========================
# 3) LLM generation (text only)
# =========================
def generate_plan_text(
    model,
    tok,
    messages,
    max_new_tokens: int = 800,
    do_sample: bool = False,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> str:
    """
    只负责从 messages 生成文本，不负责解析 JSON
    """
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
        attention_mask = torch.ones_like(
            input_ids, dtype=torch.long, device=input_ids.device
        )

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

    out = tok.decode(out_ids[0][input_ids.shape[-1] :], skip_special_tokens=True)
    return out.strip()


# =========================
# 4) JSON extraction
# =========================
def _clean_raw_text_to_first_json_object(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else ""


def extract_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned = _clean_raw_text_to_first_json_object(text)
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except Exception:
        return None


# =========================
# 5) Task Planner messages (placeholder)
# =========================
def build_level3_planner_messages(prompt_text: str) -> list:
    system = r"""You are the Task Planner of an urban geospatial editing system.

You will be given a natural-language instruction for a Level-3 urban geospatial editing task (polygon editing).
Your job is to read the instruction and produce an intent-level plan.json.

Important distribution hint:
- Most Level-3 instructions describe a GLOBAL area adjustment over MULTIPLE green polygons,
  with the goal of increasing the TOTAL green area by an approximate percentage
  (e.g., "increase green space by about 30%").
- The task is NOT to scale a single polygon independently,
  but to achieve a cumulative area increase across eligible green polygons.

Requirements:
1) Infer the overall editing goal, target scope, target area ratio, and constraints from the instruction.
   Do NOT ask questions.
2) If a percentage or ratio is mentioned (e.g., "30%", "about one third"),
   normalize it into a decimal ratio (e.g., 0.30) and store it as target_ratio.
   - If no explicit ratio is present, set target_ratio to null and explain this in intent_summary.
3) Assume the editing scope applies to ALL existing green polygons unless explicitly restricted.
4) You must clearly distinguish HARD constraints from SOFT preferences:
   - Hard constraints must never be violated.
   - Soft preferences may be satisfied only if feasible.
5) Output MUST be a single valid JSON object, no markdown, no comments, no extra text.

Output schema (Level-3 / Polygon Editing):
{
  "task_meta": {
    "level": 3,
    "task_type": "green_polygon_area_increase",
    "intent_summary": "",
    "confidence": 0.0
  },
  "parsed": {
    "target_ratio": null,
    "tolerance": 0.02,
    "spacing_m_soft": 8.0
  },
  "scope": {
    "targets": "all_green_polygons",
    "geometry_types": ["Polygon", "MultiPolygon"]
  },
  "constraints": {
    "hard": [
      "no_overlap_non_green",
      "no_merge_green_polygons",
      "no_deletion",
      "geometry_valid"
    ],
    "soft": [
      "prefer_keep_spacing"
    ]
  },
  "stop": {
    "metric": "delta_green_area",
    "target": "A0 * target_ratio",
    "tolerance": 0.02
  },
  "subtasks": [
    {
      "id": "S1",
      "type": "COLLECT_GREEN_TARGETS"
    },
    {
      "id": "S2",
      "type": "MEASURE_BASE_GREEN_AREA"
    },
    {
      "id": "S3",
      "type": "SET_AREA_TARGET",
      "inputs": {
        "target_ratio": null
      }
    },
    {
      "id": "S4",
      "type": "ITERATIVE_EXPAND_UNDER_CONSTRAINTS",
      "mode": "multi_element_cumulative"
    },
    {
      "id": "S5",
      "type": "EVALUATE_ACE",
      "ace": "|ΔA_green − (A0 × target_ratio)| / (A0 × target_ratio)"
    }
  ]
}

Rules for confidence:
- 0.9+ if a numeric target_ratio is explicit and constraints are clearly stated
- 0.6~0.8 if target_ratio is implicit but reasonably inferable
- <=0.5 if the goal or constraints are ambiguous
""".strip()

    user = f"""Raw instruction:
{prompt_text}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_plan_from_prompt(
    prompt_text: str,
    model,
    tok,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    messages = build_level3_planner_messages(prompt_text)
    raw = generate_plan_text(
        model=model,
        tok=tok,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )
    plan = extract_json_from_text(raw)
    if isinstance(plan, dict):
        plan["raw_output"] = raw
        return plan, raw
    return None, raw


# =========================
# 6) File layout mapping
# =========================
def plan_path_from_prompt_path(
    prompt_path: Path,
    data_root: Path,
    model_id: str,
) -> Path:
    """
    Input prompt:
      data_root/city/bbox/<stem>_prompt.json

    Output plan:
      data_root/city/bbox/<sanitized_model_id>/<stem>_plan.json
    """
    rel = prompt_path.resolve().relative_to(data_root.resolve())
    parts = rel.parts
    sanitized = sanitize_model_id(model_id)

    name = prompt_path.name
    if name.endswith("_prompt.json"):
        stem = name[: -len("_prompt.json")]
    else:
        stem = prompt_path.stem

    out_name = f"{stem}_plan.json"

    if len(parts) >= 3:
        city = parts[0]
        bbox = parts[1]
        return (data_root / city / bbox / sanitized / out_name).resolve()

    return (prompt_path.parent / sanitized / out_name).resolve()


# =========================
# 7) Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data", help="Root containing city/bbox/*_prompt.json")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    # 1) Load model
    tok, model = load_model_and_tokenizer(args.model_id, args.device)

    # 2) Collect all *_prompt.json
    prompt_files = sorted(data_root.rglob("*_prompt.json"))
    print(f"[Info] Found {len(prompt_files)} prompt files matching '*_prompt.json' under: {data_root}")

    global_start_time = time.time()
    processed_count = 0
    successful_count = 0
    total_count = 0

    for idx, p_path in enumerate(prompt_files, start=1):
        total_count += 1
        try:
            prompt_text = read_prompt_file(p_path)
            if not prompt_text.strip():
                raise RuntimeError("Empty prompt field.")

            plan, raw = generate_plan_from_prompt(
                prompt_text=prompt_text,
                model=model,
                tok=tok,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            out_path = plan_path_from_prompt_path(p_path, data_root, args.model_id)

            if plan is None:
                # ❌ planner failed: save raw only
                out_path.parent.mkdir(parents=True, exist_ok=True)
                raw_path = out_path.with_suffix(".raw.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(raw)
                print(f"[FAIL] ({idx}/{len(prompt_files)}) {out_path.name} | json_parse_failed")
            else:
                # ✅ planner success: save plan.json
                successful_count += 1
                save_plan_json(out_path, plan)
                print(f"[OK]   ({idx}/{len(prompt_files)}) {out_path.name} | planner_success")

            processed_count += 1
            if idx % 50 == 0:
                print(f"[Info] Progress: {idx}/{len(prompt_files)}")

        except Exception as e:
            print(f"[ERROR] ({idx}/{len(prompt_files)}) {p_path} | exception: {e}")
            traceback.print_exc()

    total_time_sec = time.time() - global_start_time
    success_rate = successful_count / total_count if total_count > 0 else 0.0

    print(f"[Done] Processed {processed_count} tasks.")

    # ===== summary file (和 level2 风格一致) =====
    results_dir = Path(os.path.dirname(__file__)) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_subdir = results_dir / sanitize_model_id(args.model_id) / "Level_3"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level3_task_planner_running_time.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-3 Task Planner Summary\n"
            f"-----------------------------\n"
            f"Model: {args.model_id}\n"
            f"Total tasks: {total_count}\n"
            f"Successful tasks: {successful_count}\n"
            f"Success rate: {success_rate:.4f}\n"
            f"Total running time (seconds): {total_time_sec:.2f}\n"
        )

    print(f"[Info] Planner runtime summary saved to {summary_file}")


if __name__ == "__main__":
    main()