import argparse
import json
import traceback
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import sys
import os
import re
from typing import Dict, Any, Optional
import time
# 初始化 GPU 环境
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

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=device)
    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


# =========================
# 2) IO Helpers
# =========================
def read_prompt_file(file_path: Path) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt_data = json.load(f)
        return prompt_data.get("prompt", "")
    except Exception as e:
        print(f"[Error] Failed to read {file_path}: {e}")
        return ""


def sanitize_model_id(model_id: str) -> str:
    # 仅用于目录命名，不影响任何任务逻辑
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_id)


def save_output(output_file: Path, plan_json: Dict[str, Any]):
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(plan_json, f, indent=2, ensure_ascii=False)
        print(f"[Info] Results saved to {output_file}")
    except Exception as e:
        print(f"[Error] Failed to save {output_file}: {e}")
        raw_output_file = output_file.with_suffix(".raw.txt")
        try:
            with open(raw_output_file, "w", encoding="utf-8") as f:
                f.write(plan_json.get("raw_output", ""))
            print(f"[Info] Raw output saved to {raw_output_file}")
        except Exception as save_raw_e:
            print(f"[Error] Failed to save raw output: {save_raw_e}")


# =========================
# 3) LLM Generation
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
        eos_token_id=tok.eos_token_id,     # 不要 None
        pad_token_id=pad_id,
        do_sample=bool(do_sample),
    )
    if do_sample:
        gen_kwargs.update(dict(temperature=float(temperature), top_p=float(top_p)))

    with torch.no_grad():
        out_ids = model.generate(**gen_kwargs)

    out = tok.decode(out_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return out.strip()

def clean_raw_text(text: str) -> str:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)  # 查找第一个大括号之间的内容

    if match:
        cleaned_text = match.group(0)  # 获取第一个匹配的 JSON 字符串
    else:
        cleaned_text = ""  # 如果没有匹配到任何有效的 JSON 数据

    return cleaned_text

def _extract_json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    cleaned_text = clean_raw_text(text)
    try:
        return json.loads(cleaned_text)
    except Exception:
        return None


# =========================
# 4) Level-1 Task Planner Prompt
# =========================
def build_level1_planner_messages(prompt_text: str) -> list:
    system = """You are the Task Planner of an urban geospatial editing system.
    You will be given a natural-language instruction for a Level-1 urban geospatial editing task (point editing).

Your job is to read the raw editing instruction and produce an intent-level plan.json.

Requirements:
1) You must infer the editing goal, targets, the editing operation, and any additional spatial requirements or constraints by reading the instruction. Do not ask questions.
2) If the instruction contains a feature ID, you must extract the numeric identifier only.
   - Do NOT include any type prefixes such as "node", "way", or "relation".
   - Do NOT include any non-numeric characters.
   - The target_id field must contain digits only.
   - If no numeric identifier is present, set target_id to null.
3) You must infer movement direction and distance (in meters) if present, and normalize units into meters where possible.
4) Output MUST be a single valid JSON object, no markdown, no comments, no extra text.

Output schema (Level-1 / Point Editing):
{
  "task_meta": {
    "level": 1,
    "task_type": "point_edit",
    "intent_summary": "",
    "confidence": 0.0
  },
  "parsed_from_instruction": {
    "target_id": null,
    "target_semantics": "",
    "operation": "",
    "direction": "",
    "distance_m": null,
    "other_requirements": []
  },
  "objectives": [
    {"name": "primary_goal", "description": ""}
  ],
  "constraints": [
    {"type": "spatial_relation", "description": ""},
    {"type": "validity", "description": "geometry must remain valid"}
  ],
  "dependencies": [],
  "subtasks": [
    {
      "subtask_id": "S1",
      "type": "TARGET_LOCALIZATION",
      "inputs": {"target_id": null, "target_semantics": ""},
      "outputs": {"target_ref": "feature_handle"}
    },
    {
      "subtask_id": "S2",
      "type": "EDIT_INTENT_SPEC",
      "inputs": {"operation": "", "direction": null, "distance_m": null, "requirements": []},
      "outputs": {"edit_spec": "intent_struct"}
    },
    {
      "subtask_id": "S3",
      "type": "VALIDATION_SPEC",
      "inputs": {"checks": ["geometry_validity"]},
      "outputs": {"validation_spec": "checklist"}
    }
  ]
}

Rules for confidence:
- 0.9+ if target_id exists and distance+direction are explicit
- 0.6~0.8 if some fields are implicit but reasonably inferable
- <=0.5 if many key fields are missing or ambiguous
"""

    user = f"""Raw instruction:
{prompt_text}
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_task_plan_from_prompt(
    prompt_text: str,
    model,
    tokenizer,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    """
    One-shot Planner:
    - 让 LLM 自己从 prompt 做信息抽取与意图推理
    - 输出 intent-level plan.json（无 tool calls）
    """
    messages = build_level1_planner_messages(prompt_text)
    raw = generate_plan_text(
        model=model,
        tok=tokenizer,
        messages=messages,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
    )

    plan = _extract_json_from_text(raw)
    if plan is None:
        # 解析失败仍然返回结构化壳 + raw_output，便于你统计失败率
        return None, raw

    if isinstance(plan, dict):
        plan["raw_output"] = raw
    return plan, None


# =========================
# 5) Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="../../data_1_2", help="Root for Level 1 data")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=800)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    # 1) Load Model
    tokenizer, model = load_model_and_tokenizer(args.model_id, args.device)

    # 2) Iterate Data
    root_path = Path(args.data_root)
    suffix = "_prompt_level_1.json"
    prompt_files = list(root_path.rglob(f"*{suffix}"))
    print(f"[Info] Found {len(prompt_files)} prompt files matching '*{suffix}'")

    global_start_time = time.time()
    processed_count = 0
    successful_count = 0
    total_count = 0

    for idx, p_path in enumerate(prompt_files, start=1):
        stem_name = p_path.name.replace(suffix, "")
        total_count += 1
        try:
            prompt_text = read_prompt_file(p_path)

            plan_json, raw_output = generate_task_plan_from_prompt(
                prompt_text=prompt_text,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            result_dir = p_path.parent / sanitize_model_id(args.model_id)
            result_dir.mkdir(parents=True, exist_ok=True)
            output_filename = f"{stem_name}_plan_level_1.json"
            output_file = result_dir / output_filename
            if plan_json is None:
                # ❌ Planner 失败：只保存 raw.txt
                raw_txt = output_file.with_suffix(".raw.txt")
                with open(raw_txt, "w", encoding="utf-8") as f:
                    f.write(raw_output)
                print(f"[Warn] Planner failed. Raw output saved to {raw_txt}")
            else:
                # ✅ Planner 成功：只保存 plan.json
                successful_count += 1
                save_output(output_file, plan_json)

            processed_count += 1
            if idx % 50 == 0:
                print(f"[Info] Progress: {idx}/{len(prompt_files)}")

        except Exception as e:
            print(f"[Fail] Error processing {stem_name}: {e}")
            traceback.print_exc()

    total_time_sec = time.time() - global_start_time
    success_rate = successful_count / total_count if total_count > 0 else 0.0

    print(f"[Done] Processed {processed_count} tasks.")

    results_dir = Path(os.path.dirname(__file__)) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_subdir = results_dir / sanitize_model_id(args.model_id) / "Level_1"
    model_subdir.mkdir(parents=True, exist_ok=True)

    summary_file = model_subdir / "level1_task_planner_running_time.txt"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(
            f"Level-1 Task Planner Summary\n"
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