import os
import sys
import json
import argparse
import re
from pathlib import Path

# --- 路径设置 ---
HERE = os.path.dirname(__file__)
CODE_DIR = os.path.abspath(os.path.join(HERE, ".."))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

# 引入你的工具和模型
from tools.utils import iter_geojson_paths
from tools.llm_chat import pipe, tok

# --- System Prompt: 核心人设 ---
SYSTEM_PROMPT = """
You are a senior GIS Data Analyst creating tasks for an automated spatial editing agent.
Your goal is to write clear, professional, and precise English instructions based on the structured data provided.

**Structure of the Output Prompt:**
1. **Identification**: Start by explicitly identifying the feature using its Type and Feature ID.
   - Example: "Locate the cycling path (ID: way/67890)..." or "Find the bus stop (ID: node/12345)..."
2. **Action**: Clearly state the geometric operation.
   - For Lines: Specify the operation (Extend/Shorten) and the specific end (Start/Head or End/Tail).
   - For Points: Specify the direction and distance.
   - Use natural approximation for numbers (e.g., "about 20 meters").
3. **Context**: Briefly mention the context (e.g., "to correct the path length", "to align with the intersection").

**Rules:**
- Return EXACTLY one JSON object with ONLY the key "prompt".
- Format: {"prompt": "Your generated text here"}
- Do NOT use internal variable names (like "change_meters" or "head").
- Keep the tone objective and directive.

**Input Data:**
(The user will provide the task details in JSON format)
"""


def extract_json_from_response(response_text):
    """清洗模型输出，提取 JSON"""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        match = re.search(r"(\{.*\})", response_text, re.DOTALL)
        json_str = match.group(1) if match else response_text

    try:
        data = json.loads(json_str)
        return data.get("prompt", "")
    except json.JSONDecodeError:
        return ""


def call_llm_generator(payload):
    """调用本地 LLM"""
    user_message = json.dumps(payload, indent=2)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
    )

    response_text = outputs[0]["generated_text"][len(prompt):].strip()
    return extract_json_from_response(response_text)


# --- 核心：针对不同任务构建 Payload ---

def construct_payload_level1(metadata):
    """构建 Level 1 (点位移动) Payload"""
    stats = metadata.get('_edit_stats', {})
    payload = {
        "task_category": "Point Feature Correction",
        "target_object": {
            "feature_id": stats.get('target_feature_id', 'unknown'),
            "feature_type": stats.get('target_type', 'point feature'),
        },
        "edit_operation": {
            "action": "Relocate / Shift",
            "distance": f"{int(stats.get('move_distance_m', 0))} meters",
            "direction": stats.get('cardinal_direction', 'unknown'),
            "constraint": "Maintain parallelism with the road"
        }
    }
    return payload


def construct_payload_level2(metadata):
    """
    构建 Level 2 (道路几何修改) Payload
    这里是专门处理“线任务”的逻辑
    """
    stats = metadata.get('_edit_stats', {})

    # 1. 提取基础信息
    f_id = stats.get('target_feature_id', 'unknown')
    action_raw = stats.get('action', 'modify')  # extend / shorten
    end_point_raw = stats.get('end_point', 'end')  # head / tail
    meters = int(stats.get('change_meters', 0))

    # 2. 语义转换 (让模型更好理解)
    # Head -> Starting point, Tail -> Ending point
    if end_point_raw == 'head':
        loc_desc = "Starting point (Head)"
    else:
        loc_desc = "Ending point (Tail)"

    # 3. 组装 Payload
    payload = {
        "task_category": "Linear Feature Geometry Modification",
        "target_object": {
            "feature_id": f_id,
            "feature_type": "cycling path",
            "geometry_type": "LineString"
        },
        "edit_operation": {
            "action": action_raw.capitalize(),  # Extend / Shorten
            "target_location": loc_desc,  # 明确告诉它是头还是尾
            "amount": f"approximately {meters} meters",
            "reasoning": "To correct geometry or connectivity issues"
        },
        "style_requirement": "Identify object by ID first, then specify the modification at the correct end."
    }
    return payload


def save_prompt_file(output_path: Path, prompt_text: str):
    """
    保存 Prompt 到指定的 JSON 文件路径
    """
    content = {
        "prompt": prompt_text
    }
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, indent=2, ensure_ascii=False)
        print(f"  -> [SAVE] 已保存至: {output_path.name}")
    except Exception as e:
        print(f"  -> [ERROR] 保存失败: {e}")

# --- 主流程 ---

def process_label_file(label_path, output_path, level):
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ERROR] 读取失败 {label_path}: {e}")
        return

    metadata = data.get('metadata', {})
    intent = metadata.get('_edit_intent', '')

    # 简单校验
    if level not in intent:
        return

    print(f"Processing Label: {label_path.name}")

    # 分发逻辑
    if level == 'level1':
        payload = construct_payload_level1(metadata)
    elif level == 'level2':
        payload = construct_payload_level2(metadata)

    # 生成
    generated_prompt = call_llm_generator(payload)

    if generated_prompt:
        print(f"  -> [PROMPT] {generated_prompt}")
        save_prompt_file(output_path, generated_prompt)
    else:
        print("  -> [FAIL] 生成为空")


def main(args):
    root_directory = Path(args.dir).resolve()

    # 提取 level 数字 (e.g. 'level1' -> '1')
    level_num = args.level[-1]

    print(f"模式: {args.level.upper()} | 目标目录: {root_directory}")

    count = 0
    for city, bbox, source_path in iter_geojson_paths(root_directory):
        source_path = Path(source_path)

        label_filename = f"{source_path.stem}_label_{level_num}.geojson"
        label_path = source_path.parent / label_filename

        output_filename = f"{source_path.stem}_prompt_level_{level_num}.json"
        output_path = source_path.parent / output_filename

        if "_label_" in source_path.name or "_prompt_" in source_path.name:
            continue

        if label_path.exists():
            process_label_file(label_path, output_path, args.level)
            count += 1

    if count == 0:
        print(f"未处理任何文件。请检查目录下是否存在对应的 *_label_{level_num}.geojson 文件。")
    else:
        print(f"\n全部完成，共生成 {count} 个 prompt 文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Natural Language Prompts using LLM")
    parser.add_argument('--level', type=str, choices=['level1', 'level2'])
    parser.add_argument('--dir', type=str, default="../../data_1_2")

    args = parser.parse_args()


    main(args)