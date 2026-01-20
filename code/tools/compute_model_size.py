from typing import List, Dict, Any
from transformers import AutoConfig, AutoModel
import importlib

def human_params(n: int) -> str:
    # 以 B/M/K 显示
    if n >= 10**9:
        return f"{n/10**9:.2f}B"
    if n >= 10**6:
        return f"{n/10**6:.2f}M"
    if n >= 10**3:
        return f"{n/10**3:.2f}K"
    return str(n)

def has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None

def count_params_from_config(model_id: str) -> Dict[str, Any]:
    """
    通过 config 构建空模型（meta device），统计参数量。
    不下载权重文件；只需要能拿到 config（通常会联网或走本地缓存）。
    """
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    if has_accelerate():
        from accelerate import init_empty_weights
        with init_empty_weights():
            model = AutoModel.from_config(cfg, trust_remote_code=True)
    else:
        # fallback：没有 accelerate 的话，还是会在 CPU 上分配参数（可能比较大）
        # 但不会下载权重，只是从 config 构建。
        model = AutoModel.from_config(cfg, trust_remote_code=True)

    # transformers 的 num_parameters
    num = model.num_parameters()
    return {
        "model_id": model_id,
        "arch": getattr(cfg, "architectures", None),
        "model_type": getattr(cfg, "model_type", None),
        "num_params": int(num),
        "num_params_h": human_params(int(num)),
    }

def main(model_ids: List[str]):
    rows = []
    for mid in model_ids:
        try:
            info = count_params_from_config(mid)
            rows.append(info)
            print(f"[OK] {mid} -> {info['num_params_h']} ({info['num_params']})")
        except Exception as e:
            rows.append({"model_id": mid, "error": str(e)})
            print(f"[FAIL] {mid} -> {e}")

    # pretty print table
    print("\n=== Summary ===")
    header = ["model_id", "model_type", "arch", "num_params_h", "num_params", "error"]
    for h in header:
        print(f"{h:28s}", end="")
    print()
    print("-" * 160)
    for r in rows:
        print(f"{str(r.get('model_id','')):28.28s}"
              f"{str(r.get('model_type','')):28.28s}"
              f"{str(r.get('arch','')):28.28s}"
              f"{str(r.get('num_params_h','')):28.28s}"
              f"{str(r.get('num_params','')):28.28s}"
              f"{str(r.get('error','')):28.28s}")
    return rows

if __name__ == "__main__":
    model_ids = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen3-8B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    ]
    main(model_ids)
