import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                 # 开启4bit量化
    bnb_4bit_quant_type="nf4",         # nf4通常更好；也可 "fp4"
    bnb_4bit_use_double_quant=False,    # 双重量化省显存
    bnb_4bit_compute_dtype=torch.bfloat16  # 计算用fp16（也可 torch.bfloat16 试试）
)

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:1",          # ✅ 自动切分/多卡/CPU offload
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",  # 可选：装了flash-attn再开
)


pipe = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tok,
device_map="cuda:1",
)


