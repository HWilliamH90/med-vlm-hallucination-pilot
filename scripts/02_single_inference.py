"""Step 2: single-image question answering with a 4-bit Qwen2.5-VL model.

This is the smallest end-to-end VLM test: load one model, read one image,
ask one question, print the answer.

Preparation:
    1. Put an image at  images/test.jpg
    2. Run:  python scripts/02_single_inference.py
"""
import os

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

# Point MODEL_PATH to the local ModelScope folder to avoid re-downloading.
MODEL_ID = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct")
IMAGE_PATH = "images/test.jpg"

# 4-bit NormalFloat quantization keeps a 3B model well inside 8 GB VRAM.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model (first run downloads the weights, please wait)...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe what you see in this image in one short sentence."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

with torch.no_grad():
    generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)

# Strip the prompt tokens so only the new answer is decoded.
trimmed = [out[len(inp):] for out, inp in zip(generated, inputs.input_ids)]
answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
print("\nANSWER:", answer)
