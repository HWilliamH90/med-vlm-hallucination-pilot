"""Step 7: measure the VRAM footprint of the 4-bit model during inference."""
import os

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

MODEL_ID = os.environ.get(
    "MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"
)

torch.cuda.reset_peak_memory_stats()
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    ),
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)

image = Image.open("images/test.jpg").convert("RGB")
messages = [{"role": "user", "content": [
    {"type": "image", "image": image},
    {"type": "text", "text": "What is in this image?"},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
with torch.no_grad():
    model.generate(**inputs, max_new_tokens=64, do_sample=False)

print("Peak VRAM (GB):", round(torch.cuda.max_memory_allocated() / 1024**3, 2))
print("Total GPU VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
