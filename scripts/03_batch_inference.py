"""Step 3: batch inference over a JSONL file of image-question pairs.

Input format (one JSON object per line, UTF-8):
    {"image": "images/x.jpg", "question": "Is there a nodule?"}
Any extra fields (e.g. ground-truth answer) are preserved in the output.

Run:
    python scripts/03_batch_inference.py --input data/queries.jsonl --output outputs/results.jsonl
"""
import argparse
import json
import os
import time

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ask_one(model, processor, image, question, max_new_tokens):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )

    trimmed = [out[len(inp):] for out, inp in zip(generated, inputs.input_ids)]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--model-id",
        default=os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-VL-3B-Instruct"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    records = load_records(args.input)
    results = []

    for i, record in enumerate(records):
        image = Image.open(record["image"]).convert("RGB")
        start = time.time()
        answer = ask_one(
            model, processor, image, record["question"], args.max_new_tokens
        )
        elapsed = round(time.time() - start, 2)

        out_record = dict(record)
        out_record["prediction"] = answer
        out_record["seconds"] = elapsed
        results.append(out_record)
        print(f"[{i + 1}/{len(records)}] {elapsed}s -> {answer[:70]}")

    with open(args.output, "w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Saved", args.output)


if __name__ == "__main__":
    main()
