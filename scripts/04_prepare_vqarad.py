"""Step 4: load VQA-RAD and build a small local subset for batch inference.

VQA-RAD is small (a few hundred radiology images), so even with a slower CDN
the download is manageable. Set HF_ENDPOINT before running if needed:
    $env:HF_ENDPOINT = "https://hf-mirror.com"

The script saves images locally and writes a JSONL file that can be fed
directly into 03_batch_inference.py.

Run:
    python scripts/04_prepare_vqarad.py --n 30
"""
import argparse
import json
import os
import random

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30, help="number of examples")
    parser.add_argument("--out-dir", default="data/vqarad")
    parser.add_argument("--jsonl", default="data/vqarad_queries.jsonl")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_dataset("flaviagiammarino/vqa-rad")
    print("Dataset structure:", dataset)
    split = "test" if "test" in dataset else list(dataset.keys())[0]
    rows = dataset[split]
    print("Columns:", rows.column_names)

    indices = list(range(len(rows)))
    random.seed(42)
    random.shuffle(indices)
    indices = indices[: args.n]

    records = []
    for k, i in enumerate(indices):
        row = rows[i]
        image = row["image"].convert("RGB")
        image_path = os.path.join(args.out_dir, f"{k:03d}.jpg")
        image.save(image_path, quality=95)

        record = {
            "image": image_path,
            "question": row["question"],
            "ground_truth": row["answer"],
        }
        records.append(record)

    with open(args.jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} examples to {args.jsonl}")


if __name__ == "__main__":
    main()
