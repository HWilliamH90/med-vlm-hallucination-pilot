"""Step 5: evaluate predictions against ground truth on the small VQA-RAD subset.

The general-purpose model answers in full sentences, while VQA-RAD ground
truth is short (yes/no or a single concept). We therefore normalize both
sides before comparing:

- yes/no questions: extract the polarity from the prediction, including
  "no signs of abnormality" style answers that are effectively affirmative.
- open questions: check whether the ground-truth concept appears in the
  prediction (case-insensitive, punctuation-insensitive).

Automatic labels are written out for manual review; the rule-based polarity
classifier is deliberately conservative.

Run:
    python scripts/05_evaluate.py --input outputs/vqarad_results.jsonl
"""
import argparse
import csv
import json
import re


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_sentence(text):
    parts = re.split(r"(?<=[.!?])\s+|[\n]", text.strip())
    return parts[0].lower()


def classify_yes_no(prediction):
    low = prediction.lower()
    first = first_sentence(prediction)

    if re.match(r"^yes\b", first):
        return "yes"
    if re.match(r"^no\b", first):
        return "no"

    # "does not show any abnormality", "no clear signs of abnormality",
    # "appears to be normal" are affirmative answers to a normal/adequate question.
    if re.search(
        r"does not show|no (clear |visible )?signs? of|without any .*abnormal|"
        r"appear to be normal|appears? to be normal|appears? normal",
        low,
    ):
        return "yes"

    if re.search(r"cannot|can't|does not|is not|\bnot\b|\bno\b", low):
        return "no"
    if re.search(r"\byes\b", low):
        return "yes"
    return "unknown"


def open_answer_matches(ground_truth, prediction):
    gt = ground_truth.lower().strip()
    pred = prediction.lower()
    # Ground truth may contain several concepts separated by tab/comma/semicolon.
    terms = [t.strip() for t in re.split(r"[\t,;]", gt) if t.strip()]
    return any(term in pred for term in terms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/vqarad_results.jsonl")
    parser.add_argument("--csv", default="outputs/evaluation_table.csv")
    parser.add_argument("--summary", default="outputs/evaluation_summary.json")
    args = parser.parse_args()

    records = load_jsonl(args.input)
    rows = []

    for record in records:
        gt = record["ground_truth"].strip().lower()
        is_yes_no = gt in ("yes", "no")

        if is_yes_no:
            normalized = classify_yes_no(record["prediction"])
            question_type = "yes/no"
        else:
            normalized = "concept_match" if open_answer_matches(
                record["ground_truth"], record["prediction"]
            ) else "concept_miss"
            question_type = "open"

        correct = (normalized == gt) if is_yes_no else (normalized == "concept_match")

        rows.append({
            "question": record["question"],
            "type": question_type,
            "ground_truth": gt,
            "normalized_prediction": normalized,
            "correct": correct,
            "raw_prediction": record["prediction"],
        })

    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def stats(subset):
        if not subset:
            return {"n": 0, "correct": 0, "accuracy": None}
        correct = sum(r["correct"] for r in subset)
        return {
            "n": len(subset),
            "correct": correct,
            "accuracy": round(correct / len(subset), 3),
        }

    yes_no_rows = [r for r in rows if r["type"] == "yes/no"]
    open_rows = [r for r in rows if r["type"] == "open"]
    summary = {
        "overall": stats(rows),
        "yes_no": stats(yes_no_rows),
        "open_ended": stats(open_rows),
        "unknown_polarity": sum(
            r["normalized_prediction"] == "unknown" for r in yes_no_rows
        ),
    }

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("\nSaved table:", args.csv)
    print("Saved summary:", args.summary)


if __name__ == "__main__":
    main()
