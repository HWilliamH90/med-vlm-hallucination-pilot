"""Step 6: apply manual review labels and map errors onto the project taxonomy.

The automatic normalizer in 05_evaluate.py has known edge cases (e.g. it
cannot tell a real answer from an abstention, and it misses synonyms such as
"fat" vs "adipose tissue"). This script records the human-reviewed verdict
and a fine-grained error category for every item, then writes a final table
and counts.

Error categories are aligned with Sub-project A's taxonomy:
  correct
  abstention                 # model declines to answer / cannot determine
  false_positive_existence   # claims a finding that is absent (GT says no)
  false_negative_existence   # calls an abnormal finding normal
  wrong_location             # wrong anatomy / laterality / plane location
  wrong_attribute            # wrong entity, pathology, property or count

Run:
    python scripts/06_manual_review.py
"""
import csv
import json

# Keyed by question; values are (manual_correct, category, note).
REVIEW = {
    "do the kidneys appear to be normal?": (True, "correct", "negated abnormality = effectively yes"),
    "what is the dark structure underneath the skin?": (False, "wrong_attribute", "said intestines, GT is fat"),
    "are the clavicles fractured?": (False, "abstention", "says modality cannot determine, not a real 'no'"),
    "is there enlargement of the pulmonary vasculature?": (False, "false_positive_existence", "fabricated congestion"),
    "is the lesion hyperdense?": (False, "false_positive_existence", "fabricated hyperdensity"),
    "is the abdominal aorta enlarged?": (False, "false_positive_existence", "fabricated enlargement"),
    "what is the hypodensity in the liver?": (False, "wrong_attribute", "listed guesses, missed gallbladder"),
    "is the colon more prominent on the patient's right or left side?": (True, "correct", "left"),
    "is this a ct?": (True, "correct", "yes"),
    "the image is taken in what plane?": (True, "correct", "axial"),
    "where is the lesion located?": (False, "wrong_location", "said pons, GT right cerebellopontine angle"),
    "does this represent adequate inspiratory effort?": (False, "abstention", "gives criteria but no clear verdict"),
    "is the trachea normal?": (False, "false_negative_existence", "called abnormal trachea normal"),
    "what skeletal joint is seen in this image?": (False, "wrong_attribute", "denied any joint, GT sacroiliac"),
    "what pathology does the image represent": (False, "wrong_attribute", "missed kidney cyst"),
    "what abnormality is present?": (False, "wrong_location", "said neck mass, GT cerebellar bleeding"),
    "is this a normal image?": (False, "abstention", "says cannot determine, not a real 'no'"),
    "is there a chest tube placed?": (False, "false_positive_existence", "fabricated chest tube"),
    "what is the mass most likely?": (False, "abstention", "declines to identify, GT kidney cyst"),
    "in which brain area is the lesion located?": (False, "wrong_location", "said parietal, GT right cerebellopontine angle"),
    "is the diaphragm clearly visualized on both sides of the thorax?": (True, "correct", "yes"),
    "can fluids be highlighted with this modality?": (True, "correct", "yes"),
    "where is the abnormal mass located with respect to the lungs?": (False, "wrong_location", "said mediastinum, GT right upper lobe"),
    "what part of the body is being imaged here?": (True, "correct", "abdomen"),
    "what structures are visible in this image?": (False, "wrong_attribute", "missed skull/cartilage/medulla"),
    "what cross section of the brain is this image?": (True, "correct", "axial"),
    "what is the condition in which the ventricles are enlarged as seen in this image?": (True, "correct", "hydrocephalus"),
    "the tissue within the hyperattenuating ovoid mass adjacent to the ascending colon is similar in density to what tissue normally found?": (True, "correct", "said fat; synonym of adipose tissue"),
    "are there multiple or just 1 metastatic focus?": (False, "wrong_attribute", "said multiple, GT one"),
    "is there mass effect?": (False, "false_positive_existence", "fabricated mass effect"),
}


def main():
    source = "outputs/evaluation_table.csv"
    final_csv = "outputs/final_reviewed_table.csv"
    summary_json = "outputs/final_review_summary.json"

    with open(source, "r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        manual_correct, category, note = REVIEW[row["question"]]
        row["manual_correct"] = manual_correct
        row["error_category"] = category
        row["review_note"] = note

    fields = list(rows[0].keys())
    with open(final_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    correct = sum(r["manual_correct"] for r in rows)
    category_counts = {}
    for row in rows:
        category_counts[row["error_category"]] = category_counts.get(
            row["error_category"], 0
        ) + 1

    summary = {
        "n": n,
        "manual_correct": correct,
        "manual_accuracy": round(correct / n, 3),
        "error_category_counts": category_counts,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print("Saved", final_csv)


if __name__ == "__main__":
    main()
