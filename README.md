# Medical VLM Hallucination Pilot — Sub-project A

This is my first engineering step of
Sub-project A ("Medical hallucination taxonomy and fine-grained benchmark"):
get a VLM running locally, run it on real medical questions, and record
predictions against ground truth so that hallucination types can later be
analyzed.

## Goal

Answer an engineering question first — *can I run a VLM end-to-end on this
machine and collect structured predictions?* — before scaling to the full
taxonomy (existence, location/laterality, attribute, negation/uncertainty,
count, fabricated comparison) and multiple models.

## My Hardware

- GPU: NVIDIA GeForce RTX 5060 Laptop, 8 GB VRAM (Blackwell, compute capability sm_120)
- CPU: Intel Core Ultra 7 255HX, 32 GB RAM
- OS: Windows

## Model download

The first model is `Qwen/Qwen2.5-VL-3B-Instruct`, loaded in 4-bit NF4 so it
fits comfortably in 8 GB.

## Pilot results (n = 30)

- Rule-based accuracy: 40% (12/30); after manual review: 33.3% (10/30).
- Peak VRAM during 4-bit inference: 2.35 GB out of 7.96 GB.
- Dominant failure mode: false-positive existence (5 of 6 "no" questions were
  answered "yes"), followed by wrong attribute (6) and wrong location (4).
- These are pilot observations on a small subset, not statistically
  significant estimates; see the evaluation report for details and caveats.

## Repository layout

```
med_vlm_work/
├── scripts/
│   ├── 01_check_gpu.py          # GPU / CUDA verification
│   ├── 00_make_test_image.py    # synthetic image with known content
│   ├── 00b_download_model.py    # ModelScope weight download
│   ├── 02_single_inference.py   # one-image, one-question test
│   ├── 03_batch_inference.py    # batch inference over JSONL
│   ├── 04_prepare_vqarad.py     # build a VQA-RAD subset
│   ├── 05_evaluate.py           # rule-based normalization and metrics
│   ├── 06_manual_review.py      # human review + fine-grained error labels
│   ├── 07_measure_vram.py       # VRAM footprint measurement
│   └── download_watchdog.ps1    # timeout/retry wrapper for the download
├── data/                        # local datasets and query files
├── images/                      # ad-hoc test images
└── outputs/                     # logs, predictions, evaluation tables
```

## Roadmap

- [x] Conda environment + cu128 PyTorch, GPU verified
- [x] Single-image inference with 4-bit Qwen2.5-VL-3B
- [x] VQA-RAD subset with ground-truth comparison and manual review
- [x] Errors mapped onto the hallucination taxonomy with per-category counts
- [ ] Scale to 7B and medical-specific models (e.g. LLaVA-Med)
- [ ] Enlarge the subset and add confidence intervals / multi-dataset coverage

## References

- POPE — arXiv:2305.10355
- CHAIR — arXiv:1809.02156
- Med-HALT — arXiv:2307.15343
- LLaVA-Med — arXiv:2306.00890
- Hallucination survey — arXiv:2402.00253
