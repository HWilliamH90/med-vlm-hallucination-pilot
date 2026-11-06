# Medical VLM Hallucination Pilot — Sub-project A

A local, reproducible pilot for **fine-grained hallucination evaluation of
medical vision-language models (VLMs)**. This is the first engineering step of
Sub-project A ("Medical hallucination taxonomy and fine-grained benchmark"):
get a VLM running locally, run it on real medical questions, and record
predictions against ground truth so that hallucination types can later be
analyzed.

## Goal

Answer an engineering question first — *can we run a VLM end-to-end on this
machine and collect structured predictions?* — before scaling to the full
taxonomy (existence, location/laterality, attribute, negation/uncertainty,
count, fabricated comparison) and multiple models.

## Hardware

- GPU: NVIDIA GeForce RTX 5060 Laptop, 8 GB VRAM (Blackwell, compute capability sm_120)
- CPU: Intel Core Ultra 7 255HX, 32 GB RAM
- OS: Windows

## Environment setup

```powershell
# 1. Create an isolated environment (Python 3.11).
conda create -n medvlm python=3.11

# 2. Install a CUDA 12.8 build of PyTorch. The cu128 wheels are required for
#    Blackwell (sm_120); older cu121/cu124 builds have no sm_120 kernels.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3. Install inference and data dependencies.
pip install transformers accelerate bitsandbytes datasets modelscope
```

Verify the GPU:

```powershell
python scripts/01_check_gpu.py
```

## Model download

The first model is `Qwen/Qwen2.5-VL-3B-Instruct`, loaded in 4-bit NF4 so it
fits comfortably in 8 GB. On this network the HuggingFace weight CDN is very
slow, so the weights are downloaded once from ModelScope (same model, fast
domestic CDN):

```powershell
python scripts/00b_download_model.py
# This prints MODEL_PATH=<local folder>; export it for the inference scripts.
$env:MODEL_PATH = "<printed local folder>"
```

## Usage

```powershell
# Point the scripts at the locally downloaded model.
$env:MODEL_PATH = "<printed local folder>"

# Optional synthetic sanity-check image (one red circle, one blue rectangle).
python scripts/00_make_test_image.py
python scripts/02_single_inference.py

# Full VQA-RAD pilot pipeline.
python scripts/04_prepare_vqarad.py --n 30
python scripts/03_batch_inference.py --input data/vqarad_queries.jsonl `
    --output outputs/vqarad_results.jsonl
python scripts/05_evaluate.py            # rule-based normalization + accuracy
python scripts/06_manual_review.py       # human review + taxonomy labels
python scripts/07_measure_vram.py        # report peak VRAM during inference
```

If the download stalls on a long-lived connection, run the watchdog, which
restarts the download with a per-attempt timeout and resumes from partial files:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_watchdog.ps1
```

## Pilot results (n = 30, descriptive only)

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
