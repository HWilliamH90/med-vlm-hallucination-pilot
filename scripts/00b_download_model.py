"""Step 0b: download the model from ModelScope.

HuggingFace redirects the large weight files to a US CDN that is very slow
from this network. ModelScope hosts the same Qwen model with a fast domestic
CDN, so I download once here and then load the model from the local path.

Run:
    python scripts/00b_download_model.py
"""
from modelscope import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

local_path = snapshot_download(MODEL_ID)
print("MODEL_PATH:", local_path)
