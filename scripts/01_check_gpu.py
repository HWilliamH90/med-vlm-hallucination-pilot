"""Step 1: verify that PyTorch can see and use the NVIDIA GPU.

Run inside the `medvlm` conda environment:
    python scripts/01_check_gpu.py
"""
import torch

print("Torch version:", torch.__version__)
print("CUDA runtime bundled with torch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    major, minor = torch.cuda.get_device_capability(0)
    print("Compute capability: sm_%d%d" % (major, minor))

    # A tiny real computation on the GPU proves kernels actually run.
    x = torch.randn(1000, 1000, device="cuda")
    y = (x @ x).sum().item()
    print("GPU matmul OK, sum =", round(y, 2))
else:
    raise SystemExit("GPU not visible; stop here and check the PyTorch CUDA build.")
