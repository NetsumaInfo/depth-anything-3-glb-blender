"""Pre-download the model weights and run a tiny CUDA inference to validate the install."""
import os
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from depth_anything_3.api import DepthAnything3

MODEL_ID = os.environ.get("DA3_MODEL", "depth-anything/DA3-LARGE-1.1")


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not available — torch build is wrong."
    print(f"downloading + loading {MODEL_ID} ...")
    model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
    model.device = torch.device("cuda")

    dummy = (np.random.rand(360, 480, 3) * 255).astype(np.uint8)
    pred = model.inference([dummy], process_res=392, export_dir=None)
    print("OK depth shape:", pred.depth.shape, "| dtype:", pred.depth.dtype)
    print("VRAM peak (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    main()
