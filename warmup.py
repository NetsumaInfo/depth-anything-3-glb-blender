"""Pre-download the model weights and run a tiny CUDA inference to validate the install."""
import os
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Reuse the app loader so the bf16 / forward setup matches exactly.
import app


def main():
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
    assert torch.cuda.is_available(), "CUDA not available — torch build is wrong."
    print(f"downloading + loading {app.MODEL_ID} ...")
    model = app.get_model()

    dummy = (np.random.rand(360, 480, 3) * 255).astype(np.uint8)
    pred = model.inference([dummy], process_res=392, export_dir=None)
    print("OK depth shape:", pred.depth.shape, "| dtype:", pred.depth.dtype)
    print("VRAM peak (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2))


if __name__ == "__main__":
    main()
