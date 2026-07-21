"""Device selection: CUDA -> Apple-Silicon MPS -> CPU.

All models in this package are pure 2D convolutional networks
(Conv2d / GroupNorm / GELU / MaxPool2d / ConvTranspose2d), which are fully
supported on every backend, so the same code path runs unchanged on a Linux
GPU box, an Apple-Silicon Mac, or plain CPU.
"""
from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Parameters
    ----------
    prefer : optionally force "cuda", "mps" or "cpu"; raises if unavailable.
    """
    if prefer is not None:
        if prefer == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        if prefer == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available")
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
