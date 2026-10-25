"""Seeding, device selection and environment/hardware reporting."""

from __future__ import annotations

import os
import platform
import random
import sys
from importlib import metadata
from typing import Any

import numpy as np

_PACKAGES = [
    "numpy", "scipy", "pandas", "scikit-learn", "librosa", "soundfile", "torch",
    "transformers", "pyarrow", "matplotlib", "PyYAML", "joblib",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover
        pass


def resolve_device(requested: str = "auto") -> str:
    """Return "cuda" or "cpu". Raises if CUDA was explicitly requested but is unavailable."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "device: cuda was requested but torch.cuda.is_available() is False. "
            f"Installed torch build: {torch.__version__}. Install a CUDA build of torch "
            "(see README) or set device: auto/cpu."
        )
    return requested


def package_versions() -> dict[str, str]:
    out = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def environment_info() -> dict[str, Any]:
    """Hardware and software description stored with every result."""
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "packages": package_versions(),
    }
    try:
        import torch

        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
    except ImportError:  # pragma: no cover
        info["cuda_available"] = False
    return info
