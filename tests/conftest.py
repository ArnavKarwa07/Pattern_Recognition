"""Shared fixtures. All audio here is synthetic and exists only to test software."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from speechdeepfake.config import load_config


@pytest.fixture
def cfg(tmp_path: Path):
    """Default config with every output path redirected into a temp directory."""
    return load_config(
        None,
        overrides={
            "paths": {"data_dir": str(tmp_path / "data"), "models_dir": str(tmp_path / "models"),
                      "results_dir": str(tmp_path / "results"), "reports_dir": str(tmp_path / "reports")},
        },
    )


@pytest.fixture
def wav_file(tmp_path: Path):
    """Factory writing a WAV file and returning its path."""

    def make(name: str = "a.wav", seconds: float = 1.0, sr: int = 16000, channels: int = 1, value=None) -> Path:
        n = int(seconds * sr)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((n, channels)).astype(np.float32) * 0.1 if value is None else np.full((n, channels), value, np.float32)
        p = tmp_path / name
        sf.write(p, x, sr, subtype="FLOAT")
        return p

    return make
