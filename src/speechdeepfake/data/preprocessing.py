"""Shared audio preprocessing used by all three pipelines.

Order of operations (documented in docs/methodology.md):

1. decode, down-mix to mono, resample to ``sample_rate``;
2. reject empty or non-finite signals (:class:`AudioError`);
3. optional silence trimming;
4. optional amplitude normalisation;
5. fixed-length crop/pad to ``max_duration``. Evaluation crops are
   deterministic (``eval_crop``); random crops are used only when
   ``train=True`` *and* ``train_random_crop`` is enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from speechdeepfake.config import AudioConfig


class AudioError(RuntimeError):
    """A recording that cannot be used (missing, undecodable, empty or non-finite)."""


def load_audio(path: Path | str, sample_rate: int = 16000, mono: bool = True) -> np.ndarray:
    """Decode an audio file to float32 at ``sample_rate``.

    Raises:
        AudioError: file missing, undecodable, empty, or containing NaN/Inf.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"missing file: {path}")
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:  # soundfile raises several error types
        raise AudioError(f"undecodable audio: {path} ({type(e).__name__}: {e})") from e
    if data.size == 0:
        raise AudioError(f"empty signal: {path}")
    if not np.isfinite(data).all():
        raise AudioError(f"non-finite samples (NaN/Inf): {path}")
    x = data.mean(axis=1) if mono else data[:, 0]
    if sr != sample_rate:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=sample_rate, res_type="soxr_hq")
    return np.ascontiguousarray(x, dtype=np.float32)


def target_num_samples(cfg: AudioConfig) -> Optional[int]:
    return None if cfg.max_duration is None else int(round(cfg.max_duration * cfg.sample_rate))


def fix_length(
    x: np.ndarray,
    n: Optional[int],
    crop: str = "start",
    pad_mode: str = "repeat",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Crop or pad ``x`` to exactly ``n`` samples (``pad_mode='none'`` only crops).

    ``crop`` is "start", "center" or "random" (the latter requires ``rng``).
    """
    if n is None:
        return x
    if len(x) > n:
        if crop == "start":
            offset = 0
        elif crop == "center":
            offset = (len(x) - n) // 2
        elif crop == "random":
            if rng is None:
                raise ValueError("random crop needs an explicit rng")
            offset = int(rng.integers(0, len(x) - n + 1))
        else:
            raise ValueError(f"unknown crop mode {crop!r}")
        return x[offset : offset + n]
    if len(x) == n or pad_mode == "none":
        return x
    if pad_mode == "repeat":
        reps = int(np.ceil(n / len(x)))
        return np.tile(x, reps)[:n]
    if pad_mode == "zero":
        return np.pad(x, (0, n - len(x)))
    raise ValueError(f"unknown pad mode {pad_mode!r}")


def trim_silence(x: np.ndarray, top_db: float) -> np.ndarray:
    import librosa

    y, _ = librosa.effects.trim(x, top_db=top_db)
    return y if len(y) else x


def normalize(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "peak":
        peak = float(np.max(np.abs(x)))
        return x / peak if peak > 1e-8 else x
    if mode == "rms":
        rms = float(np.sqrt(np.mean(x**2)))
        return x / rms if rms > 1e-8 else x
    raise ValueError(f"unknown normalization {mode!r}")


def preprocess(x: np.ndarray, cfg: AudioConfig, train: bool = False, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Apply steps 3-5 of the shared policy to an already-decoded waveform."""
    if cfg.trim_silence_db is not None:
        x = trim_silence(x, cfg.trim_silence_db)
    x = normalize(x, cfg.normalization)
    crop = "random" if (train and cfg.train_random_crop) else cfg.eval_crop
    x = fix_length(x, target_num_samples(cfg), crop=crop, pad_mode=cfg.pad_mode, rng=rng)
    return x.astype(np.float32, copy=False)


def load_and_preprocess(
    path: Path | str, cfg: AudioConfig, train: bool = False, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Decode + shared preprocessing; the single entry point used by every pipeline."""
    return preprocess(load_audio(path, cfg.sample_rate, cfg.mono), cfg, train=train, rng=rng)
