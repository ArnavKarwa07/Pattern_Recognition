"""MFCC extraction and fixed-length temporal statistics (Solution 1 features)."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Sequence, Union

import librosa
import numpy as np
from joblib import Parallel, delayed

from speechdeepfake.config import AudioConfig, MFCCConfig
from speechdeepfake.data.preprocessing import load_and_preprocess


def mfcc_matrix(wave: np.ndarray, sr: int, cfg: MFCCConfig) -> np.ndarray:
    """MFCCs of shape (n_mfcc, frames)."""
    return librosa.feature.mfcc(
        y=wave,
        sr=sr,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )


def _delta(m: np.ndarray, order: int) -> np.ndarray:
    frames = m.shape[1]
    width = min(9, frames if frames % 2 else frames - 1)
    if width < 3:
        return np.zeros_like(m)
    return librosa.feature.delta(m, width=width, order=order, mode="nearest")


def mfcc_summary(wave: np.ndarray, sr: int, cfg: MFCCConfig) -> np.ndarray:
    """[mean, std] over time of the MFCCs and (optionally) their deltas.

    Layout: ``[mean(static), std(static), mean(d1), std(d1), mean(d2), std(d2)]``
    truncated to ``delta_order``. Length is :func:`feature_dim`.
    """
    m = mfcc_matrix(wave, sr, cfg)
    streams = [m] + [_delta(m, k) for k in range(1, cfg.delta_order + 1)]
    parts = []
    for s in streams:
        parts.extend([s.mean(axis=1), s.std(axis=1)])
    return np.concatenate(parts).astype(np.float32)


def feature_dim(cfg: MFCCConfig) -> int:
    return cfg.n_mfcc * (1 + cfg.delta_order) * 2


def _one(path: Path, audio_cfg: AudioConfig, mfcc_cfg: MFCCConfig) -> Union[np.ndarray, Exception]:
    try:
        wave = load_and_preprocess(path, audio_cfg, train=False)
        return mfcc_summary(wave, audio_cfg.sample_rate, mfcc_cfg)
    except Exception as e:  # reported per file by the cache
        return RuntimeError(f"{type(e).__name__}: {e}")


class MFCCExtractor:
    """Picklable extractor usable with :func:`speechdeepfake.features.cache.cached_features`."""

    name = "mfcc_stats"

    def __init__(self, audio_cfg: AudioConfig, mfcc_cfg: MFCCConfig):
        self.audio_cfg = audio_cfg
        self.mfcc_cfg = mfcc_cfg

    def config_dict(self) -> dict:
        mfcc = dataclasses.asdict(self.mfcc_cfg)
        mfcc.pop("n_jobs")  # parallelism does not change the features
        return {"feature": self.name, "audio": dataclasses.asdict(self.audio_cfg), "mfcc": mfcc, "librosa": librosa.__version__}

    def __call__(self, paths: Sequence[Path]) -> list[Union[np.ndarray, Exception]]:
        n_jobs = self.mfcc_cfg.n_jobs
        if n_jobs == 1 or len(paths) < 8:
            return [_one(p, self.audio_cfg, self.mfcc_cfg) for p in paths]
        return Parallel(n_jobs=n_jobs)(delayed(_one)(p, self.audio_cfg, self.mfcc_cfg) for p in paths)
