"""Tiny synthetic audio fixture for SOFTWARE TESTING ONLY.

The "bona fide" clips are harmonic tones with vibrato and noise; the "spoof"
clips are the same kind of signal low-passed and given a narrow-band
high-frequency artefact. They resemble neither real speech nor real
deepfakes. Anything computed on this fixture verifies that the code runs and
produces well-formed outputs; it is never a research result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt

FIXTURE_CORPUS = "synthetic_fixture"
FIXTURE_WARNING = "SYNTHETIC SOFTWARE-TEST FIXTURE - NOT A RESEARCH RESULT"


def _voice_like(rng: np.random.Generator, n: int, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    f0 = rng.uniform(90, 220) * (1 + 0.03 * np.sin(2 * np.pi * rng.uniform(3, 6) * t))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = sum((0.6 / k) * np.sin(k * phase + rng.uniform(0, 2 * np.pi)) for k in range(1, 12))
    envelope = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(2, 4) * t + rng.uniform(0, np.pi)))
    x = x * envelope + 0.02 * rng.standard_normal(n)
    return x


def synthesize_clip(label: int, rng: np.random.Generator, sr: int = 16000, duration: float = 2.0) -> np.ndarray:
    """One synthetic clip; label 1 adds a low-pass + 6 kHz artefact signature."""
    n = int(duration * sr)
    x = _voice_like(rng, n, sr)
    if label == 1:
        x = sosfilt(butter(6, 3500, fs=sr, output="sos"), x)
        t = np.arange(n) / sr
        x = x + 0.05 * np.sin(2 * np.pi * 6000 * t) * (1 + 0.5 * rng.standard_normal(n).clip(-1, 1))
    x = 0.5 * x / (np.max(np.abs(x)) + 1e-9)
    return x.astype(np.float32)


def make_fixture(out_dir: Path, n_per_class: int = 8, splits=("train", "dev", "test", "external"), sr: int = 16000, seed: int = 0) -> Path:
    """Write WAV files plus ``manifest.csv`` (custom-CSV format) under ``out_dir``."""
    out_dir = Path(out_dir)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for split in splits:
        for label in (0, 1):
            for i in range(n_per_class):
                dur = float(rng.uniform(1.5, 5.0))
                x = synthesize_clip(label, rng, sr, dur)
                name = f"{split}_{'spoof' if label else 'bonafide'}_{i:03d}"
                path = audio_dir / f"{name}.wav"
                sf.write(path, x, sr, subtype="PCM_16")
                rows.append(
                    {
                        "record_id": f"{FIXTURE_CORPUS}/{name}",
                        "audio_path": f"audio/{name}.wav",
                        "label": "spoof" if label else "bonafide",
                        "corpus": FIXTURE_CORPUS,
                        "split": split,
                        "utterance_id": name,
                        "speaker_id": f"spk{i % 4}_{split}",
                        "source_domain": "synthetic",
                        "generator_id": (f"synth{'AB'[i % 2]}" if label else "bonafide"),
                        "group_id": name,
                    }
                )
    manifest = out_dir / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    (out_dir / "README.txt").write_text(FIXTURE_WARNING + "\n" + __doc__, encoding="utf-8")
    return manifest
