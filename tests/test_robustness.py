"""Audio degradations, robustness config checks and the extra figures (synthetic signals only)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from speechdeepfake.config import ConfigError, load_config
from speechdeepfake.evaluation import plots
from speechdeepfake.evaluation.robustness import CONDITIONS, apply_degradation, condition_tag

SR = 16000


def _tone_plus_hiss(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    rng = np.random.default_rng(0)
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


def _band_energy(x: np.ndarray, lo: float, hi: float) -> float:
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return float(spec[(f >= lo) & (f < hi)].sum())


def test_noise_has_requested_snr():
    x = _tone_plus_hiss(2.0)
    y = apply_degradation(x, SR, "noise_20db", np.random.default_rng(1))
    noise = y - x
    snr = 10 * np.log10(np.mean(x.astype(np.float64) ** 2) / np.mean(noise.astype(np.float64) ** 2))
    assert snr == pytest.approx(20.0, abs=0.3)


def test_narrowband_removes_content_above_4khz():
    x = _tone_plus_hiss()
    y = apply_degradation(x, SR, "narrowband")
    assert len(y) == len(x)
    assert _band_energy(y, 4200, 8000) < 0.01 * _band_energy(x, 4200, 8000)
    assert _band_energy(y, 300, 600) == pytest.approx(_band_energy(x, 300, 600), rel=0.1)


@pytest.mark.parametrize("cond", ["mp3_high", "mp3_low"])
def test_mp3_roundtrip_keeps_signal(cond):
    x = _tone_plus_hiss()
    y = apply_degradation(x, SR, cond)
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert len(y) >= len(x)  # decoder may add encoder-delay samples, never drops audio
    assert _band_energy(y, 400, 480) > 0.5 * _band_energy(x, 400, 480)


def test_condition_tag_depends_on_parameters(monkeypatch):
    before = condition_tag("noise_20db")
    monkeypatch.setitem(CONDITIONS, "noise_20db", {**CONDITIONS["noise_20db"], "snr_db": 10.0})
    assert condition_tag("noise_20db") != before


def test_config_rejects_unknown_condition_and_test_set():
    with pytest.raises(ConfigError):
        load_config(None, overrides={"robustness": {"conditions": ["vinyl"]}})
    with pytest.raises(ConfigError):
        load_config(None, overrides={"robustness": {"test_sets": ["nope"]}})


def _fake_preds(n: int = 200):
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(n // 2, int), np.ones(n // 2, int)]
    out = {}
    for k, m in enumerate(plots.MODEL_ORDER):
        s = rng.standard_normal(n) + (1 + k) * y
        out[(m, "t")] = pd.DataFrame({"record_id": [f"r{i}" for i in range(n)], "label": y, "score": s, "threshold": 0.5})
    return out, {f"r{i}" for i in range(n)}


def test_extra_figures_are_written(tmp_path):
    preds, common = _fake_preds()
    assert plots.plot_det(preds, "t", common, tmp_path / "det.png", "pilot").exists()
    assert plots.plot_score_distributions(preds, "t", common, tmp_path / "s.png", "pilot").exists()
    rob = pd.DataFrame([{"model": m, "dataset": "t", "audio_condition": c, "audio_condition_label": c,
                         "eer": 0.1 + 0.05 * i, "eer_ci_low": 0.05, "eer_ci_high": 0.3, "balanced_accuracy": 0.8}
                        for m in plots.MODEL_ORDER for i, c in enumerate(["clean", "mp3_low"])])
    assert plots.plot_robustness(rob, "eer", "EER", tmp_path / "r.png", "full").exists()
    assert plots.plot_pipeline_overview(tmp_path / "p.png").exists()
