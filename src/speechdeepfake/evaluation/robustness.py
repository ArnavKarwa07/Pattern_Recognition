"""Robustness of the trained detectors to common audio degradations.

The case study asks how audio compression and channel conditions affect
detection. This stage re-scores the *already trained* models on degraded
copies of the test recordings; nothing is retrained and the dev-selected
thresholds are reused unchanged, so the numbers show how a deployed detector
would behave when the incoming audio is degraded.

For each test recording the shared preprocessing (decode, 16 kHz mono, fixed
4 s window) is applied first, then one degradation, and the result is written
to disk in the degraded format (MP3 conditions are stored as real ``.mp3``
files and decoded again by the normal pipeline). Every model then scores the
degraded file through its usual prediction path.

Outputs (``results/run_data/<run>/``):

* ``robustness/<model>__<test>__<condition>.csv`` - per-recording scores;
* ``metrics/robustness.csv`` - metrics per model, test set and condition
  (``clean`` = the original recordings), on the records every model scored
  under every condition;
* ``metrics/robustness_conditions.json`` - the exact degradation parameters
  and measured MP3 bit rates.
"""

from __future__ import annotations

import hashlib
import io
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
import soxr
from tqdm import tqdm

from speechdeepfake import artifacts
from speechdeepfake.config import MODEL_NAMES, ExperimentConfig, stable_hash
from speechdeepfake.data.preprocessing import load_and_preprocess
from speechdeepfake.data.validation import resolve_audio_path
from speechdeepfake.evaluation.metrics import binary_metrics, bootstrap_ci, eer_only
from speechdeepfake.evaluation.predict import DISPLAY, PRED_COLUMNS, _model_meta, _predict, load_predictions
from speechdeepfake.utils.logging import get_logger
from speechdeepfake.utils.reproducibility import set_seed

log = get_logger("evaluation.robustness")

# Parameters are part of the degraded record ids, so changing them never reuses stale cached features.
CONDITIONS: dict[str, dict] = {
    "mp3_high": {"kind": "mp3", "compression_level": 0.5,
                 "label": "MP3, moderate compression",
                 "description": "MP3 (LAME via libsndfile, CBR, compression level 0.5; about 80 kbps at 16 kHz mono)"},
    "mp3_low": {"kind": "mp3", "compression_level": 0.9,
                "label": "MP3, aggressive compression",
                "description": "MP3 (LAME via libsndfile, CBR, compression level 0.9; about 25 kbps at 16 kHz mono)"},
    "narrowband": {"kind": "narrowband", "rate": 8000,
                   "label": "narrowband (8 kHz)",
                   "description": "telephone-style band limitation: resampled to 8 kHz and back to 16 kHz (content above 4 kHz removed)"},
    "noise_20db": {"kind": "noise", "snr_db": 20.0,
                   "label": "white noise, 20 dB SNR",
                   "description": "additive white Gaussian noise at 20 dB signal-to-noise ratio (per-recording seed)"},
}
CONDITION_ORDER = ["clean", *CONDITIONS]
CLEAN_LABEL = "clean (original audio)"


def condition_tag(condition: str) -> str:
    """Suffix for degraded record ids: condition name plus a hash of its parameters."""
    return f"{condition}-{stable_hash(CONDITIONS[condition], 6)}"


def _record_rng(record_id: str, condition: str, seed: int) -> np.random.Generator:
    digest = hashlib.sha1(f"{seed}|{condition}|{record_id}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def encode_mp3(x: np.ndarray, sr: int, compression_level: float) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.clip(x, -1.0, 1.0), sr, format="MP3", subtype="MPEG_LAYER_III",
             compression_level=compression_level, bitrate_mode="CONSTANT")
    return buf.getvalue()


def apply_degradation(x: np.ndarray, sr: int, condition: str, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Degrade a waveform in memory (used for tests and figures; scoring decodes real files)."""
    p = CONDITIONS[condition]
    if p["kind"] == "mp3":
        y, _ = sf.read(io.BytesIO(encode_mp3(x, sr, p["compression_level"])), dtype="float32")
        return y if y.ndim == 1 else y.mean(axis=1)
    if p["kind"] == "narrowband":
        return soxr.resample(soxr.resample(x, sr, p["rate"]), p["rate"], sr).astype(np.float32)[: len(x)]
    if p["kind"] == "noise":
        rng = rng or np.random.default_rng(0)
        power = float(np.mean(x.astype(np.float64) ** 2))
        noise = rng.standard_normal(len(x)) * np.sqrt(power / 10 ** (p["snr_db"] / 10))
        return (x + noise).astype(np.float32)
    raise ValueError(f"unknown degradation kind {p['kind']!r}")


def _write_degraded(x: np.ndarray, sr: int, condition: str, stem: Path, rng: np.random.Generator) -> tuple[Path, int]:
    p = CONDITIONS[condition]
    if p["kind"] == "mp3":
        data = encode_mp3(x, sr, p["compression_level"])
        out = stem.with_suffix(".mp3")
        out.write_bytes(data)
        return out, len(data)
    y = np.clip(apply_degradation(x, sr, condition, rng), -1.0, 1.0)
    out = stem.with_suffix(".flac")
    sf.write(out, y, sr, subtype="PCM_16")
    return out, out.stat().st_size


def degrade_test_set(cfg: ExperimentConfig, df: pd.DataFrame, condition: str, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Write a degraded 4 s clip for every record; returns the degraded manifest and encoding stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = condition_tag(condition)
    sr = cfg.audio.sample_rate
    rows, sizes, seconds, failed = [], [], [], 0
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"degrade:{condition}", unit="file"):
        try:
            x = load_and_preprocess(resolve_audio_path(r["audio_path"], cfg.data_dir), cfg.audio)
        except Exception as e:  # unreadable originals are logged and skipped, as in evaluation
            artifacts.error_log(cfg).add(f"robustness_{condition}", r["record_id"], str(r["audio_path"]), str(e))
            failed += 1
            continue
        stem = out_dir / hashlib.sha1(r["record_id"].encode("utf-8")).hexdigest()[:20]
        path, nbytes = _write_degraded(x, sr, condition, stem, _record_rng(r["record_id"], condition, cfg.seed))
        sizes.append(nbytes)
        seconds.append(len(x) / sr)
        row = r.copy()
        row["record_id"] = f"{r['record_id']}#{tag}"
        row["audio_path"] = str(path)
        rows.append(row)
    stats = {"n": len(rows), "n_failed": failed}
    if CONDITIONS[condition]["kind"] == "mp3" and sizes:
        stats["measured_kbps"] = float(8 * np.sum(sizes) / np.sum(seconds) / 1000)
    return pd.DataFrame(rows).reset_index(drop=True), stats


def _pred_path(cfg: ExperimentConfig, model: str, test: str, condition: str) -> Path:
    return cfg.results_dir / "robustness" / f"{model}__{test}__{condition}.csv"


def run_robustness(cfg: ExperimentConfig, data, models: list[str] = MODEL_NAMES, force: bool = False) -> pd.DataFrame:
    """Score every model on every degraded test set, then compile ``metrics/robustness.csv``."""
    set_seed(cfg.seed)
    for m in models:
        _model_meta(cfg, m)  # fail early if a model is missing
    rob = cfg.robustness
    tests = [t for t in data.tests if rob.test_sets is None or t in rob.test_sets]
    info_path = artifacts.metrics_dir(cfg) / "robustness_conditions.json"
    info = artifacts.load_json(info_path) if info_path.exists() else {}
    audio_root = cfg.cache_dir / "robustness_audio"
    for test in tests:
        df = data.tests[test]
        for cond in rob.conditions:
            todo = [m for m in models if force or not _pred_path(cfg, m, test, cond).exists()]
            if not todo:
                log.info("robustness %s/%s: predictions exist, skipping (use --force to redo)", test, cond)
                continue
            out_dir = audio_root / test / condition_tag(cond)
            ddf, stats = degrade_test_set(cfg, df, cond, out_dir)
            info.setdefault(cond, {**{k: v for k, v in CONDITIONS[cond].items()}, "tag": condition_tag(cond)})
            info[cond].setdefault("per_test_set", {})[test] = stats
            for m in todo:
                meta = _model_meta(cfg, m)
                thr = float(meta["threshold"])
                recs, scores, _ = _predict(m, cfg, ddf, f"{test}+{cond}")
                pred = recs.copy()
                pred["score"] = scores
                pred["threshold"] = thr
                pred["predicted"] = (scores >= thr).astype(int)
                pred["dataset"] = test
                pred["condition"] = cond
                pred["model"] = m
                pred = pred[[c for c in PRED_COLUMNS if c in pred.columns]]
                out = _pred_path(cfg, m, test, cond)
                out.parent.mkdir(parents=True, exist_ok=True)
                pred.to_csv(out, index=False)
                q = binary_metrics(pred["label"], pred["score"], thr)
                log.info("robustness %s on %s/%s: n=%d EER=%.4f BA=%.4f", m, test, cond, q["n"], q["eer"], q["balanced_accuracy"])
            if not rob.keep_audio:
                shutil.rmtree(out_dir, ignore_errors=True)
    artifacts.save_json(info, info_path)
    return compile_robustness(cfg)


def load_robustness_predictions(cfg: ExperimentConfig) -> dict[tuple[str, str, str], pd.DataFrame]:
    """{(model, test, condition): predictions}, with record ids mapped back to the clean ids."""
    out = {}
    d = cfg.results_dir / "robustness"
    for f in sorted(d.glob("*__*__*.csv")) if d.exists() else []:
        model, test, cond = f.stem.split("__")
        if model not in MODEL_NAMES or cond not in CONDITIONS:
            continue
        p = pd.read_csv(f, dtype={"record_id": str, "group_id": str}, keep_default_na=False)
        p["record_id"] = p["record_id"].str.rsplit("#", n=1).str[0]
        out[(model, test, cond)] = p
    return out


def compile_robustness(cfg: ExperimentConfig) -> pd.DataFrame:
    rob = load_robustness_predictions(cfg)
    if not rob:
        raise FileNotFoundError("No robustness predictions; run `speechdeepfake robustness` first.")
    clean = load_predictions(cfg)
    rows = []
    for test in sorted({t for _, t, _ in rob}):
        conds = [c for c in CONDITIONS if any(k[1] == test and k[2] == c for k in rob)]
        models = [m for m in MODEL_NAMES if (m, test) in clean and all((m, test, c) in rob for c in conds)]
        if not models:
            continue
        sets = [set(clean[(m, test)]["record_id"]) for m in models]
        sets += [set(rob[(m, test, c)]["record_id"]) for m in models for c in conds]
        common = set.intersection(*sets)
        for m in models:
            for cond in ["clean", *conds]:
                p = clean[(m, test)] if cond == "clean" else rob[(m, test, cond)]
                q = p[p["record_id"].isin(common)].sort_values("record_id")
                thr = float(q["threshold"].iloc[0])
                met = binary_metrics(q["label"], q["score"], thr)
                groups = q["group_id"].to_numpy() if "group_id" in q and (q["group_id"] != "").all() else None
                lo, hi = bootstrap_ci(q["label"].to_numpy(), q["score"].to_numpy(), eer_only,
                                      cfg.evaluation.n_bootstrap, cfg.evaluation.ci_level, groups=groups, seed=cfg.seed)
                rows.append({"model": m, "display_name": DISPLAY[m], "dataset": test, "audio_condition": cond,
                             "audio_condition_label": CLEAN_LABEL if cond == "clean" else CONDITIONS[cond]["label"],
                             "run_type": cfg.run_type, **met, "eer_ci_low": lo, "eer_ci_high": hi})
    table = pd.DataFrame(rows)
    base = table[table["audio_condition"] == "clean"].set_index(["model", "dataset"])
    table["delta_eer_vs_clean"] = [r["eer"] - base.loc[(r["model"], r["dataset"]), "eer"] for _, r in table.iterrows()]
    table["delta_balanced_accuracy_vs_clean"] = [
        r["balanced_accuracy"] - base.loc[(r["model"], r["dataset"]), "balanced_accuracy"] for _, r in table.iterrows()]
    order = {t.name: i for i, t in enumerate(cfg.test_sets())}
    table = table.sort_values(["dataset", "model", "audio_condition"],
                              key=lambda s: s.map(order) if s.name == "dataset" else (
                                  s.map({c: i for i, c in enumerate(CONDITION_ORDER)}) if s.name == "audio_condition"
                                  else s.map({m: i for i, m in enumerate(MODEL_NAMES)})),
                              kind="stable").reset_index(drop=True)
    table.to_csv(artifacts.metrics_dir(cfg) / "robustness.csv", index=False)
    return table
