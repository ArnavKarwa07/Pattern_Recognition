"""Per-file audio validation and dataset statistics.

``validate_manifest`` decodes every file once, records its status, duration,
peak level and SHA-1, runs the structural integrity checks and writes a
validated manifest (``status == "ok"`` rows are usable) plus a JSON summary.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm.auto import tqdm

from speechdeepfake.config import AudioConfig, ExperimentConfig
from speechdeepfake.data.manifests import (
    check_integrity,
    label_counts,
    manifest_digest,
    metadata_coverage,
    read_manifest,
    write_manifest,
)
from speechdeepfake.utils.logging import get_logger

log = get_logger("data.validation")

STATUS_OK = "ok"
VALIDATION_COLUMNS = ["status", "message", "sample_rate", "num_channels", "duration_s", "peak", "is_silent", "sha1"]


def resolve_audio_path(audio_path: str, data_dir: Path) -> Path:
    p = Path(audio_path)
    return p if p.is_absolute() else Path(data_dir) / p


def inspect_audio_file(path: Path, cfg: AudioConfig, compute_hash: bool = True) -> dict:
    """Decode a file and classify it as ok / missing / corrupt / empty / non_finite / too_short."""
    out = {c: "" for c in VALIDATION_COLUMNS}
    out.update(status=STATUS_OK, is_silent=False)
    if not path.exists():
        out.update(status="missing", message="file does not exist")
        return out
    try:
        if compute_hash:
            out["sha1"] = hashlib.sha1(path.read_bytes()).hexdigest()
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as e:
        out.update(status="corrupt", message=f"{type(e).__name__}: {e}")
        return out
    out.update(sample_rate=int(sr), num_channels=int(data.shape[1]))
    if data.size == 0:
        out.update(status="empty", message="zero samples", duration_s=0.0)
        return out
    duration = data.shape[0] / sr
    out["duration_s"] = round(float(duration), 4)
    if not np.isfinite(data).all():
        out.update(status="non_finite", message="NaN or Inf samples")
        return out
    peak = float(np.max(np.abs(data)))
    out["peak"] = round(peak, 6)
    out["is_silent"] = peak < cfg.silence_threshold
    if duration < cfg.min_duration:
        out.update(status="too_short", message=f"duration {duration:.3f}s < min_duration {cfg.min_duration}s")
        return out
    notes = []
    if sr != cfg.sample_rate:
        notes.append(f"sample rate {sr} Hz will be resampled to {cfg.sample_rate} Hz")
    if out["is_silent"]:
        notes.append("silent clip (peak below silence_threshold)")
    out["message"] = "; ".join(notes)
    return out


def describe(df: pd.DataFrame) -> dict:
    """Class counts, duration statistics, failures and metadata coverage of a manifest slice."""
    ok = df[df["status"] == STATUS_OK] if "status" in df.columns else df
    dur = pd.to_numeric(ok.get("duration_s", pd.Series(dtype=float)), errors="coerce").dropna()
    stats = {
        "n_records": int(len(df)),
        "n_usable": int(len(ok)),
        "class_counts_usable": label_counts(ok),
        "metadata_coverage": metadata_coverage(df),
    }
    if "status" in df.columns:
        stats["status_counts"] = {k: int(v) for k, v in df["status"].value_counts().items()}
        stats["n_silent"] = int(df["is_silent"].astype(str).str.lower().eq("true").sum())
        sr = pd.to_numeric(df["sample_rate"], errors="coerce").dropna().astype(int)
        stats["sample_rates"] = {str(k): int(v) for k, v in sr.value_counts().items()}
    if len(dur):
        stats["duration_s"] = {
            "min": round(float(dur.min()), 3),
            "mean": round(float(dur.mean()), 3),
            "median": round(float(dur.median()), 3),
            "p95": round(float(dur.quantile(0.95)), 3),
            "max": round(float(dur.max()), 3),
            "total_hours": round(float(dur.sum()) / 3600, 3),
        }
    gens = ok["generator_id"][ok["generator_id"] != ""]
    if len(gens):
        stats["generator_counts"] = {k: int(v) for k, v in gens.value_counts().sort_index().items()}
    doms = ok["source_domain"][ok["source_domain"] != ""]
    if len(doms):
        stats["source_domain_counts"] = {k: int(v) for k, v in doms.value_counts().sort_index().items()}
    return stats


def validate_manifest(
    df: pd.DataFrame, cfg: ExperimentConfig, workers: int = 8, compute_hash: bool = True
) -> tuple[pd.DataFrame, dict]:
    """Inspect every file of ``df`` and return (validated manifest, summary)."""
    paths = [resolve_audio_path(p, cfg.data_dir) for p in df["audio_path"]]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        results = list(
            tqdm(
                pool.map(lambda p: inspect_audio_file(p, cfg.audio, compute_hash), paths),
                total=len(paths),
                desc="validate audio",
                unit="file",
            )
        )
    checked = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)
    integrity = check_integrity(checked)
    summary = {
        "manifest_digest": manifest_digest(df),
        "integrity": integrity.to_dict(),
        "overall": describe(checked),
        "by_split": {f"{c}/{s}": describe(g) for (c, s), g in checked.groupby(["corpus", "split"])},
    }
    for msg in integrity.errors:
        log.error(msg)
    for msg in integrity.warnings:
        log.warning(msg)
    bad = checked[checked["status"] != STATUS_OK]
    if len(bad):
        log.warning("%d file(s) failed validation: %s", len(bad), bad["status"].value_counts().to_dict())
    return checked, summary


def validated_paths(cfg: ExperimentConfig, corpus: str) -> tuple[Path, Path]:
    d = cfg.manifests_dir / "validated"
    return d / f"{corpus}.csv", d / f"{corpus}_summary.json"


def save_validation(checked: pd.DataFrame, summary: dict, cfg: ExperimentConfig, corpus: str) -> tuple[Path, Path]:
    csv_path, json_path = validated_paths(cfg, corpus)
    write_manifest(checked, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    bad = checked[checked["status"] != STATUS_OK]
    if len(bad):
        write_manifest(bad, csv_path.with_name(f"{corpus}_failures.csv"))
    return csv_path, json_path


def load_corpus(cfg: ExperimentConfig, corpus: str) -> pd.DataFrame:
    """Load a corpus manifest, preferring its validated version.

    Raises a clear error when ``data.require_validation`` is set and the
    validated manifest is missing or stale.
    """
    raw_path = cfg.manifests_dir / f"{corpus}.csv"
    csv_path, json_path = validated_paths(cfg, corpus)
    raw = read_manifest(raw_path)
    if csv_path.exists() and json_path.exists():
        summary = json.loads(json_path.read_text(encoding="utf-8"))
        if summary.get("manifest_digest") == manifest_digest(raw):
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            df["label"] = df["label"].astype(int)
            return df
        if cfg.data.require_validation:
            raise RuntimeError(
                f"Validated manifest for {corpus!r} is stale (manifest changed). "
                "Re-run `speechdeepfake validate-data --config <config>`."
            )
    elif cfg.data.require_validation:
        raise RuntimeError(
            f"Corpus {corpus!r} has not been validated. Run `speechdeepfake validate-data --config <config>` "
            "before training/evaluation (or set data.require_validation: false)."
        )
    log.warning("Using unvalidated manifest for %s; unreadable files will be logged at feature extraction.", corpus)
    raw["status"] = STATUS_OK
    return raw


def usable(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == STATUS_OK].reset_index(drop=True)


def summary_markdown(summary: Optional[dict]) -> str:
    """Compact Markdown table of per-split statistics (used by the report)."""
    if not summary:
        return "_No validation summary available._"
    lines = [
        "| corpus/split | records | usable | bona fide | spoof | mean dur. (s) | total (h) | failures |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summary["by_split"].items():
        cc = s["class_counts_usable"]
        d = s.get("duration_s", {})
        fails = s["n_records"] - s["n_usable"]
        lines.append(
            f"| {name} | {s['n_records']} | {s['n_usable']} | {cc['bonafide']} | {cc['spoof']} | "
            f"{d.get('mean', '–')} | {d.get('total_hours', '–')} | {fails} |"
        )
    return "\n".join(lines)
