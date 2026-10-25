"""Normalised manifest schema, label normalisation and integrity checks.

A manifest is a table with one row per recording:

=============  ========  ==========================================================
column         required  meaning
=============  ========  ==========================================================
record_id      yes       globally unique id, ``<corpus>/<corpus-local id>``
audio_path     yes       path to the audio file (relative paths are relative to
                         the configured data directory)
label          yes       0 = bona fide, 1 = spoof
corpus         yes       e.g. ``asvspoof2019_la``, ``cdadd``, ``custom``
split          yes       corpus split, e.g. ``train``, ``dev``, ``eval``, ``test``
utterance_id   no        corpus utterance identifier
speaker_id     no        speaker identifier (namespaced per corpus for checks)
source_domain  no        source corpus/domain, e.g. ``vctk``, ``libritts``, ``ted``
generator_id   no        generation system / attack id, ``bonafide`` for genuine
group_id       no        recordings sharing source content (leakage unit)
=============  ========  ==========================================================
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from speechdeepfake import BONAFIDE, CLASS_NAMES, SPOOF

REQUIRED_COLUMNS = ["record_id", "audio_path", "label", "corpus", "split"]
OPTIONAL_COLUMNS = ["utterance_id", "speaker_id", "source_domain", "generator_id", "group_id"]
MANIFEST_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

_BONAFIDE_ALIASES = {"0", "bonafide", "bona-fide", "bona_fide", "bona fide", "genuine", "real", "human", "authentic"}
_SPOOF_ALIASES = {
    "1", "spoof", "spoofed", "fake", "synthetic", "generated", "ai", "ai-generated",
    "deepfake", "tts", "manipulated",
}


class ManifestError(ValueError):
    """Raised for structurally invalid manifests or unknown labels."""


def normalize_label(value) -> int:
    """Map a label variant to 0 (bona fide) or 1 (spoof); raise on anything else.

    Integers 0/1 (and floats equal to them) are accepted. Strings are compared
    case-insensitively after stripping whitespace.
    """
    if isinstance(value, bool):
        raise ManifestError(f"Boolean label {value!r} is ambiguous; use 0/1 or bonafide/spoof")
    if isinstance(value, (int, float)) and not pd.isna(value):
        if float(value) == 0.0:
            return BONAFIDE
        if float(value) == 1.0:
            return SPOOF
        raise ManifestError(f"Unknown numeric label {value!r}; expected 0 (bonafide) or 1 (spoof)")
    text = str(value).strip().lower()
    if text in _BONAFIDE_ALIASES:
        return BONAFIDE
    if text in _SPOOF_ALIASES:
        return SPOOF
    raise ManifestError(f"Unknown label {value!r}; refusing to guess. Expected one of bonafide/spoof variants.")


def normalize_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns, normalise labels and fill optional metadata with ''."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"Manifest is missing required column(s): {missing}")
    out = df.copy()
    for c in OPTIONAL_COLUMNS:
        if c not in out.columns:
            out[c] = ""
    bad = []
    labels = []
    for i, v in enumerate(out["label"].tolist()):
        try:
            labels.append(normalize_label(v))
        except ManifestError as e:
            bad.append((out["record_id"].iloc[i], str(e)))
            labels.append(-1)
    if bad:
        preview = "; ".join(f"{r}: {m}" for r, m in bad[:5])
        raise ManifestError(f"{len(bad)} record(s) have unknown labels. First: {preview}")
    out["label"] = pd.Series(labels, index=out.index, dtype="int64")
    for c in MANIFEST_COLUMNS:
        if c != "label":
            out[c] = out[c].fillna("").astype(str).str.strip()
    empty = out["record_id"].eq("") | out["audio_path"].eq("")
    if empty.any():
        raise ManifestError(f"{int(empty.sum())} row(s) have an empty record_id or audio_path")
    extra = [c for c in out.columns if c not in MANIFEST_COLUMNS]
    return out[MANIFEST_COLUMNS + extra]


@dataclass
class IntegrityReport:
    """Result of :func:`check_integrity`. ``errors`` block training; ``warnings`` do not."""

    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"errors": self.errors, "warnings": self.warnings, "info": self.info}


def find_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (duplicate rows with consistent labels, rows with conflicting labels)."""
    dup_mask = df.duplicated("record_id", keep=False)
    dups = df[dup_mask]
    if dups.empty:
        return dups, dups
    n_labels = dups.groupby("record_id")["label"].transform("nunique")
    return dups[n_labels == 1], dups[n_labels > 1]


def _cross_split(df: pd.DataFrame, column: str) -> dict[str, list[str]]:
    """Values of ``column`` (within a corpus) appearing in more than one split."""
    sub = df[df[column] != ""]
    if sub.empty:
        return {}
    key = sub["corpus"] + "::" + sub[column]
    splits = sub.groupby(key)["split"].unique()
    shared = splits[splits.map(len) > 1]
    return {k: sorted(v) for k, v in shared.items()}


def check_integrity(df: pd.DataFrame, speaker_overlap_is_error: bool = False) -> IntegrityReport:
    """Duplicate ids, conflicting labels and cross-split leakage within each corpus.

    Speakers shared between splits are reported as warnings by default because
    some official protocols intentionally reuse speakers; identical source
    groups or audio files shared across splits are errors.
    """
    rep = IntegrityReport()
    same, conflict = find_duplicates(df)
    if not conflict.empty:
        ids = sorted(conflict["record_id"].unique())
        rep.errors.append(f"{len(ids)} record_id(s) appear with conflicting labels, e.g. {ids[:5]}")
    if not same.empty:
        ids = sorted(same["record_id"].unique())
        rep.errors.append(f"{len(ids)} duplicated record_id(s), e.g. {ids[:5]}")

    for column in ("group_id", "utterance_id", "audio_path"):
        shared = _cross_split(df, column)
        if shared:
            example = next(iter(shared.items()))
            rep.errors.append(
                f"Leakage: {len(shared)} {column} value(s) occur in multiple splits, e.g. {example[0]} in {example[1]}"
            )
        elif (df[column] != "").any():
            rep.info.append(f"No {column} shared across splits.")
        else:
            rep.info.append(f"{column} not available; leakage by {column} could not be checked.")

    shared_spk = _cross_split(df, "speaker_id")
    if shared_spk:
        msg = f"{len(shared_spk)} speaker(s) occur in more than one split within a corpus"
        (rep.errors if speaker_overlap_is_error else rep.warnings).append(msg)
    if "sha1" in df.columns:
        sub = df[df["sha1"].fillna("") != ""]
        splits = sub.groupby("sha1")[["corpus", "split"]].apply(lambda g: sorted(set(g["corpus"] + "/" + g["split"])))
        shared = splits[splits.map(len) > 1]
        if len(shared):
            rep.errors.append(
                f"{len(shared)} identical audio file(s) (by SHA-1) occur in more than one corpus/split, "
                f"e.g. {list(shared.iloc[0])}"
            )
        else:
            rep.info.append("No byte-identical audio files shared across corpus/splits (SHA-1).")
    return rep


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df["label"].value_counts()
    return {CLASS_NAMES[k]: int(counts.get(k, 0)) for k in (BONAFIDE, SPOOF)}


def metadata_coverage(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of rows with a non-empty value for each optional column."""
    if df.empty:
        return {c: 0.0 for c in OPTIONAL_COLUMNS}
    return {c: round(float((df[c].astype(str) != "").mean()), 4) for c in OPTIONAL_COLUMNS}


def write_manifest(df: pd.DataFrame, path: Path) -> str:
    """Write a manifest deterministically (sorted, fixed columns); return its SHA-1."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_values("record_id", kind="mergesort").reset_index(drop=True)
    text = out.to_csv(index=False, lineterminator="\n", float_format="%.6f")
    path.write_text(text, encoding="utf-8")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run `speechdeepfake prepare-data --config <config>` first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    return normalize_manifest(df)


def manifest_digest(df: pd.DataFrame, columns: Optional[Iterable[str]] = None) -> str:
    cols = list(columns or MANIFEST_COLUMNS)
    text = df.sort_values("record_id")[cols].to_csv(index=False, lineterminator="\n")
    return hashlib.sha1(text.encode("utf-8")).hexdigest()
