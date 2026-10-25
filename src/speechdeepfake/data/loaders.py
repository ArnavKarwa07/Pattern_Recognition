"""Dataset-specific parsers that produce normalised manifests.

Supported sources:

* ASVspoof 2019 LA with its official CM protocol files.
* CD-ADD from the public Hugging Face parquet package
  (``SpeechAntiSpoofingBenchmarks/CD-ADD``), exported once to FLAC files.
* A user-provided CSV manifest referencing local audio.

Nothing in this module downloads data implicitly; the ``download_*`` functions
are only called by the explicit ``speechdeepfake download`` command.
"""

from __future__ import annotations

import json
import re
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from speechdeepfake import BONAFIDE, SPOOF
from speechdeepfake.data.manifests import ManifestError, normalize_label, normalize_manifest
from speechdeepfake.utils.logging import get_logger

log = get_logger("data.loaders")

ASVSPOOF_CORPUS = "asvspoof2019_la"
CDADD_CORPUS = "cdadd"

ASVSPOOF_LA_URL = "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"
CDADD_REPO_ID = "SpeechAntiSpoofingBenchmarks/CD-ADD"

_ASV_FILE_RE = re.compile(r"^LA_[TDE]_\d+$")
_ASV_SPEAKER_RE = re.compile(r"^LA_\d{4}$")
_ASV_SYSTEM_RE = re.compile(r"^A\d{2}$")
_ASV_KEYS = {"bonafide", "spoof"}
_ASV_SPLITS = {"train": "train.trn", "dev": "dev.trl", "eval": "eval.trl"}


def relative_to_data_dir(path: Path, data_dir: Path) -> str:
    """Store paths relative to the data directory when possible (portable manifests)."""
    try:
        return Path(path).resolve().relative_to(Path(data_dir).resolve()).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


# --------------------------------------------------------------- ASVspoof 2019


class ProtocolError(ValueError):
    pass


def parse_asvspoof_protocol(path: Path, split: str) -> tuple[pd.DataFrame, bool]:
    """Parse an ASVspoof 2019 LA CM protocol file.

    Fields are identified by their content rather than fixed positions: the
    utterance id matches ``LA_[TDE]_<digits>``, the speaker ``LA_<4 digits>``, the
    attack id ``A<2 digits>`` (``-`` for bona fide) and the key is
    ``bonafide``/``spoof``. Returns the table and whether keys were present.
    """
    rows = []
    has_keys: Optional[bool] = None
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            tokens = line.split()
            if not tokens:
                continue
            file_ids = [t for t in tokens if _ASV_FILE_RE.match(t)]
            if len(file_ids) != 1:
                raise ProtocolError(f"{path}:{lineno}: expected exactly one LA_[TDE]_ id, got {tokens}")
            speakers = [t for t in tokens if _ASV_SPEAKER_RE.match(t)]
            systems = [t for t in tokens if _ASV_SYSTEM_RE.match(t)]
            keys = [t for t in tokens if t.lower() in _ASV_KEYS]
            line_has_key = len(keys) == 1
            if has_keys is None:
                has_keys = line_has_key
            elif has_keys != line_has_key:
                raise ProtocolError(f"{path}:{lineno}: inconsistent presence of bonafide/spoof key")
            label = normalize_label(keys[0]) if line_has_key else None
            system = systems[0] if systems else "-"
            if label is not None:
                if label == SPOOF and system == "-":
                    raise ProtocolError(f"{path}:{lineno}: spoof utterance without an attack id")
                if label == BONAFIDE and system != "-":
                    raise ProtocolError(f"{path}:{lineno}: bona fide utterance with attack id {system}")
            rows.append(
                {
                    "utterance_id": file_ids[0],
                    "speaker_id": speakers[0] if speakers else "",
                    "generator_id": "bonafide" if system == "-" else system,
                    "label": label,
                }
            )
    df = pd.DataFrame(rows)
    df["split"] = split
    return df, bool(has_keys)


def find_asvspoof_protocol(root: Path, split: str) -> Optional[Path]:
    suffix = _ASV_SPLITS[split]
    hits = sorted(root.glob(f"**/ASVspoof2019.LA.cm.{suffix}.txt"))
    return hits[0] if hits else None


def load_asvspoof2019_la(root: Path, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Build the ASVspoof 2019 LA manifest from the official protocol files.

    Returns (manifest, notes). Splits whose protocol is absent or unlabelled are
    skipped with a note; labels are never invented.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"ASVspoof 2019 LA directory not found: {root}\n"
            "Download LA.zip (7.1 GB) from https://datashare.ed.ac.uk/handle/10283/3336 and extract it so that "
            f"{root}/ASVspoof2019_LA_cm_protocols exists, or run `speechdeepfake download --dataset asvspoof2019_la`."
        )
    frames, notes = [], []
    for split in ("train", "dev", "eval"):
        proto = find_asvspoof_protocol(root, split)
        if proto is None:
            notes.append(f"ASVspoof 2019 LA {split}: protocol file not found under {root}; split skipped.")
            continue
        df, has_keys = parse_asvspoof_protocol(proto, split)
        if not has_keys:
            notes.append(f"ASVspoof 2019 LA {split}: protocol {proto.name} has no bonafide/spoof keys; split skipped.")
            continue
        audio_dir = root / f"ASVspoof2019_LA_{split}" / "flac"
        df["audio_path"] = [relative_to_data_dir(audio_dir / f"{u}.flac", data_dir) for u in df["utterance_id"]]
        notes.append(f"ASVspoof 2019 LA {split}: parsed {len(df)} rows from {proto.name}.")
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No labelled ASVspoof 2019 LA protocol files found under {root}")
    df = pd.concat(frames, ignore_index=True)
    df["record_id"] = ASVSPOOF_CORPUS + "/" + df["utterance_id"]
    df["corpus"] = ASVSPOOF_CORPUS
    df["source_domain"] = "vctk"
    df["group_id"] = df["utterance_id"]
    df["speaker_id"] = df["speaker_id"]
    return normalize_manifest(df), notes


# ----------------------------------------------------------------------- CD-ADD


def _audio_extension(data: bytes) -> str:
    if data[:4] == b"fLaC":
        return ".flac"
    if data[:4] == b"RIFF":
        return ".wav"
    if data[:4] == b"OggS":
        return ".ogg"
    raise ManifestError("Unrecognised embedded audio format (not FLAC/WAV/OGG)")


def export_cdadd_parquet(cdadd_root: Path, data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """Export the CD-ADD parquet shards to audio files and build its manifest.

    Expects ``<cdadd_root>/data/test-*.parquet`` (as downloaded from the Hub).
    Audio is written to ``<cdadd_root>/audio/<source path>.<ext>``; files that
    already exist with the same size are kept, so the export is resumable.
    Labels are taken from the ``label`` column, cross-checked against
    ``notes.system`` and, when present, ``data/labels.parquet``.
    """
    import pyarrow.parquet as pq

    cdadd_root = Path(cdadd_root)
    shards = sorted((cdadd_root / "data").glob("test-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No CD-ADD parquet shards found in {cdadd_root / 'data'}.\n"
            "Run `speechdeepfake download --dataset cdadd` (about 3.9 GB) first."
        )
    reference: dict[str, int] = {}
    labels_file = cdadd_root / "data" / "labels.parquet"
    if labels_file.exists():
        t = pq.read_table(labels_file).to_pydict()
        reference = dict(zip(t["utterance_id"], (int(v) for v in t["label"])))
    audio_root = cdadd_root / "audio"
    rows, notes = [], []
    for shard in shards:
        pf = pq.ParquetFile(shard)
        for batch in tqdm(pf.iter_batches(batch_size=64), total=pf.metadata.num_row_groups, desc=f"export {shard.name}", unit="batch", leave=False):
            d = batch.to_pydict()
            for path, audio, label, notes_json in zip(d["path"], d["audio"], d["label"], d["notes"]):
                meta = json.loads(notes_json) if notes_json else {}
                lab = normalize_label(int(label))
                system = str(meta.get("system", ""))
                if system:
                    expected = BONAFIDE if system == "real" else SPOOF
                    if expected != lab:
                        raise ManifestError(f"CD-ADD {path}: label {lab} contradicts system {system!r}")
                utt = str(meta.get("utterance_id") or Path(path).with_suffix("").as_posix())
                if reference and utt in reference and reference[utt] != lab:
                    raise ManifestError(f"CD-ADD {utt}: label differs from labels.parquet")
                data = audio["bytes"]
                ext = _audio_extension(data)
                out = audio_root / Path(path).with_suffix(ext)
                if not (out.exists() and out.stat().st_size == len(data)):
                    out.parent.mkdir(parents=True, exist_ok=True)
                    tmp = out.with_suffix(out.suffix + ".part")
                    tmp.write_bytes(data)
                    tmp.replace(out)
                rows.append(
                    {
                        "record_id": f"{CDADD_CORPUS}/{utt}",
                        "audio_path": relative_to_data_dir(out, data_dir),
                        "label": lab,
                        "corpus": CDADD_CORPUS,
                        "split": "test",
                        "utterance_id": utt,
                        "speaker_id": str(meta.get("speaker_id", "") or ""),
                        "source_domain": str(meta.get("source", "") or ""),
                        "generator_id": "bonafide" if system == "real" else system,
                        "group_id": Path(path).parent.as_posix(),
                    }
                )
    df = pd.DataFrame(rows)
    if reference:
        missing = set(reference) - set(df["utterance_id"])
        if missing:
            notes.append(f"CD-ADD: {len(missing)} ids in labels.parquet were not found in the shards.")
    notes.append(f"CD-ADD: exported {len(df)} recordings from {len(shards)} shard(s) to {audio_root}.")
    return normalize_manifest(df), notes


# ----------------------------------------------------------------- custom CSV


def load_custom_csv(path: Path, data_dir: Path, corpus: str = "custom") -> pd.DataFrame:
    """Load a user CSV with at least ``audio_path``, ``label`` and ``split`` columns.

    Relative audio paths are interpreted relative to the CSV's directory.
    ``record_id`` defaults to ``<corpus>/<audio path>``.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for col in ("audio_path", "label", "split"):
        if col not in df.columns:
            raise ManifestError(f"{path}: custom manifest needs column {col!r}")
    if "corpus" not in df.columns or (df["corpus"] == "").all():
        df["corpus"] = corpus
    resolved = []
    for p in df["audio_path"]:
        q = Path(p)
        q = q if q.is_absolute() else (path.parent / q)
        resolved.append(relative_to_data_dir(q, data_dir))
    df["audio_path"] = resolved
    if "record_id" not in df.columns or (df["record_id"] == "").any():
        df["record_id"] = df["corpus"] + "/" + df["audio_path"].map(lambda s: Path(s).with_suffix("").as_posix())
    return normalize_manifest(df)


# ------------------------------------------------------------------ downloads


def download_cdadd(out_dir: Path) -> Path:
    """Download the CD-ADD parquet package (about 3.9 GB) with huggingface_hub."""
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s to %s (about 3.9 GB)...", CDADD_REPO_ID, out_dir)
    snapshot_download(
        repo_id=CDADD_REPO_ID,
        repo_type="dataset",
        local_dir=str(out_dir),
        allow_patterns=["data/*.parquet", "README.md", "LICENSE.txt"],
    )
    return out_dir


def segmented_download(url: str, dest: Path, connections: int = 8, retries: int = 50) -> Path:
    """Resumable download using ``connections`` parallel HTTP range requests.

    Edinburgh DataShare limits the rate of each connection (~25 KB/s observed in
    Oct 2026), so parallel ranges are needed for multi-GB files. Each segment is
    stored in ``<dest>.parts/`` and resumed from its current size after an
    interruption; segments are concatenated once all are complete.
    """
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as resp:
        if resp.status != 206:
            raise RuntimeError(f"{url} does not support range requests (HTTP {resp.status})")
        final_url = resp.geturl()
        total = int(resp.headers["Content-Range"].split("/")[1])
    parts = dest.with_name(dest.name + ".parts")
    parts.mkdir(parents=True, exist_ok=True)
    step = -(-total // connections)
    bounds = [(i, s, min(s + step, total) - 1) for i, s in enumerate(range(0, total, step))]
    done = sum((parts / f"{i:03d}").stat().st_size for i, _, _ in bounds if (parts / f"{i:03d}").exists())
    lock = threading.Lock()
    bar = tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=dest.name, mininterval=5)

    def fetch(i: int, start: int, end: int) -> None:
        p = parts / f"{i:03d}"
        for attempt in range(retries):
            have = p.stat().st_size if p.exists() else 0
            if have >= end - start + 1:
                return
            try:
                req = urllib.request.Request(final_url, headers={"Range": f"bytes={start + have}-{end}"})
                with urllib.request.urlopen(req, timeout=60) as r, open(p, "ab") as f:
                    if r.status != 206:
                        raise RuntimeError(f"segment {i}: HTTP {r.status}")
                    while chunk := r.read(1 << 16):
                        f.write(chunk)
                        with lock:
                            bar.update(len(chunk))
            except Exception as e:  # network hiccup: back off and resume this segment
                log.warning("segment %d attempt %d failed: %s", i, attempt + 1, e)
                time.sleep(min(60, 2 ** attempt))
        raise RuntimeError(f"segment {i} did not complete after {retries} attempts")

    with ThreadPoolExecutor(max_workers=connections) as pool:
        list(pool.map(lambda b: fetch(*b), bounds))
    bar.close()
    tmp = dest.with_name(dest.name + ".tmp")
    with open(tmp, "wb") as out:
        for i, _, _ in bounds:
            with open(parts / f"{i:03d}", "rb") as f:
                while chunk := f.read(1 << 24):
                    out.write(chunk)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"{dest.name}: assembled {tmp.stat().st_size} bytes, expected {total}")
    tmp.replace(dest)
    for i, _, _ in bounds:
        (parts / f"{i:03d}").unlink()
    parts.rmdir()
    return dest


def download_asvspoof2019_la(out_dir: Path, keep_zip: bool = False, connections: int = 12) -> Path:
    """Download LA.zip (7.6 GB) from Edinburgh DataShare (parallel, resumable), then extract it.

    The archive extracts to ``<out_dir>/LA``. The database is distributed under
    the Open Data Commons Attribution License (ODC-By). Re-running the command
    after an interruption continues where it stopped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "LA.zip"
    if not zip_path.exists():
        segmented_download(ASVSPOOF_LA_URL, zip_path, connections=connections)
    log.info("Extracting %s ...", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    if not keep_zip:
        zip_path.unlink()
    la = out_dir / "LA"
    if not la.exists():
        raise FileNotFoundError(f"Extraction finished but {la} does not exist; inspect {out_dir}")
    return la
