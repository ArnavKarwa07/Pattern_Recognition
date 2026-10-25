"""Download small, stratified *subsets* of the datasets without fetching them in full.

Intended for demonstrations and pilot runs (e.g. a course presentation) where
the full ASVspoof 2019 LA archive (7.6 GB, throttled server) and CD-ADD
(3.9 GB) are unnecessary.

* **ASVspoof 2019 LA** — ``LA.zip`` is read remotely with HTTP range requests:
  only the zip central directory, the official CM protocol files and the
  selected FLAC members are transferred. The selection is stratified per split:
  ``n_bonafide`` genuine utterances plus ``n_per_attack`` utterances of every
  attack (A01–A06 in train/dev, A07–A19 in eval). The protocol files written
  to ``ASVspoof2019_LA_cm_protocols/`` are the official lines of the selected
  utterances only; the untouched originals are kept in ``.../full/``.
* **CD-ADD** — ``n_row_groups`` random 100-row groups are read from each of the
  8 parquet shards (so LibriTTS and TED, and all generators, are represented)
  and saved as one local shard with the original schema.

Each subset directory gets a ``SUBSET.json`` describing exactly what was
selected (seed, counts, source). Results computed on these subsets must be
reported as subset/pilot results.
"""

from __future__ import annotations

import io
import json
import struct
import threading
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from speechdeepfake.data.loaders import ASVSPOOF_LA_URL, CDADD_REPO_ID, parse_asvspoof_protocol
from speechdeepfake.utils.logging import get_logger

log = get_logger("data.subsets")


def _get_range(url: str, start: int, end: int, retries: int = 8) -> bytes:
    """Bytes ``start..end`` (inclusive) of ``url``, retrying on network errors."""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise RuntimeError(f"HTTP {r.status} for range request")
                data = r.read()
            if len(data) != end - start + 1:
                raise RuntimeError(f"short read {len(data)} != {end - start + 1}")
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning("range %d-%d failed (%s); retrying", start, end, e)
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests.

    Large reads (such as a zip central directory) are split into parallel
    range requests because the server throttles each connection.
    """

    def __init__(self, url: str, connections: int = 12, split_bytes: int = 1 << 20):
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"})) as r:
            if r.status != 206:
                raise RuntimeError(f"{url} does not support range requests")
            self.url = r.geturl()
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0
        self.connections = connections
        self.split_bytes = split_bytes

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        if end <= self.pos:
            return b""
        spans = [(s, min(s + self.split_bytes, end) - 1) for s in range(self.pos, end, self.split_bytes)]
        if len(spans) == 1:
            data = _get_range(self.url, *spans[0])
        else:
            with ThreadPoolExecutor(self.connections) as pool:
                data = b"".join(pool.map(lambda sp: _get_range(self.url, *sp), spans))
        self.pos = end
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _fetch_member(url: str, info, margin: int = 4096) -> bytes:
    """Download and decompress one zip member using a single range request."""
    start = info.header_offset
    need = 30 + len(info.orig_filename.encode("utf-8")) + info.compress_size + margin
    buf = _get_range(url, start, start + need - 1)
    sig, *_ = struct.unpack("<I", buf[:4])
    if sig != 0x04034B50:
        raise RuntimeError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", buf[26:30])
    off = 30 + name_len + extra_len
    if off + info.compress_size > len(buf):
        buf += _get_range(url, start + len(buf), start + off + info.compress_size - 1)
    raw = buf[off : off + info.compress_size]
    if info.compress_type == 0:
        data = raw
    elif info.compress_type == 8:
        data = zlib.decompressobj(-15).decompress(raw)
    else:
        raise RuntimeError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise RuntimeError(f"CRC mismatch for {info.filename}")
    return data


def _stratified_ids(df: pd.DataFrame, n_bonafide: int, n_per_attack: int, rng: np.random.Generator) -> list[str]:
    out = []
    for gen, g in df.groupby("generator_id", sort=True):
        k = n_bonafide if gen == "bonafide" else n_per_attack
        idx = rng.permutation(len(g))[: min(k, len(g))]
        out.extend(g["utterance_id"].iloc[idx].tolist())
    return sorted(out)


def fetch_asvspoof2019_la_subset(
    out_dir: Path,
    n_bonafide: dict[str, int],
    n_per_attack: dict[str, int],
    seed: int = 1234,
    connections: int = 12,
) -> Path:
    """Fetch a stratified subset of ASVspoof 2019 LA into ``<out_dir>/LA`` (see module doc)."""
    import zipfile

    la = Path(out_dir) / "LA"
    log.info("Opening remote LA.zip (reading the central directory only)...")
    remote = HTTPRangeFile(ASVSPOOF_LA_URL, connections=connections)
    zf = zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16))
    infos = {i.filename: i for i in zf.infolist()}
    rng = np.random.default_rng(seed)
    summary = {"source": ASVSPOOF_LA_URL, "seed": seed, "archive_bytes": remote.size, "splits": {}}
    wanted: list[tuple[str, Path]] = []
    proto_dir = la / "ASVspoof2019_LA_cm_protocols"
    (proto_dir / "full").mkdir(parents=True, exist_ok=True)
    for split, suffix in (("train", "train.trn"), ("dev", "dev.trl"), ("eval", "eval.trl")):
        name = f"LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{suffix}.txt"
        if name not in infos:
            raise FileNotFoundError(f"{name} not found in LA.zip")
        full_path = proto_dir / "full" / Path(name).name
        full_path.write_bytes(_fetch_member(remote.url, infos[name]))
        df, has_keys = parse_asvspoof_protocol(full_path, split)
        if not has_keys:
            log.warning("%s protocol has no keys; split skipped", split)
            continue
        ids = _stratified_ids(df, n_bonafide[split], n_per_attack[split], rng)
        keep = set(ids)
        lines = [ln for ln in full_path.read_text(encoding="utf-8").splitlines() if any(t in keep for t in ln.split())]
        (proto_dir / Path(name).name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        sub = df[df["utterance_id"].isin(keep)]
        summary["splits"][split] = {
            "available_in_full_protocol": int(len(df)),
            "selected": int(len(sub)),
            "selected_by_generator": {k: int(v) for k, v in sub["generator_id"].value_counts().sort_index().items()},
        }
        for u in ids:
            member = f"LA/ASVspoof2019_LA_{split}/flac/{u}.flac"
            wanted.append((member, Path(out_dir) / member))

    todo = [(m, p) for m, p in wanted if not p.exists()]
    log.info("Fetching %d of %d selected FLAC files...", len(todo), len(wanted))
    lock = threading.Lock()
    bar = tqdm(total=len(todo), unit="file", desc="ASVspoof subset", mininterval=2)

    def get(item):
        member, path = item
        data = _fetch_member(remote.url, infos[member])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        with lock:
            bar.update(1)
        return len(data)

    with ThreadPoolExecutor(connections) as pool:
        nbytes = sum(pool.map(get, todo))
    bar.close()
    summary["downloaded_audio_bytes_this_run"] = nbytes
    (la / "SUBSET.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return la


def fetch_cdadd_subset(out_root: Path, n_row_groups: int = 1, seed: int = 1234) -> Path:
    """Read ``n_row_groups`` random row groups from each CD-ADD shard into one local shard."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    base = f"datasets/{CDADD_REPO_ID}/data"
    shards = sorted(p for p in fs.ls(base, detail=False) if Path(p).name.startswith("test-"))
    rng = np.random.default_rng(seed)
    tables, picked = [], {}
    for shard in tqdm(shards, desc="CD-ADD subset", unit="shard"):
        with fs.open(shard, "rb") as f:
            pf = pq.ParquetFile(f)
            groups = sorted(rng.choice(pf.num_row_groups, size=min(n_row_groups, pf.num_row_groups), replace=False).tolist())
            picked[Path(shard).name] = {"row_groups": groups, "of": pf.num_row_groups}
            tables.append(pf.read_row_groups(groups))
    table = pa.concat_tables(tables)
    out = Path(out_root) / "data"
    out.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, out / "test-subset-00000.parquet")
    labels = table.column("label").to_numpy()
    (Path(out_root) / "SUBSET.json").write_text(json.dumps({
        "source": f"https://huggingface.co/datasets/{CDADD_REPO_ID}", "seed": seed,
        "row_groups_per_shard": n_row_groups, "picked": picked, "rows": table.num_rows,
        "bonafide": int((labels == 0).sum()), "spoof": int((labels == 1).sum()),
    }, indent=2), encoding="utf-8")
    return out
