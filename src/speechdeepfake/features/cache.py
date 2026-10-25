"""Resumable on-disk cache for fixed-length per-recording feature vectors.

Layout::

    <cache_dir>/<feature_name>/<config hash>/config.json
    <cache_dir>/<feature_name>/<config hash>/chunk_<ids hash>.npz

Records are sorted by ``record_id`` and processed in chunks; each chunk is
written atomically as soon as it is computed, so an interrupted extraction
resumes from the last finished chunk. A chunk stores the record ids it
covers, so cached vectors are always linked to their source recordings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from speechdeepfake.config import stable_hash
from speechdeepfake.data.validation import resolve_audio_path
from speechdeepfake.utils.logging import ErrorLog, get_logger
from speechdeepfake.utils.timing import Timer

log = get_logger("features.cache")

# An extractor maps a list of audio paths to one result per path: a 1-D
# feature vector, or an Exception describing why that file failed.
Extractor = Callable[[Sequence[Path]], list[Union[np.ndarray, Exception]]]


@dataclass
class FeatureSet:
    ids: list[str]
    X: np.ndarray
    failed: dict[str, str] = field(default_factory=dict)
    # Extraction time as measured when the chunks were computed (cached or not).
    extraction_seconds: float = 0.0
    computed_now: int = 0

    def align(self, records: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        """Return the records that have features, in feature order, with their matrix."""
        idx = {rid: i for i, rid in enumerate(self.ids)}
        keep = records[records["record_id"].isin(idx)].copy()
        keep["_order"] = keep["record_id"].map(idx)
        keep = keep.sort_values("_order").drop(columns="_order").reset_index(drop=True)
        return keep, self.X[[idx[r] for r in keep["record_id"]]]


def cache_directory(cache_root: Path, feature_name: str, config: dict) -> Path:
    d = Path(cache_root) / feature_name / stable_hash(config)
    d.mkdir(parents=True, exist_ok=True)
    cfg_file = d / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text(json.dumps(config, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return d


def _chunk_name(ids: Sequence[str]) -> str:
    return "chunk_" + hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()[:16] + ".npz"


def cached_features(
    records: pd.DataFrame,
    extractor: Extractor,
    cache_root: Path,
    feature_name: str,
    config: dict,
    data_dir: Path,
    chunk_size: int = 512,
    desc: str = "features",
    error_log: Optional[ErrorLog] = None,
) -> FeatureSet:
    """Compute (or load) features for every record; failed files are logged, not hidden."""
    d = cache_directory(cache_root, feature_name, config)
    recs = records.sort_values("record_id").reset_index(drop=True)
    ids_all = recs["record_id"].tolist()
    paths_all = [resolve_audio_path(p, data_dir) for p in recs["audio_path"]]
    out_ids: list[str] = []
    blocks: list[np.ndarray] = []
    failed: dict[str, str] = {}
    seconds = 0.0
    computed = 0
    starts = range(0, len(ids_all), chunk_size)
    for start in tqdm(starts, desc=desc, unit="chunk", total=len(starts)):
        ids = ids_all[start : start + chunk_size]
        file = d / _chunk_name(ids)
        if file.exists():
            with np.load(file, allow_pickle=False) as z:
                ok_ids, X, bad_ids, bad_msg, secs = (
                    z["ids"].tolist(), z["X"], z["bad_ids"].tolist(), z["bad_msg"].tolist(), float(z["seconds"])
                )
        else:
            with Timer() as t:
                results = extractor(paths_all[start : start + chunk_size])
            secs = t.seconds
            computed += len(ids)
            ok_ids, rows, bad_ids, bad_msg = [], [], [], []
            for rid, res in zip(ids, results):
                if isinstance(res, Exception) or not np.all(np.isfinite(res)):
                    bad_ids.append(rid)
                    bad_msg.append(str(res) if isinstance(res, Exception) else "non-finite feature values")
                else:
                    ok_ids.append(rid)
                    rows.append(np.asarray(res, dtype=np.float32))
            X = np.stack(rows) if rows else np.zeros((0, 0), dtype=np.float32)
            tmp = file.with_name(file.name + ".tmp.npz")
            np.savez(tmp, ids=np.array(ok_ids, dtype=str), X=X, bad_ids=np.array(bad_ids, dtype=str),
                     bad_msg=np.array(bad_msg, dtype=str), seconds=np.array(secs))
            tmp.replace(file)
            if error_log is not None:
                path_of = dict(zip(ids, paths_all[start : start + chunk_size]))
                for rid, msg in zip(bad_ids, bad_msg):
                    error_log.add(feature_name, rid, str(path_of[rid]), msg)
        seconds += secs
        out_ids.extend(ok_ids)
        if len(ok_ids):
            blocks.append(X)
        failed.update(zip(bad_ids, bad_msg))
    if failed:
        log.warning("%s: %d recording(s) failed feature extraction (see errors.csv)", feature_name, len(failed))
    X_all = np.concatenate(blocks) if blocks else np.zeros((0, 0), dtype=np.float32)
    return FeatureSet(ids=out_ids, X=X_all, failed=failed, extraction_seconds=seconds, computed_now=computed)
