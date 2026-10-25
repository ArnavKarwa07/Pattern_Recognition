"""Assemble the experiment partitions (train / dev / test sets) from validated manifests.

Official partitions are used as-is: nothing is merged or randomly
re-partitioned. Pilot runs may take stratified random subsets, and are
labelled as such. Every assembled experiment is checked for leakage across
all of its partitions before anything expensive runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from speechdeepfake.config import ExperimentConfig, TestSetConfig
from speechdeepfake.data.manifests import check_integrity, label_counts, write_manifest
from speechdeepfake.data.validation import load_corpus, usable
from speechdeepfake.utils.logging import get_logger

log = get_logger("data.splits")


class SplitError(RuntimeError):
    pass


@dataclass
class ExperimentData:
    train: pd.DataFrame
    dev: pd.DataFrame
    tests: dict[str, pd.DataFrame] = field(default_factory=dict)
    test_configs: dict[str, TestSetConfig] = field(default_factory=dict)
    skipped_tests: dict[str, str] = field(default_factory=dict)
    integrity: dict = field(default_factory=dict)

    def partitions(self) -> dict[str, pd.DataFrame]:
        return {"train": self.train, "dev": self.dev, **self.tests}

    def summary(self) -> dict:
        out = {}
        for name, df in self.partitions().items():
            out[name] = {"n": int(len(df)), **label_counts(df)}
        return {"partitions": out, "skipped_tests": self.skipped_tests, "integrity": self.integrity}


def stratified_subsample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    """Deterministic class-stratified random subset of at most ``n`` rows."""
    if n is None or len(df) <= n:
        return df
    frac = n / len(df)
    parts = [
        g.sample(n=max(1, int(round(len(g) * frac))), random_state=seed) for _, g in df.groupby("label", sort=True)
    ]
    return pd.concat(parts).sort_values("record_id").reset_index(drop=True)


def _select(corpora: dict[str, pd.DataFrame], cfg: ExperimentConfig, corpus: str, split: str) -> pd.DataFrame:
    if corpus not in corpora:
        corpora[corpus] = load_corpus(cfg, corpus)
    df = corpora[corpus]
    sel = df[df["split"] == split]
    if sel.empty:
        available = sorted(df["split"].unique())
        raise SplitError(f"Corpus {corpus!r} has no split {split!r} (available: {available})")
    return usable(sel)


def build_experiment_data(cfg: ExperimentConfig) -> ExperimentData:
    """Load train/dev/test partitions according to ``cfg.data``."""
    corpora: dict[str, pd.DataFrame] = {}
    seed = cfg.seed
    train = _select(corpora, cfg, cfg.data.train["corpus"], cfg.data.train["split"])
    dev = _select(corpora, cfg, cfg.data.dev["corpus"], cfg.data.dev["split"])
    holdout = set(cfg.data.holdout_generators or [])
    if holdout:
        train = train[~train["generator_id"].isin(holdout)].reset_index(drop=True)
        dev = dev[~dev["generator_id"].isin(holdout)].reset_index(drop=True)
        log.info("Held-out generators removed from train/dev: %s", sorted(holdout))
    train = stratified_subsample(train, cfg.data.max_train_samples, seed)
    dev = stratified_subsample(dev, cfg.data.max_dev_samples, seed)
    for name, df in (("train", train), ("dev", dev)):
        if df["label"].nunique() < 2:
            raise SplitError(f"{name} partition must contain both classes; got {label_counts(df)}")

    data = ExperimentData(train=train, dev=dev)
    for t in cfg.test_sets():
        try:
            df = _select(corpora, cfg, t.corpus, t.split)
        except (FileNotFoundError, SplitError) as e:
            data.skipped_tests[t.name] = str(e).splitlines()[0]
            log.warning("Test set %s unavailable: %s", t.name, data.skipped_tests[t.name])
            continue
        if t.generators:
            df = df[df["generator_id"].isin(set(t.generators) | {"bonafide"})].reset_index(drop=True)
            leaked = set(t.generators) & set(train["generator_id"]) | set(t.generators) & set(dev["generator_id"])
            if t.condition == "generator_held_out" and leaked:
                raise SplitError(f"Test set {t.name}: generators {sorted(leaked)} also occur in train/dev")
        df = stratified_subsample(df, cfg.data.max_test_samples, seed)
        if df.empty:
            data.skipped_tests[t.name] = "no usable records"
            continue
        data.tests[t.name] = df
        data.test_configs[t.name] = t

    parts = []
    for name, df in data.partitions().items():
        d = df.copy()
        d["split"] = name  # treat each experiment partition as its own split for the check
        parts.append(d)
    rep = check_integrity(pd.concat(parts, ignore_index=True))
    data.integrity = rep.to_dict()
    for w in rep.warnings:
        log.warning("Split check: %s", w)
    if not rep.ok:
        raise SplitError("Experiment partitions failed integrity checks:\n  " + "\n  ".join(rep.errors))
    return data


def export_partitions(data: ExperimentData, out_dir: Path) -> None:
    """Save the exact record lists of every partition and a JSON summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.partitions().items():
        write_manifest(df, out_dir / f"{name}.csv")
    (out_dir / "split_summary.json").write_text(json.dumps(data.summary(), indent=2), encoding="utf-8")
