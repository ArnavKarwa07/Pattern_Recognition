"""Experiment stages shared by the CLI and the scripts.

Each stage reads its inputs from disk and writes its outputs to disk, so any
stage can be re-run on its own: prepare -> validate -> (extract) -> train ->
evaluate -> figures -> report.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

import yaml

from speechdeepfake import artifacts
from speechdeepfake.config import MODEL_NAMES, ExperimentConfig, deep_merge, load_config
from speechdeepfake.data import loaders
from speechdeepfake.data.manifests import manifest_digest, read_manifest, write_manifest
from speechdeepfake.data.splits import ExperimentData, build_experiment_data, export_partitions
from speechdeepfake.data.validation import save_validation, validate_manifest, validated_paths
from speechdeepfake.utils.logging import get_logger
from speechdeepfake.utils.reproducibility import set_seed

log = get_logger("pipeline")


def required_corpora(cfg: ExperimentConfig) -> list[str]:
    names = [cfg.data.train["corpus"], cfg.data.dev["corpus"], *(t.corpus for t in cfg.test_sets())]
    return list(dict.fromkeys(names))


# ------------------------------------------------------------------ prepare


def dataset_status(cfg: ExperimentConfig) -> dict[str, dict]:
    """Which datasets are present on disk, without reading audio."""
    out = {}
    root = cfg.data_path(cfg.data.asvspoof_root)
    st = {"path": str(root), "present": bool(root and root.exists())}
    if st["present"]:
        st["protocols"] = {s: str(p) if (p := loaders.find_asvspoof_protocol(root, s)) else None for s in ("train", "dev", "eval")}
    out[loaders.ASVSPOOF_CORPUS] = st
    root = cfg.data_path(cfg.data.cdadd_root)
    shards = sorted((root / "data").glob("test-*.parquet")) if root and root.exists() else []
    out[loaders.CDADD_CORPUS] = {"path": str(root), "parquet_shards": len(shards),
                                 "exported_audio": bool(root and (root / "audio").exists())}
    if cfg.data.custom_manifest:
        p = cfg.data_path(cfg.data.custom_manifest)
        out["custom"] = {"path": str(p), "present": p.exists()}
    for name in out:
        out[name]["manifest"] = (cfg.manifests_dir / f"{name}.csv").exists()
    return out


def prepare_data(cfg: ExperimentConfig, corpora: Optional[Iterable[str]] = None) -> dict[str, str]:
    """Build normalised manifests for the requested (default: all configured) sources."""
    wanted = set(corpora or [])
    results: dict[str, str] = {}

    def want(name: str) -> bool:
        return not wanted or name in wanted

    if want(loaders.ASVSPOOF_CORPUS) and cfg.data.asvspoof_root:
        try:
            df, notes = loaders.load_asvspoof2019_la(cfg.data_path(cfg.data.asvspoof_root), cfg.data_dir)
            write_manifest(df, cfg.manifests_dir / f"{loaders.ASVSPOOF_CORPUS}.csv")
            results[loaders.ASVSPOOF_CORPUS] = " ".join(notes)
        except FileNotFoundError as e:
            results[loaders.ASVSPOOF_CORPUS] = f"NOT AVAILABLE: {e}"
    if want(loaders.CDADD_CORPUS) and cfg.data.cdadd_root:
        try:
            df, notes = loaders.export_cdadd_parquet(cfg.data_path(cfg.data.cdadd_root), cfg.data_dir)
            write_manifest(df, cfg.manifests_dir / f"{loaders.CDADD_CORPUS}.csv")
            results[loaders.CDADD_CORPUS] = " ".join(notes)
        except FileNotFoundError as e:
            results[loaders.CDADD_CORPUS] = f"NOT AVAILABLE: {e}"
    if cfg.data.custom_manifest and (not wanted or "custom" in wanted):
        df = loaders.load_custom_csv(cfg.data_path(cfg.data.custom_manifest), cfg.data_dir)
        for corpus, g in df.groupby("corpus"):
            write_manifest(g, cfg.manifests_dir / f"{corpus}.csv")
            results[corpus] = f"custom manifest: {len(g)} rows"
    for k, v in results.items():
        (log.warning if v.startswith("NOT AVAILABLE") else log.info)("%s: %s", k, v)
    return results


# ----------------------------------------------------------------- validate


def validate_data(cfg: ExperimentConfig, workers: int = 8, compute_hash: bool = True, force: bool = False) -> dict[str, dict]:
    """Validate every corpus manifest the configuration needs."""
    summaries = {}
    for corpus in required_corpora(cfg):
        path = cfg.manifests_dir / f"{corpus}.csv"
        if not path.exists():
            log.warning("%s: no manifest at %s; run `speechdeepfake prepare-data` (dataset may be unavailable).", corpus, path)
            continue
        df = read_manifest(path)
        csv_path, json_path = validated_paths(cfg, corpus)
        if not force and json_path.exists():
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if old.get("manifest_digest") == manifest_digest(df):
                log.info("%s: already validated (use --force to re-check)", corpus)
                summaries[corpus] = old
                continue
        log.info("%s: validating %d files...", corpus, len(df))
        checked, summary = validate_manifest(df, cfg, workers=workers, compute_hash=compute_hash)
        save_validation(checked, summary, cfg, corpus)
        summaries[corpus] = summary
    return summaries


# ---------------------------------------------------------- train/evaluate


def load_data(cfg: ExperimentConfig) -> ExperimentData:
    data = build_experiment_data(cfg)
    export_partitions(data, artifacts.splits_dir(cfg))
    for name, df in data.partitions().items():
        log.info("partition %-22s n=%-7d bonafide=%-6d spoof=%d", name, len(df), int((df.label == 0).sum()), int((df.label == 1).sum()))
    return data


def expand_models(model: str) -> list[str]:
    return list(MODEL_NAMES) if model == "all" else [model]


def extract_features(cfg: ExperimentConfig, models: list[str], data: Optional[ExperimentData] = None) -> None:
    from speechdeepfake.training.train_svm import SVM_MODELS, extract

    data = data or load_data(cfg)
    for m in models:
        if m not in SVM_MODELS:
            log.info("%s computes log-Mel spectrograms on the fly; nothing to cache.", m)
            continue
        for name, df in data.partitions().items():
            fs = extract(m, cfg, df, name)
            log.info("%s/%s: %d vectors (%d failed), %.1fs extraction", m, name, len(fs.ids), len(fs.failed), fs.extraction_seconds)


def train(cfg: ExperimentConfig, models: list[str], resume: bool = False, data: Optional[ExperimentData] = None) -> dict:
    set_seed(cfg.seed)
    data = data or load_data(cfg)
    out = {}
    for m in models:
        if m == "logmel_cnn":
            from speechdeepfake.training.train_cnn import train_cnn

            out[m] = train_cnn(cfg, data, resume=resume)
        else:
            from speechdeepfake.training.train_svm import train_svm_model

            out[m] = train_svm_model(m, cfg, data)
    return out


def evaluate(cfg: ExperimentConfig, models: list[str], data: Optional[ExperimentData] = None):
    from speechdeepfake.evaluation.predict import compile_results, evaluate_model

    set_seed(cfg.seed)
    data = data or load_data(cfg)
    if not data.tests:
        raise RuntimeError(f"No test set is available: {data.skipped_tests}")
    for m in models:
        evaluate_model(m, cfg, data)
    return compile_results(cfg)


def figures_and_report(cfg: ExperimentConfig) -> tuple[list[Path], Path]:
    from speechdeepfake.evaluation.figures import generate_figures
    from speechdeepfake.evaluation.report import write_report

    made, _ = generate_figures(cfg)
    return made, write_report(cfg)


def run_all(cfg: ExperimentConfig, resume: bool = False) -> Path:
    data = load_data(cfg)
    train(cfg, list(MODEL_NAMES), resume=resume, data=data)
    evaluate(cfg, list(MODEL_NAMES), data=data)
    _, report = figures_and_report(cfg)
    return report


# ---------------------------------------------------------------- smoke test


def smoke_test_config(base: ExperimentConfig, out_root: Path, wav2vec_model: Optional[str] = None) -> ExperimentConfig:
    """Derive a configuration that runs every stage on the synthetic fixture."""
    from speechdeepfake.data.synthetic import FIXTURE_CORPUS

    overrides = {
        "run_name": "smoke_test",
        "run_type": "smoke_test",
        "feature_chunk_size": 8,
        "paths": {"data_dir": str(out_root / "data"), "models_dir": str(out_root / "models"),
                  "results_dir": str(out_root / "results"), "reports_dir": str(out_root / "report")},
        "mfcc": {"n_jobs": 1},
        "svm": {"C_grid": [1.0], "gamma_grid": ["scale"]},
        "cnn": {"epochs": min(base.cnn.epochs, 3), "num_workers": 0, "norm_stats_samples": 16, "batch_size": 8},
        "wav2vec": {"model_name": wav2vec_model or base.wav2vec.model_name, "batch_size": 4},
        "data": {
            "asvspoof_root": None, "cdadd_root": None, "custom_manifest": str(out_root / "data" / "fixture" / "manifest.csv"),
            "train": {"corpus": FIXTURE_CORPUS, "split": "train"},
            "dev": {"corpus": FIXTURE_CORPUS, "split": "dev"},
            "test_sets": [{"name": "synthetic_test", "corpus": FIXTURE_CORPUS, "split": "test", "condition": "in_domain"},
                          {"name": "synthetic_external", "corpus": FIXTURE_CORPUS, "split": "external",
                           "condition": "external_domain"}],
            "max_train_samples": None, "max_dev_samples": None, "max_test_samples": None, "holdout_generators": [],
        },
        "evaluation": {"n_bootstrap": 20},
    }
    values = base.to_dict()
    values.pop("config_path", None)
    tmp = out_root / "smoke_config.yaml"
    out_root.mkdir(parents=True, exist_ok=True)
    tmp.write_text(yaml.safe_dump(deep_merge(values, overrides), sort_keys=False), encoding="utf-8")
    cfg = load_config(tmp)
    cfg.project_root = base.project_root
    return cfg


def smoke_test(base: ExperimentConfig, out_root: Path, wav2vec_model: Optional[str] = None, n_per_class: int = 8) -> dict:
    """Run prepare/validate/train/evaluate/figures/report on a synthetic fixture.

    Returns a dict of checks; raises on any failure. Outputs go to
    ``out_root`` (default ``results/smoke_test``) and are clearly labelled.
    """
    from speechdeepfake.data.synthetic import make_fixture

    for sub in ("data", "models", "results", "report"):  # leave logs/ (may be open)
        if (out_root / sub).exists():
            shutil.rmtree(out_root / sub)
    cfg = smoke_test_config(base, out_root, wav2vec_model)
    make_fixture(out_root / "data" / "fixture", n_per_class=n_per_class, seed=cfg.seed)
    prepare_data(cfg)
    validate_data(cfg, workers=4)
    data = load_data(cfg)
    train(cfg, list(MODEL_NAMES), data=data)
    table = evaluate(cfg, list(MODEL_NAMES), data=data)
    made, report = figures_and_report(cfg)
    checks = {
        "models_evaluated": sorted(table["model"].unique().tolist()),
        "n_test_records": int(table["n_common"].iloc[0]),
        "metrics_finite": bool(table[["eer", "roc_auc", "balanced_accuracy", "f1"]].notna().all().all()),
        "figures": len(made),
        "report": str(report),
    }
    if checks["models_evaluated"] != sorted(MODEL_NAMES) or not checks["metrics_finite"]:
        raise RuntimeError(f"Smoke test failed: {checks}")
    return checks
