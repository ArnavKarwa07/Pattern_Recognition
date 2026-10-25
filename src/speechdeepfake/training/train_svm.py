"""Train MFCC+SVM or wav2vec+SVM: cached features -> dev-tuned SVM -> dev threshold."""

from __future__ import annotations

import dataclasses
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from speechdeepfake import artifacts
from speechdeepfake.config import ExperimentConfig
from speechdeepfake.data.splits import ExperimentData
from speechdeepfake.features.cache import FeatureSet, cached_features
from speechdeepfake.models import mfcc_svm, wav2vec_svm
from speechdeepfake.models.svm import svm_scores, tune_svm
from speechdeepfake.utils.logging import get_logger
from speechdeepfake.utils.reproducibility import environment_info, resolve_device
from speechdeepfake.utils.timing import Timer

log = get_logger("training.svm")

SVM_MODELS = {mfcc_svm.MODEL_NAME: mfcc_svm, wav2vec_svm.MODEL_NAME: wav2vec_svm}


def make_extractor(model_name: str, cfg: ExperimentConfig):
    device = resolve_device(cfg.device)
    return SVM_MODELS[model_name].make_extractor(cfg, device)


def extract(model_name: str, cfg: ExperimentConfig, df: pd.DataFrame, partition: str, extractor=None) -> FeatureSet:
    """Cached features for one partition (computed only for missing chunks)."""
    extractor = extractor or make_extractor(model_name, cfg)
    return cached_features(
        df,
        extractor,
        cache_root=cfg.cache_dir,
        feature_name=extractor.name,
        config=extractor.config_dict(),
        data_dir=cfg.data_dir,
        chunk_size=cfg.feature_chunk_size,
        desc=f"{model_name}:{partition}",
        error_log=artifacts.error_log(cfg),
    )


def train_svm_model(model_name: str, cfg: ExperimentConfig, data: ExperimentData) -> dict:
    """Fit, tune on dev, choose the dev threshold and save the model artifacts."""
    if model_name not in SVM_MODELS:
        raise ValueError(f"{model_name} is not an SVM model")
    out = artifacts.model_dir(cfg, model_name)
    out.mkdir(parents=True, exist_ok=True)
    extractor = make_extractor(model_name, cfg)
    fs_train = extract(model_name, cfg, data.train, "train", extractor)
    fs_dev = extract(model_name, cfg, data.dev, "dev", extractor)
    train_df, X_train = fs_train.align(data.train)
    dev_df, X_dev = fs_dev.align(data.dev)
    y_train, y_dev = train_df["label"].to_numpy(), dev_df["label"].to_numpy()
    log.info("%s: training on %d recordings (%d dims), tuning on %d dev recordings",
             model_name, len(y_train), X_train.shape[1], len(y_dev))

    with Timer() as t:
        pipe, tuning, best = tune_svm(X_train, y_train, X_dev, y_dev, cfg.svm, cfg.evaluation.threshold_method)
    s_dev = svm_scores(pipe, X_dev)
    joblib.dump(pipe, out / "pipeline.joblib")
    tuning.to_csv(out / "tuning.csv", index=False)
    pd.DataFrame({
        "record_id": dev_df["record_id"], "label": y_dev, "score": s_dev,
        "predicted": (s_dev >= best["threshold"]).astype(int), "threshold": best["threshold"],
    }).to_csv(out / "dev_predictions.csv", index=False)

    meta = {
        "model": model_name,
        "display_name": SVM_MODELS[model_name].DISPLAY_NAME,
        "run_name": cfg.run_name,
        "run_type": cfg.run_type,
        "feature_config": extractor.config_dict(),
        "feature_dim": int(X_train.shape[1]),
        "svm_config": dataclasses.asdict(cfg.svm),
        "selected": {k: best[k] for k in ("C", "gamma_spec", "gamma")},
        "threshold": best["threshold"],
        "threshold_method": cfg.evaluation.threshold_method,
        "threshold_fitted_on": "dev",
        "dev_metrics": best["dev_metrics"],
        "n_train": int(len(y_train)),
        "n_train_failed": len(fs_train.failed),
        "n_dev": int(len(y_dev)),
        "n_dev_failed": len(fs_dev.failed),
        "train_class_counts": {"bonafide": int((y_train == 0).sum()), "spoof": int((y_train == 1).sum())},
        "training_seconds_total_grid": t.seconds,
        "training_seconds_selected": best["fit_seconds"],
        "feature_extraction_seconds": {"train": fs_train.extraction_seconds, "dev": fs_dev.extraction_seconds},
        "environment": environment_info(),
    }
    if model_name == wav2vec_svm.MODEL_NAME:
        meta["wav2vec"] = extractor.metadata()
    artifacts.save_json(meta, out / "metadata.json")
    cfg.save_snapshot(out / "config_snapshot.yaml")
    log.info("%s saved to %s (dev EER %.4f, threshold %.4f)", model_name, out, best["dev_metrics"]["eer"], best["threshold"])
    return meta


def load_svm(model_name: str, cfg: ExperimentConfig) -> tuple[object, dict]:
    d = artifacts.model_dir(cfg, model_name)
    path = d / "pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{model_name} is not trained ({path} missing). Run `speechdeepfake train --model {model_name}`.")
    return joblib.load(path), artifacts.load_json(d / "metadata.json")


def predict_svm(model_name: str, cfg: ExperimentConfig, df: pd.DataFrame, partition: str) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Scores for ``df``: (records with features, scores, timing)."""
    pipe, meta = load_svm(model_name, cfg)
    extractor: Optional[object] = make_extractor(model_name, cfg)
    if extractor.config_dict() != meta["feature_config"]:
        raise RuntimeError(f"{model_name}: current feature config differs from the one used in training; retrain.")
    fs = extract(model_name, cfg, df, partition, extractor)
    recs, X = fs.align(df)
    with Timer() as t:
        scores = svm_scores(pipe, X)
    timing = {"feature_seconds": fs.extraction_seconds, "classifier_seconds": t.seconds, "n_failed": len(fs.failed)}
    return recs, scores, timing
