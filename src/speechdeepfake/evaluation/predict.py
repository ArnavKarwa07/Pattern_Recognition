"""Score trained models on the configured test sets and aggregate the results.

Per model and test set this writes
``results/<run>/predictions/<model>__<test>.csv`` (one row per recording) and
``results/<run>/metrics/<model>__<test>.json``. :func:`compile_results` then
builds the comparison tables on the records that *all* evaluated models
scored, so that every model is compared on identical recordings.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from speechdeepfake import artifacts
from speechdeepfake.config import MODEL_NAMES, ExperimentConfig
from speechdeepfake.data.splits import ExperimentData
from speechdeepfake.evaluation.metrics import binary_metrics, bootstrap_ci, eer_only, subgroup_metrics
from speechdeepfake.utils.logging import get_logger
from speechdeepfake.utils.reproducibility import environment_info

log = get_logger("evaluation.predict")

DISPLAY = {"mfcc_svm": "MFCC + SVM", "logmel_cnn": "log-Mel + CNN", "wav2vec_svm": "wav2vec 2.0 + SVM"}
PRED_COLUMNS = ["record_id", "label", "predicted", "score", "threshold", "dataset", "corpus", "split", "condition",
                "model", "generator_id", "source_domain", "speaker_id", "group_id"]


def _predict(model: str, cfg: ExperimentConfig, df: pd.DataFrame, partition: str):
    if model == "logmel_cnn":
        from speechdeepfake.training.train_cnn import predict_cnn

        return predict_cnn(cfg, df, partition)
    from speechdeepfake.training.train_svm import predict_svm

    return predict_svm(model, cfg, df, partition)


def _model_meta(cfg: ExperimentConfig, model: str) -> dict:
    path = artifacts.model_dir(cfg, model) / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(f"{model} has not been trained for run {cfg.run_name!r} ({path} missing).")
    return artifacts.load_json(path)


def evaluate_model(model: str, cfg: ExperimentConfig, data: ExperimentData) -> list[dict]:
    """Score every available test set with the dev-selected threshold (never re-tuned)."""
    meta = _model_meta(cfg, model)
    threshold = float(meta["threshold"])
    assert meta.get("threshold_fitted_on") == "dev", "threshold must come from the development set"
    results = []
    for name, df in data.tests.items():
        tcfg = data.test_configs[name]
        recs, scores, timing = _predict(model, cfg, df, name)
        pred = recs.copy()
        pred["score"] = scores
        pred["threshold"] = threshold
        pred["predicted"] = (scores >= threshold).astype(int)
        pred["dataset"] = name
        pred["condition"] = tcfg.condition
        pred["model"] = model
        pred = pred[[c for c in PRED_COLUMNS if c in pred.columns]]
        out = artifacts.predictions_dir(cfg) / f"{model}__{name}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        pred.to_csv(out, index=False)
        m = binary_metrics(pred["label"], pred["score"], threshold)
        n = len(pred)
        total = timing["feature_seconds"] + timing["classifier_seconds"]
        res = {
            "model": model,
            "display_name": DISPLAY[model],
            "run_name": cfg.run_name,
            "run_type": cfg.run_type,
            "dataset": name,
            "corpus": tcfg.corpus,
            "split": tcfg.split,
            "condition": tcfg.condition,
            "n_requested": int(len(df)),
            "n_failed": int(timing["n_failed"]),
            **m,
            "threshold_method": meta["threshold_method"],
            "threshold_source": "dev",
            "feature_seconds": timing["feature_seconds"],
            "classifier_seconds": timing["classifier_seconds"],
            "inference_ms_per_recording": 1000 * total / n if n else float("nan"),
            "training_seconds": meta.get("training_seconds", meta.get("training_seconds_selected")),
            "train_feature_seconds": (meta.get("feature_extraction_seconds") or {}).get("train"),
            "environment": environment_info(),
        }
        artifacts.save_json(res, artifacts.metrics_dir(cfg) / f"{model}__{name}.json")
        log.info("%s on %s: n=%d EER=%.4f AUC=%.4f BA=%.4f (threshold %.4f from dev)", model, name, n, m["eer"],
                 m["roc_auc"], m["balanced_accuracy"], threshold)
        results.append(res)
    return results


def load_predictions(cfg: ExperimentConfig, models=MODEL_NAMES) -> dict[tuple[str, str], pd.DataFrame]:
    out = {}
    d = artifacts.predictions_dir(cfg)
    for f in sorted(d.glob("*__*.csv")) if d.exists() else []:
        model, dataset = f.stem.split("__", 1)
        if model in models:
            out[(model, dataset)] = pd.read_csv(f, dtype={"record_id": str, "generator_id": str, "source_domain": str,
                                                         "speaker_id": str, "group_id": str}, keep_default_na=False)
    return out


def common_records(preds: dict[tuple[str, str], pd.DataFrame], dataset: str) -> Optional[set]:
    sets = [set(p["record_id"]) for (m, d), p in preds.items() if d == dataset]
    return set.intersection(*sets) if sets else None


def compile_results(cfg: ExperimentConfig) -> pd.DataFrame:
    """Aggregate metrics on the common record set per test set, with bootstrap CIs and subgroups."""
    preds = load_predictions(cfg)
    if not preds:
        raise FileNotFoundError(f"No predictions in {artifacts.predictions_dir(cfg)}; run `speechdeepfake evaluate-all` first.")
    rows = []
    sub_rows = {"generator_id": [], "source_domain": []}
    ev = cfg.evaluation
    for (model, dataset), p in sorted(preds.items()):
        common = common_records(preds, dataset)
        q = p[p["record_id"].isin(common)].reset_index(drop=True)
        thr = float(q["threshold"].iloc[0])
        m = binary_metrics(q["label"], q["score"], thr)
        groups = q["group_id"].to_numpy() if (q["group_id"] != "").all() else None
        lo, hi = bootstrap_ci(q["label"].to_numpy(), q["score"].to_numpy(), eer_only, ev.n_bootstrap, ev.ci_level,
                              groups=groups, seed=cfg.seed)
        per_model = artifacts.load_json(artifacts.metrics_dir(cfg) / f"{model}__{dataset}.json")
        rows.append({
            "model": model, "display_name": DISPLAY[model], "dataset": dataset, "condition": q["condition"].iloc[0],
            "run_type": cfg.run_type, "n_common": len(q), "n_model_scored": len(p), **m,
            "eer_ci_low": lo, "eer_ci_high": hi, "ci_method": f"{'cluster ' if groups is not None else ''}percentile bootstrap, {ev.n_bootstrap} resamples",
            "inference_ms_per_recording": per_model["inference_ms_per_recording"],
            "training_seconds": per_model["training_seconds"],
            "train_feature_seconds": per_model["train_feature_seconds"],
        })
        for col in sub_rows:
            s = subgroup_metrics(q, col, thr)
            if not s.empty:
                s.insert(0, "dataset", dataset)
                s.insert(0, "model", model)
                if col == "generator_id" and ev.n_bootstrap > 0:
                    cis = []
                    bona = q[q["label"] == 0]
                    for gen in s["generator_id"]:
                        sub = pd.concat([bona, q[(q["label"] == 1) & (q["generator_id"] == gen)]])
                        g = sub["group_id"].to_numpy() if groups is not None else None
                        cis.append(bootstrap_ci(sub["label"].to_numpy(), sub["score"].to_numpy(), eer_only,
                                                ev.n_bootstrap, ev.ci_level, groups=g, seed=cfg.seed))
                    s["eer_ci_low"], s["eer_ci_high"] = zip(*cis)
                sub_rows[col].append(s)
    table = pd.DataFrame(rows)
    order = {t.name: i for i, t in enumerate(cfg.test_sets())}  # configured order, in-domain first
    table = table.sort_values("dataset", key=lambda s: s.map(order).fillna(len(order)), kind="stable").reset_index(drop=True)
    mdir = artifacts.metrics_dir(cfg)
    table.to_csv(mdir / "comparison.csv", index=False)
    artifacts.save_json(table.to_dict(orient="records"), mdir / "comparison.json")
    for col, frames in sub_rows.items():
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(mdir / f"by_{col}.csv", index=False)
    for d in table["dataset"].unique():
        n = table[table["dataset"] == d]["n_common"].unique()
        scored = table[table["dataset"] == d]["n_model_scored"]
        if (scored != n[0]).any():
            log.warning("%s: models scored different record sets; comparison uses the %d common records.", d, n[0])
    return table
