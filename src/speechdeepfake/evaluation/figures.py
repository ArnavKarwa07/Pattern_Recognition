"""Generate every figure and table that the saved outputs of a run support.

Figures whose inputs do not exist yet are skipped with a reason (returned and
logged) rather than drawn from placeholder data.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from speechdeepfake import artifacts
from speechdeepfake.config import ExperimentConfig
from speechdeepfake.data.preprocessing import load_and_preprocess
from speechdeepfake.data.validation import resolve_audio_path
from speechdeepfake.evaluation import plots
from speechdeepfake.evaluation.predict import common_records, load_predictions
from speechdeepfake.features.logmel import logmel_numpy
from speechdeepfake.features.mfcc import mfcc_matrix
from speechdeepfake.utils.logging import get_logger

log = get_logger("evaluation.figures")


def _read_partitions(cfg: ExperimentConfig) -> dict[str, pd.DataFrame]:
    d = artifacts.splits_dir(cfg)
    out = {}
    if (d / "split_summary.json").exists():
        summary = artifacts.load_json(d / "split_summary.json")
        for name in summary["partitions"]:
            f = d / f"{name}.csv"
            if f.exists():
                df = pd.read_csv(f, dtype=str, keep_default_na=False)
                df["label"] = df["label"].astype(int)
                out[name] = df
    return out


def _examples(train: pd.DataFrame, cfg: ExperimentConfig, per_class: int = 1):
    rows = []
    for lab in (0, 1):
        for _, r in train[train["label"] == lab].sort_values("record_id").head(per_class).iterrows():
            wave = load_and_preprocess(resolve_audio_path(r["audio_path"], cfg.data_dir), cfg.audio)
            rows.append((r["record_id"].split("/")[-1], lab, wave))
    return rows


def generate_figures(cfg: ExperimentConfig) -> tuple[list[Path], dict[str, str]]:
    figs = artifacts.figures_dir(cfg)
    rt = cfg.run_type
    made: list[Path] = []
    skipped: dict[str, str] = {}

    # A. dataset
    parts = _read_partitions(cfg)
    summary_path = artifacts.splits_dir(cfg) / "split_summary.json"
    if parts:
        summary = artifacts.load_json(summary_path)
        conditions = {t.name: t.condition.replace("_", " ") for t in cfg.test_sets()}
        made.append(plots.plot_partition_counts(summary, conditions, figs / "partition_counts.png", rt))
        made.append(plots.plot_class_balance(summary, figs / "class_distribution.png", rt))
        p = plots.plot_duration_distribution(parts, figs / "duration_distribution.png", rt)
        if p:
            made.append(p)
        try:
            ex = _examples(parts["train"], cfg)
            made.append(plots.plot_waveforms(ex, cfg.audio.sample_rate, figs / "waveforms.png", rt))
            # B. features
            sr = cfg.audio.sample_rate
            mf = [(rid, lab, mfcc_matrix(w, sr, cfg.mfcc)) for rid, lab, w in ex]
            made.append(plots.plot_feature_pair(mf, sr, cfg.mfcc.hop_length, "MFCC index",
                                                f"MFCCs ({cfg.mfcc.n_mfcc} coefficients), genuine vs. spoof", figs / "mfcc_heatmap.png", rt))
            lm = [(rid, lab, logmel_numpy(w, sr, cfg.logmel)) for rid, lab, w in ex]
            made.append(plots.plot_feature_pair(lm, sr, cfg.logmel.hop_length, "Mel band",
                                                f"log-Mel spectrograms ({cfg.logmel.n_mels} bands), genuine vs. spoof", figs / "logmel_examples.png", rt))
        except Exception as e:  # audio may have been moved since training
            skipped["examples"] = f"could not load example audio: {e}"
        feats = {}
        for model, label in (("mfcc_svm", "MFCC statistics"), ("wav2vec_svm", "wav2vec 2.0 embeddings")):
            if (artifacts.model_dir(cfg, model) / "metadata.json").exists():
                from speechdeepfake.training.train_svm import extract

                fs = extract(model, cfg, parts["train"], "train")
                recs, X = fs.align(parts["train"])
                feats[label] = (X, recs["label"].to_numpy())
        p = plots.plot_pca(feats, figs / "pca_features.png", rt, cfg.seed)
        if p:
            made.append(p)
        else:
            skipped["pca_features"] = "no trained SVM model with cached features"
    else:
        skipped["dataset figures"] = f"{summary_path} missing (run training first, which exports the partitions)"

    # C. CNN history
    hist = artifacts.model_dir(cfg, "logmel_cnn") / "history.csv"
    if hist.exists():
        dev_name = f"{cfg.data.dev['corpus']}/{cfg.data.dev['split']}"
        made.append(plots.plot_learning_curves(pd.read_csv(hist), figs / "cnn_learning_curves.png", rt, dev_name))
    else:
        skipped["cnn_learning_curves"] = "CNN not trained"

    # D/E. comparison and generalisation
    comp = artifacts.metrics_dir(cfg) / "comparison.csv"
    preds = load_predictions(cfg)
    if comp.exists() and preds:
        table = pd.read_csv(comp)
        for dataset in table["dataset"].unique():
            common = common_records(preds, dataset)
            made += plots.plot_roc_pr(preds, dataset, common, figs, rt)
            made.append(plots.plot_confusion(table, dataset, figs / f"confusion_{dataset}.png", rt))
        made.append(plots.plot_metric_bars(table, figs / "eer_comparison.png", rt))
        made.append(plots.plot_ba_f1(table, figs / "ba_f1_comparison.png", rt))
        made.append(plots.plot_inference_time(table, figs / "inference_time.png", rt))
        p = plots.plot_generalization(table, figs / "generalization_eer.png", rt)
        if p:
            made.append(p)
        else:
            skipped["generalization_eer"] = "needs at least two evaluated test sets"
        for col in ("generator_id", "source_domain"):
            f = artifacts.metrics_dir(cfg) / f"by_{col}.csv"
            if f.exists():
                sub = pd.read_csv(f, keep_default_na=False, na_values=[""])
                for dataset in sub["dataset"].unique():
                    p = plots.plot_subgroups(sub, col, dataset, figs / f"eer_by_{col.replace('_id', '')}_{dataset}.png", rt)
                    if p:
                        made.append(p)
        tdir = artifacts.tables_dir(cfg)
        tdir.mkdir(parents=True, exist_ok=True)
        for f in artifacts.metrics_dir(cfg).glob("*.csv"):
            shutil.copy2(f, tdir / f.name)
    else:
        skipped["comparison figures"] = "no evaluation results yet (run `speechdeepfake evaluate-all`)"
    for k, v in skipped.items():
        log.info("skipped %s: %s", k, v)
    return made, skipped
