"""Publication-style Matplotlib figures, generated only from saved outputs.

Every figure is built from files on disk (manifests, caches, histories,
prediction CSVs), never from hard-coded numbers. Figures from pilot or
smoke-test runs carry a visible banner so they cannot be mistaken for final
results. Datasets are always named in titles or legends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from speechdeepfake.evaluation.metrics import compute_eer  # noqa: E402
from speechdeepfake.utils.logging import get_logger  # noqa: E402

log = get_logger("evaluation.plots")

# Categorical slots 1-3 of the reference palette (validated colour-blind safe as a set).
MODEL_COLORS = {"mfcc_svm": "#2a78d6", "logmel_cnn": "#eb6834", "wav2vec_svm": "#1baf7a"}
MODEL_LABELS = {"mfcc_svm": "MFCC + SVM", "logmel_cnn": "log-Mel + CNN", "wav2vec_svm": "wav2vec 2.0 + SVM"}
MODEL_ORDER = ["mfcc_svm", "logmel_cnn", "wav2vec_svm"]
CLASS_COLORS = {0: "#2a78d6", 1: "#eb6834"}
CLASS_LABELS = {0: "bona fide", 1: "spoof"}
TEXT, TEXT_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#ffffff", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
SPEC_CMAP = LinearSegmentedColormap.from_list(
    "spec_blue", ["#0d366b", "#184f95", "#256abf", "#3987e5", "#86b6ef", "#cde2fb", "#ffffff"]
)

BANNERS = {
    "pilot": "PILOT RUN - stratified subset of the data; not final research results",
    "smoke_test": "SYNTHETIC SOFTWARE-TEST FIXTURE - not research results",
}


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": TEXT_2, "axes.labelcolor": TEXT, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    })


_style()


def _banner(fig: plt.Figure, run_type: str, y: float = -0.01) -> None:
    """Non-final runs get a red banner below the plot area (never over titles)."""
    if run_type in BANNERS:
        fig.text(0.5, y, BANNERS[run_type], ha="center", va="top", fontsize=8.5, color="#b3261e", weight="bold")


def _save(fig: plt.Figure, path: Path, run_type: str = "full") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    _banner(fig, run_type)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log.info("figure: %s", path)
    return path


# ------------------------------------------------------------------ dataset


def plot_partition_counts(summary: dict, conditions: dict, path: Path, run_type: str) -> Path:
    """Grouped bars of bona fide / spoof counts for every experiment partition."""
    parts = summary["partitions"]
    names = list(parts)
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(1.6 * len(names) + 3, 4))
    for k, (cls, off) in enumerate(((0, -w / 2), (1, w / 2))):
        vals = [parts[n][CLASS_LABELS[cls].replace(" ", "")] for n in names]
        bars = ax.bar(x + off, vals, w * 0.94, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls])
        ax.bar_label(bars, labels=[f"{v:,}" for v in vals], fontsize=8, color=TEXT_2, padding=2)
    roles = {"train": "training", "dev": "validation", **conditions}
    ax.set_xticks(x, [f"{n}\n({roles.get(n, 'test')})" for n in names])
    ax.set_ylabel("recordings")
    ax.set_title("Recordings per experiment partition and class")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return _save(fig, path, run_type)


def plot_class_balance(summary: dict, path: Path, run_type: str) -> Path:
    """100 % stacked horizontal bars: share of spoof recordings per partition."""
    parts = summary["partitions"]
    names = list(parts)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(names) + 1.5))
    bona = np.array([parts[n]["bonafide"] / max(parts[n]["n"], 1) for n in names])
    ax.barh(names, bona, color=CLASS_COLORS[0], label="bona fide", height=0.6)
    ax.barh(names, 1 - bona, left=bona, color=CLASS_COLORS[1], label="spoof", height=0.6)
    for i, b in enumerate(bona):
        ax.text(0.01, i, f"{b:.1%}", va="center", color="white", fontsize=8.5, weight="bold")
        ax.text(0.99, i, f"{1 - b:.1%}", va="center", ha="right", color="white", fontsize=8.5, weight="bold")
    ax.set_xlim(0, 1)
    ax.set_xlabel("share of recordings")
    ax.set_title("Class distribution per partition")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.grid(False)
    return _save(fig, path, run_type)


def plot_duration_distribution(partitions: dict[str, pd.DataFrame], path: Path, run_type: str) -> Optional[Path]:
    """Small multiples: duration histograms per partition, by class."""
    frames = {k: v for k, v in partitions.items() if "duration_s" in v.columns}
    if not frames:
        return None
    fig, axes = plt.subplots(1, len(frames), figsize=(3.6 * len(frames), 3.2), squeeze=False)
    for ax, (name, df) in zip(axes[0], frames.items()):
        d = pd.to_numeric(df["duration_s"], errors="coerce")
        hi = float(np.nanquantile(d, 0.995)) if d.notna().any() else 1.0
        bins = np.linspace(0, hi, 40)
        for cls in (0, 1):
            ax.hist(d[df["label"] == cls].clip(upper=hi), bins=bins, histtype="step", lw=1.6,
                    color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], density=True)
        ax.set_title(name)
        ax.set_xlabel("duration (s)")
    axes[0][0].set_ylabel("density")
    axes[0][0].legend()
    fig.suptitle("Recording duration by partition and class (before cropping/padding)", y=1.02, fontsize=11)
    return _save(fig, path, run_type)


def plot_waveforms(examples: list[tuple[str, int, np.ndarray]], sr: int, path: Path, run_type: str) -> Path:
    """Example waveforms; ``examples`` = [(record_id, label, wave), ...]."""
    fig, axes = plt.subplots(len(examples), 1, figsize=(9, 1.7 * len(examples)), sharex=True, squeeze=False)
    for ax, (rid, lab, w) in zip(axes[:, 0], examples):
        t = np.arange(len(w)) / sr
        ax.plot(t, w, color=CLASS_COLORS[lab], lw=0.6)
        ax.set_title(f"{CLASS_LABELS[lab]}: {rid}", fontsize=9, loc="left")
        ax.set_ylabel("amplitude")
    axes[-1, 0].set_xlabel("time (s)")
    fig.suptitle("Example genuine and spoofed waveforms (training partition)", fontsize=11)
    return _save(fig, path, run_type)


# ----------------------------------------------------------------- features


def plot_feature_pair(mats: list[tuple[str, int, np.ndarray]], sr: int, hop: int, ylabel: str, title: str,
                      path: Path, run_type: str, cmap=SPEC_CMAP) -> Path:
    """Side-by-side time-frequency matrices (MFCC or log-Mel) for genuine vs spoof examples."""
    fig, axes = plt.subplots(1, len(mats), figsize=(5 * len(mats), 3.4), squeeze=False)
    vmin = min(np.percentile(m, 1) for _, _, m in mats)
    vmax = max(np.percentile(m, 99) for _, _, m in mats)
    for ax, (rid, lab, m) in zip(axes[0], mats):
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       extent=(0, m.shape[1] * hop / sr, 0, m.shape[0]))
        ax.set_title(f"{CLASS_LABELS[lab]}: {rid}", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.grid(False)
    axes[0][0].set_ylabel(ylabel)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="value (shared scale)")
    fig.suptitle(title, fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    _banner(fig, run_type, y=-0.06)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pca(feature_sets: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, run_type: str, seed: int = 0) -> Optional[Path]:
    """2-D PCA of cached training features (one panel per feature type), by class."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if not feature_sets:
        return None
    fig, axes = plt.subplots(1, len(feature_sets), figsize=(5 * len(feature_sets), 4.2), squeeze=False)
    rng = np.random.default_rng(seed)
    for ax, (name, (X, y)) in zip(axes[0], feature_sets.items()):
        idx = rng.permutation(len(y))[:3000]
        Xs = StandardScaler().fit_transform(X[idx])
        Z = PCA(n_components=2, random_state=seed).fit(Xs)
        P = Z.transform(Xs)
        for cls in (0, 1):
            m = y[idx] == cls
            ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.5, color=CLASS_COLORS[cls], label=CLASS_LABELS[cls],
                       edgecolors="white", linewidths=0.3)
        ev = Z.explained_variance_ratio_
        ax.set_xlabel(f"PC 1 ({ev[0]:.1%} var.)")
        ax.set_ylabel(f"PC 2 ({ev[1]:.1%} var.)")
        ax.set_title(f"{name} (training subset, n={len(idx)})", fontsize=10)
    axes[0][0].legend(markerscale=2)
    fig.suptitle("PCA projection of standardised features", fontsize=11)
    return _save(fig, path, run_type)


# ---------------------------------------------------------------------- CNN


def plot_learning_curves(history: pd.DataFrame, path: Path, run_type: str, dev_name: str = "development set") -> Path:
    best = history.loc[history["is_best"].astype(bool), "epoch"]
    best_epoch = int(best.iloc[-1]) if len(best) else None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    a1.plot(history["epoch"], history["train_loss"], color=MODEL_COLORS["mfcc_svm"], marker="o", ms=4, label="training loss")
    a1.plot(history["epoch"], history["dev_loss"], color=MODEL_COLORS["logmel_cnn"], marker="o", ms=4, label="validation loss")
    a1.set_xlabel("epoch")
    a1.set_ylabel("weighted cross-entropy")
    a1.set_title("Loss")
    a2.plot(history["epoch"], history["dev_eer"], color=MODEL_COLORS["mfcc_svm"], marker="o", ms=4, label="validation EER")
    a2.plot(history["epoch"], 1 - history["dev_roc_auc"], color=MODEL_COLORS["logmel_cnn"], marker="o", ms=4,
            label="validation 1 - ROC-AUC")
    a2.set_xlabel("epoch")
    a2.set_ylabel("error rate (lower is better)")
    a2.set_title(f"Validation metrics ({dev_name})")
    for ax in (a1, a2):
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if best_epoch is not None:
            ax.axvline(best_epoch, color=TEXT_2, ls="--", lw=1)
            ax.text(best_epoch, ax.get_ylim()[1], " selected checkpoint", va="top", fontsize=8, color=TEXT_2)
        ax.legend()
    fig.suptitle("log-Mel CNN training history", fontsize=11)
    return _save(fig, path, run_type)


# --------------------------------------------------------------- comparison


def _models_in(preds: dict, dataset: str) -> list[str]:
    return [m for m in MODEL_ORDER if (m, dataset) in preds]


def plot_roc_pr(preds: dict, dataset: str, common: set, out_dir: Path, run_type: str) -> list[Path]:
    paths = []
    fig, ax = plt.subplots(figsize=(5.2, 5))
    fig2, ax2 = plt.subplots(figsize=(5.2, 5))
    for m in _models_in(preds, dataset):
        p = preds[(m, dataset)]
        p = p[p["record_id"].isin(common)]
        fpr, tpr, _ = roc_curve(p["label"], p["score"])
        eer, _ = compute_eer(p["label"], p["score"])
        ax.plot(fpr, tpr, color=MODEL_COLORS[m], label=f"{MODEL_LABELS[m]} (EER {eer:.1%})")
        prec, rec, _ = precision_recall_curve(p["label"], p["score"])
        ax2.plot(rec, prec, color=MODEL_COLORS[m], label=MODEL_LABELS[m])
    n = len(common)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1, ls="--")
    ax.plot([0, 1], [1, 0], color=TEXT_2, lw=0.8, ls=":", label="EER line (FPR = FNR)")
    ax.set(xlabel="false positive rate (bona fide flagged as spoof)", ylabel="true positive rate (spoof detected)",
           title=f"ROC - {dataset} (n={n:,})", xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(loc="lower right")
    paths.append(_save(fig, out_dir / f"roc_{dataset}.png", run_type))
    first = preds[(_models_in(preds, dataset)[0], dataset)]
    prevalence = float(first[first["record_id"].isin(common)]["label"].mean())
    ax2.axhline(prevalence, color=TEXT_2, lw=0.8, ls=":", label=f"chance (spoof share {prevalence:.1%})")
    ax2.set(xlabel="recall (spoof)", ylabel="precision (spoof)", title=f"Precision-recall - {dataset} (n={n:,})",
            xlim=(0, 1), ylim=(0, 1.01))
    ax2.legend(loc="lower left")
    paths.append(_save(fig2, out_dir / f"pr_{dataset}.png", run_type))
    return paths


def plot_confusion(table: pd.DataFrame, dataset: str, path: Path, run_type: str) -> Path:
    rows = table[table["dataset"] == dataset].set_index("model")
    models = [m for m in MODEL_ORDER if m in rows.index]
    fig, axes = plt.subplots(1, len(models), figsize=(3.8 * len(models), 3.6), squeeze=False)
    for ax, m in zip(axes[0], models):
        r = rows.loc[m]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]], dtype=float)
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=SEQ_BLUE, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j]):,}\n({norm[i, j]:.1%})", ha="center", va="center",
                        color="white" if norm[i, j] > 0.55 else TEXT, fontsize=9)
        ax.set_xticks([0, 1], ["bona fide", "spoof"])
        ax.set_yticks([0, 1], ["bona fide", "spoof"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{MODEL_LABELS[m]}\nthreshold {r['threshold']:.3g} (from dev)", fontsize=9.5)
        ax.grid(False)
    fig.suptitle(f"Confusion matrices - {dataset} (row-normalised shading)", fontsize=11)
    return _save(fig, path, run_type)


def _grouped_bars(ax, table: pd.DataFrame, metric: str, datasets: list[str], ci: bool = False, pct: bool = True):
    models = [m for m in MODEL_ORDER if m in set(table["model"])]
    w = 0.8 / max(len(models), 1)
    x = np.arange(len(datasets))
    for k, m in enumerate(models):
        sub = table[table["model"] == m].set_index("dataset")
        vals = np.array([sub.loc[d, metric] if d in sub.index else np.nan for d in datasets], dtype=float)
        kw = {}
        if ci and "eer_ci_low" in sub.columns:
            lo = np.array([sub.loc[d, "eer_ci_low"] if d in sub.index else np.nan for d in datasets], dtype=float)
            hi = np.array([sub.loc[d, "eer_ci_high"] if d in sub.index else np.nan for d in datasets], dtype=float)
            kw = {"yerr": np.vstack([vals - lo, hi - vals]).clip(min=0), "capsize": 3, "error_kw": {"lw": 1, "ecolor": TEXT_2}}
        bars = ax.bar(x + (k - (len(models) - 1) / 2) * w, vals, w * 0.92, color=MODEL_COLORS[m], label=MODEL_LABELS[m], **kw)
        ax.bar_label(bars, labels=[f"{v:.1%}" if pct else f"{v:.3g}" for v in vals], fontsize=7.5, color=TEXT_2, padding=2)
    ax.set_xticks(x, datasets)


def plot_metric_bars(table: pd.DataFrame, path: Path, run_type: str) -> Path:
    datasets = list(dict.fromkeys(table["dataset"]))
    fig, ax = plt.subplots(figsize=(2.6 * len(datasets) + 3, 4))
    _grouped_bars(ax, table, "eer", datasets, ci=True)
    ax.set_ylabel("EER (lower is better)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_title("Equal error rate by test set (error bars: bootstrap CI)")
    ax.legend()
    return _save(fig, path, run_type)


def plot_ba_f1(table: pd.DataFrame, path: Path, run_type: str) -> Path:
    datasets = list(dict.fromkeys(table["dataset"]))
    fig, axes = plt.subplots(1, 2, figsize=(2 * (2.4 * len(datasets) + 2), 4), sharey=True)
    for ax, metric, label in ((axes[0], "balanced_accuracy", "balanced accuracy"), (axes[1], "f1", "F1-score (spoof)")):
        _grouped_bars(ax, table, metric, datasets)
        ax.set_title(f"{label} at the dev-selected threshold")
        ax.set_ylim(0, 1.08)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].set_ylabel("score (higher is better)")
    axes[0].legend(loc="lower left")
    return _save(fig, path, run_type)


def plot_inference_time(table: pd.DataFrame, path: Path, run_type: str) -> Path:
    t = table.groupby("model")["inference_ms_per_recording"].mean().reindex([m for m in MODEL_ORDER if m in set(table["model"])])
    fig, ax = plt.subplots(figsize=(6.5, 3))
    bars = ax.barh([MODEL_LABELS[m] for m in t.index], t.values, color=[MODEL_COLORS[m] for m in t.index], height=0.55)
    ax.bar_label(bars, labels=[f"{v:.1f} ms" for v in t.values], fontsize=8.5, color=TEXT_2, padding=3)
    ax.set_xlabel("wall-clock time per recording (ms), incl. audio loading and features")
    ax.set_title("Inference time (mean over test sets)")
    ax.invert_yaxis()
    return _save(fig, path, run_type)


def plot_generalization(table: pd.DataFrame, path: Path, run_type: str) -> Optional[Path]:
    """Slope chart: EER on each in-domain test set vs each external test set, per model."""
    conds = table.drop_duplicates("dataset").set_index("dataset")["condition"]
    order = list(conds[conds == "in_domain"].index) + list(conds[conds != "in_domain"].index)
    if len(order) < 2:
        return None
    fig, ax = plt.subplots(figsize=(2.2 * len(order) + 3, 4.2))
    x = np.arange(len(order))
    for m in [m for m in MODEL_ORDER if m in set(table["model"])]:
        sub = table[table["model"] == m].set_index("dataset").reindex(order)
        ax.errorbar(x, sub["eer"], yerr=np.vstack([sub["eer"] - sub["eer_ci_low"], sub["eer_ci_high"] - sub["eer"]]).clip(min=0),
                    color=MODEL_COLORS[m], marker="o", ms=7, capsize=3, label=MODEL_LABELS[m])
        ax.annotate(f"{sub['eer'].iloc[-1]:.1%}", (x[-1], sub["eer"].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TEXT_2)
    ax.set_xticks(x, [f"{d}\n({conds[d].replace('_', ' ')})" for d in order])
    ax.set_xlim(-0.4, len(order) - 0.4)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("EER (lower is better)")
    ax.set_title("In-domain vs. external-domain EER (bootstrap CI)")
    ax.legend()
    return _save(fig, path, run_type)


def plot_subgroups(sub: pd.DataFrame, column: str, dataset: str, path: Path, run_type: str) -> Optional[Path]:
    """EER per generation system (each vs. all bona fide) or per source domain, per model."""
    s = sub[(sub["dataset"] == dataset)]
    if s.empty or "eer" not in s.columns or s["eer"].isna().all():
        return None
    groups = sorted(s[column].unique())
    models = [m for m in MODEL_ORDER if m in set(s["model"])]
    fig, ax = plt.subplots(figsize=(min(12, max(6, 0.7 * len(groups) + 2)), 4.2))
    w = 0.8 / len(models)
    x = np.arange(len(groups))
    for k, m in enumerate(models):
        d = s[s["model"] == m].set_index(column).reindex(groups)
        kw = {}
        if "eer_ci_low" in d.columns and d["eer_ci_low"].notna().any():
            kw = {"yerr": np.vstack([d["eer"] - d["eer_ci_low"], d["eer_ci_high"] - d["eer"]]).clip(min=0),
                  "capsize": 2, "error_kw": {"lw": 0.8, "ecolor": TEXT_2}}
        bars = ax.bar(x + (k - (len(models) - 1) / 2) * w, d["eer"], w * 0.92, color=MODEL_COLORS[m], label=MODEL_LABELS[m], **kw)
        ax.bar_label(bars, labels=[f"{v:.0%}" if pd.notna(v) else "" for v in d["eer"]], fontsize=7, color=TEXT_2, padding=2)
    ax.set_xticks(x, groups, rotation=30 if len(groups) > 6 else 0, ha="right" if len(groups) > 6 else "center")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("EER (lower is better)")
    what = "generation system (vs. all bona fide)" if column == "generator_id" else "source domain"
    ax.set_title(f"EER by {what} - {dataset}")
    ax.legend()
    return _save(fig, path, run_type)
