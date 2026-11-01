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


# ------------------------------------------------------- additional analyses


def plot_det(preds: dict, dataset: str, common: set, path: Path, run_type: str) -> Path:
    """Detection error trade-off curves (FNR vs FPR on normal-deviate axes), the ASVspoof convention."""
    from scipy.stats import norm

    ticks = np.array([0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95])
    lim = norm.ppf([0.0008, 0.97])
    eps = 1e-4
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    for m in _models_in(preds, dataset):
        p = preds[(m, dataset)]
        p = p[p["record_id"].isin(common)]
        fpr, tpr, _ = roc_curve(p["label"], p["score"])
        eer = compute_eer(p["label"], p["score"])[0]
        ax.plot(norm.ppf(np.clip(fpr, eps, 1 - eps)), norm.ppf(np.clip(1 - tpr, eps, 1 - eps)), color=MODEL_COLORS[m],
                label=f"{MODEL_LABELS[m]} (EER {eer:.1%})")
        e = norm.ppf(np.clip(eer, eps, 1 - eps))
        ax.plot(e, e, "o", color=MODEL_COLORS[m], ms=8, mec="white", mew=2)
    ax.plot(lim, lim, color=TEXT_2, lw=0.8, ls=":", label="EER line (FPR = FNR)")
    labels = [f"{100 * t:g}%" for t in ticks]
    ax.set_xticks(norm.ppf(ticks), labels, rotation=45)
    ax.set_yticks(norm.ppf(ticks), labels)
    ax.set(xlim=lim, ylim=lim, xlabel="false positive rate (bona fide flagged as spoof)",
           ylabel="false negative rate (spoof missed)", title=f"DET curves - {dataset} (n={len(common):,})")
    ax.legend(loc="upper right", fontsize=8.5)
    return _save(fig, path, run_type)


def plot_score_distributions(preds: dict, dataset: str, common: set, path: Path, run_type: str) -> Path:
    """Per-model score histograms by class, with the dev-selected threshold."""
    models = _models_in(preds, dataset)
    fig, axes = plt.subplots(1, len(models), figsize=(4.3 * len(models), 3.5), squeeze=False)
    for ax, m in zip(axes[0], models):
        p = preds[(m, dataset)]
        p = p[p["record_id"].isin(common)]
        s = p["score"].to_numpy(dtype=float)
        lo, hi = np.quantile(s, [0.005, 0.995])
        bins = np.linspace(lo, hi, 40)
        for cls in (0, 1):
            ax.hist(np.clip(s[p["label"].to_numpy() == cls], lo, hi), bins=bins, density=True, alpha=0.5,
                    color=CLASS_COLORS[cls], label=CLASS_LABELS[cls], edgecolor="white", linewidth=0.5)
        thr = float(p["threshold"].iloc[0])
        ax.axvline(thr, color=TEXT, lw=1.2, ls="--")
        ax.text(thr, ax.get_ylim()[1], " dev threshold", va="top", fontsize=8, color=TEXT_2)
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.set_xlabel("score (higher = more spoof-like)")
    axes[0][0].set_ylabel("density")
    axes[0][0].legend(loc="upper left")
    fig.suptitle(f"Score distributions by class - {dataset}", fontsize=11)
    return _save(fig, path, run_type)


GAMMA_ORDER = ("scale*0.3", "scale", "scale*3")


def plot_svm_tuning(tunings: dict[str, tuple[pd.DataFrame, dict]], path: Path, run_type: str) -> Optional[Path]:
    """Dev-set EER over the (C, gamma) grid for each SVM; the selected cell is outlined."""
    if not tunings:
        return None
    fig, axes = plt.subplots(1, len(tunings), figsize=(4.6 * len(tunings), 3.6), squeeze=False)
    vmax = max(float(t["dev_eer"].max()) for t, _ in tunings.values())
    for ax, (m, (t, sel)) in zip(axes[0], tunings.items()):
        piv = t.pivot_table(index="C", columns="gamma_spec", values="dev_eer", aggfunc="first")
        cols = [c for c in GAMMA_ORDER if c in piv.columns] + [c for c in piv.columns if c not in GAMMA_ORDER]
        piv = piv[cols].sort_index(ascending=False)
        ax.imshow(piv.to_numpy(), cmap=SEQ_BLUE, vmin=0, vmax=vmax, aspect="auto")
        for i, c_val in enumerate(piv.index):
            for j, g in enumerate(piv.columns):
                v = piv.iloc[i, j]
                ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=9, color="white" if v > 0.55 * vmax else TEXT)
                if float(c_val) == float(sel["C"]) and g == sel["gamma_spec"]:
                    ax.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94, fill=False, ec=TEXT, lw=2.2))
        ax.set_xticks(range(len(piv.columns)), piv.columns)
        ax.set_yticks(range(len(piv.index)), [f"{c:g}" for c in piv.index])
        ax.set_xlabel("gamma (multiple of sklearn 'scale')")
        ax.set_ylabel("C")
        ax.set_title(f"{MODEL_LABELS[m]}: dev EER", fontsize=10)
        ax.grid(False)
    fig.suptitle("SVM hyperparameter search on the development set (outlined = selected)", fontsize=11)
    return _save(fig, path, run_type)


def plot_generator_heatmap(sub: pd.DataFrame, dataset: str, path: Path, run_type: str) -> Optional[Path]:
    """Models x generation systems EER heat map (each system vs. all bona fide)."""
    s = sub[sub["dataset"] == dataset]
    if s.empty or s["eer"].isna().all():
        return None
    piv = s.pivot_table(index="model", columns="generator_id", values="eer", aggfunc="first")
    piv = piv.reindex([m for m in MODEL_ORDER if m in piv.index])
    vmax = max(0.5, float(np.nanmax(piv.to_numpy())))
    fig, ax = plt.subplots(figsize=(max(6, 0.75 * piv.shape[1] + 2.5), 0.7 * len(piv) + 1.8))
    ax.imshow(piv.to_numpy(), cmap=SEQ_BLUE, vmin=0, vmax=vmax, aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5, color="white" if v > 0.55 * vmax else TEXT)
    rot = piv.shape[1] > 6
    ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30 if rot else 0, ha="right" if rot else "center")
    ax.set_yticks(range(piv.shape[0]), [MODEL_LABELS[m] for m in piv.index])
    ax.set_title(f"EER per generation system - {dataset} (darker = harder to detect)")
    ax.grid(False)
    return _save(fig, path, run_type)


def plot_tradeoff(table: pd.DataFrame, path: Path, run_type: str) -> Path:
    """Accuracy-cost trade-off: inference time (log scale) vs EER, one panel per test set."""
    datasets = list(dict.fromkeys(table["dataset"]))
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.8 * len(datasets), 3.9), squeeze=False, sharey=True)
    for ax, d in zip(axes[0], datasets):
        sub = table[table["dataset"] == d].sort_values("eer")
        for k, (_, r) in enumerate(sub.iterrows()):
            m = r["model"]
            ax.errorbar(r["inference_ms_per_recording"], r["eer"],
                        yerr=[[max(r["eer"] - r["eer_ci_low"], 0)], [max(r["eer_ci_high"] - r["eer"], 0)]],
                        fmt="none", capsize=3, ecolor=MODEL_COLORS[m], elinewidth=1.2)
            ax.plot(r["inference_ms_per_recording"], r["eer"], "o", ms=9, color=MODEL_COLORS[m], mec="white", mew=2)
            # stagger direct labels (sorted by EER) so close points do not collide
            ax.annotate(f"{MODEL_LABELS[m]} ({r['eer']:.1%})", (r["inference_ms_per_recording"], r["eer"]),
                        xytext=(10, -14 if k % 2 == 0 else 10), textcoords="offset points", fontsize=8, color=TEXT_2)
        ax.set_xscale("log")
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0 / 1.5, x1 * 4)
        ax.set_xlabel("inference time per recording (ms, log scale)")
        ax.set_title(f"{d} ({sub['condition'].iloc[0].replace('_', ' ')})", fontsize=10)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0][0].set_ylim(bottom=0)
    axes[0][0].set_ylabel("EER (lower is better)")
    fig.suptitle("Detection error vs. computational cost (bottom-left is best)", fontsize=11)
    return _save(fig, path, run_type)


def plot_robustness(rob: pd.DataFrame, metric: str, ylabel: str, path: Path, run_type: str) -> Optional[Path]:
    """Grouped bars per audio condition (clean first), one panel per test set."""
    if rob is None or rob.empty:
        return None
    datasets = list(dict.fromkeys(rob["dataset"]))
    conds = list(dict.fromkeys(rob["audio_condition"]))
    labels = rob.drop_duplicates("audio_condition").set_index("audio_condition")["audio_condition_label"]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(1.7 * len(conds) + 3.5, 3.7 * len(datasets)), squeeze=False)
    for ax, d in zip(axes[:, 0], datasets):
        s = rob[rob["dataset"] == d]
        models = [m for m in MODEL_ORDER if m in set(s["model"])]
        w = 0.8 / len(models)
        x = np.arange(len(conds))
        for k, m in enumerate(models):
            v = s[s["model"] == m].set_index("audio_condition").reindex(conds)
            kw = {}
            if metric == "eer":
                kw = {"yerr": np.vstack([v["eer"] - v["eer_ci_low"], v["eer_ci_high"] - v["eer"]]).clip(min=0),
                      "capsize": 2, "error_kw": {"lw": 0.8, "ecolor": TEXT_2}}
            bars = ax.bar(x + (k - (len(models) - 1) / 2) * w, v[metric], w * 0.92, color=MODEL_COLORS[m],
                          label=MODEL_LABELS[m], **kw)
            ax.bar_label(bars, labels=[f"{t:.0%}" for t in v[metric]], fontsize=7, color=TEXT_2, padding=2)
        ax.set_xticks(x, [labels[c].replace(", ", ",\n").replace(" (", "\n(") for c in conds], fontsize=8.5)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1 if metric != "eer" else max(0.6, float((s["eer_ci_high"]).max()) + 0.08))
        ax.axhline(0.5, color=TEXT_2, lw=0.8, ls=":")
        ax.text(-0.45, 0.5, "chance", va="bottom", ha="left", fontsize=7.5, color=TEXT_2)
        ax.set_title(f"{d}: {ylabel} under audio degradations (same records, dev thresholds)", fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=8, ncol=len(models))
        ax.set_xlim(-0.5, len(conds) - 0.5)
    return _save(fig, path, run_type)


def plot_degradation_examples(mats: list[tuple[str, np.ndarray]], sr: int, hop: int, path: Path, run_type: str) -> Path:
    """log-Mel spectrogram of one recording under each audio condition."""
    fig, axes = plt.subplots(1, len(mats), figsize=(3.2 * len(mats), 3.2), squeeze=False, sharey=True)
    vmin = min(np.percentile(m, 1) for _, m in mats)
    vmax = max(np.percentile(m, 99) for _, m in mats)
    for ax, (name, m) in zip(axes[0], mats):
        im = ax.imshow(m, origin="lower", aspect="auto", cmap=SPEC_CMAP, vmin=vmin, vmax=vmax,
                       extent=(0, m.shape[1] * hop / sr, 0, m.shape[0]))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("time (s)")
        ax.grid(False)
    axes[0][0].set_ylabel("Mel band")
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="log power (shared scale)")
    fig.suptitle("One test recording under each audio condition (log-Mel)", fontsize=11, y=1.06)
    path.parent.mkdir(parents=True, exist_ok=True)
    _banner(fig, run_type, y=-0.06)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pipeline_overview(path: Path) -> Path:
    """Static block diagram of the three pipelines and the shared evaluation protocol."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis("off")

    def box(x, y, w, h, text, ec=TEXT_2, fc="#f4f3ef"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6, color=TEXT)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops={"arrowstyle": "-|>", "color": TEXT_2, "lw": 1.2})

    box(0.1, 3.6, 2.3, 1.2, "Datasets\nASVspoof 2019 LA\ntrain / dev / eval\nCD-ADD test (external)")
    box(0.1, 1.7, 2.3, 1.4, "Shared preprocessing\ndecode, mono, 16 kHz\nfixed 4 s window\nstart crop / repeat pad")
    arrow(1.25, 3.6, 1.25, 3.1)
    rows = [("mfcc_svm", 3.8, "40 MFCCs per frame\nmean + std over time\n-> 80-D vector", "StandardScaler\n+ RBF SVM"),
            ("logmel_cnn", 2.2, "log-Mel spectrogram\n80 bands x 401 frames\nper-band standardised", "4-block 2-D CNN\n~98k parameters"),
            ("wav2vec_svm", 0.6, "frozen wav2vec 2.0 base\nlast layer, mean-pooled\n-> 768-D vector", "StandardScaler\n+ RBF SVM")]
    for m, y, feat, clf in rows:
        c = MODEL_COLORS[m]
        box(3.0, y, 2.5, 1.05, feat, ec=c, fc="white")
        box(6.0, y, 2.0, 1.05, clf, ec=c, fc="white")
        ax.text(3.0, y + 1.13, MODEL_LABELS[m], fontsize=9.5, color=TEXT, weight="bold")
        arrow(2.4, 2.4, 3.0, y + 0.52)
        arrow(5.5, y + 0.52, 6.0, y + 0.52)
        arrow(8.0, y + 0.52, 8.5, 2.4)
    box(8.5, 1.7, 1.6, 1.4, "score\n(higher = spoof)\nvs. threshold\nchosen on dev")
    arrow(10.1, 2.4, 10.4, 2.4)
    box(10.4, 1.0, 1.55, 2.8, "Evaluation\n\nEER + 95% CI\nROC / DET / PR\nAUC, BA, F1\nper generator\nrobustness\ncost")
    ax.text(6, 5.2, "Three detectors, one protocol: fit on train, tune + threshold on dev, test sets untouched",
            ha="center", fontsize=11, color=TEXT, weight="bold")
    return _save(fig, path)
