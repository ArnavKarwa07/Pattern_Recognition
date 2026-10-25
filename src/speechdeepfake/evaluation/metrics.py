"""Metric definitions shared by every model.

Conventions: spoof (1) is the positive class; a higher score means "more
likely spoof"; a recording is predicted spoof when ``score >= threshold``.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _check(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape:
        raise ValueError(f"labels {y.shape} and scores {s.shape} differ in shape")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN/Inf")
    if set(np.unique(y)) - {0, 1}:
        raise ValueError("labels must be 0/1")
    if len(np.unique(y)) < 2:
        raise ValueError("EER/ROC need both classes present")
    return y, s


def compute_eer(y_true, scores) -> tuple[float, float]:
    """Equal Error Rate and the score threshold at which it occurs.

    The ROC curve is evaluated at every distinct score. FNR(t) = 1 - TPR(t)
    falls and FPR(t) rises as the threshold decreases; the EER is the point
    where the two curves cross, found by linear interpolation between the two
    adjacent ROC operating points that bracket the crossing.
    """
    y, s = _check(y_true, scores)
    fpr, tpr, thr = roc_curve(y, s, pos_label=1, drop_intermediate=False)
    fnr = 1.0 - tpr
    diff = fnr - fpr  # +1 at the first point (thr=inf), -1 at the last
    idx = int(np.flatnonzero(diff <= 0)[0])
    if diff[idx] == 0 or idx == 0:
        return float(fpr[idx]), float(thr[idx])
    w = diff[idx - 1] / (diff[idx - 1] - diff[idx])
    eer = fpr[idx - 1] + w * (fpr[idx] - fpr[idx - 1])
    if np.isinf(thr[idx - 1]):
        threshold = thr[idx]
    else:
        threshold = thr[idx - 1] + w * (thr[idx] - thr[idx - 1])
    return float(eer), float(threshold)


def binary_metrics(y_true, scores, threshold: float) -> dict:
    """All operating-point and threshold-free metrics for one score set."""
    y, s = _check(y_true, scores)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    eer, eer_thr = compute_eer(y, s)
    return {
        "n": int(len(y)),
        "n_bonafide": int((y == 0).sum()),
        "n_spoof": int((y == 1).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "eer": eer,
        "eer_threshold": eer_thr,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(
    y_true,
    scores,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 200,
    level: float = 0.95,
    groups: Optional[np.ndarray] = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval.

    When ``groups`` is given (e.g. CD-ADD source utterances shared by a genuine
    clip and its spoofs), whole groups are resampled (cluster bootstrap) so that
    correlated recordings do not shrink the interval artificially.
    """
    y, s = np.asarray(y_true).astype(int), np.asarray(scores, dtype=float)
    if n_boot <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    if groups is not None:
        codes, uniq = pd.factorize(np.asarray(groups))
        members = [np.flatnonzero(codes == g) for g in range(len(uniq))]
    values = []
    for _ in range(n_boot):
        if groups is None:
            idx = rng.integers(0, len(y), len(y))
        else:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(metric(y[idx], s[idx]))
    if not values:
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1 - alpha))


def eer_only(y, s) -> float:
    return compute_eer(y, s)[0]


def subgroup_metrics(pred: pd.DataFrame, column: str, threshold: float) -> pd.DataFrame:
    """Per-subgroup results from a prediction table (columns label, score, ``column``).

    * ``generator_id``: each spoofing system is scored against *all* bona fide
      recordings of the test set (the per-attack EER convention of ASVspoof).
    * any other column (e.g. ``source_domain``): rows of that subgroup only;
      threshold-free metrics are reported only when both classes are present.
    """
    rows = []
    if column not in pred.columns:
        return pd.DataFrame()
    if column == "generator_id":
        bona = pred[pred["label"] == 0]
        for gen, g in pred[(pred["label"] == 1) & (pred[column] != "")].groupby(column):
            sub = pd.concat([bona, g])
            row = {column: gen, "n_spoof": len(g), "n_bonafide": len(bona),
                   "spoof_detection_rate": float((g["score"] >= threshold).mean())}
            if len(bona):
                row["eer"], _ = compute_eer(sub["label"], sub["score"])
                row["roc_auc"] = float(roc_auc_score(sub["label"], sub["score"]))
            rows.append(row)
    else:
        for val, g in pred[pred[column] != ""].groupby(column):
            row = {column: val, "n": len(g), "n_bonafide": int((g["label"] == 0).sum()), "n_spoof": int((g["label"] == 1).sum())}
            if g["label"].nunique() == 2:
                row.update({k: v for k, v in binary_metrics(g["label"], g["score"], threshold).items()
                            if k in ("eer", "roc_auc", "balanced_accuracy", "f1", "accuracy")})
            else:
                row["accuracy"] = float(((g["score"] >= threshold).astype(int) == g["label"]).mean())
            rows.append(row)
    return pd.DataFrame(rows)
