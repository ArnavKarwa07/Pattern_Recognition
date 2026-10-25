"""RBF-SVM pipeline shared by Solutions 1 and 3, with development-set tuning.

The pipeline is ``StandardScaler -> SVC(kernel="rbf")``. Every candidate is
fitted on the training partition only (so the scaler statistics come from
training data only) and scored on the development partition; no
cross-validation over pooled data is performed.
"""

from __future__ import annotations

import itertools
from typing import Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from speechdeepfake.config import SVMConfig
from speechdeepfake.evaluation.metrics import binary_metrics
from speechdeepfake.evaluation.thresholds import select_threshold
from speechdeepfake.utils.logging import get_logger
from speechdeepfake.utils.timing import Timer

log = get_logger("models.svm")


def build_svm_pipeline(C: float = 1.0, gamma: Union[str, float] = "scale", class_weight=None, cache_size_mb: int = 2000, max_iter: int = -1) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight, cache_size=cache_size_mb, max_iter=max_iter)),
        ]
    )


def resolve_gamma(spec: Union[str, float], X_train: np.ndarray) -> Union[str, float]:
    """``"scale*<k>"`` -> k times sklearn's "scale" value on standardised training features."""
    if isinstance(spec, str) and spec.startswith("scale*"):
        factor = float(spec.split("*", 1)[1])
        Xs = StandardScaler().fit_transform(X_train)
        var = float(Xs.var())
        return factor / (X_train.shape[1] * var) if var > 0 else factor
    if isinstance(spec, str) and spec not in ("scale", "auto"):
        raise ValueError(f"Bad gamma spec {spec!r}")
    return spec


def svm_scores(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Signed distance to the boundary; positive = spoof side (classes are [0, 1])."""
    classes = list(pipeline.classes_)
    if classes != [0, 1]:
        raise ValueError(f"Unexpected class order {classes}")
    return pipeline.decision_function(X)


def tune_svm(
    X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray, cfg: SVMConfig, threshold_method: str
) -> tuple[Pipeline, pd.DataFrame, dict]:
    """Grid-search C and gamma on the development set.

    Returns the selected fitted pipeline, the tuning table and a dict with the
    selected hyperparameters, dev threshold, dev metrics and fit time.
    """
    rows = []
    best = None
    for C, g in itertools.product(cfg.C_grid, cfg.gamma_grid):
        gamma = resolve_gamma(g, X_train)
        pipe = build_svm_pipeline(float(C), gamma, cfg.class_weight, cfg.cache_size_mb, cfg.max_iter)
        with Timer() as t:
            pipe.fit(X_train, y_train)
        s_dev = svm_scores(pipe, X_dev)
        thr = select_threshold(y_dev, s_dev, threshold_method)
        m = binary_metrics(y_dev, s_dev, thr)
        rows.append({"C": C, "gamma_spec": str(g), "gamma": gamma, "fit_seconds": t.seconds,
                     "n_support": int(pipe.named_steps["svm"].n_support_.sum()),
                     **{f"dev_{k}": m[k] for k in ("eer", "roc_auc", "balanced_accuracy", "f1")}})
        log.info("SVM C=%s gamma=%s: dev EER=%.4f AUC=%.4f (%.1fs)", C, g, m["eer"], m["roc_auc"], t.seconds)
        key = {"eer": -m["eer"], "roc_auc": m["roc_auc"], "balanced_accuracy": m["balanced_accuracy"]}[cfg.selection_metric]
        if best is None or key > best[0]:
            best = (key, pipe, {"C": C, "gamma_spec": str(g), "gamma": gamma, "threshold": thr,
                                "dev_metrics": m, "fit_seconds": t.seconds})
    assert best is not None, "empty SVM grid"
    return best[1], pd.DataFrame(rows), best[2]
