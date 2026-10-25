"""Operating-threshold selection on development data.

Thresholds are chosen from development-set scores only and then applied,
unchanged, to every test set. Nothing here should ever be called with test
labels; :func:`select_threshold` records the split it was fitted on so that
the provenance is saved with the model.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from speechdeepfake.evaluation.metrics import _check, compute_eer

METHODS = ("eer", "youden", "max_f1", "balanced_accuracy")


def select_threshold(y_dev, scores_dev, method: str = "eer") -> float:
    """Return the dev-set threshold for ``method``.

    * ``eer``: threshold where dev FPR equals dev FNR.
    * ``youden`` / ``balanced_accuracy``: maximise TPR - FPR, which is
      equivalent to maximising balanced accuracy.
    * ``max_f1``: maximise the F1-score of the spoof class.
    """
    y, s = _check(y_dev, scores_dev)
    if method == "eer":
        return compute_eer(y, s)[1]
    if method in ("youden", "balanced_accuracy"):
        fpr, tpr, thr = roc_curve(y, s, drop_intermediate=False)
        finite = np.isfinite(thr)
        j = np.argmax((tpr - fpr)[finite])
        return float(thr[finite][j])
    if method == "max_f1":
        prec, rec, thr = precision_recall_curve(y, s)
        f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
        return float(thr[int(np.argmax(f1))])
    raise ValueError(f"Unknown threshold method {method!r}; choose from {METHODS}")
