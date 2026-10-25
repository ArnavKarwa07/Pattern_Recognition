"""EER, threshold selection, metric generation and scaler leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from speechdeepfake.config import SVMConfig
from speechdeepfake.evaluation.metrics import binary_metrics, bootstrap_ci, compute_eer, eer_only, subgroup_metrics
from speechdeepfake.evaluation.thresholds import select_threshold
from speechdeepfake.models.svm import build_svm_pipeline, resolve_gamma, svm_scores, tune_svm


def test_eer_perfect_separation():
    eer, thr = compute_eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert eer == 0.0
    assert 0.2 < thr <= 0.8


def test_eer_identical_scores_is_half():
    eer, _ = compute_eer([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    assert eer == pytest.approx(0.5)


def test_eer_fully_inverted_is_one():
    eer, _ = compute_eer([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    assert eer == pytest.approx(1.0)


def test_eer_hand_computed_case():
    # bona fide: 0.1 0.4 0.6 ; spoof: 0.3 0.7 0.9.  At t=0.65: FPR 0/3, FNR 1/3;
    # at t=0.5: FPR 1/3, FNR 1/3 -> EER exactly 1/3.
    eer, thr = compute_eer([0, 0, 0, 1, 1, 1], [0.1, 0.4, 0.6, 0.3, 0.7, 0.9])
    assert eer == pytest.approx(1 / 3)


def test_eer_matches_bruteforce_on_random_scores():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(400), np.ones(600)].astype(int)
    s = np.r_[rng.normal(0, 1, 400), rng.normal(1.5, 1, 600)]
    eer, thr = compute_eer(y, s)
    grid = np.sort(s)
    fpr = np.array([(s[y == 0] >= t).mean() for t in grid])
    fnr = np.array([(s[y == 1] < t).mean() for t in grid])
    i = np.argmin(np.abs(fpr - fnr))
    assert eer == pytest.approx((fpr[i] + fnr[i]) / 2, abs=0.01)
    # at the returned threshold FPR and FNR are both close to the EER
    assert (s[y == 0] >= thr).mean() == pytest.approx(eer, abs=0.01)
    assert (s[y == 1] < thr).mean() == pytest.approx(eer, abs=0.01)


def test_eer_requires_both_classes():
    with pytest.raises(ValueError):
        compute_eer([1, 1], [0.2, 0.3])


@pytest.mark.parametrize("method", ["eer", "youden", "max_f1", "balanced_accuracy"])
def test_threshold_selection_separates_clean_data(method):
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([-2.0, -1.5, -1.0, 1.0, 1.5, 2.0])
    thr = select_threshold(y, s, method)
    assert -1.0 < thr <= 1.0
    assert ((s >= thr).astype(int) == y).all()


def test_unknown_threshold_method():
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.0, 1.0], "test_labels")


def test_binary_metrics_confusion_and_values():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99])
    m = binary_metrics(y, s, threshold=0.5)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 5)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(5 / 6)
    assert m["recall"] == pytest.approx(5 / 6)
    assert m["balanced_accuracy"] == pytest.approx((3 / 4 + 5 / 6) / 2)
    assert m["f1"] == pytest.approx(5 / 6)
    assert m["n_bonafide"] == 4 and m["n_spoof"] == 6 and m["threshold"] == 0.5
    assert 0 <= m["eer"] <= 1 and 0 <= m["roc_auc"] <= 1 and 0 <= m["average_precision"] <= 1


def test_bootstrap_ci_contains_point_estimate_and_supports_groups():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(200), np.ones(200)].astype(int)
    s = np.r_[rng.normal(0, 1, 200), rng.normal(1, 1, 200)]
    lo, hi = bootstrap_ci(y, s, eer_only, n_boot=100, seed=0)
    assert lo <= eer_only(y, s) <= hi
    groups = np.r_[np.arange(200), np.arange(200)]  # pair bona fide i with spoof i
    glo, ghi = bootstrap_ci(y, s, eer_only, n_boot=100, groups=groups, seed=0)
    assert glo <= ghi


def test_subgroup_metrics_by_generator():
    pred = pd.DataFrame({"label": [0, 0, 1, 1, 1, 1], "score": [0.1, 0.2, 0.9, 0.8, 0.15, 0.05],
                         "generator_id": ["bonafide", "bonafide", "A", "A", "B", "B"]})
    sub = subgroup_metrics(pred, "generator_id", threshold=0.5).set_index("generator_id")
    assert sub.loc["A", "eer"] == 0.0 and sub.loc["A", "spoof_detection_rate"] == 1.0
    # B: bona fide 0.1/0.2 vs spoof 0.05/0.15 -> at t=0.15 FPR = FNR = 1/2
    assert sub.loc["B", "eer"] == pytest.approx(0.5) and sub.loc["B", "spoof_detection_rate"] == 0.0


# -------------------------------------------------------------- SVM leakage


def test_scaler_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(0, 1, (60, 5))
    y_tr = np.r_[np.zeros(30), np.ones(30)].astype(int)
    X_tr[y_tr == 1] += 1.0
    X_dev = rng.normal(100, 10, (20, 5))  # wildly different statistics
    y_dev = np.r_[np.zeros(10), np.ones(10)].astype(int)
    pipe, table, best = tune_svm(X_tr, y_tr, X_dev, y_dev, SVMConfig(C_grid=[1.0, 10.0], gamma_grid=["scale"]), "eer")
    scaler = pipe.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_tr.mean(axis=0))
    assert np.allclose(scaler.scale_, X_tr.std(axis=0))
    assert len(table) == 2 and best["threshold"] is not None


def test_svm_score_direction_positive_for_spoof():
    rng = np.random.default_rng(0)
    X = np.r_[rng.normal(-2, 0.5, (30, 3)), rng.normal(2, 0.5, (30, 3))]
    y = np.r_[np.zeros(30), np.ones(30)].astype(int)
    pipe = build_svm_pipeline().fit(X, y)
    s = svm_scores(pipe, X)
    assert s[y == 1].mean() > 0 > s[y == 0].mean()


def test_gamma_spec_resolution():
    X = np.random.default_rng(0).normal(size=(50, 10))
    assert resolve_gamma("scale", X) == "scale"
    assert resolve_gamma("scale*2", X) == pytest.approx(2 / 10, rel=1e-6)
    with pytest.raises(ValueError):
        resolve_gamma("bogus", X)
