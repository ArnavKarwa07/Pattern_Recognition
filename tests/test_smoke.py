"""End-to-end smoke test of all three pipelines on a tiny SYNTHETIC fixture.

This verifies software correctness only. The fixture is not speech and the
numbers it produces are not research results.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from speechdeepfake import artifacts
from speechdeepfake.config import MODEL_NAMES, load_config
from speechdeepfake.pipeline import smoke_test


@pytest.mark.slow
def test_three_pipelines_end_to_end(tmp_path):
    base = load_config(None)
    out = tmp_path / "smoke"
    checks = smoke_test(base, out, wav2vec_model="random-tiny", n_per_class=6)
    assert checks["models_evaluated"] == sorted(MODEL_NAMES)
    assert checks["metrics_finite"]
    assert checks["n_test_records"] == 12

    from speechdeepfake.pipeline import smoke_test_config

    cfg = smoke_test_config(base, out, "random-tiny")
    # predictions: one row per record with the dev threshold, identical record sets
    ids = None
    for m in MODEL_NAMES:
        p = pd.read_csv(artifacts.predictions_dir(cfg) / f"{m}__synthetic_test.csv")
        assert {"record_id", "label", "predicted", "score", "threshold", "dataset", "model"} <= set(p.columns)
        assert p["threshold"].nunique() == 1
        meta = json.loads((artifacts.model_dir(cfg, m) / "metadata.json").read_text())
        assert meta["threshold_fitted_on"] == "dev"
        assert p["threshold"].iloc[0] == pytest.approx(meta["threshold"])
        assert ((p["score"] >= p["threshold"]).astype(int) == p["predicted"]).all()
        ids = set(p["record_id"]) if ids is None else ids
        assert set(p["record_id"]) == ids
    # artefacts
    assert (artifacts.model_dir(cfg, "logmel_cnn") / "best.pt").exists()
    assert (artifacts.model_dir(cfg, "mfcc_svm") / "pipeline.joblib").exists()
    assert (artifacts.metrics_dir(cfg) / "comparison.csv").exists()
    report = (cfg.reports_dir / "final_report.md").read_text(encoding="utf-8")
    assert "SYNTHETIC SOFTWARE TEST" in report
    assert (artifacts.figures_dir(cfg) / "roc_synthetic_test.png").exists()
    assert (artifacts.figures_dir(cfg) / "cnn_learning_curves.png").exists()
    assert (artifacts.figures_dir(cfg) / "generalization_eer.png").exists()
    table = pd.read_csv(artifacts.metrics_dir(cfg) / "comparison.csv")
    assert set(table["dataset"]) == {"synthetic_test", "synthetic_external"}
    assert set(table["condition"]) == {"in_domain", "external_domain"}
