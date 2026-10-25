"""Locations of every experiment output, derived from the configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from speechdeepfake.config import ExperimentConfig
from speechdeepfake.utils.logging import ErrorLog


def model_dir(cfg: ExperimentConfig, model: str) -> Path:
    return cfg.models_dir / model


def predictions_dir(cfg: ExperimentConfig) -> Path:
    return cfg.results_dir / "predictions"


def metrics_dir(cfg: ExperimentConfig) -> Path:
    return cfg.results_dir / "metrics"


def splits_dir(cfg: ExperimentConfig) -> Path:
    return cfg.results_dir / "splits"


def logs_dir(cfg: ExperimentConfig) -> Path:
    return cfg.results_dir / "logs"


def figures_dir(cfg: ExperimentConfig) -> Path:
    return cfg.reports_dir / "figures"


def tables_dir(cfg: ExperimentConfig) -> Path:
    return cfg.reports_dir / "tables"


def error_log(cfg: ExperimentConfig) -> ErrorLog:
    return ErrorLog(logs_dir(cfg) / "errors.csv")


def _default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
