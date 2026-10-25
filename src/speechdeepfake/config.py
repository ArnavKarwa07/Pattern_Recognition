"""Experiment configuration: typed dataclasses loaded from YAML.

Unknown keys are rejected so that a typo in a YAML file fails loudly instead of
silently falling back to a default. Relative paths are resolved against the
project root (the parent of the ``configs/`` directory that holds the YAML file,
or the ``SPEECHDEEPFAKE_ROOT`` environment variable when set).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml

MODEL_NAMES = ("mfcc_svm", "logmel_cnn", "wav2vec_svm")
RUN_TYPES = ("full", "pilot", "smoke_test")


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


@dataclass
class AudioConfig:
    """Shared preprocessing applied identically by all three pipelines."""

    sample_rate: int = 16000
    mono: bool = True
    # Fixed analysis duration in seconds. Long audio is cropped, short audio is
    # padded according to ``pad_mode``. ``null`` keeps the full recording.
    max_duration: Optional[float] = 4.0
    # Deterministic crop position used for dev/test (and for train unless
    # ``train_random_crop`` is enabled): "start" or "center".
    eval_crop: str = "start"
    # Random crop position for training samples only (CNN augmentation).
    train_random_crop: bool = False
    # "repeat" tiles the signal, "zero" appends zeros, "none" leaves it short.
    pad_mode: str = "repeat"
    # "none", "peak" (max |x| = 1) or "rms" (unit RMS).
    normalization: str = "none"
    # Trim leading/trailing silence quieter than this many dB below peak (null = off).
    trim_silence_db: Optional[float] = None
    # Clips whose peak amplitude is below this are flagged as silent.
    silence_threshold: float = 1e-4
    min_duration: float = 0.1


@dataclass
class MFCCConfig:
    n_mfcc: int = 40
    n_mels: int = 64
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    fmin: float = 20.0
    fmax: Optional[float] = None
    # 0 = static only, 1 = +delta, 2 = +delta and delta-delta.
    delta_order: int = 0
    n_jobs: int = 4


@dataclass
class LogMelConfig:
    n_mels: int = 80
    n_fft: int = 512
    win_length: int = 400
    hop_length: int = 160
    fmin: float = 20.0
    fmax: Optional[float] = None
    log_offset: float = 1e-6


@dataclass
class SVMConfig:
    C_grid: list = field(default_factory=lambda: [0.1, 1.0, 10.0])
    # Values may be floats, "scale", "auto", or "scale*<factor>".
    gamma_grid: list = field(default_factory=lambda: ["scale"])
    class_weight: Optional[str] = "balanced"
    # Development-set criterion used to pick hyperparameters: eer, roc_auc, balanced_accuracy.
    selection_metric: str = "eer"
    cache_size_mb: int = 2000
    max_iter: int = -1


@dataclass
class CNNConfig:
    channels: list = field(default_factory=lambda: [16, 32, 64, 128])
    dropout: float = 0.3
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "plateau"  # "plateau" or "none"
    lr_factor: float = 0.5
    lr_patience: int = 2
    early_stopping_patience: int = 5
    class_weighted_loss: bool = True
    # Training clips used to estimate per-Mel-band normalisation statistics.
    norm_stats_samples: int = 2000
    num_workers: int = 2
    # Development-set criterion used for checkpoint selection: eer or loss.
    selection_metric: str = "eer"


@dataclass
class Wav2VecConfig:
    model_name: str = "facebook/wav2vec2-base"
    # Hidden-state index: -1 = final transformer layer, 0 = CNN feature projection.
    layer: int = -1
    pooling: str = "mean"
    batch_size: int = 8
    # Allow downloading the checkpoint from the Hugging Face Hub.
    allow_download: bool = True


@dataclass
class TestSetConfig:
    name: str
    corpus: str
    split: str
    # in_domain, external_domain or generator_held_out.
    condition: str = "in_domain"
    generators: Optional[list] = None


@dataclass
class DataConfig:
    asvspoof_root: Optional[str] = "raw/ASVspoof2019/LA"
    cdadd_root: Optional[str] = "raw/cdadd"
    custom_manifest: Optional[str] = None
    train: dict = field(default_factory=lambda: {"corpus": "asvspoof2019_la", "split": "train"})
    dev: dict = field(default_factory=lambda: {"corpus": "asvspoof2019_la", "split": "dev"})
    test_sets: list = field(default_factory=list)
    # Generators removed from train and dev (for a generator-held-out experiment).
    holdout_generators: list = field(default_factory=list)
    # Pilot-run limits. They must be null for run_type "full".
    max_train_samples: Optional[int] = None
    max_dev_samples: Optional[int] = None
    max_test_samples: Optional[int] = None
    require_validation: bool = True


@dataclass
class PathsConfig:
    # data_dir may be absolute (recommended outside OneDrive for multi-GB data).
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    reports_dir: str = "reports"


@dataclass
class EvalConfig:
    # Development-set rule for the operating threshold: eer, youden, max_f1, balanced_accuracy.
    threshold_method: str = "eer"
    n_bootstrap: int = 200
    ci_level: float = 0.95
    batch_size: int = 32


@dataclass
class ExperimentConfig:
    run_name: str = "default"
    run_type: str = "full"
    seed: int = 1234
    device: str = "auto"
    feature_chunk_size: int = 512
    audio: AudioConfig = field(default_factory=AudioConfig)
    mfcc: MFCCConfig = field(default_factory=MFCCConfig)
    logmel: LogMelConfig = field(default_factory=LogMelConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    wav2vec: Wav2VecConfig = field(default_factory=Wav2VecConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    # Set by the loader, not by YAML.
    project_root: str = "."
    config_path: Optional[str] = None

    # ----------------------------------------------------------------- paths
    def _resolve(self, p: Union[str, Path]) -> Path:
        path = Path(os.path.expandvars(str(p))).expanduser()
        return path if path.is_absolute() else Path(self.project_root) / path

    @property
    def data_dir(self) -> Path:
        override = os.environ.get("SPEECHDEEPFAKE_DATA_DIR")
        return self._resolve(override) if override else self._resolve(self.paths.data_dir)

    def data_path(self, p: Optional[str]) -> Optional[Path]:
        """Resolve a dataset path relative to ``data_dir``."""
        if p is None:
            return None
        path = Path(os.path.expandvars(str(p))).expanduser()
        return path if path.is_absolute() else self.data_dir / path

    @property
    def manifests_dir(self) -> Path:
        return self.data_dir / "manifests"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def models_dir(self) -> Path:
        return self._resolve(self.paths.models_dir) / self.run_name

    @property
    def results_dir(self) -> Path:
        return self._resolve(self.paths.results_dir) / self.run_name

    @property
    def reports_dir(self) -> Path:
        return self._resolve(self.paths.reports_dir)

    def test_sets(self) -> list[TestSetConfig]:
        return [TestSetConfig(**t) for t in self.data.test_sets]

    # ------------------------------------------------------------- snapshots
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("project_root", None)
        return d

    def save_snapshot(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def validate(self) -> None:
        if self.run_type not in RUN_TYPES:
            raise ConfigError(f"run_type must be one of {RUN_TYPES}, got {self.run_type!r}")
        limits = [self.data.max_train_samples, self.data.max_dev_samples, self.data.max_test_samples]
        if self.run_type == "full" and any(v is not None for v in limits):
            raise ConfigError(
                "run_type 'full' must not subsample data: set data.max_*_samples to null, "
                "or set run_type: pilot so results are labelled as pilot results."
            )
        a = self.audio
        if a.pad_mode not in ("repeat", "zero", "none"):
            raise ConfigError(f"audio.pad_mode must be repeat|zero|none, got {a.pad_mode!r}")
        if a.eval_crop not in ("start", "center"):
            raise ConfigError(f"audio.eval_crop must be start|center, got {a.eval_crop!r}")
        if a.normalization not in ("none", "peak", "rms"):
            raise ConfigError(f"audio.normalization must be none|peak|rms, got {a.normalization!r}")
        if self.mfcc.n_mfcc > self.mfcc.n_mels:
            raise ConfigError("mfcc.n_mfcc must not exceed mfcc.n_mels")
        if self.mfcc.delta_order not in (0, 1, 2):
            raise ConfigError("mfcc.delta_order must be 0, 1 or 2")
        if self.evaluation.threshold_method not in ("eer", "youden", "max_f1", "balanced_accuracy"):
            raise ConfigError(f"Unknown evaluation.threshold_method {self.evaluation.threshold_method!r}")
        if self.svm.selection_metric not in ("eer", "roc_auc", "balanced_accuracy"):
            raise ConfigError(f"Unknown svm.selection_metric {self.svm.selection_metric!r}")
        if self.cnn.selection_metric not in ("eer", "loss"):
            raise ConfigError(f"Unknown cnn.selection_metric {self.cnn.selection_metric!r}")
        if self.wav2vec.pooling != "mean":
            raise ConfigError("Only wav2vec.pooling: mean is implemented")
        names = [t["name"] for t in self.data.test_sets]
        if len(names) != len(set(names)):
            raise ConfigError(f"Duplicate test-set names: {names}")
        for t in self.data.test_sets:
            TestSetConfig(**t)  # raises TypeError on bad keys
            if t.get("condition", "in_domain") not in ("in_domain", "external_domain", "generator_held_out"):
                raise ConfigError(f"Bad condition for test set {t['name']!r}")


def stable_hash(obj: Any, length: int = 12) -> str:
    """Deterministic short hash of a JSON-serialisable object."""
    payload = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:length]


def _build(cls, values: dict, where: str):
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise ConfigError(f"Section {where!r} must be a mapping")
    known = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ConfigError(f"Unknown key(s) in {where!r}: {sorted(unknown)}")
    kwargs = {}
    for name, value in values.items():
        sub = _SECTIONS.get(name) if cls is ExperimentConfig else None
        kwargs[name] = _build(sub, value, name) if sub else value
    return cls(**kwargs)


_SECTIONS = {
    "audio": AudioConfig,
    "mfcc": MFCCConfig,
    "logmel": LogMelConfig,
    "svm": SVMConfig,
    "cnn": CNNConfig,
    "wav2vec": Wav2VecConfig,
    "data": DataConfig,
    "paths": PathsConfig,
    "evaluation": EvalConfig,
}


def _infer_project_root(config_path: Optional[Path]) -> Path:
    env = os.environ.get("SPEECHDEEPFAKE_ROOT")
    if env:
        return Path(env).resolve()
    if config_path is not None and config_path.parent.name == "configs":
        return config_path.parent.parent.resolve()
    return Path.cwd().resolve()


def load_config(path: Optional[Union[str, Path]] = None, overrides: Optional[dict] = None) -> ExperimentConfig:
    """Load and validate an experiment configuration.

    Args:
        path: YAML file. ``None`` returns the defaults.
        overrides: nested dict merged on top of the YAML values.
    """
    values: dict = {}
    config_path = None
    if path is not None:
        config_path = Path(path).resolve()
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        with open(config_path, encoding="utf-8") as f:
            values = yaml.safe_load(f) or {}
    if overrides:
        values = deep_merge(values, overrides)
    cfg = _build(ExperimentConfig, values, "<root>")
    cfg.project_root = str(_infer_project_root(config_path))
    cfg.config_path = str(config_path) if config_path else None
    cfg.validate()
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out
