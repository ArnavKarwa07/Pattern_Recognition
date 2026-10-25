"""Solution 1: MFCC temporal statistics + RBF-SVM (classical baseline)."""

from __future__ import annotations

from typing import Optional

from speechdeepfake.config import ExperimentConfig
from speechdeepfake.features.mfcc import MFCCExtractor

MODEL_NAME = "mfcc_svm"
DISPLAY_NAME = "MFCC + SVM"


def make_extractor(cfg: ExperimentConfig, device: Optional[str] = None) -> MFCCExtractor:
    """MFCC features run on CPU worker processes; ``device`` is ignored."""
    return MFCCExtractor(cfg.audio, cfg.mfcc)
