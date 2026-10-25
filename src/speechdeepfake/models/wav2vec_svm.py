"""Solution 3: frozen wav2vec 2.0 mean-pooled embeddings + RBF-SVM.

The pretrained network is used only as a feature extractor; it is not an
audio-deepfake detector and is never fine-tuned here. The SVM is trained for
this project's bona fide vs. spoof task.
"""

from __future__ import annotations

from typing import Optional

from speechdeepfake.config import ExperimentConfig
from speechdeepfake.features.wav2vec_embeddings import Wav2VecEmbedder

MODEL_NAME = "wav2vec_svm"
DISPLAY_NAME = "wav2vec 2.0 + SVM"


def make_extractor(cfg: ExperimentConfig, device: Optional[str] = None) -> Wav2VecEmbedder:
    return Wav2VecEmbedder(cfg.wav2vec, cfg.audio, device=device or "cpu")
