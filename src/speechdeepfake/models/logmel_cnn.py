"""Solution 2: compact 2-D CNN on log-Mel spectrograms.

Architecture (default ``channels = [16, 32, 64, 128]``)::

    waveform -> log-Mel (n_mels x frames) -> per-band standardisation
    -> 4 x [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2]
    -> global average pool -> Dropout -> Linear(2)

The per-band mean/std buffers are estimated on training data and stored in
the checkpoint. The decision score is the logit difference
``z_spoof - z_bonafide`` (the log-odds of spoof under the softmax).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from speechdeepfake.config import LogMelConfig
from speechdeepfake.features.logmel import LogMelSpectrogram

MODEL_NAME = "logmel_cnn"
DISPLAY_NAME = "log-Mel + CNN"


class LogMelCNN(nn.Module):
    def __init__(self, sample_rate: int, logmel_cfg: LogMelConfig, channels: Sequence[int] = (16, 32, 64, 128), dropout: float = 0.3):
        super().__init__()
        self.frontend = LogMelSpectrogram(sample_rate, logmel_cfg)
        self.register_buffer("band_mean", torch.zeros(logmel_cfg.n_mels, 1))
        self.register_buffer("band_std", torch.ones(logmel_cfg.n_mels, 1))
        layers: list[nn.Module] = []
        c_in = 1
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, 2))

    def spectrogram(self, wave: torch.Tensor) -> torch.Tensor:
        """Normalised log-Mel input, shape (B, 1, n_mels, frames)."""
        spec = self.frontend(wave)
        return ((spec - self.band_mean) / self.band_std).unsqueeze(1)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(self.spectrogram(wave)))

    @torch.no_grad()
    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.band_mean.copy_(mean.reshape_as(self.band_mean))
        self.band_std.copy_(std.clamp(min=1e-5).reshape_as(self.band_std))


def logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    """Spoof score = z_spoof - z_bonafide (higher = more likely spoof)."""
    return logits[:, 1] - logits[:, 0]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
