"""Log-Mel spectrogram front end (Solution 2 input), computed in PyTorch.

Computing the spectrogram on the training device avoids a multi-GB on-disk
cache and makes the same transform available for training, evaluation and
plotting.
"""

from __future__ import annotations

import librosa
import numpy as np
import torch
from torch import nn

from speechdeepfake.config import LogMelConfig


class LogMelSpectrogram(nn.Module):
    """Waveforms (B, T) -> log-Mel spectrograms (B, n_mels, frames).

    Uses a Hann window, ``center=True`` STFT, power spectrum, a Slaney-normalised
    librosa Mel filterbank and ``log(mel + log_offset)``. Frames equal
    ``1 + T // hop_length``.
    """

    def __init__(self, sample_rate: int, cfg: LogMelConfig):
        super().__init__()
        self.n_fft = cfg.n_fft
        self.hop_length = cfg.hop_length
        self.win_length = cfg.win_length
        self.log_offset = cfg.log_offset
        fb = librosa.filters.mel(sr=sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax)
        self.register_buffer("mel_fb", torch.from_numpy(fb).float(), persistent=False)
        self.register_buffer("window", torch.hann_window(cfg.win_length), persistent=False)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = torch.stft(
            wave,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        power = spec.real**2 + spec.imag**2
        mel = torch.matmul(self.mel_fb, power)
        return torch.log(mel + self.log_offset)


def num_frames(n_samples: int, hop_length: int) -> int:
    return 1 + n_samples // hop_length


def logmel_numpy(wave: np.ndarray, sample_rate: int, cfg: LogMelConfig) -> np.ndarray:
    """Convenience CPU version for plots and tests: (n_mels, frames)."""
    fe = LogMelSpectrogram(sample_rate, cfg)
    with torch.no_grad():
        return fe(torch.from_numpy(np.asarray(wave, dtype=np.float32))[None])[0].numpy()
