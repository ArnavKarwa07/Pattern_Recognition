"""Frozen wav2vec 2.0 utterance embeddings (Solution 3 features).

Each waveform is normalised by the checkpoint's feature extractor *on its own*
(so padding never affects normalisation statistics), zero-padded to the
longest clip in the batch, passed through the frozen model, and mean-pooled
over the frames that correspond to real audio only. The number of valid frames
is computed with the model's own convolutional output-length formula.

For checkpoints whose feature encoder uses group normalisation (such as
``facebook/wav2vec2-base``), Hugging Face recommends *not* passing an attention
mask; we follow that and rely on masked pooling. With the default fixed-length
preprocessing (``audio.pad_mode: repeat``) all clips in a batch have the same
length, so no padding occurs at all.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from speechdeepfake.config import AudioConfig, Wav2VecConfig
from speechdeepfake.data.preprocessing import AudioError, load_and_preprocess

TINY_RANDOM_MODEL = "random-tiny"  # randomly initialised model for unit tests only


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``hidden`` (B, T, H) over time using a boolean/0-1 ``mask`` (B, T).

    Padded frames (mask 0) contribute nothing, whatever their values.
    """
    m = mask.to(hidden.dtype).unsqueeze(-1)
    summed = (hidden * m).sum(dim=1)
    counts = m.sum(dim=1).clamp(min=1.0)
    return summed / counts


def _tiny_model():
    from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model

    config = Wav2Vec2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        conv_dim=(32,) * 7,
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
    )
    torch.manual_seed(0)
    model = Wav2Vec2Model(config)
    fe = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True)
    return model, fe


class Wav2VecEmbedder:
    """Batched, frozen wav2vec 2.0 embedding extractor."""

    name = "wav2vec2_meanpool"

    def __init__(self, cfg: Wav2VecConfig, audio_cfg: AudioConfig, device: str = "cpu"):
        if audio_cfg.sample_rate != 16000:
            raise ValueError("wav2vec 2.0 checkpoints expect 16 kHz input; set audio.sample_rate: 16000")
        self.cfg = cfg
        self.audio_cfg = audio_cfg
        self.device = device
        self._model = None
        self._fe = None
        self.revision: Optional[str] = None

    # Lazy loading keeps the object cheap to construct (e.g. for cache keys).
    def _load(self) -> None:
        if self._model is not None:
            return
        if self.cfg.model_name == TINY_RANDOM_MODEL:
            model, fe = _tiny_model()
        else:
            from transformers import AutoFeatureExtractor, Wav2Vec2Model

            kwargs = {"local_files_only": not self.cfg.allow_download}
            try:
                fe = AutoFeatureExtractor.from_pretrained(self.cfg.model_name, **kwargs)
                model = Wav2Vec2Model.from_pretrained(self.cfg.model_name, **kwargs)
            except OSError as e:
                raise RuntimeError(
                    f"Could not load {self.cfg.model_name!r}: {e}\n"
                    "Check the internet connection (first use downloads ~380 MB) or set wav2vec.allow_download: true."
                ) from e
        model.eval().to(self.device)
        for p in model.parameters():
            p.requires_grad_(False)
        self._model, self._fe = model, fe
        self.revision = getattr(model.config, "_commit_hash", None)

    @property
    def model(self):
        self._load()
        return self._model

    @property
    def hidden_size(self) -> int:
        return int(self.model.config.hidden_size)

    def config_dict(self) -> dict:
        return {
            "feature": self.name,
            "model_name": self.cfg.model_name,
            "layer": self.cfg.layer,
            "pooling": self.cfg.pooling,
            "audio": dataclasses.asdict(self.audio_cfg),
        }

    def metadata(self) -> dict:
        import transformers

        self._load()
        return {
            **self.config_dict(),
            "checkpoint_revision": self.revision,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "feat_extract_norm": self._model.config.feat_extract_norm,
            "hidden_size": self.hidden_size,
            "frozen": True,
            "device": self.device,
        }

    @torch.inference_mode()
    def embed_waveforms(self, waves: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of 16 kHz waveforms -> (B, hidden_size) float32."""
        self._load()
        normed = [
            np.asarray(self._fe(w, sampling_rate=16000, return_tensors="np").input_values[0], dtype=np.float32)
            for w in waves
        ]
        lengths = torch.tensor([len(w) for w in normed], dtype=torch.long)
        batch = torch.zeros(len(normed), int(lengths.max()), dtype=torch.float32)
        for i, w in enumerate(normed):
            batch[i, : len(w)] = torch.from_numpy(w)
        batch = batch.to(self.device)
        kwargs = {}
        if self._model.config.feat_extract_norm == "layer":
            kwargs["attention_mask"] = (torch.arange(batch.shape[1])[None] < lengths[:, None]).long().to(self.device)
        want_hidden = self.cfg.layer != -1
        out = self._model(batch, output_hidden_states=want_hidden, **kwargs)
        hidden = out.hidden_states[self.cfg.layer] if want_hidden else out.last_hidden_state
        frame_lengths = self._model._get_feat_extract_output_lengths(lengths).to(self.device)
        frame_mask = torch.arange(hidden.shape[1], device=self.device)[None] < frame_lengths[:, None]
        return masked_mean(hidden, frame_mask).float().cpu().numpy()

    def __call__(self, paths: Sequence[Path]) -> list[Union[np.ndarray, Exception]]:
        results: list[Union[np.ndarray, Exception]] = [None] * len(paths)  # type: ignore[list-item]
        loaded: list[tuple[int, np.ndarray]] = []
        for i, p in enumerate(paths):
            try:
                loaded.append((i, load_and_preprocess(p, self.audio_cfg, train=False)))
            except AudioError as e:
                results[i] = e
        # Sorting by length minimises padding when clips are variable-length.
        loaded.sort(key=lambda t: len(t[1]))
        bs = self.cfg.batch_size
        for s in range(0, len(loaded), bs):
            group = loaded[s : s + bs]
            try:
                emb = self.embed_waveforms([w for _, w in group])
                for (i, _), e in zip(group, emb):
                    results[i] = e
            except RuntimeError as err:  # e.g. CUDA OOM: report every file of the batch
                for i, _ in group:
                    results[i] = err
        return results
