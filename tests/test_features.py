"""Feature dimensionality, log-Mel shapes, wav2vec pooling/padding and caching."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from speechdeepfake.config import LogMelConfig, MFCCConfig, Wav2VecConfig
from speechdeepfake.features.cache import cached_features
from speechdeepfake.features.logmel import LogMelSpectrogram, logmel_numpy, num_frames
from speechdeepfake.features.mfcc import MFCCExtractor, feature_dim, mfcc_summary
from speechdeepfake.features.wav2vec_embeddings import TINY_RANDOM_MODEL, Wav2VecEmbedder, masked_mean
from speechdeepfake.models.logmel_cnn import LogMelCNN, logits_to_score

SR = 16000


def _wave(seconds=1.0, seed=0):
    return (np.random.default_rng(seed).standard_normal(int(SR * seconds)) * 0.1).astype(np.float32)


@pytest.mark.parametrize("n_mfcc,delta", [(40, 0), (40, 1), (20, 2)])
def test_mfcc_dimension(n_mfcc, delta):
    cfg = MFCCConfig(n_mfcc=n_mfcc, delta_order=delta)
    v = mfcc_summary(_wave(), SR, cfg)
    assert v.shape == (feature_dim(cfg),) == (n_mfcc * (1 + delta) * 2,)
    assert np.isfinite(v).all()


def test_mfcc_delta_on_very_short_clip():
    v = mfcc_summary(_wave(0.02), SR, MFCCConfig(delta_order=2))
    assert v.shape == (240,) and np.isfinite(v).all()


@pytest.mark.parametrize("n_mels", [64, 80])
def test_logmel_shape(n_mels):
    cfg = LogMelConfig(n_mels=n_mels)
    fe = LogMelSpectrogram(SR, cfg)
    out = fe(torch.from_numpy(np.stack([_wave(4.0), _wave(4.0, 1)])))
    assert out.shape == (2, n_mels, num_frames(64000, cfg.hop_length)) == (2, n_mels, 401)
    assert torch.isfinite(out).all()
    assert logmel_numpy(_wave(1.0), SR, cfg).shape == (n_mels, 101)


def test_cnn_forward_and_score_direction():
    model = LogMelCNN(SR, LogMelConfig(n_mels=64), channels=[4, 8], dropout=0.1).eval()
    x = torch.from_numpy(np.stack([_wave(1.0), _wave(1.0, 2)]))
    logits = model(x)
    assert logits.shape == (2, 2)
    assert model.spectrogram(x).shape == (2, 1, 64, 101)
    s = logits_to_score(torch.tensor([[0.0, 3.0], [2.0, 0.0]]))
    assert s.tolist() == [3.0, -2.0]


def test_masked_mean_ignores_padding():
    h = torch.zeros(2, 5, 3)
    h[0, :3] = 1.0
    h[0, 3:] = 1e6  # padded frames with garbage
    h[1] = 2.0
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)
    out = masked_mean(h, mask)
    assert torch.allclose(out, torch.tensor([[1.0] * 3, [2.0] * 3]))


@pytest.fixture(scope="module")
def tiny_embedder():
    from speechdeepfake.config import AudioConfig

    return Wav2VecEmbedder(Wav2VecConfig(model_name=TINY_RANDOM_MODEL, batch_size=2), AudioConfig(), "cpu")


def test_wav2vec_embedding_dimension_and_frozen(tiny_embedder):
    emb = tiny_embedder.embed_waveforms([_wave(1.0), _wave(1.0, 1)])
    assert emb.shape == (2, tiny_embedder.hidden_size) == (2, 32)
    assert np.isfinite(emb).all()
    assert not any(p.requires_grad for p in tiny_embedder.model.parameters())


def test_wav2vec_padding_excluded_from_pooling(tiny_embedder, monkeypatch):
    """A short clip padded in a batch must be pooled over its own frames only."""
    model = tiny_embedder.model
    captured = {}
    orig = model.forward

    def spy(*a, **k):
        out = orig(*a, **k)
        captured["hidden"] = out.last_hidden_state.detach().clone()
        return out

    monkeypatch.setattr(model, "forward", spy)
    short, long = _wave(0.5), _wave(1.5, 3)
    emb = tiny_embedder.embed_waveforms([short, long])
    n_valid = int(model._get_feat_extract_output_lengths(torch.tensor([len(short)]))[0])
    hidden = captured["hidden"]
    assert hidden.shape[1] > n_valid  # the short clip really was padded
    expected = hidden[0, :n_valid].mean(dim=0).numpy()
    assert np.allclose(emb[0], expected, atol=1e-5)
    assert not np.allclose(emb[0], hidden[0].mean(dim=0).numpy(), atol=1e-5)


def test_wav2vec_metadata_records_versions(tiny_embedder):
    meta = tiny_embedder.metadata()
    for key in ("model_name", "transformers_version", "pooling", "layer", "hidden_size", "frozen"):
        assert key in meta


def test_feature_cache_resumes_and_logs_failures(tmp_path, wav_file, cfg):
    paths = [wav_file(f"f{i}.wav", seconds=0.5 + 0.1 * i) for i in range(5)]
    records = pd.DataFrame({"record_id": [f"c/{i}" for i in range(5)] + ["c/missing"],
                            "audio_path": [str(p) for p in paths] + [str(tmp_path / "missing.wav")],
                            "label": [0, 1, 0, 1, 0, 1]})
    ext = MFCCExtractor(cfg.audio, MFCCConfig(n_jobs=1))
    calls = {"n": 0}

    def counting(p):
        calls["n"] += len(p)
        return ext(p)

    kw = dict(cache_root=tmp_path / "cache", feature_name="mfcc", config=ext.config_dict(), data_dir=tmp_path, chunk_size=2)
    fs = cached_features(records, counting, **kw)
    assert len(fs.ids) == 5 and "c/missing" in fs.failed and fs.X.shape == (5, 80)
    first = calls["n"]
    fs2 = cached_features(records, counting, **kw)
    assert calls["n"] == first  # everything served from cache
    assert np.array_equal(fs.X, fs2.X) and fs2.computed_now == 0
    recs, X = fs.align(records)
    assert recs["record_id"].tolist() == fs.ids and X.shape == (5, 80)
