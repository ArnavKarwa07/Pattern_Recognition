"""Label normalisation, manifest parsing, validation and leakage checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from speechdeepfake.config import ConfigError, load_config
from speechdeepfake.data.loaders import load_asvspoof2019_la, load_custom_csv, parse_asvspoof_protocol, ProtocolError
from speechdeepfake.data.manifests import ManifestError, check_integrity, normalize_label, normalize_manifest
from speechdeepfake.data.preprocessing import AudioError, fix_length, load_audio, preprocess
from speechdeepfake.data.splits import stratified_subsample
from speechdeepfake.data.validation import inspect_audio_file


# ------------------------------------------------------------------- labels


@pytest.mark.parametrize("value", ["bonafide", "Bona-Fide", " genuine ", "REAL", "human", 0, "0", 0.0])
def test_bonafide_variants(value):
    assert normalize_label(value) == 0


@pytest.mark.parametrize("value", ["spoof", "FAKE", "synthetic", "deepfake", "AI-generated", 1, "1"])
def test_spoof_variants(value):
    assert normalize_label(value) == 1


@pytest.mark.parametrize("value", ["unknown", "maybe", 2, -1, "", True, float("nan")])
def test_unknown_labels_rejected(value):
    with pytest.raises(ManifestError):
        normalize_label(value)


def _manifest(**over):
    base = {"record_id": ["c/a", "c/b"], "audio_path": ["a.wav", "b.wav"], "label": ["bonafide", "spoof"],
            "corpus": ["c", "c"], "split": ["train", "train"]}
    base.update(over)
    return pd.DataFrame(base)


def test_normalize_manifest_fills_optional_columns():
    df = normalize_manifest(_manifest())
    assert df["label"].tolist() == [0, 1]
    assert (df["speaker_id"] == "").all() and "generator_id" in df.columns


def test_normalize_manifest_rejects_unknown_label_and_missing_column():
    with pytest.raises(ManifestError, match="unknown labels"):
        normalize_manifest(_manifest(label=["bonafide", "weird"]))
    with pytest.raises(ManifestError, match="missing required"):
        normalize_manifest(_manifest().drop(columns="split"))


# ---------------------------------------------------------------- integrity


def test_duplicates_and_conflicting_labels_detected():
    df = normalize_manifest(_manifest(record_id=["c/a", "c/a"], audio_path=["a.wav", "a2.wav"]))
    rep = check_integrity(df)
    assert not rep.ok and any("conflicting labels" in e for e in rep.errors)
    df2 = normalize_manifest(_manifest(record_id=["c/a", "c/a"], label=["spoof", "spoof"], audio_path=["a.wav", "a2.wav"]))
    assert any("duplicated" in e for e in check_integrity(df2).errors)


def test_cross_split_leakage_detected_by_group_and_hash():
    df = normalize_manifest(_manifest(split=["train", "eval"], group_id=["g1", "g1"]))
    assert any("group_id" in e for e in check_integrity(df).errors)
    df = normalize_manifest(_manifest(split=["train", "eval"]))
    df["sha1"] = ["x", "x"]
    assert any("SHA-1" in e for e in check_integrity(df).errors)
    clean = normalize_manifest(_manifest(split=["train", "eval"], group_id=["g1", "g2"]))
    assert check_integrity(clean).ok


def test_speaker_overlap_is_a_warning():
    df = normalize_manifest(_manifest(split=["train", "eval"], speaker_id=["s1", "s1"]))
    rep = check_integrity(df)
    assert rep.ok and rep.warnings


# ----------------------------------------------------------------- ASVspoof


PROTO_TRAIN = """LA_0079 LA_T_1138215 - - bonafide
LA_0079 LA_T_1271820 - A01 spoof
LA_0080 LA_T_1000001 - A06 spoof
"""


def _asv_tree(root: Path, eval_text: str):
    p = root / "LA" / "ASVspoof2019_LA_cm_protocols"
    p.mkdir(parents=True)
    (p / "ASVspoof2019.LA.cm.train.trn.txt").write_text(PROTO_TRAIN)
    (p / "ASVspoof2019.LA.cm.dev.trl.txt").write_text("LA_0069 LA_D_1047731 - - bonafide\nLA_0069 LA_D_1105538 - A02 spoof\n")
    (p / "ASVspoof2019.LA.cm.eval.trl.txt").write_text(eval_text)
    return root / "LA"


def test_parse_asvspoof_protocol(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text(PROTO_TRAIN)
    df, has_keys = parse_asvspoof_protocol(f, "train")
    assert has_keys
    assert df["label"].tolist() == [0, 1, 1]
    assert df["generator_id"].tolist() == ["bonafide", "A01", "A06"]
    assert df["speaker_id"].tolist() == ["LA_0079", "LA_0079", "LA_0080"]


def test_asvspoof_protocol_inconsistent_key_rejected(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("LA_0079 LA_T_1 - A01 bonafide\n")
    with pytest.raises(ProtocolError):
        parse_asvspoof_protocol(f, "train")


def test_asvspoof_manifest_and_unlabelled_eval_is_skipped(tmp_path):
    root = _asv_tree(tmp_path, "LA_E_1000001\nLA_E_1000002\n")  # eval without keys
    df, notes = load_asvspoof2019_la(root, tmp_path)
    assert set(df["split"]) == {"train", "dev"}
    assert any("no bonafide/spoof keys" in n for n in notes)
    assert df["record_id"].str.startswith("asvspoof2019_la/").all()
    assert df.loc[df["utterance_id"] == "LA_T_1138215", "audio_path"].item() == "LA/ASVspoof2019_LA_train/flac/LA_T_1138215.flac"


def test_asvspoof_labelled_eval_included(tmp_path):
    root = _asv_tree(tmp_path, "LA_0001 LA_E_1000001 - - bonafide\nLA_0001 LA_E_1000002 - A17 spoof\n")
    df, _ = load_asvspoof2019_la(root, tmp_path)
    assert (df["split"] == "eval").sum() == 2


def test_missing_dataset_gives_instructions(tmp_path):
    with pytest.raises(FileNotFoundError, match="datashare"):
        load_asvspoof2019_la(tmp_path / "nope", tmp_path)


def _cdadd_parquet(root: Path, rows):
    """Write a tiny parquet shard with the public CD-ADD schema (path, audio{bytes,path}, label, notes)."""
    import io
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq
    import soundfile as sf

    recs = []
    for path, label, system in rows:
        buf = io.BytesIO()
        sf.write(buf, np.zeros(1600, np.float32) + 0.01, 16000, format="FLAC")
        notes = {"utterance_id": path[:-4], "source": "libritts", "system": system, "speaker_id": "1089"}
        recs.append({"path": path, "audio": {"bytes": buf.getvalue(), "path": path}, "label": label, "notes": json.dumps(notes)})
    (root / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(recs), root / "data" / "test-00000-of-00001.parquet")


def test_cdadd_export(tmp_path):
    from speechdeepfake.data.loaders import export_cdadd_parquet

    root = tmp_path / "cdadd"
    _cdadd_parquet(root, [("dataset_LibriTTS/test-clean/1089/u1/real.wav", 0, "real"),
                          ("dataset_LibriTTS/test-clean/1089/u1/valle.wav", 1, "valle")])
    df, notes = export_cdadd_parquet(root, tmp_path)
    assert df["label"].tolist() == [0, 1]
    assert df["generator_id"].tolist() == ["bonafide", "valle"]
    assert df["group_id"].nunique() == 1 and (df["split"] == "test").all()
    assert all((tmp_path / p).exists() and p.endswith(".flac") for p in df["audio_path"])


def test_cdadd_label_system_conflict_rejected(tmp_path):
    from speechdeepfake.data.loaders import export_cdadd_parquet

    root = tmp_path / "cdadd"
    _cdadd_parquet(root, [("d/u1/real.wav", 1, "real")])
    with pytest.raises(ManifestError, match="contradicts"):
        export_cdadd_parquet(root, tmp_path)


@pytest.mark.parametrize("method", ["stored", "deflated"])
def test_remote_zip_member_extraction(monkeypatch, method):
    """_fetch_member must reproduce zipfile's own extraction from raw byte ranges."""
    import io
    import zipfile

    from speechdeepfake.data import subsets

    payload = bytes(range(256)) * 400
    buf = io.BytesIO()
    comp = zipfile.ZIP_STORED if method == "stored" else zipfile.ZIP_DEFLATED
    with zipfile.ZipFile(buf, "w", compression=comp) as zf:
        zf.writestr("LA/a.txt", b"hello")
        zf.writestr("LA/ASVspoof2019_LA_train/flac/LA_T_1.flac", payload)
    blob = buf.getvalue()
    monkeypatch.setattr(subsets, "_get_range", lambda url, s, e: blob[s : min(e, len(blob) - 1) + 1])
    info = zipfile.ZipFile(io.BytesIO(blob)).getinfo("LA/ASVspoof2019_LA_train/flac/LA_T_1.flac")
    assert subsets._fetch_member("mem://", info) == payload


def test_stratified_subset_selection_covers_every_attack():
    from speechdeepfake.data.subsets import _stratified_ids

    df = pd.DataFrame({"utterance_id": [f"u{i}" for i in range(100)],
                       "generator_id": ["bonafide"] * 40 + ["A01"] * 30 + ["A02"] * 30})
    ids = _stratified_ids(df, n_bonafide=10, n_per_attack=5, rng=np.random.default_rng(0))
    chosen = df[df["utterance_id"].isin(ids)]["generator_id"].value_counts().to_dict()
    assert chosen == {"bonafide": 10, "A01": 5, "A02": 5}


def test_custom_csv(tmp_path):
    csv = tmp_path / "m.csv"
    pd.DataFrame({"audio_path": ["x/a.wav"], "label": ["fake"], "split": ["test"]}).to_csv(csv, index=False)
    df = load_custom_csv(csv, tmp_path)
    assert df.loc[0, "label"] == 1 and df.loc[0, "corpus"] == "custom" and df.loc[0, "audio_path"] == "x/a.wav"


# --------------------------------------------------------------- validation


def test_inspect_audio_detects_missing_corrupt_silent_and_ok(tmp_path, wav_file, cfg):
    assert inspect_audio_file(tmp_path / "missing.wav", cfg.audio)["status"] == "missing"
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF this is not audio")
    assert inspect_audio_file(bad, cfg.audio)["status"] == "corrupt"
    silent = inspect_audio_file(wav_file("s.wav", value=0.0), cfg.audio)
    assert silent["status"] == "ok" and silent["is_silent"]
    ok = inspect_audio_file(wav_file("ok.wav", seconds=1.5), cfg.audio)
    assert ok["status"] == "ok" and abs(ok["duration_s"] - 1.5) < 1e-3 and len(ok["sha1"]) == 40
    short = inspect_audio_file(wav_file("short.wav", seconds=0.01), cfg.audio)
    assert short["status"] == "too_short"


def test_inspect_audio_detects_nan(tmp_path, wav_file, cfg):
    assert inspect_audio_file(wav_file("nan.wav", value=np.nan), cfg.audio)["status"] == "non_finite"
    with pytest.raises(AudioError):
        load_audio(tmp_path / "nan.wav")


def test_load_audio_mono_and_resample(wav_file):
    x = load_audio(wav_file("st.wav", seconds=1.0, sr=22050, channels=2), sample_rate=16000)
    assert x.ndim == 1 and abs(len(x) - 16000) <= 1 and x.dtype == np.float32


# ------------------------------------------------------------ preprocessing


def test_fix_length_policies():
    x = np.arange(10, dtype=np.float32)
    assert fix_length(x, 4, "start").tolist() == [0, 1, 2, 3]
    assert fix_length(x, 4, "center").tolist() == [3, 4, 5, 6]
    assert fix_length(x[:3], 7, pad_mode="repeat").tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert fix_length(x[:3], 5, pad_mode="zero").tolist() == [0, 1, 2, 0, 0]
    assert len(fix_length(x[:3], 5, pad_mode="none")) == 3
    with pytest.raises(ValueError):
        fix_length(x, 4, "random")


def test_eval_crop_is_deterministic_even_if_train_random_crop_enabled(cfg):
    cfg.audio.train_random_crop = True
    x = np.random.default_rng(0).standard_normal(16000 * 6).astype(np.float32)
    a = preprocess(x, cfg.audio, train=False)
    b = preprocess(x, cfg.audio, train=False)
    assert np.array_equal(a, b) and np.array_equal(a, x[:64000])
    c = preprocess(x, cfg.audio, train=True, rng=np.random.default_rng(1))
    assert len(c) == 64000


def test_stratified_subsample_keeps_both_classes():
    df = pd.DataFrame({"record_id": [f"r{i}" for i in range(100)], "label": [0] * 10 + [1] * 90})
    s = stratified_subsample(df, 20, seed=0)
    assert len(s) == 20 and set(s["label"]) == {0, 1}
    assert s.equals(stratified_subsample(df, 20, seed=0))


def test_full_run_cannot_subsample():
    with pytest.raises(ConfigError, match="must not subsample"):
        load_config(None, overrides={"run_type": "full", "data": {"max_train_samples": 10}})


def test_unknown_config_key_rejected():
    with pytest.raises(ConfigError, match="Unknown key"):
        load_config(None, overrides={"audio": {"sample_rat": 8000}})


def test_shipped_configs_load():
    root = Path(__file__).resolve().parents[1]
    for name in ("default.yaml", "quick_test.yaml"):
        cfg = load_config(root / "configs" / name)
        assert cfg.project_root == str(root)
