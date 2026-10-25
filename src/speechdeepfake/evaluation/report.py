"""Markdown research report generated from the saved outputs of a run.

Every number in the report is read from result files. Anything that has not
been run is written as "Not yet measured." and no ranking statement is made
unless all compared models have measured results on that test set.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from speechdeepfake import artifacts
from speechdeepfake.config import MODEL_NAMES, ExperimentConfig
from speechdeepfake.data.validation import summary_markdown, validated_paths
from speechdeepfake.evaluation.predict import DISPLAY, load_predictions

NYM = "Not yet measured."

REFERENCES = """\
1. Li, Y., Zhang, M., Ren, M., Qiao, X., Ma, M., Wei, D., & Yang, H. (2024). Cross-Domain Audio Deepfake Detection: Dataset and Analysis. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 4977-4983. https://doi.org/10.18653/v1/2024.emnlp-main.286 (dataset repository: https://github.com/leolya/CD-ADD; public package used here: https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD, CC BY 4.0).
2. Todisco, M., Wang, X., Vestman, V., Sahidullah, M., Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., & Lee, K. A. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. *Proc. Interspeech 2019*, 1008-1012. https://doi.org/10.21437/Interspeech.2019-2249
3. Yamagishi, J., Todisco, M., Sahidullah, M., Delgado, H., Wang, X., Evans, N., Kinnunen, T., Lee, K. A., Vestman, V., & Nautsch, A. (2019). ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database [sound]. University of Edinburgh, CSTR. https://doi.org/10.7488/ds/2555 (ODC-By licence; challenge site https://www.asvspoof.org/index2019.html).
4. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://papers.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html (checkpoint: https://huggingface.co/facebook/wav2vec2-base).
5. Gao, Y., Vuong, T., Elyasi, M., Bharaj, G., & Singh, R. (2021). Generalized Spoofing Detection Inspired from Audio Generation Artifacts. *Proc. Interspeech 2021*, 4184-4188. https://doi.org/10.21437/Interspeech.2021-1705
6. Müller, N. M., Dieckmann, F., Czempin, P., Canals, R., Böttinger, K., & Williams, J. (2021). Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn? *Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge*, 55-60. https://doi.org/10.21437/ASVSPOOF.2021-9
7. Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. *Proc. IEEE ICASSP 2015*, 5206-5210. https://doi.org/10.1109/ICASSP.2015.7178964
"""


def _fmt(v, pct: bool = False, digits: int = 4) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, str):
        return v
    if pct:
        return f"{100 * float(v):.2f}%"
    if isinstance(v, (int,)) or (isinstance(v, float) and float(v).is_integer() and abs(v) > 100):
        return f"{int(v):,}"
    return f"{float(v):.{digits}g}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _model_meta(cfg: ExperimentConfig) -> dict[str, Optional[dict]]:
    out = {}
    for m in MODEL_NAMES:
        f = artifacts.model_dir(cfg, m) / "metadata.json"
        out[m] = artifacts.load_json(f) if f.exists() else None
    return out


def _results_section(table: Optional[pd.DataFrame], cfg: ExperimentConfig) -> str:
    sections = []
    for t in cfg.test_sets():
        sections.append(f"#### {t.name} ({t.condition.replace('_', ' ')}; {t.corpus}/{t.split})\n")
        sub = None if table is None else table[table["dataset"] == t.name]
        if sub is None or sub.empty:
            sections.append(NYM + "\n")
            continue
        rows = []
        for m in MODEL_NAMES:
            r = sub[sub["model"] == m]
            if r.empty:
                rows.append({"model": DISPLAY[m], "n": NYM})
                continue
            r = r.iloc[0]
            rows.append({
                "model": DISPLAY[m], "n (bona fide / spoof)": f"{int(r['n']):,} ({int(r['n_bonafide']):,} / {int(r['n_spoof']):,})",
                "threshold (dev)": _fmt(r["threshold"]), "accuracy": _fmt(r["accuracy"], True),
                "balanced acc.": _fmt(r["balanced_accuracy"], True), "precision": _fmt(r["precision"], True),
                "recall": _fmt(r["recall"], True), "F1": _fmt(r["f1"], True), "ROC-AUC": _fmt(r["roc_auc"]),
                "AP": _fmt(r["average_precision"]),
                "EER [95% CI]": f"{_fmt(r['eer'], True)} [{_fmt(r['eer_ci_low'], True)}, {_fmt(r['eer_ci_high'], True)}]",
            })
        sections.append(_md_table(pd.DataFrame(rows).fillna("")) + "\n")
        sections.append(f"Records compared (common to all evaluated models): {int(sub['n_common'].iloc[0]):,}. "
                        f"CI method: {sub['ci_method'].iloc[0]}.\n")
    return "\n".join(sections)


def _observations(table: Optional[pd.DataFrame]) -> str:
    if table is None or table.empty:
        return NYM + " No comparative conclusion is drawn before the experiments have been run."
    lines = []
    for d, sub in table.groupby("dataset", sort=False):
        if set(sub["model"]) != set(MODEL_NAMES):
            lines.append(f"- {d}: not all three models have been evaluated; no ranking is stated.")
            continue
        sub = sub.sort_values("eer")
        best, second = sub.iloc[0], sub.iloc[1]
        if best["eer"] >= 0.4:
            lines.append(
                f"- {d} ({best['condition'].replace('_', ' ')}): all three models are close to or worse than chance "
                f"(EER {_fmt(sub['eer'].min(), True)}–{_fmt(sub['eer'].max(), True)}; ROC-AUC "
                f"{_fmt(sub['roc_auc'].min())}–{_fmt(sub['roc_auc'].max())}), so no meaningful ranking exists. "
                "An AUC below 0.5 means that model scores genuine recordings as *more* spoof-like than the TTS "
                "recordings on this data. Balanced accuracy is about 50%, i.e. the dev-selected thresholds "
                f"label nearly every recording spoof; the high plain accuracy only reflects the {_fmt(best['n_spoof'] / best['n'], True)} spoof share."
            )
            continue
        if best["eer"] == second["eer"]:
            tied = sub[sub["eer"] == best["eer"]]["display_name"].tolist()
            lines.append(f"- {d}: {', '.join(tied)} tied at the lowest measured EER ({_fmt(best['eer'], True)}).")
            continue
        overlap = best["eer_ci_high"] >= second["eer_ci_low"]
        lines.append(
            f"- {d} ({best['condition'].replace('_', ' ')}): lowest measured EER was {best['display_name']} "
            f"({_fmt(best['eer'], True)}), followed by {second['display_name']} ({_fmt(second['eer'], True)}). "
            + ("Their bootstrap confidence intervals overlap, so this ordering is not statistically resolved."
               if overlap else "Their bootstrap confidence intervals do not overlap.")
        )
    return "\n".join(lines)


def _subgroup_tables(cfg: ExperimentConfig) -> str:
    out = []
    for col, title in (("generator_id", "generation system (each scored against all bona fide recordings)"),
                       ("source_domain", "source domain")):
        f = artifacts.metrics_dir(cfg) / f"by_{col}.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, keep_default_na=False, na_values=[""])
        if "eer" not in s.columns:
            continue
        for d, g in s.groupby("dataset", sort=False):
            piv = g.pivot_table(index=col, columns="model", values="eer", aggfunc="first")
            piv = piv[[m for m in MODEL_NAMES if m in piv.columns]]
            piv.columns = [f"EER {DISPLAY[m]}" for m in piv.columns]
            piv = piv.map(lambda v: _fmt(v, True)).reset_index()
            out.append(f"**EER by {title} — {d}**\n\n" + _md_table(piv) + "\n")
    return "\n".join(out) if out else NYM


def _error_analysis(cfg: ExperimentConfig) -> str:
    preds = load_predictions(cfg)
    if not preds:
        return NYM
    lines = []
    datasets = sorted({d for _, d in preds})
    for d in datasets:
        models = [m for m in MODEL_NAMES if (m, d) in preds]
        frames = [preds[(m, d)].set_index("record_id")[["label", "predicted"]].rename(columns={"predicted": m}) for m in models]
        j = frames[0][["label"]].join([f[[m]] for f, m in zip(frames, models)], how="inner")
        wrong = pd.DataFrame({m: j[m] != j["label"] for m in models})
        lines.append(f"**{d}** (n = {len(j):,} common records)\n")
        for m in models:
            fp = int(((j[m] == 1) & (j["label"] == 0)).sum())
            fn = int(((j[m] == 0) & (j["label"] == 1)).sum())
            lines.append(f"- {DISPLAY[m]}: {fp:,} bona fide recordings flagged as spoof (false positives), "
                         f"{fn:,} spoofs missed (false negatives).")
        if len(models) > 1:
            all_wrong = int(wrong.all(axis=1).sum())
            any_wrong = int(wrong.any(axis=1).sum())
            lines.append(f"- Misclassified by every evaluated model: {all_wrong:,}; by at least one: {any_wrong:,}.")
        lines.append("")
    return "\n".join(lines)


def _figure(figs: Path, base: Path, name: str, caption: str) -> str:
    f = figs / name
    if f.exists():
        return f"![{caption}]({_rel(f, base)})\n\n*{caption}*\n"
    return f"*{caption}: {NYM}*\n"


def write_report(cfg: ExperimentConfig) -> Path:
    base = cfg.reports_dir
    base.mkdir(parents=True, exist_ok=True)
    figs = artifacts.figures_dir(cfg)
    path = base / "final_report.md"
    comp = artifacts.metrics_dir(cfg) / "comparison.csv"
    table = pd.read_csv(comp) if comp.exists() else None
    metas = _model_meta(cfg)
    splits = artifacts.splits_dir(cfg) / "split_summary.json"
    split_summary = artifacts.load_json(splits) if splits.exists() else None

    banner = {
        "full": "",
        "pilot": "> **PILOT RUN.** Models were trained and evaluated on stratified random subsets "
                 "(`data.max_*_samples`). These numbers check the pipeline end to end and are **not** final research results.\n\n",
        "smoke_test": "> **SYNTHETIC SOFTWARE TEST.** All numbers below come from a synthetic audio fixture and are **not** research results.\n\n",
    }[cfg.run_type]

    ds_stats = []
    for corpus in sorted({cfg.data.train["corpus"], cfg.data.dev["corpus"], *(t.corpus for t in cfg.test_sets())}):
        _, js = validated_paths(cfg, corpus)
        if js.exists():
            s = json.loads(js.read_text(encoding="utf-8"))
            ds_stats.append(f"**{corpus}** (validated manifest)\n\n" + summary_markdown(s) + "\n")
            integ = s["integrity"]
            if integ["errors"] or integ["warnings"]:
                ds_stats.append("Integrity findings: " + "; ".join(integ["errors"] + integ["warnings"]) + "\n")
        else:
            ds_stats.append(f"**{corpus}**: not yet validated ({NYM})\n")
    if split_summary:
        rows = [{"partition": k, "records": f"{v['n']:,}", "bona fide": f"{v['bonafide']:,}", "spoof": f"{v['spoof']:,}"}
                for k, v in split_summary["partitions"].items()]
        ds_stats.append("**Partitions used in this run**\n\n" + _md_table(pd.DataFrame(rows)) + "\n")
        for k, why in split_summary.get("skipped_tests", {}).items():
            ds_stats.append(f"- Test set `{k}` was not evaluated: {why}\n")

    cost_rows = []
    for m in MODEL_NAMES:
        meta = metas[m]
        if meta is None:
            cost_rows.append({"model": DISPLAY[m], "training (s)": NYM, "train feature extraction (s)": "–", "inference (ms/recording)": "–"})
            continue
        train_s = meta.get("training_seconds", meta.get("training_seconds_selected"))
        feat = (meta.get("feature_extraction_seconds") or {}).get("train")
        inf = "–"
        if table is not None and (table["model"] == m).any():
            inf = _fmt(table[table["model"] == m]["inference_ms_per_recording"].mean())
        cost_rows.append({"model": DISPLAY[m], "training (s)": _fmt(train_s),
                          "train feature extraction (s)": _fmt(feat) if feat is not None else "on the fly (in training time)",
                          "inference (ms/recording)": inf})
    env = next((meta["environment"] for meta in metas.values() if meta), None)
    env_line = (f"Hardware/software: {env['platform']}, Python {env['python']}, CUDA available: {env.get('cuda_available')}"
                f"{', GPU ' + env['gpu'] if env.get('gpu') else ''}; packages: "
                + ", ".join(f"{k} {v}" for k, v in env["packages"].items()) if env else "Hardware/software: " + NYM)

    w2v = metas["wav2vec_svm"].get("wav2vec") if metas["wav2vec_svm"] else None
    w2v_line = (f"Checkpoint `{w2v['model_name']}` (revision `{w2v.get('checkpoint_revision')}`), Transformers {w2v['transformers_version']}, "
                f"hidden layer {w2v['layer']}, {w2v['pooling']} pooling over valid frames, embedding size {w2v['hidden_size']}."
                if w2v else f"Extraction metadata: {NYM}")

    def sel(m):
        meta = metas[m]
        if meta is None:
            return NYM
        if m == "logmel_cnn":
            return (f"best epoch {meta['best_epoch']} of {meta['epochs_run']} (by dev {meta['selection_metric']}), "
                    f"{meta['n_parameters']:,} parameters, dev EER {_fmt(meta['dev_metrics']['eer'], True)}, threshold {_fmt(meta['threshold'])}.")
        return (f"selected C = {meta['selected']['C']}, gamma = {meta['selected']['gamma_spec']} ({_fmt(meta['selected']['gamma'])}), "
                f"feature dimension {meta['feature_dim']}, dev EER {_fmt(meta['dev_metrics']['eer'], True)}, threshold {_fmt(meta['threshold'])}.")

    a = cfg.audio
    test_desc = "; ".join(f"`{t.name}` = {t.corpus}/{t.split} ({t.condition.replace('_', ' ')})" for t in cfg.test_sets())
    text = f"""# Robust Detection of AI-Generated Speech: A Comparative Study of Classical Features, Deep Learning, and Pretrained Speech Representations

*Generated {time.strftime('%Y-%m-%d %H:%M')} from run `{cfg.run_name}` (run type: {cfg.run_type}) by `speechdeepfake report`. All numbers are read from saved result files.*

{banner}## 1. Abstract

This study compares three pattern-recognition pipelines for classifying a recording as bona fide (genuine human) speech or spoofed (AI-generated or manipulated) speech: MFCC statistics with an RBF-kernel SVM, a compact CNN trained on log-Mel spectrograms, and frozen wav2vec 2.0 embeddings with an RBF-kernel SVM. All models are fitted on `{cfg.data.train['corpus']}/{cfg.data.train['split']}`, tuned and calibrated on `{cfg.data.dev['corpus']}/{cfg.data.dev['split']}`, and evaluated on: {test_desc}. In the planned full study these are the official ASVspoof 2019 LA partitions, with the LA evaluation partition as the in-domain test and the CD-ADD test split (speech from five zero-shot TTS systems) as the external test. The research question itself is not new; the contribution is a reproducible, like-for-like comparison under both conditions.

Key observations: {('see Section 12.' if table is not None else NYM)}

## 2. Problem statement

AI-generated and voice-cloned speech can be used for impersonation, fraudulent telephone calls and fabricated recordings. We ask: *how effectively can different pattern-recognition approaches distinguish genuine from AI-generated speech, particularly for recordings from speech-generation systems not encountered during training?* Class 0 is bona fide, class 1 is spoof; spoof is the positive class.

## 3. Objectives

1. Implement three detectors that span handcrafted features, learned spectro-temporal features and transferred self-supervised representations.
2. Evaluate them with identical partitions, preprocessing and metrics.
3. Quantify the gap between in-domain and external-domain performance, overall and per generation system.
4. Compare training, feature-extraction and inference cost.

## 4. Dataset sources and licensing notes

- **ASVspoof 2019 Logical Access** (Edinburgh DataShare, doi:10.7488/ds/2555), Open Data Commons Attribution License (ODC-By). Derived from VCTK. Official train/dev/eval partitions with speaker-disjoint partitions; dev uses the same attack algorithms as train (A01–A06), eval contains A07–A19, of which A16 and A19 reuse the A04 and A06 algorithms. CM keys for all three partitions ship with the database (`ASVspoof2019_LA_cm_protocols`).
- **CD-ADD** (Li et al., 2024), public package `SpeechAntiSpoofingBenchmarks/CD-ADD` (CC BY 4.0): LibriTTS test-clean and TED-sourced utterances, each with a genuine recording and spoofs from up to five zero-shot TTS systems (OpenVoice, VALL-E, WhisperSpeech, YourTTS, Seamless). The package provides a single designated `test` split, used here only for external evaluation — never for fitting, scaling, tuning or threshold selection.
- **Pretraining overlap (wav2vec 2.0).** `facebook/wav2vec2-base` was pretrained on the 960 h LibriSpeech training subsets. LibriTTS test-clean is derived from LibriSpeech test-clean, which is not part of those 960 h by the published split definitions; so no *recording-level* overlap is expected, but the bona fide CD-ADD LibriTTS audio shares the audiobook source domain with the pretraining data. Recording-level overlap was not verified by audio fingerprinting (LibriSpeech is not downloaded by this project). Results are therefore also reported per source domain (LibriTTS vs. TED) where metadata allows.

## 5. Dataset statistics

{chr(10).join(ds_stats)}

## 6. Preprocessing

All pipelines share one function (`speechdeepfake.data.preprocessing.load_and_preprocess`): decode, down-mix to mono, resample to {a.sample_rate} Hz, reject empty/non-finite signals, {'trim silence (' + str(a.trim_silence_db) + ' dB), ' if a.trim_silence_db is not None else ''}normalisation `{a.normalization}`, and fix the duration to {a.max_duration} s — longer recordings are cropped deterministically at the `{a.eval_crop}`, shorter ones are padded by `{a.pad_mode}`. Training-time random cropping is {'enabled for CNN training only' if a.train_random_crop else 'disabled'}; evaluation preprocessing is always deterministic. Files are validated before training (existence, decodability, empty, NaN/Inf, too short, silent); failures are logged, not scored.

## 7. Methodology — Solution 1: MFCC + SVM

{cfg.mfcc.n_mfcc} MFCCs ({cfg.mfcc.n_mels} Mel bands, {cfg.mfcc.win_length}-sample window, {cfg.mfcc.hop_length}-sample hop){' plus deltas up to order ' + str(cfg.mfcc.delta_order) if cfg.mfcc.delta_order else ''} are summarised by their mean and standard deviation over time into a fixed-length vector, cached to disk. A scikit-learn Pipeline (StandardScaler fitted on training data only → RBF SVC, class_weight={cfg.svm.class_weight}) is fitted on the training partition for each (C, gamma) in the grid C ∈ {cfg.svm.C_grid}, gamma ∈ {cfg.svm.gamma_grid}, and selected by development-set {cfg.svm.selection_metric}. Scores are SVM decision values.

Selected model: {sel('mfcc_svm')}

## 8. Methodology — Solution 2: log-Mel spectrogram + CNN

{cfg.logmel.n_mels}-band log-Mel spectrograms ({cfg.logmel.n_fft}-point FFT, {cfg.logmel.win_length}-sample Hann window, {cfg.logmel.hop_length}-sample hop) are computed on the fly and standardised per band with statistics from {cfg.cnn.norm_stats_samples} training clips. The CNN has {len(cfg.cnn.channels)} blocks of Conv3×3–BatchNorm–ReLU–MaxPool (channels {cfg.cnn.channels}), global average pooling, dropout {cfg.cnn.dropout} and a 2-way linear head, trained with {'class-weighted ' if cfg.cnn.class_weighted_loss else ''}cross-entropy, AdamW (lr {cfg.cnn.learning_rate}, weight decay {cfg.cnn.weight_decay}), {cfg.cnn.lr_scheduler} LR schedule, up to {cfg.cnn.epochs} epochs with early stopping (patience {cfg.cnn.early_stopping_patience}) on development {cfg.cnn.selection_metric}. The score is the logit difference z_spoof − z_bonafide.

Selected checkpoint: {sel('logmel_cnn')}

## 9. Methodology — Solution 3: wav2vec 2.0 embeddings + SVM

The pretrained `{cfg.wav2vec.model_name}` model is used as a frozen feature extractor (not fine-tuned; it is not a deepfake detector). Each waveform is normalised by the checkpoint's feature extractor, batched (batch size {cfg.wav2vec.batch_size}), and the hidden states of layer {cfg.wav2vec.layer} are mean-pooled over valid (non-padded) frames into one embedding, cached with its record id. The same SVM pipeline and dev-set tuning as Solution 1 is then applied. {w2v_line}

Selected model: {sel('wav2vec_svm')}

## 10. Experimental design and split integrity

- Train: {cfg.data.train['corpus']}/{cfg.data.train['split']}; development (tuning, checkpoint selection, threshold): {cfg.data.dev['corpus']}/{cfg.data.dev['split']}; test sets: {test_desc}.
- Official partitions are used unchanged — never merged or re-split. {'Held-out generators removed from train/dev: ' + ', '.join(cfg.data.holdout_generators) + '.' if cfg.data.holdout_generators else 'No generator hold-out beyond the official protocol.'}
- Before each run the partitions are checked for duplicated ids, conflicting labels, and shared utterance/source-group ids, audio paths and byte-identical files (SHA-1) across partitions. Findings: {('errors: ' + '; '.join(split_summary['integrity']['errors']) if split_summary and split_summary['integrity']['errors'] else 'no blocking issues') if split_summary else NYM}{('; warnings: ' + '; '.join(split_summary['integrity']['warnings'])) if split_summary and split_summary['integrity']['warnings'] else ''}.
- All three models are compared on the records that every model scored.

## 11. Evaluation metrics

Spoof is the positive class and higher scores mean "more likely spoof". Accuracy, balanced accuracy, precision, recall and F1 are computed at one operating threshold per model, chosen on the development set by the `{cfg.evaluation.threshold_method}` rule and applied unchanged to every test set. ROC-AUC, average precision and the equal error rate (EER; ROC crossing of FPR and FNR, linearly interpolated between adjacent operating points) are threshold-free summaries of the test score distribution — the EER is *not* a claim that a threshold was tuned on test data. EER confidence intervals use a {cfg.evaluation.n_bootstrap}-resample percentile bootstrap; when source-group ids exist (e.g. a CD-ADD genuine utterance and its spoofs) whole groups are resampled. Inference time is wall-clock per recording including audio loading and feature computation.

## 12. Quantitative results

{_results_section(table, cfg)}

Observations:

{_observations(table)}

## 13. Graphical comparisons

{_figure(figs, base, 'partition_counts.png', 'Recordings per partition and class')}
{_figure(figs, base, 'class_distribution.png', 'Class distribution per partition')}
{_figure(figs, base, 'duration_distribution.png', 'Duration distribution per partition')}
{_figure(figs, base, 'waveforms.png', 'Example genuine and spoofed waveforms')}
{_figure(figs, base, 'mfcc_heatmap.png', 'MFCCs of representative recordings')}
{_figure(figs, base, 'logmel_examples.png', 'Genuine vs. spoof log-Mel spectrograms')}
{_figure(figs, base, 'pca_features.png', 'PCA of MFCC statistics and wav2vec 2.0 embeddings')}
{_figure(figs, base, 'cnn_learning_curves.png', 'CNN training and validation curves')}
{chr(10).join(_figure(figs, base, f'{kind}_{t.name}.png', f'{title} — {t.name}') for t in cfg.test_sets() for kind, title in (('roc', 'ROC curves'), ('pr', 'Precision-recall curves'), ('confusion', 'Confusion matrices')))}
{_figure(figs, base, 'eer_comparison.png', 'EER by model and test set')}
{_figure(figs, base, 'ba_f1_comparison.png', 'Balanced accuracy and F1 by model and test set')}
{_figure(figs, base, 'inference_time.png', 'Inference time per recording')}

## 14. Generalization analysis

{_figure(figs, base, 'generalization_eer.png', 'In-domain vs. external-domain EER')}
{chr(10).join(_figure(figs, base, f'eer_by_{c}_{t.name}.png', f'EER by {c.replace("_", " ")} — {t.name}') for t in cfg.test_sets() for c in ('generator', 'source_domain'))}

{_subgroup_tables(cfg)}

## 15. Computational-cost analysis

{_md_table(pd.DataFrame(cost_rows))}

SVM feature extraction times are measured when features are first computed (later runs reuse the cache). The CNN computes spectrograms inside its training/inference time. {env_line}

## 16. Error analysis

{_error_analysis(cfg)}

## 17. Limitations

- ASVspoof 2019 LA is known to contain dataset artefacts such as differences in leading/trailing silence between classes (Müller et al., 2021); detectors may exploit them. Silence trimming is available (`audio.trim_silence_db`) for an ablation.
- A fixed {a.max_duration} s analysis window discards the remainder of longer recordings.
- The CD-ADD package provides only a test split, so no target-domain data are used for adaptation; class balance differs from ASVspoof, which affects precision/F1 and the dev-selected threshold.
- Possible source-domain overlap between wav2vec 2.0 pretraining (LibriSpeech) and CD-ADD LibriTTS audio is documented but not verified at the recording level.
- Hyperparameter grids are small for computational reasons.
{('- ' + chr(10).join('Test set `' + k + '` unavailable: ' + v for k, v in split_summary['skipped_tests'].items())) if split_summary and split_summary.get('skipped_tests') else ''}
{'- This is a pilot/smoke run on subsets; numbers are not final.' if cfg.run_type != 'full' else ''}

## 18. Conclusion

{_observations(table) if table is not None else NYM + ' Conclusions will be written from measured results once all experiments have run.'}

## 19. References

{REFERENCES}
"""
    path.write_text(text, encoding="utf-8")
    return path
