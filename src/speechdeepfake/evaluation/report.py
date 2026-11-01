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

import numpy as np
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


def _size(path: Path) -> str:
    if not path.exists():
        return "–"
    b = path.stat().st_size
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.0f} kB"


def _selected_row(cfg: ExperimentConfig, m: str, meta: dict) -> Optional[pd.Series]:
    f = artifacts.model_dir(cfg, m) / "tuning.csv"
    if not f.exists():
        return None
    t = pd.read_csv(f)
    hit = t[(t["C"] == meta["selected"]["C"]) & (t["gamma_spec"] == meta["selected"]["gamma_spec"])]
    return hit.iloc[0] if len(hit) else None


REPRESENTATION = {
    "mfcc_svm": ("MFCC mean + std (handcrafted)", "RBF SVM"),
    "logmel_cnn": ("log-Mel spectrogram (learned by CNN)", "4-block 2-D CNN"),
    "wav2vec_svm": ("wav2vec 2.0 base, mean-pooled (self-supervised, frozen)", "RBF SVM"),
}


def _summary_table(cfg: ExperimentConfig, table: Optional[pd.DataFrame], metas: dict) -> str:
    """One row per model: what it is, how big it is, dev/test EER and cost."""
    tests = cfg.test_sets()
    rows = []
    for m in MODEL_NAMES:
        meta = metas[m]
        rep, clf = REPRESENTATION[m]
        row = {"model": DISPLAY[m], "representation": rep, "classifier": clf}
        if meta is None:
            row["dev EER"] = NYM
            rows.append(row)
            continue
        d = artifacts.model_dir(cfg, m)
        if m == "logmel_cnn":
            row["input / feature size"] = f"{cfg.logmel.n_mels} × {1 + int(cfg.audio.max_duration * cfg.audio.sample_rate) // cfg.logmel.hop_length}"
            row["trained parameters"] = f"{meta['n_parameters']:,} weights"
            row["model file"] = _size(d / "best.pt")
        else:
            sel = _selected_row(cfg, m, meta)
            row["input / feature size"] = f"{meta['feature_dim']}-D"
            row["trained parameters"] = f"{int(sel['n_support']):,} support vectors" if sel is not None else "–"
            row["model file"] = _size(d / "pipeline.joblib") + (" (+ ~95 M-param frozen backbone)" if m == "wav2vec_svm" else "")
        row["dev EER"] = _fmt(meta["dev_metrics"]["eer"], True)
        for t in tests:
            r = None if table is None else table[(table["model"] == m) & (table["dataset"] == t.name)]
            row[f"EER {t.name}"] = _fmt(r.iloc[0]["eer"], True) if r is not None and len(r) else NYM
        if table is not None and (table["model"] == m).any():
            r = table[table["model"] == m].iloc[0]
            tr = r["training_seconds"]
            fe = r["train_feature_seconds"]
            row["training time (s)"] = _fmt(tr) + (f" + {_fmt(fe)} features" if pd.notna(fe) else "")
            row["inference (ms/rec.)"] = _fmt(table[table["model"] == m]["inference_ms_per_recording"].mean())
        rows.append(row)
    return _md_table(pd.DataFrame(rows).fillna("–"))


def _dev_table(metas: dict) -> str:
    rows = []
    for m in MODEL_NAMES:
        meta = metas[m]
        if meta is None:
            rows.append({"model": DISPLAY[m], "n": NYM})
            continue
        d = meta["dev_metrics"]
        rows.append({"model": DISPLAY[m], "n (bona fide / spoof)": f"{d['n']:,} ({d['n_bonafide']:,} / {d['n_spoof']:,})",
                     "threshold": _fmt(meta["threshold"]), "accuracy": _fmt(d["accuracy"], True),
                     "balanced acc.": _fmt(d["balanced_accuracy"], True), "precision": _fmt(d["precision"], True),
                     "recall": _fmt(d["recall"], True), "F1": _fmt(d["f1"], True), "ROC-AUC": _fmt(d["roc_auc"]),
                     "EER": _fmt(d["eer"], True)})
    return _md_table(pd.DataFrame(rows).fillna(""))


def _ranking_table(table: Optional[pd.DataFrame]) -> str:
    """Rank of every model per metric and test set (1 = best)."""
    if table is None or table.empty:
        return NYM
    metrics = [("eer", True, "EER"), ("roc_auc", False, "ROC-AUC"), ("balanced_accuracy", False, "balanced acc."),
               ("f1", False, "F1"), ("inference_ms_per_recording", True, "inference time (ms)")]
    rows = []
    for d, sub in table.groupby("dataset", sort=False):
        for col, asc, name in metrics:
            ranks = sub.set_index("model")[col].rank(ascending=asc, method="min")
            rows.append({"test set": d, "metric": name,
                         **{DISPLAY[m]: f"{int(ranks[m])} ({_fmt(sub.set_index('model').loc[m, col], col != 'inference_ms_per_recording' and col != 'roc_auc')})"
                            for m in MODEL_NAMES if m in ranks.index}})
    return _md_table(pd.DataFrame(rows))


def _robustness_section(cfg: ExperimentConfig, figs: Path, base: Path) -> str:
    from speechdeepfake.evaluation.robustness import CONDITIONS

    f = artifacts.metrics_dir(cfg) / "robustness.csv"
    if not f.exists():
        return (f"{NYM} Run `speechdeepfake robustness --config <config>` to score the trained models on "
                "MP3-compressed, narrowband and noisy copies of the test sets.")
    rob = pd.read_csv(f)
    info_f = artifacts.metrics_dir(cfg) / "robustness_conditions.json"
    info = artifacts.load_json(info_f) if info_f.exists() else {}
    cond_rows = []
    for c in dict.fromkeys(rob["audio_condition"]):
        if c == "clean":
            continue
        kbps = [s.get("measured_kbps") for s in info.get(c, {}).get("per_test_set", {}).values() if s.get("measured_kbps")]
        cond_rows.append({"condition": f"`{c}`", "description": CONDITIONS[c]["description"],
                          "measured bit rate": f"{np.mean(kbps):.1f} kbps" if kbps else "–"})
    out = ["The trained models are **not** retrained or re-thresholded: each degraded copy of a test recording "
           "(shared 4 s preprocessing first, then the degradation, stored in the degraded format and decoded again "
           "by the normal pipeline) is scored with the dev-selected threshold. All conditions are compared on the "
           "same records.\n", _md_table(pd.DataFrame(cond_rows)) + "\n"]
    for d, s in rob.groupby("dataset", sort=False):
        piv = s.pivot_table(index="audio_condition_label", columns="display_name", values="eer", aggfunc="first", sort=False)
        dl = s.pivot_table(index="audio_condition_label", columns="display_name", values="delta_eer_vs_clean", aggfunc="first", sort=False)
        ba = s.pivot_table(index="audio_condition_label", columns="display_name", values="balanced_accuracy", aggfunc="first", sort=False)
        cols = [DISPLAY[m] for m in MODEL_NAMES if DISPLAY[m] in piv.columns]
        rows = []
        for cond in piv.index:
            row = {"audio condition": cond}
            for c in cols:
                delta = "" if dl.loc[cond, c] == 0 else f" ({'+' if dl.loc[cond, c] > 0 else '−'}{abs(100 * dl.loc[cond, c]):.1f} pp)"
                row[f"EER {c}"] = _fmt(piv.loc[cond, c], True) + delta
            for c in cols:
                row[f"BA {c}"] = _fmt(ba.loc[cond, c], True)
            rows.append(row)
        out.append(f"**{d}** (n = {int(s['n'].iloc[0]):,}; EER change vs. clean in percentage points; BA = balanced accuracy at the dev threshold)\n")
        out.append(_md_table(pd.DataFrame(rows)) + "\n")
        worst = s[s["audio_condition"] != "clean"].sort_values("delta_eer_vs_clean", ascending=False).iloc[0]
        best_clean = s[s["audio_condition"] == "clean"].sort_values("eer").iloc[0]
        robust = s[s["audio_condition"] != "clean"].groupby("display_name")["delta_eer_vs_clean"].mean().sort_values()
        out.append(f"- Largest degradation on {d}: {worst['display_name']} under *{worst['audio_condition_label']}* "
                   f"(EER {_fmt(worst['eer'] - worst['delta_eer_vs_clean'], True)} → {_fmt(worst['eer'], True)}).")
        out.append(f"- Mean EER change across the degradations: " + "; ".join(f"{k} {'+' if v >= 0 else '−'}{abs(100 * v):.1f} pp" for k, v in robust.items())
                   + f". Best clean-audio model: {best_clean['display_name']} ({_fmt(best_clean['eer'], True)}).\n")
    out.append("How to read these tables: the EER measures whether the scores still *separate* the classes, balanced "
               "accuracy measures whether the *dev-selected threshold* still sits in the right place. A balanced accuracy "
               "near 50 % with a much lower EER means the degradation shifted all scores to one side of the threshold "
               "(a calibration failure that re-thresholding could repair), not that the information was lost. On a test set "
               "where the clean EER is already near chance, apparent improvements under degradation are not evidence of a "
               "better detector.\n")
    out.append(_figure(figs, base, "robustness_examples.png", "One test recording under each audio condition"))
    out.append(_figure(figs, base, "robustness_eer.png", "EER under audio degradations"))
    out.append(_figure(figs, base, "robustness_balanced_accuracy.png", "Balanced accuracy under audio degradations"))
    return "\n".join(out)


def _appendix(cfg: ExperimentConfig) -> str:
    out = []
    for m in ("mfcc_svm", "wav2vec_svm"):
        f = artifacts.model_dir(cfg, m) / "tuning.csv"
        if f.exists():
            t = pd.read_csv(f)
            t = pd.DataFrame({"C": t["C"], "gamma": t["gamma_spec"], "support vectors": t["n_support"],
                              "fit (s)": t["fit_seconds"].map(lambda v: f"{v:.3f}"),
                              "dev EER": t["dev_eer"].map(lambda v: _fmt(v, True)), "dev ROC-AUC": t["dev_roc_auc"].map(_fmt),
                              "dev BA": t["dev_balanced_accuracy"].map(lambda v: _fmt(v, True))})
            out.append(f"**{DISPLAY[m]} — full hyperparameter grid (development set)**\n\n{_md_table(t)}\n")
    f = artifacts.model_dir(cfg, "logmel_cnn") / "history.csv"
    if f.exists():
        h = pd.read_csv(f)
        h = pd.DataFrame({"epoch": h["epoch"], "train loss": h["train_loss"].map(lambda v: f"{v:.4f}"),
                          "dev loss": h["dev_loss"].map(lambda v: f"{v:.4f}"), "dev EER": h["dev_eer"].map(lambda v: _fmt(v, True)),
                          "dev ROC-AUC": h["dev_roc_auc"].map(_fmt), "learning rate": h["lr"].map(lambda v: f"{v:g}"),
                          "epoch time (s)": h["epoch_seconds"].map(lambda v: f"{v:.1f}"),
                          "selected": h["is_best"].map(lambda v: "✓" if str(v) == "True" else "")})
        out.append(f"**log-Mel + CNN — training history** (✓ = new best dev EER; the last ✓ is the saved checkpoint)\n\n{_md_table(h)}\n")
    return "\n".join(out) if out else NYM


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
    fig = lambda name, caption: _figure(figs, base, name, caption)  # noqa: E731
    per_test = lambda kind, title: "\n".join(fig(f"{kind}_{t.name}.png", f"{title} — {t.name}") for t in cfg.test_sets())  # noqa: E731
    text = f"""# Robust Detection of AI-Generated Speech: A Comparative Study of Classical Features, Deep Learning, and Pretrained Speech Representations

*Generated {time.strftime('%Y-%m-%d %H:%M')} from run `{cfg.run_name}` (run type: {cfg.run_type}) by `speechdeepfake report`. All numbers are read from saved result files.*

{banner}## Contents

1. Abstract · 2. Results at a glance · 3. Problem statement · 4. Objectives · 5. Datasets · 6. Dataset statistics · 7. Preprocessing · 8–10. Methodology of Solutions 1–3 · 11. Experimental design · 12. Evaluation metrics · 13. Quantitative results · 14. Graphical comparisons · 15. Generalization analysis · 16. Robustness to audio degradations · 17. Computational cost · 18. Error analysis · 19. Comparison of the approaches · 20. Limitations · 21. Conclusion · 22. References · Appendix A

## 1. Abstract

This study compares three pattern-recognition pipelines for classifying a recording as bona fide (genuine human) speech or spoofed (AI-generated or manipulated) speech: MFCC statistics with an RBF-kernel SVM, a compact CNN trained on log-Mel spectrograms, and frozen wav2vec 2.0 embeddings with an RBF-kernel SVM. All models are fitted on `{cfg.data.train['corpus']}/{cfg.data.train['split']}`, tuned and calibrated on `{cfg.data.dev['corpus']}/{cfg.data.dev['split']}`, and evaluated on: {test_desc}. In the planned full study these are the official ASVspoof 2019 LA partitions, with the LA evaluation partition as the in-domain test and the CD-ADD test split (speech from five zero-shot TTS systems) as the external test. The trained models are additionally re-scored on MP3-compressed, narrowband and noisy copies of the test recordings. The research question itself is not new; the contribution is a reproducible, like-for-like comparison under all of these conditions.

Key observations: {('see Sections 2, 13 and 21.' if table is not None else NYM)}

## 2. Results at a glance

{fig('pipeline_overview.png', 'The three pipelines and the shared protocol')}
**Model summary** (EER = equal error rate, lower is better; training time on the configured hardware; inference time includes audio loading and feature extraction)

{_summary_table(cfg, table, metas)}

{fig('eer_comparison.png', 'EER by model and test set')}
{fig('generalization_eer.png', 'In-domain vs. external-domain EER')}

## 3. Problem statement

AI-generated and voice-cloned speech can be used for impersonation, fraudulent telephone calls and fabricated recordings. We ask: *how effectively can different pattern-recognition approaches distinguish genuine from AI-generated speech, particularly for recordings from speech-generation systems not encountered during training?* Class 0 is bona fide, class 1 is spoof; spoof is the positive class.

## 4. Objectives

1. Implement three detectors that span handcrafted features, learned spectro-temporal features and transferred self-supervised representations.
2. Evaluate them with identical partitions, preprocessing and metrics.
3. Quantify the gap between in-domain and external-domain performance, overall and per generation system.
4. Measure how audio compression, band limitation and noise change detection performance.
5. Compare training, feature-extraction and inference cost.

## 5. Dataset sources and licensing notes

- **ASVspoof 2019 Logical Access** (Edinburgh DataShare, doi:10.7488/ds/2555), Open Data Commons Attribution License (ODC-By). Derived from VCTK. Official train/dev/eval partitions with speaker-disjoint partitions; dev uses the same attack algorithms as train (A01–A06), eval contains A07–A19, of which A16 and A19 reuse the A04 and A06 algorithms. CM keys for all three partitions ship with the database (`ASVspoof2019_LA_cm_protocols`).
- **CD-ADD** (Li et al., 2024), public package `SpeechAntiSpoofingBenchmarks/CD-ADD` (CC BY 4.0): LibriTTS test-clean and TED-sourced utterances, each with a genuine recording and spoofs from up to five zero-shot TTS systems (OpenVoice, VALL-E, WhisperSpeech, YourTTS, Seamless). The package provides a single designated `test` split, used here only for external evaluation — never for fitting, scaling, tuning or threshold selection.
- **Pretraining overlap (wav2vec 2.0).** `facebook/wav2vec2-base` was pretrained on the 960 h LibriSpeech training subsets. LibriTTS test-clean is derived from LibriSpeech test-clean, which is not part of those 960 h by the published split definitions; so no *recording-level* overlap is expected, but the bona fide CD-ADD LibriTTS audio shares the audiobook source domain with the pretraining data. Recording-level overlap was not verified by audio fingerprinting (LibriSpeech is not downloaded by this project). Results are therefore also reported per source domain (LibriTTS vs. TED) where metadata allows.

## 6. Dataset statistics

{chr(10).join(ds_stats)}
{fig('partition_counts.png', 'Recordings per partition and class')}
{fig('class_distribution.png', 'Class distribution per partition')}
{fig('duration_distribution.png', 'Duration distribution per partition')}

## 7. Preprocessing

All pipelines share one function (`speechdeepfake.data.preprocessing.load_and_preprocess`): decode, down-mix to mono, resample to {a.sample_rate} Hz, reject empty/non-finite signals, {'trim silence (' + str(a.trim_silence_db) + ' dB), ' if a.trim_silence_db is not None else ''}normalisation `{a.normalization}`, and fix the duration to {a.max_duration} s — longer recordings are cropped deterministically at the `{a.eval_crop}`, shorter ones are padded by `{a.pad_mode}`. Training-time random cropping is {'enabled for CNN training only' if a.train_random_crop else 'disabled'}; evaluation preprocessing is always deterministic. Files are validated before training (existence, decodability, empty, NaN/Inf, too short, silent); failures are logged, not scored.

{fig('waveforms.png', 'Example genuine and spoofed waveforms')}

## 8. Methodology — Solution 1: MFCC + SVM

{cfg.mfcc.n_mfcc} MFCCs ({cfg.mfcc.n_mels} Mel bands, {cfg.mfcc.win_length}-sample window, {cfg.mfcc.hop_length}-sample hop){' plus deltas up to order ' + str(cfg.mfcc.delta_order) if cfg.mfcc.delta_order else ''} are summarised by their mean and standard deviation over time into a fixed-length vector, cached to disk. A scikit-learn Pipeline (StandardScaler fitted on training data only → RBF SVC, class_weight={cfg.svm.class_weight}) is fitted on the training partition for each (C, gamma) in the grid C ∈ {cfg.svm.C_grid}, gamma ∈ {cfg.svm.gamma_grid}, and selected by development-set {cfg.svm.selection_metric}. Scores are SVM decision values.

Selected model: {sel('mfcc_svm')}

{fig('mfcc_heatmap.png', 'MFCCs of representative recordings')}
{fig('svm_tuning.png', 'Development-set EER over the SVM hyperparameter grid (Solutions 1 and 3)')}

## 9. Methodology — Solution 2: log-Mel spectrogram + CNN

{cfg.logmel.n_mels}-band log-Mel spectrograms ({cfg.logmel.n_fft}-point FFT, {cfg.logmel.win_length}-sample Hann window, {cfg.logmel.hop_length}-sample hop) are computed on the fly and standardised per band with statistics from {cfg.cnn.norm_stats_samples} training clips. The CNN has {len(cfg.cnn.channels)} blocks of Conv3×3–BatchNorm–ReLU–MaxPool (channels {cfg.cnn.channels}), global average pooling, dropout {cfg.cnn.dropout} and a 2-way linear head, trained with {'class-weighted ' if cfg.cnn.class_weighted_loss else ''}cross-entropy, AdamW (lr {cfg.cnn.learning_rate}, weight decay {cfg.cnn.weight_decay}), {cfg.cnn.lr_scheduler} LR schedule, up to {cfg.cnn.epochs} epochs with early stopping (patience {cfg.cnn.early_stopping_patience}) on development {cfg.cnn.selection_metric}. The score is the logit difference z_spoof − z_bonafide.

Selected checkpoint: {sel('logmel_cnn')}

{fig('logmel_examples.png', 'Genuine vs. spoof log-Mel spectrograms')}
{fig('cnn_learning_curves.png', 'CNN training and validation curves')}

## 10. Methodology — Solution 3: wav2vec 2.0 embeddings + SVM

The pretrained `{cfg.wav2vec.model_name}` model is used as a frozen feature extractor (not fine-tuned; it is not a deepfake detector). Each waveform is normalised by the checkpoint's feature extractor, batched (batch size {cfg.wav2vec.batch_size}), and the hidden states of layer {cfg.wav2vec.layer} are mean-pooled over valid (non-padded) frames into one embedding, cached with its record id. The same SVM pipeline and dev-set tuning as Solution 1 is then applied. {w2v_line}

Selected model: {sel('wav2vec_svm')}

{fig('pca_features.png', 'PCA of MFCC statistics and wav2vec 2.0 embeddings')}

## 11. Experimental design and split integrity

- Train: {cfg.data.train['corpus']}/{cfg.data.train['split']}; development (tuning, checkpoint selection, threshold): {cfg.data.dev['corpus']}/{cfg.data.dev['split']}; test sets: {test_desc}.
- Official partitions are used unchanged — never merged or re-split. {'Held-out generators removed from train/dev: ' + ', '.join(cfg.data.holdout_generators) + '.' if cfg.data.holdout_generators else 'No generator hold-out beyond the official protocol.'}
- Before each run the partitions are checked for duplicated ids, conflicting labels, and shared utterance/source-group ids, audio paths and byte-identical files (SHA-1) across partitions. Findings: {('errors: ' + '; '.join(split_summary['integrity']['errors']) if split_summary and split_summary['integrity']['errors'] else 'no blocking issues') if split_summary else NYM}{('; warnings: ' + '; '.join(split_summary['integrity']['warnings'])) if split_summary and split_summary['integrity']['warnings'] else ''}.
- All three models are compared on the records that every model scored.

## 12. Evaluation metrics

Spoof is the positive class and higher scores mean "more likely spoof". Accuracy, balanced accuracy, precision, recall and F1 are computed at one operating threshold per model, chosen on the development set by the `{cfg.evaluation.threshold_method}` rule and applied unchanged to every test set. ROC-AUC, average precision and the equal error rate (EER; ROC crossing of FPR and FNR, linearly interpolated between adjacent operating points) are threshold-free summaries of the test score distribution — the EER is *not* a claim that a threshold was tuned on test data. EER confidence intervals use a {cfg.evaluation.n_bootstrap}-resample percentile bootstrap; when source-group ids exist (e.g. a CD-ADD genuine utterance and its spoofs) whole groups are resampled. Inference time is wall-clock per recording including audio loading and feature computation.

| metric | definition | why it is reported |
|---|---|---|
| EER | error rate where FPR = FNR | standard ASVspoof summary; threshold-free |
| ROC-AUC | P(score of random spoof > score of random bona fide) | threshold-free ranking quality; 0.5 = chance |
| average precision | area under the precision-recall curve | sensitive to class imbalance (CD-ADD is ~82 % spoof) |
| accuracy | (TP + TN) / n | intuitive, but misleading under imbalance |
| balanced accuracy | (TPR + TNR) / 2 | accuracy corrected for imbalance; 50 % = chance |
| precision / recall / F1 | spoof-class precision, recall and their harmonic mean | cost of false alarms vs. missed spoofs |
| FPR / FNR | bona fide flagged as spoof / spoof missed | the two error types behind the EER |

## 13. Quantitative results

### 13.1 Development set (used for tuning and threshold selection)

{_dev_table(metas)}

These numbers are optimistic by construction (the dev set selected the hyperparameters, checkpoint and threshold); the test sets below are the unbiased estimates.

### 13.2 Test sets

{_results_section(table, cfg)}

Observations:

{_observations(table)}

### 13.3 Ranking per metric

Rank 1 = best (lowest EER / inference time, highest otherwise); the measured value is in brackets. Ranks on the external set are not meaningful where all models are near chance.

{_ranking_table(table)}

## 14. Graphical comparisons

### 14.1 Threshold-free curves

{per_test('roc', 'ROC curves')}
{per_test('det', 'DET curves')}
{per_test('pr', 'Precision-recall curves')}

### 14.2 Score distributions and operating point

{per_test('scores', 'Score distributions by class')}
{per_test('confusion', 'Confusion matrices')}

### 14.3 Summary bars

{fig('ba_f1_comparison.png', 'Balanced accuracy and F1 by model and test set')}

## 15. Generalization analysis

{fig('generalization_eer.png', 'In-domain vs. external-domain EER')}
{chr(10).join(fig(f'eer_heatmap_generator_{t.name}.png', f'EER per generation system — {t.name}') for t in cfg.test_sets())}
{chr(10).join(fig(f'eer_by_{c}_{t.name}.png', f'EER by {c.replace("_", " ")} — {t.name}') for t in cfg.test_sets() for c in ('generator', 'source_domain'))}

{_subgroup_tables(cfg)}

## 16. Robustness to audio degradations

{_robustness_section(cfg, figs, base)}

## 17. Computational-cost analysis

{_md_table(pd.DataFrame(cost_rows))}

SVM feature extraction times are measured when features are first computed (later runs reuse the cache). The CNN computes spectrograms inside its training/inference time. The wav2vec 2.0 SVM also needs the ~95 M-parameter frozen backbone (≈ 360 MB) at inference time. {env_line}

{fig('inference_time.png', 'Inference time per recording')}
{fig('tradeoff_eer_vs_time.png', 'Detection error vs. computational cost')}

## 18. Error analysis

{_error_analysis(cfg)}

## 19. Comparison of the approaches

| criterion | MFCC + SVM | log-Mel + CNN | wav2vec 2.0 + SVM |
|---|---|---|---|
| representation | handcrafted cepstral statistics (time order discarded) | learned from the spectrogram (time-frequency structure kept) | self-supervised speech representation, frozen |
| what is trained | SVM only | whole network | SVM only |
| hardware needed | CPU | CPU feasible, GPU helps | GPU helps (backbone forward pass dominates cost) |
| data needed | small | largest (risk of overfitting) | small (backbone already pretrained) |
| interpretability | high (coefficient statistics) | medium (saliency maps possible) | low |
| main risk | limited capacity for subtle artefacts | learns dataset-specific cues | domain / pretraining bias, highest cost |

Measured results for these criteria are in Sections 2, 13, 16 and 17.

## 20. Limitations

- ASVspoof 2019 LA is known to contain dataset artefacts such as differences in leading/trailing silence between classes (Müller et al., 2021); detectors may exploit them. Silence trimming is available (`audio.trim_silence_db`) for an ablation.
- A fixed {a.max_duration} s analysis window discards the remainder of longer recordings.
- The CD-ADD package provides only a test split, so no target-domain data are used for adaptation; class balance differs from ASVspoof, which affects precision/F1 and the dev-selected threshold.
- Possible source-domain overlap between wav2vec 2.0 pretraining (LibriSpeech) and CD-ADD LibriTTS audio is documented but not verified at the recording level.
- Degradations are applied only at test time; no model was trained with augmentation, so the robustness numbers describe off-the-shelf behaviour, not the best achievable robustness.
- Hyperparameter grids are small for computational reasons.
{('- ' + chr(10).join('Test set `' + k + '` unavailable: ' + v for k, v in split_summary['skipped_tests'].items())) if split_summary and split_summary.get('skipped_tests') else ''}
{'- This is a pilot/smoke run on subsets; numbers are not final.' if cfg.run_type != 'full' else ''}

## 21. Conclusion

{_observations(table) if table is not None else NYM + ' Conclusions will be written from measured results once all experiments have run.'}

## 22. References

{REFERENCES}

## Appendix A. Hyperparameter search and training history

{_appendix(cfg)}
"""
    path.write_text(text, encoding="utf-8")
    return path
