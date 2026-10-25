# Robust Detection of AI-Generated Speech

### A comparative study of classical features, deep learning, and pretrained speech representations

This project compares three approaches for distinguishing genuine human speech (class 0, *bona fide*) from AI-generated or manipulated speech (class 1, *spoof*). It focuses on whether systems trained on known examples generalise to recordings from unfamiliar speech-generation systems and different audio conditions. The research question is not new; the contribution is a reproducible, like-for-like comparison under in-domain and cross-domain evaluation.

| Solution | Audio representation | Classifier | Purpose |
| --- | --- | --- | --- |
| 1 | MFCC mean/std statistics | RBF-kernel SVM | Classical, computationally light baseline |
| 2 | Log-Mel spectrograms | Compact 2-D CNN | Learn time-frequency patterns from audio |
| 3 | Frozen wav2vec 2.0 embeddings (mean-pooled) | RBF-kernel SVM | Evaluate transfer-learned speech representations |

**Status (2026-10-25).** All three pipelines, the data layer, evaluation, figures and report generator are implemented and tested on a synthetic software-test fixture. **No model has been trained on real data yet** — the datasets have not been downloaded. No research results exist; `reports/final_report.md` reports every metric as "Not yet measured."

## Experimental design

- **Train:** official ASVspoof 2019 LA training partition. **Validation:** official LA development partition — used for SVM hyperparameters, CNN checkpoint selection/early stopping, and the operating threshold.
- **In-domain test:** official LA evaluation partition (keys ship with the database).
- **External test:** CD-ADD `test` split (LibriTTS test-clean + TED; OpenVoice, VALL-E, WhisperSpeech, YourTTS, Seamless). Never used for fitting, scaling, tuning or thresholds.
- Same preprocessing (16 kHz mono, fixed 4 s window, deterministic evaluation crop) and the same evaluated records for all three models.
- Metrics: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision, EER (with cluster-bootstrap CIs), confusion matrices, per-generator and per-source-domain EER, training/feature/inference time.

Details: [docs/methodology.md](docs/methodology.md) · datasets: [docs/dataset_setup.md](docs/dataset_setup.md) · team roles: [docs/team_work_split.md](docs/team_work_split.md).

## Installation

Python 3.10–3.12 in a virtual environment.

```bash
# bash / Git Bash
python -m venv .venv
source .venv/Scripts/activate        # Windows Git Bash; on Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
```

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

**GPU (recommended for wav2vec 2.0 extraction and CNN training).** `pip` installs the CPU build of PyTorch by default. If you have an NVIDIA GPU, install a CUDA build first, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu126`, then check with `speechdeepfake inspect` (`"cuda_available": true`). Everything also runs on CPU, only slower: on the development laptop's CPU wav2vec 2.0 extraction took roughly 0.25 s per 4 s clip, i.e. on the order of 10 hours for all ~140k recordings.

## Commands

All commands accept `--config` (default `configs/default.yaml`) and `--help`. The syntax is identical in bash and PowerShell.

| Step | Command |
|---|---|
| Environment and hardware | `speechdeepfake inspect --config configs/default.yaml` |
| Which datasets are present | `speechdeepfake check-data --config configs/default.yaml` |
| Download (explicit only) | `speechdeepfake download --dataset asvspoof2019_la` · `speechdeepfake download --dataset cdadd` |
| Build manifests | `speechdeepfake prepare-data --config configs/default.yaml` |
| Validate audio, labels, leakage | `speechdeepfake validate-data --config configs/default.yaml` |
| Software smoke test (synthetic) | `speechdeepfake smoke-test --config configs/quick_test.yaml` |
| Cache features | `speechdeepfake extract-features --model all --config configs/default.yaml` |
| Train one model | `speechdeepfake train --model mfcc_svm` · `--model logmel_cnn` · `--model wav2vec_svm` |
| Train all three | `speechdeepfake train --model all --config configs/default.yaml` |
| Resume interrupted CNN | `speechdeepfake train --model logmel_cnn --resume` |
| Evaluate (same test records) | `speechdeepfake evaluate-all --config configs/default.yaml` |
| Figures only | `speechdeepfake plots --config configs/default.yaml` |
| Figures + Markdown report | `speechdeepfake report --config configs/default.yaml` |
| Everything after validation | `speechdeepfake run-all --config configs/default.yaml` |
| Whole pipeline (script) | `python scripts/run_all_experiments.py --config configs/default.yaml` |

**Quick demo on small subsets (~300 MB, under an hour on a CPU laptop):**

```bash
speechdeepfake download-subset --config configs/demo.yaml   # ~1.8k ASVspoof clips + ~800 CD-ADD clips via HTTP range requests
speechdeepfake prepare-data    --config configs/demo.yaml
speechdeepfake validate-data   --config configs/demo.yaml
speechdeepfake run-all         --config configs/demo.yaml   # report: reports/final_report.md (labelled PILOT)
```

`download-subset` never downloads the full archives: it reads only the needed parts of `LA.zip` (protocols plus a stratified sample per split and attack) and a few 100-row blocks of each CD-ADD parquet shard. The exact selection is recorded in `data/raw/*/SUBSET.json`.

Recommended order for the full study:

```bash
speechdeepfake download --dataset asvspoof2019_la
speechdeepfake download --dataset cdadd
speechdeepfake prepare-data  --config configs/default.yaml
speechdeepfake validate-data --config configs/default.yaml
# 1) pilot on subsets (minutes-to-an-hour; outputs labelled PILOT, written to reports/pilot/)
speechdeepfake run-all --config configs/quick_test.yaml
# 2) full experiment (hours; final figures and report in reports/)
speechdeepfake run-all --config configs/default.yaml
```

To keep multi-GB data out of OneDrive, set the data directory first — PowerShell: `$env:SPEECHDEEPFAKE_DATA_DIR = "D:\speechdeepfake-data"`; bash: `export SPEECHDEEPFAKE_DATA_DIR=/d/speechdeepfake-data`.

Tests: `pytest` (all) or `pytest -m "not slow"` (skip the end-to-end test).

## Outputs

| What | Where |
|---|---|
| Manifests, validation summaries, failures | `data/manifests/`, `data/manifests/validated/` |
| Cached MFCC / wav2vec 2.0 features (resumable chunks) | `data/processed/<feature>/<config-hash>/` |
| SVM pipelines, CNN checkpoints, metadata, tuning tables, dev predictions, histories | `models/<run>/<model>/` |
| Exact partitions used + split summary | `results/<run>/splits/` |
| Per-recording predictions | `results/<run>/predictions/<model>__<test>.csv` |
| Metrics (JSON per model/test, `comparison.csv/json`, `by_generator_id.csv`, `by_source_domain.csv`) | `results/<run>/metrics/` |
| Logs, `errors.csv`, config snapshots | `results/<run>/logs/`, `results/<run>/config_snapshots/` |
| Figures and tables | `reports/figures/`, `reports/tables/` (pilot: `reports/pilot/…`) |
| Report | `reports/final_report.md` (pilot: `reports/pilot/final_report.md`) |
| Smoke-test outputs (synthetic, not results) | `results/smoke_test/` |

Data, caches, models, results and generated figures are git-ignored.

## Repository layout

```
configs/            default.yaml (full run), quick_test.yaml (pilot + smoke-test settings)
src/speechdeepfake/ cli.py, pipeline.py, config.py, artifacts.py
  data/             loaders, manifests, validation, splits, preprocessing, synthetic (test fixture only)
  features/         mfcc, logmel, wav2vec_embeddings, cache
  models/           svm, mfcc_svm, logmel_cnn, wav2vec_svm
  training/         train_svm, train_cnn
  evaluation/       metrics, thresholds, predict, plots, figures, report
  utils/            logging, reproducibility, timing
scripts/            create_manifest.py, validate_data.py, run_all_experiments.py
tests/              test_data, test_features, test_metrics, test_smoke
docs/               dataset_setup, methodology, team_work_split
```

Planning documents: [case study](CaseStudy_ProblemDefinition.md) · [Solution 1](Solution1_MFCCFeatures+SVM.md) · [Solution 2](Solution2_Log-MelSpectrogram+CNN.md) · [Solution 3](Solution3_wav2vec2.0Embeddings+SVM.md).

## Tested environment

Windows 11, Python 3.12.9, numpy 1.26.4, scipy 1.15.1, pandas 2.3.3, scikit-learn 1.7.2, librosa 0.11.0, soundfile 0.13.1, torch 2.8.0 (CPU build), transformers 4.41.2, pyarrow 23.0.1, matplotlib 3.10.7, PyYAML 6.0.2, pytest 8.4.2. Not yet tested: CUDA execution, Linux/macOS.

## Team

Implementation is group work by four students; see [docs/team_work_split.md](docs/team_work_split.md) for module ownership and interfaces.

## References

Li et al. (2024), *Cross-Domain Audio Deepfake Detection: Dataset and Analysis*, EMNLP 2024 · ASVspoof 2019 (Todisco et al., Interspeech 2019; database doi:10.7488/ds/2555) · Baevski et al. (2020), *wav2vec 2.0*, NeurIPS 2020 · Gao et al. (2021), *Generalized Spoofing Detection Inspired from Audio Generation Artifacts*, Interspeech 2021. Full list in the generated report.
