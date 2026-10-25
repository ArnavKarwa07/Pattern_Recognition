# Team work split (four members)

The split below assigns *ownership*, not exclusive work: every module has one
owner who answers for it and one reviewer from another sub-team.

## Member 1 — Dataset and preprocessing

- Verify dataset downloads, labels, metadata and licences (ASVspoof 2019 LA:
  ODC-By; CD-ADD package: CC BY 4.0) — see [dataset_setup.md](dataset_setup.md).
- Own manifest generation and validation; run `prepare-data` / `validate-data`
  and review the `*_summary.json` and `*_failures.csv` outputs.
- Own the shared preprocessing and dataset statistics; check for train/test
  leakage (integrity report in `results/<run>/splits/split_summary.json`).
- Optional ablation: silence trimming (`audio.trim_silence_db`).

| Owns | Path |
|---|---|
| ASVspoof / CD-ADD / CSV parsers, downloads | `src/speechdeepfake/data/loaders.py` |
| manifest schema, label normalisation, integrity checks | `src/speechdeepfake/data/manifests.py` |
| per-file validation, dataset statistics | `src/speechdeepfake/data/validation.py` |
| experiment partitions, pilot subsets, hold-outs | `src/speechdeepfake/data/splits.py` |
| shared audio preprocessing | `src/speechdeepfake/data/preprocessing.py` |
| tests | `tests/test_data.py` |

## Member 2 — Classical approach (Solution 1)

- Own MFCC extraction and the MFCC + SVM pipeline; run its experiments
  (`speechdeepfake train --model mfcc_svm`) and analyse its errors.
- Optional ablations: deltas (`mfcc.delta_order`), larger C/gamma grid.
- Produce the MFCC heat-map and PCA figures.

| Owns | Path |
|---|---|
| MFCC features | `src/speechdeepfake/features/mfcc.py` |
| SVM pipeline + dev-set tuning (shared with Member 4) | `src/speechdeepfake/models/svm.py`, `models/mfcc_svm.py` |
| SVM training script | `src/speechdeepfake/training/train_svm.py` |

## Member 3 — Deep-learning approach (Solution 2)

- Own the log-Mel front end, the CNN and its training loop; run training and
  checkpoint selection (`speechdeepfake train --model logmel_cnn`).
- Optional ablations: `audio.train_random_crop`, 64 vs. 80 Mel bands,
  channel widths.
- Produce the spectrogram and learning-curve figures.

| Owns | Path |
|---|---|
| log-Mel front end | `src/speechdeepfake/features/logmel.py` |
| CNN architecture | `src/speechdeepfake/models/logmel_cnn.py` |
| training, checkpoints, resume, early stopping | `src/speechdeepfake/training/train_cnn.py` |

## Member 4 — Transfer learning, evaluation and integration (Solution 3)

- Own wav2vec 2.0 embedding extraction, caching and the wav2vec + SVM model.
- Coordinate shared metrics, comparison plots, the report generator and the CLI.
- Document the pretraining-overlap limitation (LibriSpeech vs. CD-ADD LibriTTS).

| Owns | Path |
|---|---|
| wav2vec 2.0 embeddings, masked pooling | `src/speechdeepfake/features/wav2vec_embeddings.py`, `models/wav2vec_svm.py` |
| feature cache | `src/speechdeepfake/features/cache.py` |
| metrics, thresholds, prediction | `src/speechdeepfake/evaluation/{metrics,thresholds,predict}.py` |
| figures and report | `src/speechdeepfake/evaluation/{plots,figures,report}.py` |
| CLI and pipeline stages | `src/speechdeepfake/cli.py`, `src/speechdeepfake/pipeline.py` |
| tests | `tests/test_metrics.py`, `tests/test_features.py`, `tests/test_smoke.py` |

## How the modules connect

```
loaders.py ──► manifests.py ──► validation.py ──► splits.py ──► ExperimentData(train, dev, tests)
                                                                     │
                     preprocessing.load_and_preprocess (shared) ◄────┤
                                                                     │
   mfcc.py ──┐                                                       │
             ├─► cache.py ──► models/svm.py (train-only scaler, dev tuning, dev threshold)
wav2vec ─────┘                                                       │
logmel.py ──► logmel_cnn.py ──► train_cnn.py (dev checkpoint, dev threshold)
                                                                     │
          predict.py: score each test set with the saved dev threshold
                                   │
          compile_results: metrics on records common to all models, bootstrap CIs, subgroups
                                   │
          figures.py / plots.py ──► reports/figures     report.py ──► reports/final_report.md
```

Interface contracts every member relies on:

1. A manifest row = one recording with `record_id`, `audio_path`, `label ∈ {0,1}`, `corpus`, `split` (+ optional metadata).
2. All audio goes through `load_and_preprocess(path, cfg.audio, train=...)`.
3. Every model writes `models/<run>/<model>/metadata.json` containing `threshold` and `threshold_fitted_on: "dev"`.
4. Scores: higher = more likely spoof.
5. Predictions: `results/<run>/predictions/<model>__<test_set>.csv`.

## Shared responsibilities (all four)

- Review the data protocol and agree on class labels and experiment splits
  (sign off on `configs/default.yaml` before the full run).
- Run `pytest` and `speechdeepfake smoke-test` before merging changes.
- Integrate the modules and cross-check results: each member re-runs one other
  member's model from the saved config snapshot and confirms the metrics.
- Review the generated report for any statement not supported by the result files.
