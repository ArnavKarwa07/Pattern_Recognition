# Methodology and implementation notes

This document records the decisions behind the code so that the group can
defend them. The planning documents in the repository root
([case study](../CaseStudy_ProblemDefinition.md), [Solution 1](../Solution1_MFCCFeatures+SVM.md),
[Solution 2](../Solution2_Log-MelSpectrogram+CNN.md), [Solution 3](../Solution3_wav2vec2.0Embeddings+SVM.md))
describe the motivation; this file describes what is actually implemented.

## Classes and scores

- `0` = bona fide, `1` = spoof. Spoof is the positive class in every metric.
- Every model outputs a real-valued score where **higher = more likely spoof**:
  SVM `decision_function` (classes ordered `[0, 1]`), CNN logit difference
  `z_spoof − z_bonafide`. A recording is predicted spoof when `score ≥ threshold`.

## Data partitions

| Role | Default source | Used for |
|---|---|---|
| train | ASVspoof 2019 LA train (official) | fitting scalers, SVMs, CNN weights, CNN normalisation stats |
| dev | ASVspoof 2019 LA dev (official) | SVM hyperparameters, CNN checkpoint/early stopping, LR schedule, operating threshold |
| `asvspoof19_la_eval` | ASVspoof 2019 LA eval (official, keys in database) | in-domain test (contains unseen attacks A07–A15, A17, A18) |
| `cdadd_test` | CD-ADD `test` | external-domain test |

- Official partitions are never merged or re-split.
- Test labels are used only to compute metrics after all choices are frozen.
- `data.holdout_generators` removes listed generators from train and dev; a test
  set with `condition: generator_held_out` and `generators: [...]` is then
  checked to contain only generators absent from train/dev.
- Pilot runs (`run_type: pilot`) may take stratified random subsets
  (`data.max_*_samples`). A `run_type: full` config with any subsampling is
  rejected at load time. Pilot/smoke outputs are bannered as such.

## Shared preprocessing (`data/preprocessing.py`)

1. Decode (soundfile), average channels to mono, resample to 16 kHz (soxr).
2. Reject empty / NaN / Inf signals (`AudioError`, logged to `errors.csv`).
3. Optional silence trimming (`audio.trim_silence_db`, default off).
4. Optional normalisation (`none` default, `peak`, `rms`).
5. Fixed 4.0 s window (64,000 samples): longer audio is cropped at the start
   (`eval_crop: start`, deterministic); shorter audio is repeat-padded
   (`pad_mode: repeat`, the convention used by many ASVspoof baselines).
   `train_random_crop: true` enables random crops **only** for CNN training
   samples; evaluation is always deterministic.

All three pipelines call the same `load_and_preprocess`, so each model sees
exactly the same 4 s of every recording.

## Solution 1 — MFCC + SVM

- `librosa.feature.mfcc`: 40 coefficients, 64 Mel bands, 512-point FFT,
  25 ms window, 10 ms hop. Optional Δ/ΔΔ (`mfcc.delta_order`).
- Per-coefficient mean and standard deviation over time → 80-D vector
  (160-D with Δ, 240-D with ΔΔ).
- `Pipeline([StandardScaler, SVC(kernel="rbf", class_weight="balanced")])`.
  Each grid point is fitted on **train only** and scored on dev; the scaler
  statistics therefore never see dev or test data (unit-tested).
- Grid: `C ∈ {0.1, 1, 10}`, `gamma ∈ {0.3, 1, 3} × "scale"` where "scale" is
  `1 / (n_features · Var(X_train_standardised))`. Selection by dev EER.

## Solution 2 — log-Mel CNN

- Log-Mel front end in PyTorch (`features/logmel.py`): 80 bands, 512-point
  FFT, 400-sample Hann window, 160-sample hop, `center=True`, power spectrum,
  Slaney Mel filterbank, `log(x + 1e-6)` → input 80 × 401 for 4 s.
  Computing it on the device avoids a multi-GB spectrogram cache.
- Per-band standardisation with mean/std from 2,000 random training clips,
  stored as buffers in the checkpoint.
- Architecture: 4 × [Conv3×3 → BatchNorm → ReLU → MaxPool 2×2] with channels
  16-32-64-128, global average pooling, dropout 0.3, linear 128→2
  (~98k parameters).
- Class-weighted cross-entropy (`n / (2 · n_c)`), AdamW (lr 1e-3, wd 1e-4),
  ReduceLROnPlateau on dev EER, early stopping after 5 epochs without dev-EER
  improvement, maximum 30 epochs. `best.pt` = best dev EER; `last.pt` allows
  `--resume` after interruption.

## Solution 3 — wav2vec 2.0 + SVM

- `facebook/wav2vec2-base` (pretrained on LibriSpeech 960 h, self-supervised;
  **not** a deepfake detector), frozen, `eval()` mode, `requires_grad=False`.
- Each clip is normalised by the checkpoint's own feature extractor
  individually (zero mean, unit variance), then zero-padded to the longest
  clip in the batch.
- Hidden states of `wav2vec.layer` (default −1 = last transformer layer) are
  mean-pooled over **valid frames only**; the number of valid frames comes from
  the model's convolutional output-length formula
  (`_get_feat_extract_output_lengths`). A unit test injects huge values into
  padded frames and checks they do not affect the embedding.
- `wav2vec2-base` uses group normalisation in its feature encoder, for which
  Hugging Face recommends not passing an attention mask; we follow that. With
  the default fixed 4 s inputs there is no padding at all.
- Embeddings (768-D) are cached per record id; the SVM stage is identical to
  Solution 1. Metadata records the checkpoint name and commit hash,
  Transformers/PyTorch versions, layer and pooling.
- **Known spurious warning.** Loading the checkpoint with Transformers 4.41 and
  PyTorch ≥ 2.1 prints “Some weights … `pos_conv_embed.conv.parametrizations.weight.original0/1`
  … newly initialized”. We verified on 2026-10-25 that these tensors are
  equal to the checkpoint's `weight_g`/`weight_v` after loading (PyTorch's
  weight-norm parametrisation renames them), so the weights *are* pretrained.

### Pretraining-data overlap

LibriSpeech 960 h = train-clean-100 + train-clean-360 + train-other-500.
LibriTTS test-clean is derived from LibriSpeech test-clean, which is outside
those subsets, so no recording-level overlap with the CD-ADD LibriTTS
bona fide audio is expected — but it is the same audiobook domain. This was
not verified by fingerprinting. CD-ADD results are additionally broken down by
`source_domain` (LibriTTS vs. TED) for all three models.

## Thresholds and metrics (`evaluation/`)

- **Operating threshold**: chosen on dev scores (`evaluation.threshold_method`,
  default `eer` = dev threshold where FPR = FNR; alternatives `youden`,
  `max_f1`), saved in the model's `metadata.json` with
  `threshold_fitted_on: dev`, and applied unchanged to every test set.
- **EER**: ROC computed at every distinct score; the crossing of FNR and FPR is
  linearly interpolated between the two bracketing operating points. It is a
  threshold-free summary of the test score distribution.
- Accuracy, balanced accuracy, precision, recall, F1 (spoof), ROC-AUC, average
  precision, confusion counts, FPR/FNR.
- **Confidence intervals**: percentile bootstrap of the EER (200 resamples).
  When `group_id` is available, whole groups are resampled (cluster
  bootstrap) because a CD-ADD genuine utterance and its spoofs are correlated.
- **Per-generator EER**: each spoofing system against *all* bona fide
  recordings of that test set (ASVspoof per-attack convention).
- **Inference time**: wall-clock per recording including audio decoding and
  feature computation, under each pipeline's configured parallelism
  (MFCC uses `mfcc.n_jobs` processes; feature times are measured when features
  are first computed and stored in the cache).
- Comparisons use only the records scored by **all** models on that test set.

## Reproducibility

- Global seed (`seed`) for Python, NumPy, PyTorch and data-loader shuffling;
  deterministic per-sample RNG for training crops.
- Config snapshots in `results/<run>/config_snapshots/` and next to each model.
- `environment` block (Python, OS, CPU/GPU, package versions) in every
  `metadata.json` and metrics JSON.
- Feature caches are keyed by a hash of the feature and audio configuration;
  changing any parameter creates a new cache instead of reusing stale vectors.
