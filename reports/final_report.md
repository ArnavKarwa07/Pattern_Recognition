# Robust Detection of AI-Generated Speech: A Comparative Study of Classical Features, Deep Learning, and Pretrained Speech Representations

*Generated 2026-10-25 18:03 from run `demo` (run type: pilot) by `speechdeepfake report`. All numbers are read from saved result files.*

> **PILOT RUN.** Models were trained and evaluated on stratified random subsets (`data.max_*_samples`). These numbers check the pipeline end to end and are **not** final research results.

## 1. Abstract

This study compares three pattern-recognition pipelines for classifying a recording as bona fide (genuine human) speech or spoofed (AI-generated or manipulated) speech: MFCC statistics with an RBF-kernel SVM, a compact CNN trained on log-Mel spectrograms, and frozen wav2vec 2.0 embeddings with an RBF-kernel SVM. All models are fitted on `asvspoof2019_la/train`, tuned and calibrated on `asvspoof2019_la/dev`, and evaluated on: `asvspoof19_la_eval` = asvspoof2019_la/eval (in domain); `cdadd_test` = cdadd/test (external domain). In the planned full study these are the official ASVspoof 2019 LA partitions, with the LA evaluation partition as the in-domain test and the CD-ADD test split (speech from five zero-shot TTS systems) as the external test. The research question itself is not new; the contribution is a reproducible, like-for-like comparison under both conditions.

Key observations: see Section 12.

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

**asvspoof2019_la** (validated manifest)

| corpus/split | records | usable | bona fide | spoof | mean dur. (s) | total (h) | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| asvspoof2019_la/dev | 600 | 600 | 300 | 300 | 3.492 | 0.582 | 0 |
| asvspoof2019_la/eval | 625 | 625 | 300 | 325 | 3.217 | 0.559 | 0 |
| asvspoof2019_la/train | 600 | 600 | 300 | 300 | 3.364 | 0.561 | 0 |

**cdadd** (validated manifest)

| corpus/split | records | usable | bona fide | spoof | mean dur. (s) | total (h) | failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| cdadd/test | 796 | 796 | 140 | 656 | 9.686 | 2.142 | 0 |

**Partitions used in this run**

| partition | records | bona fide | spoof |
|---|---|---|---|
| train | 600 | 300 | 300 |
| dev | 600 | 300 | 300 |
| asvspoof19_la_eval | 625 | 300 | 325 |
| cdadd_test | 796 | 140 | 656 |


## 6. Preprocessing

All pipelines share one function (`speechdeepfake.data.preprocessing.load_and_preprocess`): decode, down-mix to mono, resample to 16000 Hz, reject empty/non-finite signals, normalisation `none`, and fix the duration to 4.0 s — longer recordings are cropped deterministically at the `start`, shorter ones are padded by `repeat`. Training-time random cropping is disabled; evaluation preprocessing is always deterministic. Files are validated before training (existence, decodability, empty, NaN/Inf, too short, silent); failures are logged, not scored.

## 7. Methodology — Solution 1: MFCC + SVM

40 MFCCs (64 Mel bands, 400-sample window, 160-sample hop) are summarised by their mean and standard deviation over time into a fixed-length vector, cached to disk. A scikit-learn Pipeline (StandardScaler fitted on training data only → RBF SVC, class_weight=balanced) is fitted on the training partition for each (C, gamma) in the grid C ∈ [0.1, 1.0, 10.0], gamma ∈ ['scale*0.3', 'scale', 'scale*3'], and selected by development-set eer. Scores are SVM decision values.

Selected model: selected C = 10.0, gamma = scale (scale), feature dimension 80, dev EER 14.33%, threshold 0.4583.

## 8. Methodology — Solution 2: log-Mel spectrogram + CNN

80-band log-Mel spectrograms (512-point FFT, 400-sample Hann window, 160-sample hop) are computed on the fly and standardised per band with statistics from 600 training clips. The CNN has 4 blocks of Conv3×3–BatchNorm–ReLU–MaxPool (channels [16, 32, 64, 128]), global average pooling, dropout 0.3 and a 2-way linear head, trained with class-weighted cross-entropy, AdamW (lr 0.001, weight decay 0.0001), plateau LR schedule, up to 30 epochs with early stopping (patience 5) on development eer. The score is the logit difference z_spoof − z_bonafide.

Selected checkpoint: best epoch 15 of 21 (by dev eer), 97,650 parameters, dev EER 4.67%, threshold -0.777.

## 9. Methodology — Solution 3: wav2vec 2.0 embeddings + SVM

The pretrained `facebook/wav2vec2-base` model is used as a frozen feature extractor (not fine-tuned; it is not a deepfake detector). Each waveform is normalised by the checkpoint's feature extractor, batched (batch size 8), and the hidden states of layer -1 are mean-pooled over valid (non-padded) frames into one embedding, cached with its record id. The same SVM pipeline and dev-set tuning as Solution 1 is then applied. Checkpoint `facebook/wav2vec2-base` (revision `0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8`), Transformers 4.41.2, hidden layer -1, mean pooling over valid frames, embedding size 768.

Selected model: selected C = 1.0, gamma = scale*0.3 (0.0003906), feature dimension 768, dev EER 5.67%, threshold -0.1869.

## 10. Experimental design and split integrity

- Train: asvspoof2019_la/train; development (tuning, checkpoint selection, threshold): asvspoof2019_la/dev; test sets: `asvspoof19_la_eval` = asvspoof2019_la/eval (in domain); `cdadd_test` = cdadd/test (external domain).
- Official partitions are used unchanged — never merged or re-split. No generator hold-out beyond the official protocol.
- Before each run the partitions are checked for duplicated ids, conflicting labels, and shared utterance/source-group ids, audio paths and byte-identical files (SHA-1) across partitions. Findings: no blocking issues.
- All three models are compared on the records that every model scored.

## 11. Evaluation metrics

Spoof is the positive class and higher scores mean "more likely spoof". Accuracy, balanced accuracy, precision, recall and F1 are computed at one operating threshold per model, chosen on the development set by the `eer` rule and applied unchanged to every test set. ROC-AUC, average precision and the equal error rate (EER; ROC crossing of FPR and FNR, linearly interpolated between adjacent operating points) are threshold-free summaries of the test score distribution — the EER is *not* a claim that a threshold was tuned on test data. EER confidence intervals use a 200-resample percentile bootstrap; when source-group ids exist (e.g. a CD-ADD genuine utterance and its spoofs) whole groups are resampled. Inference time is wall-clock per recording including audio loading and feature computation.

## 12. Quantitative results

#### asvspoof19_la_eval (in domain; asvspoof2019_la/eval)

| model | n (bona fide / spoof) | threshold (dev) | accuracy | balanced acc. | precision | recall | F1 | ROC-AUC | AP | EER [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| MFCC + SVM | 625 (300 / 325) | 0.4583 | 82.88% | 83.32% | 93.25% | 72.31% | 81.46% | 0.9061 | 0.9314 | 18.15% [15.38%, 21.32%] |
| log-Mel + CNN | 625 (300 / 325) | -0.777 | 80.80% | 81.37% | 94.37% | 67.08% | 78.42% | 0.9108 | 0.9327 | 16.62% [13.84%, 18.98%] |
| wav2vec 2.0 + SVM | 625 (300 / 325) | -0.1869 | 92.00% | 92.00% | 92.57% | 92.00% | 92.28% | 0.9776 | 0.9828 | 8.00% [5.55%, 9.88%] |

Records compared (common to all evaluated models): 625. CI method: cluster percentile bootstrap, 200 resamples.

#### cdadd_test (external domain; cdadd/test)

| model | n (bona fide / spoof) | threshold (dev) | accuracy | balanced acc. | precision | recall | F1 | ROC-AUC | AP | EER [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| MFCC + SVM | 796 (140 / 656) | 0.4583 | 82.04% | 49.77% | 82.35% | 99.54% | 90.13% | 0.5597 | 0.8625 | 46.49% [42.96%, 51.44%] |
| log-Mel + CNN | 796 (140 / 656) | -0.777 | 82.29% | 49.92% | 82.39% | 99.85% | 90.28% | 0.5309 | 0.8387 | 47.86% [44.31%, 52.82%] |
| wav2vec 2.0 + SVM | 796 (140 / 656) | -0.1869 | 81.03% | 49.44% | 82.25% | 98.17% | 89.51% | 0.3923 | 0.7833 | 57.14% [53.19%, 61.33%] |

Records compared (common to all evaluated models): 796. CI method: cluster percentile bootstrap, 200 resamples.


Observations:

- asvspoof19_la_eval (in domain): lowest measured EER was wav2vec 2.0 + SVM (8.00%), followed by log-Mel + CNN (16.62%). Their bootstrap confidence intervals do not overlap.
- cdadd_test (external domain): all three models are close to or worse than chance (EER 46.49%–57.14%; ROC-AUC 0.3923–0.5597), so no meaningful ranking exists. An AUC below 0.5 means that model scores genuine recordings as *more* spoof-like than the TTS recordings on this data. Balanced accuracy is about 50%, i.e. the dev-selected thresholds label nearly every recording spoof; the high plain accuracy only reflects the 82.41% spoof share.

## 13. Graphical comparisons

![Recordings per partition and class](figures/partition_counts.png)

*Recordings per partition and class*

![Class distribution per partition](figures/class_distribution.png)

*Class distribution per partition*

![Duration distribution per partition](figures/duration_distribution.png)

*Duration distribution per partition*

![Example genuine and spoofed waveforms](figures/waveforms.png)

*Example genuine and spoofed waveforms*

![MFCCs of representative recordings](figures/mfcc_heatmap.png)

*MFCCs of representative recordings*

![Genuine vs. spoof log-Mel spectrograms](figures/logmel_examples.png)

*Genuine vs. spoof log-Mel spectrograms*

![PCA of MFCC statistics and wav2vec 2.0 embeddings](figures/pca_features.png)

*PCA of MFCC statistics and wav2vec 2.0 embeddings*

![CNN training and validation curves](figures/cnn_learning_curves.png)

*CNN training and validation curves*

![ROC curves — asvspoof19_la_eval](figures/roc_asvspoof19_la_eval.png)

*ROC curves — asvspoof19_la_eval*

![Precision-recall curves — asvspoof19_la_eval](figures/pr_asvspoof19_la_eval.png)

*Precision-recall curves — asvspoof19_la_eval*

![Confusion matrices — asvspoof19_la_eval](figures/confusion_asvspoof19_la_eval.png)

*Confusion matrices — asvspoof19_la_eval*

![ROC curves — cdadd_test](figures/roc_cdadd_test.png)

*ROC curves — cdadd_test*

![Precision-recall curves — cdadd_test](figures/pr_cdadd_test.png)

*Precision-recall curves — cdadd_test*

![Confusion matrices — cdadd_test](figures/confusion_cdadd_test.png)

*Confusion matrices — cdadd_test*

![EER by model and test set](figures/eer_comparison.png)

*EER by model and test set*

![Balanced accuracy and F1 by model and test set](figures/ba_f1_comparison.png)

*Balanced accuracy and F1 by model and test set*

![Inference time per recording](figures/inference_time.png)

*Inference time per recording*


## 14. Generalization analysis

![In-domain vs. external-domain EER](figures/generalization_eer.png)

*In-domain vs. external-domain EER*

![EER by generator — asvspoof19_la_eval](figures/eer_by_generator_asvspoof19_la_eval.png)

*EER by generator — asvspoof19_la_eval*

![EER by source domain — asvspoof19_la_eval](figures/eer_by_source_domain_asvspoof19_la_eval.png)

*EER by source domain — asvspoof19_la_eval*

![EER by generator — cdadd_test](figures/eer_by_generator_cdadd_test.png)

*EER by generator — cdadd_test*

![EER by source domain — cdadd_test](figures/eer_by_source_domain_cdadd_test.png)

*EER by source domain — cdadd_test*


**EER by generation system (each scored against all bona fide recordings) — asvspoof19_la_eval**

| generator_id | EER MFCC + SVM | EER log-Mel + CNN | EER wav2vec 2.0 + SVM |
|---|---|---|---|
| A07 | 2.00% | 2.00% | 1.00% |
| A08 | 1.33% | 0.00% | 0.00% |
| A09 | 4.00% | 0.33% | 0.00% |
| A10 | 3.33% | 16.00% | 3.33% |
| A11 | 4.00% | 12.33% | 0.00% |
| A12 | 14.00% | 16.00% | 0.00% |
| A13 | 12.00% | 36.00% | 0.00% |
| A14 | 4.33% | 8.00% | 0.67% |
| A15 | 12.00% | 17.33% | 3.67% |
| A16 | 20.00% | 6.67% | 8.00% |
| A17 | 60.00% | 40.00% | 16.00% |
| A18 | 28.00% | 12.00% | 4.00% |
| A19 | 32.00% | 12.00% | 21.67% |

**EER by generation system (each scored against all bona fide recordings) — cdadd_test**

| generator_id | EER MFCC + SVM | EER log-Mel + CNN | EER wav2vec 2.0 + SVM |
|---|---|---|---|
| openvoice | 36.43% | 42.54% | 55.97% |
| seamless | 56.20% | 63.57% | 45.71% |
| valle | 51.85% | 54.63% | 67.59% |
| whisperSpeech | 55.71% | 52.86% | 56.43% |
| yourTTS | 39.86% | 24.64% | 61.43% |

**EER by source domain — asvspoof19_la_eval**

| source_domain | EER MFCC + SVM | EER log-Mel + CNN | EER wav2vec 2.0 + SVM |
|---|---|---|---|
| vctk | 18.15% | 16.62% | 8.00% |

**EER by source domain — cdadd_test**

| source_domain | EER MFCC + SVM | EER log-Mel + CNN | EER wav2vec 2.0 + SVM |
|---|---|---|---|
| libritts | 47.54% | 45.90% | 56.56% |
| ted | 32.93% | 66.67% | 63.41% |


## 15. Computational-cost analysis

| model | training (s) | train feature extraction (s) | inference (ms/recording) |
|---|---|---|---|
| MFCC + SVM | 0.008562 | 3.993 | 4.176 |
| log-Mel + CNN | 196.2 | on the fly (in training time) | 5.532 |
| wav2vec 2.0 + SVM | 0.04276 | 87.99 | 145 |

SVM feature extraction times are measured when features are first computed (later runs reuse the cache). The CNN computes spectrograms inside its training/inference time. Hardware/software: Windows-11-10.0.26300-SP0, Python 3.12.9, CUDA available: False; packages: numpy 1.26.4, scipy 1.15.1, pandas 2.3.3, scikit-learn 1.7.2, librosa 0.11.0, soundfile 0.13.1, torch 2.8.0, transformers 4.41.2, pyarrow 23.0.1, matplotlib 3.10.7, PyYAML 6.0.2, joblib 1.5.2

## 16. Error analysis

**asvspoof19_la_eval** (n = 625 common records)

- MFCC + SVM: 17 bona fide recordings flagged as spoof (false positives), 90 spoofs missed (false negatives).
- log-Mel + CNN: 13 bona fide recordings flagged as spoof (false positives), 107 spoofs missed (false negatives).
- wav2vec 2.0 + SVM: 24 bona fide recordings flagged as spoof (false positives), 26 spoofs missed (false negatives).
- Misclassified by every evaluated model: 15; by at least one: 193.

**cdadd_test** (n = 796 common records)

- MFCC + SVM: 140 bona fide recordings flagged as spoof (false positives), 3 spoofs missed (false negatives).
- log-Mel + CNN: 140 bona fide recordings flagged as spoof (false positives), 1 spoofs missed (false negatives).
- wav2vec 2.0 + SVM: 139 bona fide recordings flagged as spoof (false positives), 12 spoofs missed (false negatives).
- Misclassified by every evaluated model: 139; by at least one: 156.


## 17. Limitations

- ASVspoof 2019 LA is known to contain dataset artefacts such as differences in leading/trailing silence between classes (Müller et al., 2021); detectors may exploit them. Silence trimming is available (`audio.trim_silence_db`) for an ablation.
- A fixed 4.0 s analysis window discards the remainder of longer recordings.
- The CD-ADD package provides only a test split, so no target-domain data are used for adaptation; class balance differs from ASVspoof, which affects precision/F1 and the dev-selected threshold.
- Possible source-domain overlap between wav2vec 2.0 pretraining (LibriSpeech) and CD-ADD LibriTTS audio is documented but not verified at the recording level.
- Hyperparameter grids are small for computational reasons.

- This is a pilot/smoke run on subsets; numbers are not final.

## 18. Conclusion

- asvspoof19_la_eval (in domain): lowest measured EER was wav2vec 2.0 + SVM (8.00%), followed by log-Mel + CNN (16.62%). Their bootstrap confidence intervals do not overlap.
- cdadd_test (external domain): all three models are close to or worse than chance (EER 46.49%–57.14%; ROC-AUC 0.3923–0.5597), so no meaningful ranking exists. An AUC below 0.5 means that model scores genuine recordings as *more* spoof-like than the TTS recordings on this data. Balanced accuracy is about 50%, i.e. the dev-selected thresholds label nearly every recording spoof; the high plain accuracy only reflects the 82.41% spoof share.

## 19. References

1. Li, Y., Zhang, M., Ren, M., Qiao, X., Ma, M., Wei, D., & Yang, H. (2024). Cross-Domain Audio Deepfake Detection: Dataset and Analysis. *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 4977-4983. https://doi.org/10.18653/v1/2024.emnlp-main.286 (dataset repository: https://github.com/leolya/CD-ADD; public package used here: https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/CD-ADD, CC BY 4.0).
2. Todisco, M., Wang, X., Vestman, V., Sahidullah, M., Delgado, H., Nautsch, A., Yamagishi, J., Evans, N., Kinnunen, T., & Lee, K. A. (2019). ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection. *Proc. Interspeech 2019*, 1008-1012. https://doi.org/10.21437/Interspeech.2019-2249
3. Yamagishi, J., Todisco, M., Sahidullah, M., Delgado, H., Wang, X., Evans, N., Kinnunen, T., Lee, K. A., Vestman, V., & Nautsch, A. (2019). ASVspoof 2019: The 3rd Automatic Speaker Verification Spoofing and Countermeasures Challenge database [sound]. University of Edinburgh, CSTR. https://doi.org/10.7488/ds/2555 (ODC-By licence; challenge site https://www.asvspoof.org/index2019.html).
4. Baevski, A., Zhou, Y., Mohamed, A., & Auli, M. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations. *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*. https://papers.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html (checkpoint: https://huggingface.co/facebook/wav2vec2-base).
5. Gao, Y., Vuong, T., Elyasi, M., Bharaj, G., & Singh, R. (2021). Generalized Spoofing Detection Inspired from Audio Generation Artifacts. *Proc. Interspeech 2021*, 4184-4188. https://doi.org/10.21437/Interspeech.2021-1705
6. Müller, N. M., Dieckmann, F., Czempin, P., Canals, R., Böttinger, K., & Williams, J. (2021). Speech is Silver, Silence is Golden: What do ASVspoof-trained Models Really Learn? *Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge*, 55-60. https://doi.org/10.21437/ASVSPOOF.2021-9
7. Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). LibriSpeech: An ASR corpus based on public domain audio books. *Proc. IEEE ICASSP 2015*, 5206-5210. https://doi.org/10.1109/ICASSP.2015.7178964

