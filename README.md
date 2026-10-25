# Robust Detection of AI-Generated Speech

### A pattern recognition study of generalization across unseen speech-generation systems

This project compares three approaches for distinguishing genuine human speech from AI-generated or manipulated speech. It focuses on whether systems trained on known examples can generalize to recordings from unfamiliar speech-generation systems and different audio conditions.

The project is currently in the planning/documentation stage. The methods and evaluations below describe the work to be implemented; no experimental results are claimed yet.

## Project goals

- Implement and compare three speech-classification solutions.
- Evaluate them consistently on benchmark and independent external data.
- Measure predictive performance, computational cost, and practical limitations.
- Visualize model behavior and compare results, including performance on unseen generators.

## Planned solutions

| Solution | Audio representation | Classifier | Purpose |
| --- | --- | --- | --- |
| 1 | MFCC summary features | RBF-kernel SVM | Classical, computationally light baseline |
| 2 | Log-Mel spectrograms | CNN | Learn time-frequency patterns from audio |
| 3 | Frozen wav2vec 2.0 embeddings | RBF-kernel SVM | Evaluate transfer-learned speech representations |

Each approach will use consistent audio preprocessing and the same data partitions where applicable. Scaling, tuning, early stopping, and decision-threshold selection will use training/development data only; final test data will be held out from model fitting and tuning.

## Data and evaluation plan

- **ASVspoof 2019 Logical Access (LA):** use the official training and development partitions for model development and reserve the official evaluation partition for final in-domain evaluation.
- **CD-ADD:** use as an independent external test of generalization to newer zero-shot text-to-speech systems, subject to its evaluation protocol and available metadata. Do not use its test data for fitting or hyperparameter tuning.
- **Optional condition analysis:** where feasible, examine the effect of audio compression. For wav2vec 2.0, report the TED-sourced external subset separately when metadata allows, because the pretrained model's LibriSpeech pretraining may overlap in source with LibriTTS-derived CD-ADD recordings. Use the same applicable test subsets for all three approaches.

Report accuracy, precision, recall, F1-score, Equal Error Rate (EER), and ROC-AUC. Also document confusion matrices and relevant computational measurements, such as training, feature/embedding extraction, inference time, and model storage. Results should be separated by benchmark and external test set.

## Planned graphical representations

The implemented solutions will be accompanied by plots and visualizations that make their behavior and trade-offs clear, including:

- Representative genuine and synthetic log-Mel spectrograms.
- CNN training and validation learning curves.
- ROC curves and confusion matrices for each approach.
- Side-by-side performance comparisons across all three solutions and test sets, using shared metrics and clearly labeled axes.
- Comparisons of computational cost, such as training/extraction and inference time, where measured.
- Where data and protocol permit, comparisons across external test subsets or audio-compression conditions.

Visualizations will be generated from measured results and labeled with the corresponding dataset split and evaluation conditions. No expected ranking or performance advantage is assumed in advance.

## Team implementation

Implementation of all three solutions is planned as group work by a team of four students. The team will coordinate shared preprocessing, data partitioning, evaluation definitions, and result reporting so that comparisons remain reproducible and fair. The three methods and supporting project work can be divided across the team, with a shared review of evaluation and plots.

## Repository documentation

- [Case study problem definition](CaseStudy_ProblemDefinition.md) — motivation, objectives, datasets, methodology, and project-level evaluation plan.
- [Solution 1: MFCC + SVM](Solution1_MFCCFeatures+SVM.md) — handcrafted acoustic features and classical baseline.
- [Solution 2: Log-Mel spectrogram + CNN](Solution2_Log-MelSpectrogram+CNN.md) — spectrogram-based neural classifier.
- [Solution 3: wav2vec 2.0 embeddings + SVM](Solution3_wav2vec2.0Embeddings+SVM.md) — frozen pretrained representations and transfer-learning considerations.

## Project status

The current repository contains planning documents for the case study and three proposed solutions. Code, trained models, generated plots, and measured results will be added as implementation progresses. Once available, this README should be updated with setup instructions, data preparation steps, commands for running each solution, and links to the final results.

## Expected deliverables

- Implementations of all three classification pipelines.
- A consistent evaluation of in-domain and external-test performance.
- Plots and visual comparisons of predictive performance and computational cost.
- A comparative report discussing generalization, limitations, and evaluation conditions.

## References

See the references sections in the [case study](CaseStudy_ProblemDefinition.md) and the individual solution documents for source material on ASVspoof 2019, CD-ADD, MFCC-based detection, spectrogram-based detection, and wav2vec 2.0.
