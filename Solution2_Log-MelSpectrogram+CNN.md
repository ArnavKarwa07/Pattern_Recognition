# Solution 2: Log-Mel Spectrogram + Convolutional Neural Network
### Learning Acoustic Patterns Through Deep Learning

## 1. Overview

A log-Mel spectrogram represents how audio energy is distributed across frequency bands over time. Unlike a compact summary of manually extracted features, it preserves time-frequency patterns that may help distinguish genuine speech from synthetic speech.

A Convolutional Neural Network (CNN) can learn discriminative patterns directly from spectrograms. This approach investigates whether a neural network trained on these representations can detect useful acoustic patterns that are not captured by the MFCC + SVM baseline.

## 2. Proposed Methodology

**Step 1: Audio Preprocessing**

Load and resample recordings consistently, convert them to mono where necessary, and handle variable durations using a documented cropping or padding strategy.

**Step 2: Log-Mel Spectrogram Generation**

Convert each waveform into a log-Mel spectrogram using fixed parameters for sample rate, window length, hop length, and number of Mel frequency bands. Normalize the inputs consistently using training-set statistics where applicable.

**Step 3: CNN Architecture**

Develop a compact CNN containing convolutional layers, nonlinear activation functions, pooling, and dropout. The network will learn features from the spectrogram and use a final classification layer to predict the genuine or synthetic class.

**Step 4: Model Training**

Train the CNN on labelled training recordings. Use the development set for hyperparameter tuning and early stopping. Apply class weighting if needed to address the class imbalance.

Any augmentation, such as moderate background noise or compression, must be restricted to training data and applied without creating overlap between training and evaluation sources.

**Step 5: Prediction**

Transform unseen recordings through the identical preprocessing and spectrogram pipeline before obtaining class predictions and decision scores.

## 3. Evaluation

Evaluate the CNN on exactly the same benchmark and external test recordings used for Solution 1.

Metrics will include:
- Accuracy, precision, recall, and F1-score.
- Equal Error Rate (EER) and ROC-AUC.
- Confusion matrix and ROC curve.
- Training time and inference time.

Visualizations will include representative genuine and synthetic speech spectrograms, training and validation learning curves, and comparative ROC curves for the three approaches.

## 4. Strengths and Limitations

**Strengths**
- Learns local time-frequency patterns automatically.
- Preserves more temporal and spectral information than pooled MFCC features.
- Provides useful visual material for understanding acoustic differences between recordings.

**Limitations**
- Generally requires more computation than the SVM baseline.
- May overfit to the specific datasets or generation systems used during training.
- High performance on familiar samples does not necessarily imply generalization to unseen systems.

## 5. Expected Contribution

This solution will examine whether learned spectrogram representations provide an advantage over handcrafted acoustic features and how the CNN behaves when evaluated on independent speech-generation systems.

## 6. References

1. Gao, Y., Vuong, T., Elyasi, M., Bharaj, G., and Singh, R. (2021). "Generalized Spoofing Detection Inspired from Audio Generation Artifacts." Interspeech 2021, pp. 4184–4188.
   https://doi.org/10.21437/Interspeech.2021-1705

2. Li, Y. et al. (2024). "Cross-Domain Audio Deepfake Detection: Dataset and Analysis." EMNLP 2024.
   https://aclanthology.org/2024.emnlp-main.286/