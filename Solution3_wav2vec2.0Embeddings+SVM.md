# Solution 3: Pretrained wav2vec 2.0 Embeddings + Support Vector Machine
### Transfer Learning with Frozen Speech Representations

## 1. Overview

wav2vec 2.0 is a self-supervised speech representation model designed to learn useful representations from large amounts of speech audio. It can provide learned features without requiring a new neural network to be trained from scratch for the classification task.

In this approach, a pretrained wav2vec 2.0 model will act as a frozen feature extractor. The representations obtained from its hidden layers will be aggregated into fixed-length embeddings, which will then be supplied to an SVM to classify recordings as genuine or AI-generated.

This method investigates whether pretrained speech representations offer useful information beyond the handcrafted features and spectrogram-based features used in Solutions 1 and 2.

## 2. Proposed Methodology

**Step 1: Audio Preprocessing**

Prepare recordings according to the pretrained model's requirements. The public `facebook/wav2vec2-base` checkpoint expects audio sampled at 16 kHz. Apply consistent duration handling and normalization.

**Step 2: Embedding Extraction**

Pass each waveform through the frozen pretrained model to obtain frame-level hidden representations. Apply mean pooling across time to produce a fixed-length embedding for each recording.

Cache the embeddings to avoid repeatedly running the pretrained model during SVM tuning.

**Step 3: Feature Scaling**

Standardize embeddings using statistics calculated from the training set only. Reuse the same transformation for development and test recordings.

**Step 4: SVM Training**

Train an SVM, initially with an RBF kernel, on the extracted embeddings. Tune the hyperparameters and classification threshold using the development set only.

**Step 5: Prediction**

Extract embeddings from unseen recordings using the same checkpoint and preprocessing pipeline. Use the trained SVM to obtain the final prediction and decision score.

## 3. Evaluation

Use the same training, development, and final test partitions as the other two solutions.

Evaluation metrics will include:
- Accuracy, precision, recall, and F1-score.
- Equal Error Rate (EER) and ROC-AUC.
- Confusion matrix and ROC curve.
- Embedding-extraction time, SVM inference time, and model-storage requirements.

An important limitation must be considered: the public wav2vec 2.0 base checkpoint was pretrained using LibriSpeech, while the public CD-ADD test package includes LibriTTS-derived recordings as well as TED-sourced recordings. Report results on a TED-sourced subset separately where the metadata permits, alongside the overall external-test result, to make potential source overlap transparent. Apply the same test subsets to all three approaches.

## 4. Strengths and Limitations

**Strengths**
- Uses representations learned from large amounts of speech audio.
- Avoids training a large feature-extraction network from scratch.
- Combines transfer learning with a relatively lightweight classical classifier.

**Limitations**
- Embedding extraction requires additional computation and storage.
- The pretrained representations may retain speaker-, language-, or dataset-specific characteristics.
- Strong speech representations do not guarantee strong synthetic-speech detection.
- Potential overlap between pretraining and evaluation sources must be documented.

## 5. Expected Contribution

This solution will test whether pretrained speech representations provide a useful balance between handcrafted acoustic features and a CNN trained directly on spectrograms. It will also examine whether any observed benefits persist on independent test recordings.

## 6. References

1. Baevski, A., Zhou, Y., Mohamed, A., and Auli, M. (2020). "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." Advances in Neural Information Processing Systems 33.
   https://papers.neurips.cc/paper_files/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html

2. Meta AI. "facebook/wav2vec2-base: Model Card."
   https://huggingface.co/facebook/wav2vec2-base

3. Li, Y. et al. (2024). "Cross-Domain Audio Deepfake Detection: Dataset and Analysis."
   https://aclanthology.org/2024.emnlp-main.286/