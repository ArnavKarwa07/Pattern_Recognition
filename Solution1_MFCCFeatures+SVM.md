# Solution 1: MFCC Features + Support Vector Machine
### Classical Statistical Pattern Recognition Baseline

## 1. Overview

Mel-Frequency Cepstral Coefficients (MFCCs) represent important spectral characteristics of speech in a compact numerical form. They are widely used in speech-processing tasks and provide a practical starting point for conventional audio classification.

A Support Vector Machine (SVM) is a supervised learning algorithm that constructs a decision boundary between classes. In this approach, MFCC features extracted from each recording will be used to train an SVM to classify audio as genuine human speech or AI-generated/manipulated speech.

This solution serves as the classical baseline for comparison with the two deep-learning-based approaches.

## 2. Proposed Methodology

**Step 1: Audio Preprocessing**

Load the labelled recordings, convert them to mono where necessary, and standardize the sampling rate and duration. Apply consistent preprocessing to all three approaches.

**Step 2: MFCC Feature Extraction**

Extract MFCCs from successive audio frames. Calculate the mean and standard deviation of each coefficient across time to produce a fixed-length feature vector for each recording. Delta coefficients may be investigated as an optional extension.

**Step 3: Feature Scaling**

Standardize the feature vectors using statistics calculated from the training data only. Apply the same fitted transformation to the development and evaluation data.

**Step 4: Model Training**

Train an SVM, initially using a Radial Basis Function (RBF) kernel. Tune hyperparameters using the development set and consider class weighting to address the imbalance between genuine and spoofed recordings.

**Step 5: Classification**

Use the trained model to produce class predictions and decision scores for unseen recordings. Select any classification threshold using development data, not the final test set.

## 3. Evaluation

Use the same training, development, and final test partitions as Solutions 2 and 3.

Evaluation will include:
- Accuracy, precision, recall, and F1-score.
- Equal Error Rate (EER) and ROC-AUC.
- Confusion matrix and ROC curve.
- Feature-extraction and inference time.

Results will be reported separately for the benchmark and independent external test sets. The external test will examine whether the classifier remains effective when exposed to speech from unfamiliar generation systems.

## 4. Strengths and Limitations

**Strengths**
- Relatively low computational requirements.
- Fast training and straightforward implementation.
- Provides a clear example of feature vectors, supervised learning, statistical classification, and decision boundaries.

**Limitations**
- Temporal pooling may discard useful information about changes across speech frames.
- Handcrafted features may not capture every artifact introduced by modern speech-generation systems.
- Performance may deteriorate under unseen generators or altered audio conditions.

## 5. Expected Contribution

This solution will establish a reproducible conventional baseline. Its results will help determine how far compact, handcrafted acoustic features can support robust AI-generated speech detection.

## 6. References

1. Hamza, A. et al. (2022). "Deepfake Audio Detection via MFCC Features Using Machine Learning." IEEE Access, 10.
   https://doi.org/10.1109/ACCESS.2022.3231480

2. ASVspoof 2019. "Automatic Speaker Verification Spoofing and Countermeasures Challenge: Evaluation Plan."
   https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf