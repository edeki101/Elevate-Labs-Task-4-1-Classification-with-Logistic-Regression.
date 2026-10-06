# Elevate-Labs-Task-4-1-Classification-with-Logistic-Regression.
# Task 4 – Classification with Logistic Regression

Binary classifier that predicts whether a breast tumour is **malignant (M)** or **benign (B)** from the
Breast Cancer Wisconsin (Diagnostic) dataset, built with **scikit-learn, pandas and matplotlib**.

## What I did
1. **Data** – 569 samples, 30 numeric features (357 benign / 212 malignant). Dropped the `id` column and an empty `Unnamed: 32` column; encoded `M = 1`, `B = 0`.
2. **Split** – 80/20 stratified train/test split (`random_state=42`).
3. **Standardize** – `StandardScaler` fit on the training set only, wrapped in a `Pipeline` with the model so there is no data leakage.
4. **Model** – `LogisticRegression(max_iter=1000)`.
5. **Evaluate** – confusion matrix, precision, recall, F1, ROC-AUC.
6. **Tune threshold** – chosen on the *training* data using 5-fold out-of-fold probabilities (test set untouched). Rule: highest precision while keeping recall ≥ 0.97, because missing a malignant tumour is costlier than a false alarm.
7. **Sigmoid** – plotted the sigmoid curve with the test predictions placed on it.

## Results (test set, 114 samples)

| Threshold | TN | FP | FN | TP | Accuracy | Precision | Recall | F1 |
|-----------|----|----|----|----|----------|-----------|--------|-----|
| 0.50 (default) | 71 | 1 | 3 | 39 | 0.965 | 0.975 | 0.929 | 0.951 |
| 0.30 (tuned)   | 71 | 1 | 1 | 41 | 0.983 | 0.976 | 0.976 | 0.976 |

**ROC-AUC = 0.996.** Lowering the threshold from 0.50 to ~0.30 removed two missed malignant cases without adding false positives.
Caveat: the test set is small (42 malignant cases), so one or two samples move the numbers noticeably.

## Plots

### Confusion matrix (default threshold 0.5)
![Confusion matrix](images/confusion_matrix.png)

### Confusion matrix (tuned threshold ~0.30)
![Tuned confusion matrix](images/confusion_matrix_tuned.png)

### ROC curve
![ROC curve](images/roc_curve.png)

### Precision / recall vs threshold
![Threshold tuning](images/threshold_tuning.png)

### Top coefficients
![Coefficients](images/coefficients.png)

## Files
- `logistic_regression.py` – full code (run `python logistic_regression.py`)
- `data.csv` – dataset
- `images/` – confusion matrices (default and tuned), ROC curve, threshold tuning plot, sigmoid plot, coefficients
- `requirements.txt`

## Sigmoid function
Logistic regression computes a linear score `z = w·x + b` and squashes it into a probability:

`σ(z) = 1 / (1 + e^(−z))`

σ(0) = 0.5, large positive z → close to 1, large negative z → close to 0. `z` is the log-odds, so each coefficient
is the change in log-odds per one standard deviation of that feature. The classification threshold is applied to σ(z).

![sigmoid](images/sigmoid.png)

## Interview questions
1. **Logistic vs linear regression** – Linear regression predicts a continuous value and minimises squared error. Logistic regression predicts a probability for a class by passing a linear score through the sigmoid, and is fit by maximising likelihood (minimising log-loss).
2. **Sigmoid function** – `1 / (1 + e^(−z))`; an S-shaped curve mapping any real number to (0, 1).
3. **Precision vs recall** – Precision = TP / (TP + FP): of the cases flagged positive, how many really are. Recall = TP / (TP + FN): of the real positives, how many were caught.
4. **ROC-AUC** – The ROC curve plots true positive rate against false positive rate across all thresholds. AUC is the area under it: the probability that a random positive is scored higher than a random negative (0.5 = chance, 1.0 = perfect).
5. **Confusion matrix** – A table of TP, FP, TN, FN counts comparing predictions with actual labels; all the other metrics derive from it.
6. **Imbalanced classes** – Accuracy becomes misleading and the model favours the majority class. Use precision/recall/F1/PR-AUC, `class_weight="balanced"`, resampling (e.g. SMOTE), stratified splits, and threshold tuning.
7. **Choosing the threshold** – Depends on the cost of FN vs FP. Inspect the precision-recall trade-off or ROC curve, then pick the point meeting the business need (e.g. a recall target, max F1, or Youden's J). Choose it on validation/CV data, not the test set.
8. **Multi-class** – Yes: one-vs-rest (a binary model per class) or multinomial/softmax regression (`LogisticRegression` in scikit-learn handles this natively).
