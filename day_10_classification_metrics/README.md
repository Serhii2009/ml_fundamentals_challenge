# 📘 LESSON 10: CLASSIFICATION METRICS

## 1. Theory: Why Do We Need Different Metrics?

### 🔹 The Problem with Accuracy Alone

When we train a classification model, we want to understand: how well does it predict?

The naive metric is **Accuracy** (proportion of correct answers). But it has a major flaw:

**Example:** Imagine a medical test where 95% of patients are healthy and 5% are sick.
If a model always predicts "healthy," its accuracy = 95%, but it never finds sick patients!

### 🔹 Different Metrics for Different Perspectives

- **Accuracy** → overall proportion of correct predictions
- **Precision** → how "clean" are positive predictions
- **Recall** → how completely do we find positive examples
- **F1** → balance between Precision and Recall
- **ROC-AUC** → overall classification quality across different thresholds

📌 **Key Insight:** Different metrics reveal different aspects of model performance. A complete evaluation requires multiple perspectives.

### ✅ Quick Check:

Why isn't accuracy always a good metric for classification?

---

## 2. Theory: Confusion Matrix

### 🔹 The Foundation of All Metrics

The key to understanding metrics is the **confusion matrix**:

```
                Predicted
              0       1
Actual   0   TN      FP
         1   FN      TP
```

**Definitions:**

- **TP (True Positive):** model correctly found "positive" cases
- **FP (False Positive):** model mistakenly said "positive" when object was negative
- **FN (False Negative):** model missed a "positive" case
- **TN (True Negative):** model correctly said "negative"

### 🔹 Metal Detector Analogy

Think of a metal detector searching for gold:

- **TP** → found real gold
- **FP** → alarm sounded, but it was trash
- **FN** → there was gold, but detector missed it
- **TN** → correctly showed no metal when there was none

### ✅ Quick Check:

Explain in your own words what FN (False Negative) means.

---

## 3. Theory: Accuracy

### 🔹 Formula and Interpretation

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Shows the proportion of correct answers overall.

**Good for:** Balanced classes where all errors are equally important.
**Bad for:** Imbalanced datasets where rare class is more important.

### 🔹 The Imbalanced Data Problem

**Example:**

- 1000 patients: 950 healthy (0), 50 sick (1)
- Model always predicts "healthy"
- Accuracy = 950/1000 = 95%
- But model never finds sick patients → completely useless!

### ✅ Quick Check:

Give an example where accuracy = 95% but the model is terrible.

---

## 4. Theory: Precision and Recall

### 🔹 Precision (Positive Predictive Value)

```
Precision = TP / (TP + FP)
```

**Interpretation:** Of all predicted "positive" cases, how many are actually positive?

**Analogy:** Gold purity. If metal is 90% pure, then in 100 pieces of "gold," only 90 are real and 10 are trash.

### 🔹 Recall (Sensitivity, True Positive Rate)

```
Recall = TP / (TP + FN)
```

**Interpretation:** Of all actual "positive" cases, how many did the model find?

**Analogy:** Mining efficiency. If we found only 70 out of 100 gold pieces in a mine, recall = 70%.

### 🔹 The Precision-Recall Tradeoff

| High Precision Focus     | High Recall Focus         |
| ------------------------ | ------------------------- |
| Few false alarms         | Don't miss positive cases |
| Conservative predictions | Liberal predictions       |
| Quality over quantity    | Quantity over quality     |

### ✅ Quick Check:

For medical disease testing, which is more important: Precision or Recall? Why?

---

## 5. Theory: F1-Score

### 🔹 Harmonic Mean of Precision and Recall

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Balances precision and recall into a single metric.

### 🔹 Why Harmonic Mean?

**Harmonic mean "punishes" large imbalances:**

| Precision | Recall | Arithmetic Mean | Harmonic Mean (F1) |
| --------- | ------ | --------------- | ------------------ |
| 1.0       | 0.1    | 0.55            | 0.18               |
| 0.9       | 0.9    | 0.90            | 0.90               |

F1 honestly shows when one metric is much worse than the other.

### ✅ Quick Check:

Why does F1 use harmonic mean instead of arithmetic mean?

---

## 6. Theory: ROC and AUC

### 🔹 ROC Curve

**ROC (Receiver Operating Characteristic)** plots:

- **Y-axis:** True Positive Rate (Recall)
- **X-axis:** False Positive Rate = FP / (FP + TN)

Shows model performance across all classification thresholds.

### 🔹 AUC (Area Under Curve)

**Interpretation of AUC values:**

- **AUC = 0.5** → random model (no predictive power)
- **AUC = 1.0** → perfect model
- **AUC = 0.85** → 85% of the time, model ranks a positive example higher than a negative example

📌 **Key Advantage:** AUC is threshold-independent and works well with imbalanced data.

### ✅ Quick Check:

What does AUC = 0.85 mean in practical terms?

---

## 7. Python Practice

### 7.1 Data Generation and Model Training

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

# Generate imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10,
                          weights=[0.9, 0.1], random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
```

### 7.2 Manual Metrics Calculation

```python
# Confusion matrix
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:")
print(cm)

# Extract components
TN, FP, FN, TP = cm.ravel()

# Manual calculations
accuracy_manual = (TP + TN) / (TP + TN + FP + FN)
precision_manual = TP / (TP + FP)
recall_manual = TP / (TP + FN)
f1_manual = 2 * precision_manual * recall_manual / (precision_manual + recall_manual)

print(f"\nManual Calculations:")
print(f"Accuracy: {accuracy_manual:.3f}")
print(f"Precision: {precision_manual:.3f}")
print(f"Recall: {recall_manual:.3f}")
print(f"F1-Score: {f1_manual:.3f}")
```

### 7.3 Sklearn Verification

```python
# Verify with sklearn
print(f"\nSklearn Verification:")
print(f"Accuracy: {accuracy_score(y, y_pred):.3f}")
print(f"Precision: {precision_score(y, y_pred):.3f}")
print(f"Recall: {recall_score(y, y_pred):.3f}")
print(f"F1-Score: {f1_score(y, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y, y_proba):.3f}")
```

### 7.4 ROC Curve Visualization

```python
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y, y_proba):.3f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

📌 This demonstrates how to calculate all major classification metrics both manually and using sklearn, helping understand the underlying mathematics.

---

## 8. When to Use Which Metric

### 🔹 Metric Selection Guide

| Scenario                               | Primary Metric    | Why                                      |
| -------------------------------------- | ----------------- | ---------------------------------------- |
| **Balanced classes, all errors equal** | Accuracy          | Simple and interpretable                 |
| **Imbalanced classes**                 | F1-Score, ROC-AUC | Accounts for class imbalance             |
| **Cost of false positives is high**    | Precision         | Minimizes incorrect positive predictions |
| **Cost of false negatives is high**    | Recall            | Ensures we don't miss positive cases     |
| **Need single balanced metric**        | F1-Score          | Balances precision and recall            |
| **Threshold-independent evaluation**   | ROC-AUC           | Works across all decision thresholds     |

### 🔹 Real-World Examples

**High Recall Priority:**

- **Medical diagnosis:** Don't miss disease cases
- **Fraud detection:** Catch all fraudulent transactions
- **Security systems:** Detect all threats

**High Precision Priority:**

- **Email spam filtering:** Don't mark legitimate emails as spam
- **Product recommendations:** Only suggest relevant items
- **Legal document review:** High confidence in flagged documents

---

## 9. Understanding Challenge

### 🎤 Your Tasks:

1. What's the difference between Precision and Recall?
2. What does F1-score show and when is it needed?
3. Why is ROC-AUC considered more reliable than Accuracy for imbalanced classes?
4. Give a real example where Recall is more important than Precision
5. Give an example where Precision is more important than Recall

## Key Takeaways

- **Confusion Matrix** is the foundation for understanding all classification metrics
- **Accuracy** can be misleading with imbalanced data
- **Precision** focuses on the quality of positive predictions
- **Recall** focuses on finding all positive cases
- **F1-Score** balances precision and recall using harmonic mean
- **ROC-AUC** provides threshold-independent performance evaluation
- **Metric choice** depends on the specific costs of different types of errors
- **Multiple metrics** give a complete picture of model performance

_Happy Learning! 🚀_
