# 📘 LESSON 9: LOGISTIC REGRESSION

## 1. Theory: Why Do We Need Logistic Regression?

### 🔹 The Classification Problem

Linear regression predicts continuous numerical values (any real numbers). This works well for tasks like "predicting apartment prices."

But in classification tasks (0/1, yes/no, dog/cat) we need to predict the **probability** of belonging to a class.

**Logistic regression solves binary classification:**

```
ŷ = P(y = 1 | x)
```

Where ŷ represents the probability that the example belongs to class 1.

### 🔹 Why Not Use Linear Regression for Classification?

Linear regression can output any value (-∞ to +∞), but probabilities must be between 0 and 1. Also, the relationship between features and class membership is often non-linear.

📌 **Example:** If linear regression predicts 1.5 for "spam detection" — what does that mean? We need probabilities!

### ✅ Quick Check:

Why can't we simply use linear regression for classification tasks?

---

## 2. Theory: Activation Function - Sigmoid

### 🔹 The Sigmoid Function

**Formula:**

```
σ(z) = 1 / (1 + e^(-z)), where z = w^T x + b
```

- **Input:** Linear combination of features (z)
- **Output:** Always a number in range [0, 1]
- **Interpretation:** This number represents probability

### 🔹 Sigmoid Properties

- **Symmetric** around z = 0
- **Large positive z** → output ≈ 1
- **Large negative z** → output ≈ 0
- **Smooth and differentiable** → good for gradient descent
- **S-shaped curve** → natural for binary decisions

📌 **Analogy:** Think of sigmoid as a "soft switch" that gradually transitions from OFF (0) to ON (1) instead of a hard binary switch.

### ✅ Quick Check:

If we get σ(z) = 0.9, how should we interpret this result?

---

## 3. Theory: Loss Function - Cross-Entropy

### 🔹 Why Not Use MSE?

MSE works poorly for classification because:

- **Non-convex** when combined with sigmoid → multiple local minima
- **Slow learning** when model is very wrong
- **Not designed** for probability distributions

### 🔹 Cross-Entropy Loss

**Formula:**

```
L(y, ŷ) = -(1/m) × Σᵢ₌₁ᵐ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]
```

**Behavior:**

- **Correct and confident prediction** → loss ≈ 0
- **Wrong and confident prediction** → loss explodes

### 🔹 Intuitive Examples

| True Label | Prediction | Loss       | Interpretation              |
| ---------- | ---------- | ---------- | --------------------------- |
| 1          | 0.99       | Very small | Good: correct and confident |
| 1          | 0.51       | Medium     | Okay: correct but uncertain |
| 1          | 0.01       | Very large | Bad: wrong and confident    |

📌 **Key Insight:** Cross-entropy severely punishes "confident mistakes" — exactly what we want!

### ✅ Quick Check:

Why can't we use MSE instead of Cross-Entropy for logistic regression?

---

## 4. Theory: Gradient Descent Optimization

### 🔹 Parameter Updates

We update parameters using gradients of the loss function:

**Update rules:**

```
w := w - η × (∂L/∂w)
b := b - η × (∂L/∂b)
```

**Gradients:**

```
∂L/∂w = (1/m) × Σᵢ₌₁ᵐ (ŷᵢ - yᵢ) × xᵢ
∂L/∂b = (1/m) × Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)
```

### 🔹 Similarity to Linear Regression

The gradient formulas look almost identical to linear regression! The key difference is:

- **Linear regression:** ŷ comes from w^T x + b directly
- **Logistic regression:** ŷ comes from σ(w^T x + b)

The magic happens through the sigmoid function and cross-entropy loss combination.

### ✅ Quick Check:

Why is the difference (ŷᵢ - yᵢ) important for weight updates?

---

## 5. Analogies for Understanding

- **Sigmoid Function:** "A smooth translator that converts any number into a probability"
- **Cross-Entropy:** "A harsh judge that punishes confident mistakes much more than hesitant ones"
- **Decision Boundary:** "Drawing a line (or curve) that separates cats from dogs"

### 🔹 The Confidence Penalty Analogy

Imagine a student taking a test:

- **Correct answer with 99% confidence** → small penalty
- **Wrong answer with 99% confidence** → huge penalty
- **Wrong answer with 51% confidence** → moderate penalty

This encourages the model to be humble when uncertain!

### ✅ Quick Check:

Why should "confident errors" be punished more severely than "hesitant errors"?

---

## 6. Python Practice

### 6.1 Core Functions

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# 1. Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Cross-entropy loss function
def compute_loss(y, y_hat):
    m = len(y)
    return -(1/m) * np.sum(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
```

### 6.2 Custom Logistic Regression Implementation

```python
class MyLogisticRegression:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr            # Learning rate
        self.n_iters = n_iters  # Number of iterations

    def fit(self, X, y):
        m, n = X.shape
        self.w = np.zeros(n)    # Initialize weights
        self.b = 0              # Initialize bias

        for _ in range(self.n_iters):
            # Forward pass
            z = np.dot(X, self.w) + self.b
            y_hat = sigmoid(z)

            # Compute gradients
            dw = (1/m) * np.dot(X.T, (y_hat - y))
            db = (1/m) * np.sum(y_hat - y)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
```

### 6.3 Training and Comparison

```python
# Generate synthetic classification data
X, y = make_classification(n_samples=200, n_features=2,
                          n_classes=2, n_redundant=0,
                          n_informative=2, random_state=42)

# Train our custom model
model = MyLogisticRegression(lr=0.1, n_iters=1000)
model.fit(X, y)
y_pred_custom = model.predict(X)

# Compare with sklearn
clf = LogisticRegression()
clf.fit(X, y)
y_pred_sklearn = clf.predict(X)

# Calculate accuracies
accuracy_custom = np.mean(y_pred_custom == y)
accuracy_sklearn = np.mean(y_pred_sklearn == y)

print(f"Custom Model Accuracy: {accuracy_custom:.3f}")
print(f"Sklearn Model Accuracy: {accuracy_sklearn:.3f}")
```

### 6.4 Visualization

```python
# Create decision boundary visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.7)

# Plot decision boundary
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2_boundary = -(model.w[0] * x1_range + model.b) / model.w[1]

plt.plot(x1_range, x2_boundary, 'k-', linewidth=2,
         label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Classification")
plt.legend()
plt.show()
```

📌 Here we implement logistic regression from scratch and compare it with sklearn's optimized version to understand the underlying mechanics.

### ✅ Quick Check:

How do the weights from our custom implementation compare to sklearn's?

---

## 7. Additional Explanations

### 🔹 Key Differences: Linear vs Logistic Regression

| Aspect            | Linear Regression | Logistic Regression |
| ----------------- | ----------------- | ------------------- |
| **Output**        | Continuous values | Probabilities [0,1] |
| **Activation**    | None (linear)     | Sigmoid             |
| **Loss Function** | MSE               | Cross-Entropy       |
| **Use Case**      | Regression        | Classification      |
| **Decision**      | Direct prediction | Threshold-based     |

### 🔹 Why Logistic Regression Works

The combination of sigmoid activation and cross-entropy loss creates a **convex optimization problem** → guaranteed to find global minimum with gradient descent.

---

## 8. Understanding Challenge

### 🎤 Your Tasks:

1. Why can't we use MSE loss function with logistic regression?
2. What's the purpose of the sigmoid function when we could use linear equations?
3. What does ŷ = 0.7 mean for a specific example?
4. Why does Cross-Entropy punish "confident mistakes" more severely?
5. What are the similarities and differences between linear and logistic regression?

## Key Takeaways

- **Logistic Regression** solves binary classification by predicting probabilities
- **Sigmoid Function** maps any real number to probability range [0, 1]
- **Cross-Entropy Loss** severely punishes confident wrong predictions
- **Gradient Descent** works similarly to linear regression but with different activation and loss
- **Decision Boundary** separates classes in feature space
- **Foundation** for understanding neural networks and deep learning

_Happy Learning! 🚀_
