# 📘 LESSON 5: REGULARIZATION

## 1. Theory: Why Do We Need Regularization?

### 🔹 Overfitting (Overtraining)

The model fits the training data too closely.

Everything looks perfect on train, but on test → large error.

### 🔹 The Cause

The loss function is minimized so that the model even learns noise.

### 🔹 The Idea of Regularization

Add a penalty for model complexity (weights that are too large).

The simpler the model (smaller weights) → the better it generalizes.

📌 **Analogy:** Like a student who doesn't just memorize answers to specific questions, but understands the essence.

### ✅ Quick Check:

Why might a model with very small training error be bad on test data?

---

## 2. L2 Regularization (Ridge)

**Formula:**

```
L_total = L_data + λ × Σ(w_i²)
```

Where:

- `L_data` → regular loss (MSE)
- `λ` → regularization strength
- `Σ(w_i²)` → sum of squared weights

### 🔹 Effect:

Weights decrease uniformly.

### 🔹 Geometry:

Constraint in the form of a sphere.

📌 **Analogy:** Wealth tax — the bigger the weight, the stronger the penalty.

### ✅ Quick Check:

If λ → ∞, what happens to the weights?

---

## 3. L1 Regularization (Lasso)

**Formula:**

```
L_total = L_data + λ × Σ|w_i|
```

Instead of squares — absolute values.

**Property:** Many weights become exactly zero.

Used for feature selection.

### 🔹 Geometry:

Diamond-shaped constraints.

### 🔹 Effect:

The model automatically selects important features.

📌 **Analogy:** House cleaning — you throw away unnecessary items, keep only what you need.

### ✅ Quick Check:

How does L1 differ from L2 in its effect on weights?

---

## 4. Elastic Net

**Formula:**

```
L_total = L_data + λ₁ × Σ|w_i| + λ₂ × Σ(w_i²)
```

Combination of L1 + L2.

**Balance:** Both feature selection (L1) and stability (L2).

📌 **Analogy:** Both tidying up and not letting "wealthy" weights grow too large.

---

## 5. Dropout (in Neural Networks)

During training, we randomly "turn off" neurons.

The model cannot rely on one neuron, learns to use different combinations.

```
p = P(neuron is active)
```

📌 **Analogy:** Sports team — each time some players sit on the bench, the rest must cope.

### ✅ Quick Check:

Why is Dropout applied only during training, not during inference?

---

## 6. Analogies for Understanding

- **L2:** Wealth tax (penalty for large weights)
- **L1:** Cleaning out unnecessary things (keeping only what's needed)
- **Elastic Net:** Both balance and cleaning
- **Dropout:** Team play where players learn to replace each other

---

## 7. Python Practice

We'll create data with linear dependence + noise, then compare models:

- Without regularization
- Ridge (L2)
- Lasso (L1)
- Elastic Net

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Synthetic data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3*X.squeeze() + 5 + np.random.randn(100) * 2

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "Linear": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = {}
plt.figure(figsize=(10,6))
plt.scatter(X, y, color="gray", alpha=0.5, label="Data")

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = (model.coef_[0], model.intercept_, mse)
    plt.plot(X_test, y_pred, label=f"{name} (MSE={mse:.2f})")

plt.legend()
plt.title("Regularization: Linear vs Ridge vs Lasso vs ElasticNet")
plt.show()

print(results)
```

📌 **What we'll see:**

- **LinearRegression** → large weights, risk of overfitting
- **Ridge** → smaller weights, more stable model
- **Lasso** → some weights zeroed out
- **Elastic Net** → compromise

---

## 8. Additional Explanations

### 🔹 Gradients:

**L2:** `∇L_total = ∇L_data + 2λw_i`

**L1:** `∇L_total = ∇L_data + λ × sign(w_i)`

### 🔹 Bayesian Interpretation:

- **L2** = Gaussian prior (normal distribution on weights)
- **L1** = Laplace prior (with sharp peak at zero)

---

## 9. Understanding Challenge

### 🎤 Your Tasks:

1. Explain the difference between L1 and L2 regularization
2. What happens to the model if λ = 0?
3. Why can L1 zero out weights while L2 cannot?
4. Why is Dropout needed in neural networks?
5. When should you use Elastic Net instead of L1 or L2?

---

## Key Takeaways

- **Regularization** prevents overfitting by penalizing model complexity
- **L2 (Ridge)** shrinks weights uniformly, provides stability
- **L1 (Lasso)** performs feature selection by zeroing out weights
- **Elastic Net** combines benefits of both L1 and L2
- **Dropout** prevents neural networks from over-relying on specific neurons
- **Goal:** Create models that generalize well to unseen data

_Happy Learning! 🚀_
