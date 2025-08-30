# 📘 LESSON 1: LOSS FUNCTION

## 1. Theory: What is a Loss Function?

### 🔹 Definition

A Loss Function is a mathematical measure of how much the model's predictions differ from the correct answers.

📌 If predictions are perfect → loss = 0  
📌 The further predictions are from truth → the higher the loss value

### Formulas

- **For a single object (one example):**
  ```
  L(y, ŷ) = difference between y and ŷ
  ```
- **For the entire dataset (N examples):**
  ```
  Loss = (1/N) × Σ(i=1 to N) L(yi, ŷi)
  ```

### Why is loss always ≥ 0?

Because:

- Most functions (MSE, Cross-Entropy) use squared differences or logarithms of probabilities
- Squares are always ≥ 0
- Negative logarithm of probability is also always ≥ 0, since log(p) ≤ 0 when 0 ≤ p ≤ 1

⚡ **Therefore:** loss cannot be negative → it always shows the "magnitude of error"

### ✅ Quick Understanding Check:

How would you explain the difference between loss for one example vs. loss for the entire dataset?

---

## 2. Two Key Types of Loss Functions

### A. Mean Squared Error (MSE)

**Formula:**

```
MSE = (1/N) × Σ(i=1 to N) (yi - ŷi)²
```

🔎 **Breaking down the elements:**

- `yi` → true value (target)
- `ŷi` → model prediction
- `(yi - ŷi)²` → squared error for each example
- `1/N` → averaging across all examples

### 📊 Example (for one case):

Let's say:

- True value: y = 5
- Prediction: ŷ = 7

Then:

```
L = (5 - 7)² = (-2)² = 4
```

### 🔥 Why are large errors penalized more heavily?

Because the error is squared:

- Error of 2 → penalty of 4
- Error of 10 → penalty of 100

⚡ The model "fears" making large mistakes

### ✅ Check:

Why do errors of 10 and 2 differ not by a factor of 5, but by a factor of 25 in MSE?

---

### B. Cross-Entropy

Used for classification tasks.

**Formula (binary classification):**

```
L(y, ŷ) = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

🔎 **Breaking it down:**

- `y ∈ {0,1}` → correct class
- `ŷ` → probability of "class 1" (e.g., 0.9 or 0.1)
- If y=1: we take `-log(ŷ)`
- If y=0: we take `-log(1-ŷ)`

### 📊 Example:

- True class: y = 1
- Model says: ŷ = 0.9

```
L = -log(0.9) ≈ 0.105
```

Now if:

- Model is confidently wrong: ŷ = 0.1

```
L = -log(0.1) ≈ 2.302
```

⚡ **Conclusion:** confident mistake → large penalty

### **Multi-class formula (softmax):**

```
L = -Σ(c=1 to C) yc × log(ŷc)
```

Where:

- `yc` = one-hot encoding (e.g., for class 3 out of 5: [0,0,1,0,0])
- `ŷc` = prediction probability for class c

### ✅ Check:

Why does Cross-Entropy penalize more heavily when the model was confident but wrong?

---

## 3. Why We Minimize Loss

📌 **Our goal in ML:** Find model parameters (weights w, bias b) that make predictions as close to truth as possible.

**Mathematically:**

```
min(w,b) Loss(y, ŷ(x; w,b))
```

We adjust model parameters to minimize loss.

### 🎯 Analogy:

Imagine a dartboard:

- Center → perfect predictions (loss=0)
- Edge hits → errors
- Our task → get closer to the center with each step (training iteration)

### ✅ Check:

If loss decreases with each training iteration — what does this mean for our model?

---

## 4. Python Practice

```python
import numpy as np

# MSE
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Cross-Entropy (binary)
def binary_cross_entropy(y_true, y_pred):
    # To avoid log(0), add small epsilon
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- Examples ---
y_true_reg = np.array([3, -0.5, 2, 7])
y_pred_reg = np.array([2.5, 0.0, 2, 8])

print("MSE:", mse(y_true_reg, y_pred_reg))

y_true_cls = np.array([1, 0, 1, 1])
y_pred_cls = np.array([0.9, 0.1, 0.8, 0.4])

print("Binary Cross-Entropy:", binary_cross_entropy(y_true_cls, y_pred_cls))
```

---

## 5. Understanding Challenge

### 🎤 Your Task:

Explain to a 10-year-old what a Loss Function is and why we minimize it.

(Imagine the child loves playing darts 🎯)

---

## Key Takeaways

- **Loss Function** measures how wrong our predictions are
- **MSE** is great for regression problems (predicting numbers)
- **Cross-Entropy** is perfect for classification problems (predicting categories)
- **Goal**: Always minimize loss to improve model accuracy
- **Visualization**: Think of it as hitting the bullseye on a dartboard

_Happy Learning! 🚀_
