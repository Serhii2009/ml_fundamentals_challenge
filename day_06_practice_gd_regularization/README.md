# 📘 LESSON 6: GRADIENT DESCENT + REGULARIZATION

## 1. Theory: Gradient Descent with Regularization

### 🔹 Gradient Descent Review

We move in the direction of the negative gradient:

```
w_{t+1} = w_t - η * ∂L/∂w
```

Where η is the learning rate.

### 🔹 Regularization Types

**L2 (Ridge) Regularization:**

```
L_total = L_data + λ * Σ(w_i²)
```

- Adds `+2λw` to the gradient
- Shrinks weights uniformly but doesn't zero them out
- Creates smooth, stable solutions

**L1 (Lasso) Regularization:**

```
L_total = L_data + λ * Σ|w_i|
```

- Adds `λ * sign(w)` to the gradient
- Can zero out weights completely
- Performs automatic feature selection

**Elastic Net:**

```
L_total = L_data + λ * (α * Σ|w_i| + (1-α) * Σ(w_i²))
```

- Combines both L1 and L2 regularization
- Balances feature selection and stability

📌 **Key Idea:** Regularization = penalty for large/unnecessary weights → simpler model → better generalization

### ✅ Mini Question:

What happens when λ = 0? The model reduces to standard gradient descent with no regularization.

---

## 2. File Structure

```
day_06_gradient_descent_regularization/
├── README.md
├── gradient_descent_basic.py          # Basic gradient descent implementation
├── gradient_descent_l2.py             # L2 (Ridge) regularization
├── gradient_descent_l1.py             # L1 (Lasso) regularization
├── gradient_descent_elastic.py        # Elastic Net regularization
└── train_test_vs_lambda.py           # Lambda parameter analysis
```

---

## 3. Practice Files Overview

### 3.1 `gradient_descent_basic.py`

- Implements standard gradient descent without regularization
- Creates synthetic linear data with noise
- Shows basic MSE loss optimization
- **Key Observation:** Weights can grow large, potential overfitting

### 3.2 `gradient_descent_l2.py`

- Adds L2 regularization penalty to gradient descent
- Demonstrates weight shrinkage effect
- **Key Observation:** Weights become smaller, more stable convergence

### 3.3 `gradient_descent_l1.py`

- Implements L1 regularization with sign function
- Shows sparse solution capabilities
- **Key Observation:** Some weights may approach zero

### 3.4 `gradient_descent_elastic.py`

- Combines L1 and L2 regularization
- Uses l1_ratio parameter to balance between penalties
- **Key Observation:** Compromise between stability and sparsity

### 3.5 `train_test_vs_lambda.py`

- Analyzes how regularization strength (λ) affects model weights
- Shows relationship between lambda and weight magnitude
- **Key Observation:** Higher λ → smaller weights → potential underfitting

---

## 4. Key Experiments & Results

### 4.1 Weight Behavior Comparison

```python
# Expected results for different regularization types:
# No Reg:     w ≈ [bias, ~4.0] (close to true slope)
# L2 λ=0.1:   w ≈ [bias, ~3.8] (slightly shrunk)
# L2 λ=10:    w ≈ [bias, ~2.0] (heavily shrunk)
# L1 λ=0.1:   w ≈ [bias, ~3.7] (may have sparse elements)
# Elastic:    w ≈ [bias, ~3.6] (balanced shrinkage)
```

### 4.2 Lambda Effects on L2 Regularization

- **λ = 0:** No regularization, weights unconstrained
- **λ = 0.1:** Mild regularization, slight weight reduction
- **λ = 1:** Moderate regularization, noticeable shrinkage
- **λ = 10:** Strong regularization, significant weight reduction

### 4.3 Bias-Variance Tradeoff

- **Low λ:** Lower bias, higher variance (potential overfitting)
- **High λ:** Higher bias, lower variance (potential underfitting)
- **Optimal λ:** Balanced bias-variance, best generalization

---

## 5. Advanced Concepts

### 5.1 Gradient Modifications

**L2 Regularization:**

```python
grad = (2/m) * X.T.dot(y_pred - y) + 2*lambda*w
```

**L1 Regularization:**

```python
grad = (2/m) * X.T.dot(y_pred - y) + lambda*np.sign(w)
```

**Elastic Net:**

```python
grad = (2/m) * X.T.dot(y_pred - y) + lambda*(l1_ratio*np.sign(w) + (1-l1_ratio)*2*w)
```

### 5.2 Dropout (Bonus Concept)

```python
def dropout(X, p=0.5):
    mask = (np.random.rand(*X.shape) > p).astype(float)
    return X * mask / (1-p)
```

- Randomly "turns off" features during training
- Prevents over-reliance on specific features
- Commonly used in neural networks

---

## 6. Practical Exercises

### Exercise 1: Parameter Sensitivity

Run `train_test_vs_lambda.py` with different lambda values:

- Try λ = [0, 0.01, 0.1, 1, 10, 100]
- Observe weight magnitude changes
- Find optimal λ that balances train/test error

### Exercise 2: Regularization Comparison

Modify the basic gradient descent to include all three regularization types:

```python
def gradient_descent_unified(X, y, reg_type="none", lam=0.0):
    # Implement unified function with regularization choice
    pass
```

### Exercise 3: Real Data Application

- Load a real dataset (e.g., Boston housing, California housing)
- Apply different regularization techniques
- Compare generalization performance

---

## 7. Understanding Check

### 🎤 Questions to Test Your Knowledge:

1. **How does regularization modify gradient descent?**
   👉 Adds penalty terms to the gradient, shrinking/zeroing weights

2. **What's the key difference between L1 and L2?**
   👉 L1 creates sparse solutions (zeros out weights), L2 only shrinks them

3. **Why does large λ cause underfitting?**
   👉 Weights become too small to capture the underlying pattern

4. **How do bias-variance tradeoff and regularization relate?**
   👉 λ increases bias but decreases variance, finding optimal balance

5. **When would you choose Elastic Net over L1 or L2?**
   👉 When you want both feature selection (L1) and stability (L2)

---

## 8. Key Takeaways

- **Regularization** prevents overfitting by penalizing model complexity
- **L2 (Ridge)** shrinks weights uniformly, providing stability
- **L1 (Lasso)** can zero out weights, performing feature selection
- **Elastic Net** combines benefits of both L1 and L2
- **Lambda (λ)** controls regularization strength: higher λ → simpler model
- **Optimal λ** balances bias-variance tradeoff for best generalization
- **Gradient descent** easily incorporates regularization through penalty terms

### 🚀 Next Steps:

- Experiment with different datasets and regularization strengths
- Try implementing coordinate descent for Lasso regression
- Explore regularization in neural networks (dropout, batch normalization)

---

## 9. Mathematical Foundations

### Regularization Gradient Derivations:

**L2 Penalty:** `∂/∂w[λ * w²] = 2λw`
**L1 Penalty:** `∂/∂w[λ * |w|] = λ * sign(w)`

### Geometric Interpretation:

- **L2:** Spherical constraint region
- **L1:** Diamond-shaped constraint region
- **Elastic Net:** Combination of both constraint shapes

_Happy Learning! 🎯_
