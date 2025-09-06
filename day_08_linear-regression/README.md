# 📘 LESSON 8: LINEAR REGRESSION

## 1. Theory: What is Linear Regression?

### 🔹 Definition

We want to predict a target variable y as a linear combination of features:

**Prediction formula:**

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ
```

Where:

- `ŷ` → model prediction
- `xᵢ` → features (input variables)
- `wᵢ` → weights (coefficients)
- `w₀` → bias term (intercept)

**Learning Task:** Find weights w such that the error between ŷ and actual y is minimized.

### 🔹 The Bias Term (w₀)

The bias term w₀ allows our line to shift up or down, not just pass through the origin. Without it, we'd be forced to fit a line that always goes through (0,0).

📌 **Analogy:** Imagine drawing the best line through data points. The slope is determined by weights w₁, w₂, etc., but you also need to position the line at the right height — that's w₀.

### ✅ Quick Check:

What does w₀ represent and why do we need it?

---

## 2. Theory: Loss Function

### 🔹 Mean Squared Error (MSE)

We measure how well our model fits the data using MSE:

**Formula:**

```
MSE = (1/m) × Σᵢ₌₁ᵐ (ŷᵢ - yᵢ)²
```

Where:

- `m` → number of examples
- `ŷᵢ` → model prediction for example i
- `yᵢ` → actual value for example i

### 🔹 Why Square the Error?

- **Penalizes large errors** more than small ones
- **Always positive** → errors don't cancel out
- **Mathematically convenient** → smooth, differentiable function
- **Geometric interpretation** → minimizes perpendicular distance to the line

### ✅ Quick Check:

Why do we square the error instead of using absolute values?

---

## 3. Theory: Gradient Descent Solution

### 🔹 Weight Updates

We iteratively adjust weights to minimize MSE:

**Update rule:**

```
wⱼ := wⱼ - η × (∂MSE/∂wⱼ)
```

**Partial derivative:**

```
∂MSE/∂wⱼ = (2/m) × Σᵢ₌₁ᵐ (ŷᵢ - yᵢ) × xᵢⱼ
```

Where:

- `η` → learning rate (step size)
- We gradually shift w to reduce error

### 🔹 Learning Rate Impact

- **Too small** → very slow convergence, many iterations needed
- **Optimal** → steady progress toward minimum
- **Too large** → algorithm will overshoot and oscillate, may not converge

📌 **Analogy:** Like adjusting the volume on a radio — small steps get you closer to the perfect level, big jumps make you overshoot constantly.

### ✅ Quick Check:

What happens if we choose a learning rate that's too large?

---

## 4. Theory: Normal Equation Solution

### 🔹 Analytical Solution

Instead of iterative optimization, we can solve directly:

**Formula:**

```
ŵ = (XᵀX)⁻¹Xᵀy
```

Where:

- `X` → feature matrix (m × n)
- `y` → target vector
- `ŵ` → optimal weights

### 🔹 Comparison: Gradient Descent vs Normal Equation

| Method               | Speed               | Scalability         | Accuracy             |
| -------------------- | ------------------- | ------------------- | -------------------- |
| **Normal Equation**  | Fast for small data | Slow for large data | Exact solution       |
| **Gradient Descent** | Iterative           | Scales well         | Approximate solution |

**When Normal Equation fails:** When XᵀX is not invertible (singular matrix) — happens with perfect multicollinearity.

### ✅ Quick Check:

In what case might the normal equation not work?

---

## 5. Analogies for Understanding

- **Linear Regression:** "Stretching a rubber band through data points to find the best fit"
- **MSE:** "Average squared distance from points to our line"
- **Gradient Descent:** "Rolling a ball downhill to find the lowest point, adjusting our line bit by bit"
- **Normal Equation:** "Using math to calculate the exact best position for our line in one step"

### 🔹 Mountain Climbing Analogy

Gradient Descent is like walking down a mountain with a blindfold:

- **Gradient** → tells you the steepest downhill direction
- **Learning Rate** → how big steps you take
- **Convergence** → reaching the valley (minimum error)

### ✅ Quick Check:

How is gradient descent similar to climbing down a mountain?

---

## 6. Python Practice

### 6.1 Synthetic Data Generation

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate data: y = 3x + noise
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 3 * X + 4 + np.random.randn(100, 1)

# Add bias column
X_b = np.c_[np.ones((100, 1)), X]
```

### 6.2 Gradient Descent Implementation

```python
def mse(y, y_pred):
    return np.mean((y - y_pred) ** 2)

def gradient_descent(X, y, lr=0.1, n_iters=1000):
    m, n = X.shape
    w = np.random.randn(n, 1)
    losses = []

    for _ in range(n_iters):
        y_pred = X.dot(w)
        error = y_pred - y
        grad = (2/m) * X.T.dot(error)
        w -= lr * grad
        losses.append(mse(y, y_pred))
    return w, losses

w_gd, losses = gradient_descent(X_b, y)
print("GD Weights:", w_gd.ravel())

plt.plot(losses)
plt.title("Loss vs Iterations (Gradient Descent)")
plt.xlabel("Iterations")
plt.ylabel("MSE Loss")
plt.show()
```

### 6.3 Normal Equation Implementation

```python
w_normal = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print("Normal Equation Weights:", w_normal.ravel())
```

### 6.4 Real Dataset Example

```python
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
boston = load_boston()
X, y = boston.data, boston.target

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# sklearn implementation
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Train MSE:", mean_squared_error(y_train, y_pred_train))
print("Test MSE:", mean_squared_error(y_test, y_pred_test))
```

📌 Here we compare manual implementation with sklearn's optimized version and see how they perform on real housing data.

### ✅ Quick Check:

Whose weights will be closer to the "truth" — gradient descent or normal equation?

---

## 7. Additional Explanations

- **Linear Regression** assumes a linear relationship between features and target
- **MSE** provides a smooth, differentiable objective function for optimization
- **Gradient Descent** is the foundation for training neural networks
- **Normal Equation** gives exact solution but doesn't scale to big data

### 🔹 Why Linear Regression is Fundamental

Linear regression is the "hello world" of machine learning because:

- **Simple to understand** and implement
- **Fast to train** and predict
- **Baseline model** for comparison
- **Foundation** for more complex algorithms
- **Interpretable** coefficients show feature importance

---

## 8. Understanding Challenge

### 🎤 Your Tasks:

1. What's the difference between gradient descent and normal equation approaches?
2. Why is linear regression considered a baseline ML model?
3. What happens if features are highly correlated (multicollinearity)?
4. Why do we use MSE instead of MAE (Mean Absolute Error)?
5. When does the normal equation become impractical?

---

## Key Takeaways

- **Linear Regression** finds the best linear relationship between features and target
- **MSE Loss Function** measures prediction quality by penalizing errors quadratically
- **Gradient Descent** iteratively optimizes weights, scales to large datasets
- **Normal Equation** provides exact solution but limited to smaller problems
- **Foundation** for understanding more complex ML algorithms
- **Interpretability** makes it valuable for understanding feature relationships

_Happy Learning! 🚀_
