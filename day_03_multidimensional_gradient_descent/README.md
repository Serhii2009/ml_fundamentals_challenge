# 📘 LESSON 3: MULTIDIMENSIONAL GRADIENT DESCENT

## 1. Theory: How Gradient Descent Generalizes to Multidimensional Cases

### 🔹 One-Dimensional Gradient Descent Review

In the one-dimensional case, we have a loss function `L(w)` and one parameter `w`.

We calculate the derivative `dL/dw` to understand which direction to move to minimize loss.

**Update formula:**

```
w(t+1) = w(t) - η × dL/dw(t)
```

where `η` is the learning rate.

### 🔹 Generalization to Multidimensional Case

Now we have multiple parameters `w = [w₀, w₁, ..., wₙ]ᵀ`.

Loss is a function of many variables: `L(w)`.

To understand how to move in parameter space, we use the **gradient**:

```
∇w L(w) = [∂L/∂w₀, ∂L/∂w₁, ..., ∂L/∂wₙ]ᵀ
```

**Parameter update formula becomes vectorized:**

```
w(t+1) = w(t) - η × ∇w L(w(t))
```

📌 **Breaking down the elements:**

| Symbol       | Meaning                                                                               |
| ------------ | ------------------------------------------------------------------------------------- |
| `w`          | Vector of model parameters                                                            |
| `η`          | Learning rate - speed of movement along gradient direction                            |
| `∇w L(w(t))` | Gradient: vector of all partial derivatives, points toward steepest function increase |

### ⚡ Geometric Interpretation:

- Gradient shows where function `L(w)` grows fastest
- We go in the opposite direction to decrease loss
- In multidimensional case, gradient is an arrow in n-dimensional space pointing "where the mountain climbs steepest"

### ✅ Quick Understanding Check:

If the gradient equals zero, what does this mean for the loss function?

---

## 2. Analogy for Understanding

### 💡 "Walking in Mountains with Blindfolded Eyes"

- You can only feel the steepness at your current location (gradient)
- To go downhill → walk in the opposite direction
- When there are multiple axes (X, Y, Z...) → gradient becomes a vector showing how to move simultaneously in all directions
- Learning rate `η` is your "step size" at each movement

### ✅ Understanding Check:

How will your path change if the step size is too large? If too small?

---

## 3. Practice: Gradient Descent for Linear Regression

### Linear regression with two parameters:

```
ŷ = w₁x + w₀
```

### Loss Function (MSE):

```
L = (1/N) × Σ(i=1 to N) (yᵢ - ŷᵢ)²
```

### Gradients:

```
∂L/∂w₁ = -(2/N) × Σ(i=1 to N) xᵢ(yᵢ - ŷᵢ)
∂L/∂w₀ = -(2/N) × Σ(i=1 to N) (yᵢ - ŷᵢ)
```

### Parameter Updates:

```
w₁ ← w₁ - η × ∂L/∂w₁
w₀ ← w₀ - η × ∂L/∂w₀
```

---

## 4. Python Implementation

```python
import numpy as np

# Data
x = np.array([1, 2, 3, 4])
y = np.array([2, 3, 5, 7])

# Parameters
w0, w1 = 0.0, 0.0
eta = 0.01
epochs = 1000
N = len(x)

# Gradient descent
for epoch in range(epochs):
    y_pred = w1 * x + w0
    dw1 = (-2/N) * np.sum(x * (y - y_pred))
    dw0 = (-2/N) * np.sum(y - y_pred)
    w1 -= eta * dw1
    w0 -= eta * dw0

    if epoch % 200 == 0:
        loss = np.mean((y - y_pred)**2)
        print(f"Epoch {epoch}: Loss = {loss:.4f}, w0 = {w0:.4f}, w1 = {w1:.4f}")

print(f"Final parameters: w0 = {w0:.4f}, w1 = {w1:.4f}")
```

⚡ **Key Insight:**

- With each step, parameters `w₀, w₁` approach optimal values
- One gradient step updates all parameters simultaneously

### ✅ Check:

Why is updating all parameters simultaneously more important than updating them one by one?

---

## 5. Manual Calculation Exercise

### Given Data:

```
(x₁, y₁) = (1, 2), (x₂, y₂) = (2, 3)
```

### Initial Parameters:

```
w₀ = 0, w₁ = 0, η = 0.1
```

### Step 1: Calculate Predictions

```
ŷ₁ = 0, ŷ₂ = 0
```

### Step 2: Calculate Gradients

```
dw₁ = -(2/2)(1×(2-0) + 2×(3-0)) = -(2+6) = -8
dw₀ = -(2/2)((2-0) + (3-0)) = -(2+3) = -5
```

### Step 3: Update Parameters

```
w₁ ← 0 - 0.1×(-8) = 0 + 0.8 = 0.8
w₀ ← 0 - 0.1×(-5) = 0 + 0.5 = 0.5
```

✅ **Result:** After one gradient step, parameters became `w₀ = 0.5, w₁ = 0.8`

---

## 6. Additional Explanations

### 🔹 Key Concepts:

- **Gradient vector** is a direction pointer for steepest loss increase
- **Simultaneous parameter update** is crucial for correct minimization in multidimensional space
- In neural networks, each weight and bias is a component of vector `w`
- **One gradient step** means moving in the direction opposite to gradient for all weights simultaneously

---

## 7. Understanding Challenge

### 🎤 Your Task:

Explain in your own words:
👉 What is multidimensional gradient descent?
👉 How does it differ from one-dimensional gradient descent?
👉 Why does it work for models with many parameters?

(Use the blindfolded mountain walking analogy 🏔️)

---

## Key Takeaways

- **Multidimensional Gradient Descent** optimizes multiple parameters simultaneously
- **Gradient Vector** points toward steepest increase, we move opposite direction
- **Parameter Updates** happen simultaneously for all weights and biases
- **Linear Regression** is a perfect example with two parameters (slope and intercept)
- **Manual Calculation** helps understand the mathematical foundation

_Happy Learning! 🚀_
