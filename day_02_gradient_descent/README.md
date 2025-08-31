# 📘 LESSON 2: GRADIENT DESCENT

## 1. Theory: What is Gradient Descent?

### 🔹 Definition

Gradient Descent is an optimization method used to minimize the loss function. The idea: we change model parameters step by step in the direction that reduces error until we reach the minimum.

📌 **Formula (for one variable):**

```
x(t+1) = x(t) - η × dL/dx(x(t))
```

🔎 **Breaking it down:**

- `x(t)` → current parameter value (e.g., model weight)
- `η` → learning rate (step size, speed of movement)
- `dL/dx(x(t))` → derivative of loss function (gradient) at point x(t)
- The "−" sign → we go against the gradient because it shows the direction of function increase

### 📌 Why do we go against the gradient?

- Gradient shows where the function grows fastest
- To minimize loss, we need to go in the opposite direction

⚡ **Result:** we move parameters so that the loss value becomes smaller and smaller

### ✅ Quick Understanding Check:

If the gradient is positive (dL/dx > 0), will the new x become smaller or larger?

---

## 2. Analogy for Understanding

### 🎢 Imagine a hill:

- Height = loss function value
- You = a ball rolling down the slope
- Goal = roll to the lowest point (minimum)

📌 **Elements:**

- Loss → mountain height
- Gradient → slope steepness
- Learning rate → step size (rolling distance)

### ⚡ What happens if learning rate is wrong?

- **Too small** → tiny steps, takes very long to reach the bottom
- **Too large** → ball might jump over the minimum and start bouncing back and forth, never stopping

### ✅ Check:

What happens if learning rate = 0?

---

## 3. Python Practice

Let's try a simple function:

```
f(x) = (x - 3)²
```

The minimum of this function → at x = 3, where f(x) = 0

```python
import numpy as np
import matplotlib.pyplot as plt

# Function and its derivative
def f(x):
    return (x - 3)**2

def df(x):
    return 2 * (x - 3)

# Gradient descent
x = -5           # starting point
eta = 0.1        # learning rate
history = [x]    # save points

for _ in range(20):
    grad = df(x)
    x = x - eta * grad
    history.append(x)

print("Approaching minimum:", x)

# Visualization
xs = np.linspace(-6, 6, 200)
ys = f(xs)

plt.plot(xs, ys, label="f(x) = (x-3)^2")
plt.scatter(history, [f(h) for h in history], color="red")
plt.title("Gradient Descent")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.legend()
plt.show()
```

📊 The graph will show red points demonstrating how we move step by step toward x = 3

---

## 4. Additional Explanations

### 🔹 Convergence

The process of the algorithm approaching the minimum. If learning rate is good → we smoothly move downward.

### 🔹 Local vs Global Minimum

- **Local minimum** → a point that seems low, but there are lower points elsewhere
- **Global minimum** → the lowest point (true minimum)

⚡ For function f(x) = (x-3)², the minimum is global and unique. But in neural networks, the loss function is complex with many local minima.

### 🔹 Connection with Loss Function

Yesterday we talked about loss showing error. Today → we learn to find parameters that minimize loss.

**Mathematically:**

```
w(t+1) = w(t) - η × ∂L/∂w
```

(same concept, but now parameters can be vectors)

### ✅ Check:

If learning rate is too large, what will happen to the loss?

---

## 5. Understanding Challenge

### 🎤 Your Task:

Explain in your own words to a 10-year-old:
👉 What is gradient descent, why do we need it, and what does learning rate mean?

(You can use the example with a ball and hill 🎢)

---

## Key Takeaways

- **Gradient Descent** finds the best parameters by minimizing loss
- **Gradient** shows the steepest direction of function increase
- **Learning Rate** controls how big steps we take toward the minimum
- **Goal**: Move against the gradient to reach the lowest loss
- **Visualization**: Think of a ball rolling down to the bottom of a hill

_Happy Learning! 🚀_
