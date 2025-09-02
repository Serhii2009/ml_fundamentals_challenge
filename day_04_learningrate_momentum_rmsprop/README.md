# 📘 LESSON 4: LEARNING RATE, MOMENTUM AND RMSPROP

## 1. Theory: Learning Rate (Step Size)

### 🔹 Definition

In gradient descent, we search for the minimum of the loss function by moving in the direction opposite to the gradient. The speed at which we take steps is called the learning rate (denoted η).

**Update formula:**

```
w(t+1) = w(t) - η × ∇L(w(t))
```

Where:

- `w(t)` → parameters at step t
- `η` → learning rate
- `∇L(w(t))` → gradient of loss function with respect to parameters

### 🔹 Effect of η:

- **Too small** → slow convergence, may get "stuck" in local minima
- **Optimal** → fast and stable approach to minimum
- **Too large** → steps too aggressive, algorithm will "bounce" and may not converge

📌 **Analogy:** Imagine walking toward a pit with your eyes closed:

- Small step → takes very long to reach bottom
- Normal step → quickly get to the bottom
- Giant step → you'll keep jumping over the pit

### ✅ Quick Check:

What happens if η = 0?

---

## 2. Theory: Momentum (Acceleration)

### 🔹 The Idea

Regular gradient descent can oscillate, especially if the function surface has "zigzag" slopes. Momentum introduces "impulse" that helps accumulate movement direction.

**Formulas:**

```
v(t) = β × v(t-1) + (1-β) × ∇L(w(t))
w(t+1) = w(t) - η × v(t)
```

Where:

- `v(t)` → momentum (smoothed gradient)
- `β` → decay coefficient (usually 0.9)
- `η` → learning rate

### 🔹 Intuition

- **Without Momentum:** ball rolls down, bounces on every pebble
- **With Momentum:** ball gains speed and smoothly rolls down without getting stuck on small obstacles

### ✅ Quick Check:

What happens if β = 0?

---

## 3. Theory: RMSProp

### 🔹 The Problem

Functions can have different steepness in different directions. Using the same learning rate for all parameters makes optimization inefficient.

### 🔹 The Solution

RMSProp makes adaptive steps: large gradients → step decreases, small gradients → step increases.

**Formulas:**

```
s(t) = β × s(t-1) + (1-β) × (∇L(w(t)))²
w(t+1) = w(t) - (η / √(s(t) + ε)) × ∇L(w(t))
```

Where:

- `s(t)` → moving average of squared gradients
- `ε` → small number to avoid division by zero

### 🔹 Intuition

Imagine walking on different paths:

- On steep slopes → take small steps
- On gentle slopes → can take big steps

### ✅ Quick Check:

What happens if we remove ε?

---

## 4. Analogies for Understanding

- **Learning Rate:** "Step length when walking blindfolded toward a target"
- **Momentum:** "Rolling a ball down a hill — it gains speed and doesn't stop at small bumps"
- **RMSProp:** "Different roads: on steep slopes take small steps, on gentle ones — big steps"

---

## 5. Python Practice

Let's use function: `f(x,y) = (x-3)² + (y+2)²`

Minimum at point (3, -2)

```python
import numpy as np
import matplotlib.pyplot as plt

# Function and gradient
def f(x, y):
    return (x - 3)**2 + (y + 2)**2

def grad(x, y):
    return np.array([2*(x-3), 2*(y+2)])

# Parameters
eta = 0.1
epochs = 50

# ------------------ 1. Regular GD ------------------
x, y = 0.0, 0.0
trajectory_gd = [(x, y)]
for _ in range(epochs):
    g = grad(x, y)
    x, y = (x - eta*g[0], y - eta*g[1])
    trajectory_gd.append((x, y))

# ------------------ 2. Momentum ------------------
x, y = 0.0, 0.0
v = np.array([0.0, 0.0])
beta = 0.9
trajectory_momentum = [(x, y)]
for _ in range(epochs):
    g = grad(x, y)
    v = beta*v + (1-beta)*g
    x, y = (x - eta*v[0], y - eta*v[1])
    trajectory_momentum.append((x, y))

# ------------------ 3. RMSProp ------------------
x, y = 0.0, 0.0
s = np.array([0.0, 0.0])
beta = 0.9
eps = 1e-8
trajectory_rmsprop = [(x, y)]
for _ in range(epochs):
    g = grad(x, y)
    s = beta*s + (1-beta)*(g**2)
    x, y = (x - eta/np.sqrt(s[0]+eps)*g[0],
            y - eta/np.sqrt(s[1]+eps)*g[1])
    trajectory_rmsprop.append((x, y))

# ------------------ Visualization ------------------
def plot_trajectory(traj, label):
    xs, ys = zip(*traj)
    plt.plot(xs, ys, marker="o", label=label)

plt.figure(figsize=(8,6))
plot_trajectory(trajectory_gd, "GD")
plot_trajectory(trajectory_momentum, "Momentum")
plot_trajectory(trajectory_rmsprop, "RMSProp")
plt.scatter([3], [-2], color="red", marker="*", s=200, label="Minimum")
plt.legend()
plt.title("Trajectories of Optimization Methods")
plt.show()
```

📌 Here we compare three methods and see different trajectories:

- GD may "oscillate"
- Momentum smooths the path
- RMSProp adapts the steps

---

## 6. Additional Explanations

- **Regular GD:** simply moves in gradient direction
- **Momentum:** considers past movement, smooths oscillations
- **RMSProp:** adapts learning rate for different directions

### 🔹 Connection to Modern Optimizers

**Adam = Momentum + RMSProp.** It combines "acceleration" and "adaptive step."

---

## 7. Understanding Challenge

### 🎤 Your Tasks:

1. Explain in your own words the difference between regular gradient descent, Momentum, and RMSProp
2. What happens if learning rate is too large?
3. Why does Momentum help on "zigzag" functions?
4. Why does RMSProp decrease steps for large gradients and increase for small ones?

---

## Key Takeaways

- **Learning Rate** controls how big steps we take toward the minimum
- **Momentum** smooths oscillations by remembering past directions
- **RMSProp** adapts step size based on gradient magnitude
- **Goal:** Find optimal parameters faster and more stably
- **Modern approach:** Adam optimizer combines the best of both worlds

_Happy Learning! 🚀_
