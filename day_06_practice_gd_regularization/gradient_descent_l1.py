import numpy as np                                                                               # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                  # Import Matplotlib for plotting

print("="*80)                                                                                    # Separator line
print("GRADIENT DESCENT WITH L1 (LASSO) REGULARIZATION")                                         # Header
print("="*80)                                                                                    # Separator line

# ===== DATA CREATION =====
np.random.seed(42)                                                                               # Set seed for reproducibility
n_samples = 100                                                                                  # Number of data points
X = np.random.rand(n_samples, 1) * 10                                                            # Single feature scaled 0-10
true_w = 3.5                                                                                     # True slope
true_b = 2.0                                                                                     # True intercept
y = true_w * X + true_b + np.random.randn(n_samples, 1) * 2                                      # Linear relation + noise

# ===== INITIALIZATION =====
w = np.random.randn(1, 1)                                                                        # Random initial weight
b = 0.0                                                                                          # Bias initialized to zero
lr = 0.01                                                                                        # Learning rate
epochs = 100                                                                                     # Number of iterations
lambda_reg = 0.1                                                                                 # L1 regularization strength

# ===== GRADIENT DESCENT WITH L1 =====
for epoch in range(epochs):                                                                      # Loop over epochs
    y_pred = X @ w + b                                                                           # Prediction
    error = y_pred - y                                                                           # Compute error
    grad_w = (2/n_samples) * X.T @ error + lambda_reg * np.sign(w)                               # Gradient with L1 penalty
    grad_b = (2/n_samples) * np.sum(error)                                                       # Gradient for bias
    w -= lr * grad_w                                                                             # Update weight
    b -= lr * grad_b                                                                             # Update bias
    if epoch % 10 == 0:                                                                          # Every 10 epochs
        loss = np.mean(error**2) + lambda_reg * np.sum(np.abs(w))                                # Compute MSE + L1 penalty
        print(f"Epoch {epoch:3d}, Loss: {loss:.4f}, w: {w[0][0]:.4f}, b: {b:.4f}")               # Print progress

# ===== VISUALIZATION =====
plt.scatter(X, y, color="gray", alpha=0.5, label="Data") # Plot data points
plt.plot(X, X @ w + b, color="green", label="L1 Prediction") # Plot prediction line
plt.xlabel("X") # X-axis label
plt.ylabel("y") # Y-axis label
plt.title("Gradient Descent with L1 Regularization") # Plot title
plt.legend() # Show legend
plt.show() # Display plot
