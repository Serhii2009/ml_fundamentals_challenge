import numpy as np                                                                                 # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                    # Import Matplotlib for plotting

print("="*80)                                                                                      # Separator line
print("TRAIN-TEST VS LAMBDA ANALYSIS")                                                             # Header
print("="*80)                                                                                      # Separator line

# ===== DATA CREATION =====
np.random.seed(42)                                                                                 # Set seed for reproducibility
n_samples = 100                                                                                    # Number of data points
X = np.random.rand(n_samples, 1) * 10                                                              # Single feature scaled 0-10
true_w = 3.5                                                                                       # True slope
true_b = 2.0                                                                                       # True intercept
y = true_w * X + true_b + np.random.randn(n_samples, 1) * 2                                        # Linear relation + noise

# ===== INITIALIZATION =====               
w_init = 0.0                                                                                       # Initial weight
b_init = 0.0                                                                                       # Initial bias
lr = 0.01                                                                                          # Learning rate
epochs = 100                                                                                       # Number of iterations
lambdas = [0, 0.01, 0.1, 1, 10]                                                                    # List of lambda values to test
results = []                                                                                       # Store results for each lambda

# ===== LAMBDA EXPERIMENT =====
for lambda_reg in lambdas:                                                                         # Loop over lambda values
    w = np.array([[w_init]])                                                                       # Reset weight for each lambda
    b = b_init                                                                                     # Reset bias for each lambda
    for epoch in range(epochs):                                                                    # Loop over epochs
        y_pred = X @ w + b                                                                         # Prediction
        error = y_pred - y                                                                         # Compute error
        grad_w = (2/n_samples) * X.T @ error + 2*lambda_reg*w                                      # Gradient with L2 penalty
        grad_b = (2/n_samples) * np.sum(error)                                                     # Gradient for bias
        w -= lr * grad_w                                                                           # Update weight
        b -= lr * grad_b                                                                           # Update bias
    results.append((lambda_reg, w[0][0], b))                                                       # Store final weight and bias

# ===== PRINT RESULTS =====
print("Lambda Analysis Results:") # Header
for lam, w_val, b_val in results: # Loop over results
    print(f"Lambda={lam:6.2f}, Weight={w_val:.4f}, Bias={b_val:.4f}") # Print lambda, weight, bias

# ===== VISUALIZATION =====
for lam, w_val, b_val in results: # Loop over results
    plt.scatter(lam, w_val, color="red") # Weight vs lambda
    plt.scatter(lam, b_val, color="blue") # Bias vs lambda

plt.xlabel("Lambda (L2 Strength)") # X-axis label
plt.ylabel("Final Weight / Bias") # Y-axis label
plt.title("Effect of Lambda on Weight and Bias") # Plot title
plt.show() # Display plot