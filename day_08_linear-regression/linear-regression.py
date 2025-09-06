import numpy as np                                                                  # Import NumPy library for mathematical operations
import matplotlib.pyplot as plt                                                     # Import matplotlib for plotting and visualization
from sklearn.linear_model import LinearRegression                                   # Import sklearn's Linear Regression implementation

np.random.seed(42)                                                                  # Set random seed for reproducibility
X = 2 * np.random.rand(100, 1)                                                      # Generate random X values between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)                                             # Generate y = 4 + 3*X + noise (true parameters: intercept=4, slope=3)

print(f"Generated {len(X)} data points")                                            # Print number of generated data points
print(f"True parameters: intercept = 4, slope = 3")                                 # Print true underlying parameters

def gradient_descent(X, y, lr=0.1, n_iter=1000):                                    # Define gradient descent function
    m = len(y)                                                                      # Get number of training examples
    X_b = np.c_[np.ones((m, 1)), X]                                                 # Add bias column (x0=1) to feature matrix
    theta = np.random.randn(2, 1)                                                   # Initialize random weights (intercept and slope)
    
    for i in range(n_iter):                                                         # Loop through optimization iterations
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)                             # Calculate gradients using MSE derivative
        theta -= lr * gradients                                                     # Update parameters using gradient descent rule
    
    return theta                                                                    # Return optimized parameters

theta_gd = gradient_descent(X, y)                                                   # Train linear regression using gradient descent
print(f"\nGradient Descent completed with learning rate = 0.1")                     # Print completion message

X_b = np.c_[np.ones((len(X), 1)), X]                                                # Add bias column for normal equation
theta_ne = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)                          # Calculate optimal parameters using normal equation: (X^T X)^-1 X^T y
print(f"Normal Equation solved analytically")                                       # Print completion message

lin_reg = LinearRegression()                                                        # Create sklearn LinearRegression instance
lin_reg.fit(X, y)                                                                   # Fit the model to training data
print(f"Sklearn LinearRegression fitted")                                           # Print completion message



# ===== PARAMETER COMPARISON =====
print("\n" + "="*60) # New line and separator
print("PARAMETER COMPARISON") # Parameters section header
print("="*60) # Separator line for formatting

print("Method comparison results:") # Comparison header
print(f"  - Gradient Descent θ: [{theta_gd[0][0]:.6f}, {theta_gd[1][0]:.6f}]") # Print gradient descent parameters
print(f"  - Normal Equation θ:  [{theta_ne[0][0]:.6f}, {theta_ne[1][0]:.6f}]") # Print normal equation parameters
print(f"  - Sklearn θ:          [{lin_reg.intercept_[0]:.6f}, {lin_reg.coef_[0][0]:.6f}]") # Print sklearn parameters
print(f"  - True parameters:    [4.000000, 3.000000]") # Print true underlying parameters

# ===== ERROR ANALYSIS =====
print("\n" + "="*60) # New line and separator
print("ERROR ANALYSIS") # Error analysis section header
print("="*60) # Separator line for formatting

# Calculate Mean Squared Error for each method
X_b_full = np.c_[np.ones((len(X), 1)), X] # Full feature matrix with bias
y_pred_gd_full = X_b_full.dot(theta_gd) # Full predictions using gradient descent
y_pred_ne_full = X_b_full.dot(theta_ne) # Full predictions using normal equation
y_pred_sklearn_full = lin_reg.predict(X) # Full predictions using sklearn

mse_gd = np.mean((y - y_pred_gd_full) ** 2) # Calculate MSE for gradient descent
mse_ne = np.mean((y - y_pred_ne_full) ** 2) # Calculate MSE for normal equation
mse_sklearn = np.mean((y - y_pred_sklearn_full.reshape(-1, 1)) ** 2) # Calculate MSE for sklearn

print("Mean Squared Error comparison:") # MSE comparison header
print(f"  - Gradient Descent MSE: {mse_gd:.8f}") # Print gradient descent MSE
print(f"  - Normal Equation MSE:  {mse_ne:.8f}") # Print normal equation MSE
print(f"  - Sklearn MSE:          {mse_sklearn:.8f}") # Print sklearn MSE

# ===== VISUALIZATION SETUP =====
plt.figure(figsize=(12, 8)) # Create figure with specified size
plt.scatter(X, y, color="blue", alpha=0.6, label="Training Data") # Plot original data points

# Generate prediction lines for comparison
X_new = np.array([[0], [2]]) # Create range of X values for plotting lines
X_new_b = np.c_[np.ones((2, 1)), X_new] # Add bias column to prediction range

# Calculate predictions for each method
y_pred_gd = X_new_b.dot(theta_gd) # Predictions using gradient descent parameters
y_pred_ne = X_new_b.dot(theta_ne) # Predictions using normal equation parameters
y_pred_sklearn = lin_reg.predict(X_new) # Predictions using sklearn model

# Plot regression lines
plt.plot(X_new, y_pred_gd, "r-", linewidth=2, label="Gradient Descent") # Plot gradient descent line in red
plt.plot(X_new, y_pred_ne, "g--", linewidth=2, label="Normal Equation") # Plot normal equation line in green dashed
plt.plot(X_new, y_pred_sklearn, "k:", linewidth=3, label="Sklearn LinearRegression") # Plot sklearn line in black dotted

# Customize plot appearance
plt.xlabel("X coordinate", fontsize=12) # Set x-axis label with font size
plt.ylabel("Y coordinate", fontsize=12) # Set y-axis label with font size
plt.title("Linear Regression: Gradient Descent vs Normal Equation vs Sklearn", fontsize=14) # Set plot title
plt.legend(fontsize=11) # Add legend with custom font size
plt.grid(True, alpha=0.3) # Add grid for better readability
plt.show() # Display the plot