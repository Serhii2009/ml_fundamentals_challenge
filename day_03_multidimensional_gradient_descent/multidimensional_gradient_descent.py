import numpy as np                                                                  # Import NumPy for numerical operations

def mse(y_true, y_pred):                                                            # Function to calculate Mean Squared Error
    return np.mean((y_true - y_pred) ** 2)                                          # Average squared difference

def accuracy(y_true, y_pred, tol=1.0):                                              # Function to calculate "accuracy" (tolerance-based)
    return np.mean(np.abs(y_true - y_pred) < tol)                                   # % of predictions close to actual values

np.random.seed(42)                                                                  # For reproducibility
x = np.array([1, 2, 3, 4, 5, 6])                                                    # Input feature values
y = np.array([3, 5, 7, 9, 11, 13]) + np.random.normal(0, 1.0, size=6)               # Add small random noise

w0, w1 = 0.0, 0.0                                                                   # Initial model parameters
learning_rate = 0.01                                                                # Learning rate (step size)
epochs = 1000                                                                       # Number of iterations
N = len(x)                                                                          # Number of data points

initial_pred = w1 * x + w0                                                          # Initial predictions before training
initial_mse = mse(y, initial_pred)                                                  # Initial error
initial_acc = accuracy(y, initial_pred)                                             # Initial accuracy

for epoch in range(1, epochs + 1):                                                  # Run for specified number of epochs
    y_pred = w1 * x + w0                                                            # Current predictions
    dw1 = (-2/N) * np.sum(x * (y - y_pred))                                         # Gradient wrt w1
    dw0 = (-2/N) * np.sum(y - y_pred)                                               # Gradient wrt w0
    w1 -= learning_rate * dw1                                                       # Update weight w1
    w0 -= learning_rate * dw0                                                       # Update weight w0

    if epoch % 200 == 0:                                                            # Print metrics every 200 epochs
        curr_mse = mse(y, y_pred)                                                   # Current MSE
        curr_acc = accuracy(y, y_pred)                                              # Current Accuracy
        print(f"Epoch {epoch:4d} | MSE = {curr_mse:.4f} | Accuracy = {curr_acc:.4f}")  # Print current MSE and Accuracy


# ===== FINAL RESULTS =====
y_pred = w1 * x + w0 # Final predictions
final_mse = mse(y, y_pred) # Final error
final_acc = accuracy(y, y_pred) # Final accuracy

print("\n" + "="*50)
print("FINAL MODEL RESULTS")
print("="*50)
print(f"Actual values: {y}") # Print real values
print(f"Predicted values: {y_pred.round(4)}") # Print predicted values
print(f"Optimized params: w0 = {w0:.4f}, w1 = {w1:.4f}") # Print final parameters
print(f"Final Mean Squared Error: {final_mse:.4f}") # Print final MSE
print(f"Final Accuracy: {final_acc:.4f}") # Print final Accuracy
