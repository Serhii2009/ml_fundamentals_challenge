import numpy as np                                                                  # Import NumPy library for mathematical operations
                    
def mse(y_true, y_pred):                                                            # Function to calculate Mean Squared Error
    return np.mean((y_true - y_pred) ** 2)                                          # Calculate mean of squared differences
                    
def bce(y_true, y_pred):                                                            # Function to calculate Binary Cross Entropy
    eps = 1e-15                                                                     # Very small number to avoid log(0)
    y_pred = np.clip(y_pred, eps, 1-eps)                                            # Clip predictions between eps and 1-eps
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))    # BCE formula


# ===== MSE EXAMPLES (REGRESSION) =====
print("="*50)  # Separator line for formatting
print("MSE EXAMPLES (REGRESSION)")  # MSE section header
print("="*50)  # Separator line for formatting

# Example 1: Perfect predictions
y_true_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # True values
y_pred_1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Exactly same predictions
mse_1 = mse(y_true_1, y_pred_1)  # Calculate MSE
print(f"Perfect predictions: MSE = {mse_1:.4f}")  # Print result

# Example 2: Small errors
y_true_2 = np.array([10, 20, 30, 40, 50])  # True values
y_pred_2 = np.array([11, 19, 31, 39, 51])  # Predictions with small errors
mse_2 = mse(y_true_2, y_pred_2)  # Calculate MSE
print(f"Small errors: MSE = {mse_2:.4f}")  # Print result

# Example 3: One big mistake
y_true_3 = np.array([2, 4, 6, 8, 10])  # True values
y_pred_3 = np.array([2, 4, 6, 8, 100])  # One very bad prediction
mse_3 = mse(y_true_3, y_pred_3)  # Calculate MSE
print(f"One big mistake: MSE = {mse_3:.4f}")  # Print result

# ===== BCE EXAMPLES (BINARY CLASSIFICATION) =====
print("\n" + "="*50)  # New line and separator line
print("BCE EXAMPLES (BINARY CLASSIFICATION)")  # BCE section header
print("="*50)  # Separator line for formatting

# Example 1: Great classification
y_true_cls1 = np.array([1, 0, 1, 0, 1])  # True classes (0 or 1)
y_pred_cls1 = np.array([0.99, 0.01, 0.98, 0.02, 0.97])  # Very confident correct predictions
bce_1 = bce(y_true_cls1, y_pred_cls1)  # Calculate BCE
print(f"Great classification: BCE = {bce_1:.4f}")  # Print result

# Example 2: Random predictions (50/50)
y_true_cls2 = np.array([1, 0, 1, 0, 1])  # True classes
y_pred_cls2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])  # Complete uncertainty
bce_2 = bce(y_true_cls2, y_pred_cls2)  # Calculate BCE
print(f"Random predictions: BCE = {bce_2:.4f}")  # Print result

# Example 3: Confident but wrong predictions
y_true_cls3 = np.array([1, 0, 1, 0, 1])  # True classes
y_pred_cls3 = np.array([0.1, 0.9, 0.2, 0.8, 0.15])  # Confident but completely wrong
bce_3 = bce(y_true_cls3, y_pred_cls3)  # Calculate BCE
print(f"Confident mistakes: BCE = {bce_3:.4f}")  # Print result

# ===== CONCLUSIONS =====
print("\n" + "="*50)  # New line and separator line
print("CONCLUSIONS:")  # Conclusions header
print("="*50)  # Separator line for formatting
print("MSE:")  # MSE header
print("- Larger error = larger penalty (quadratic)")  # MSE explanation
print("- One big mistake heavily increases loss")  # MSE feature
print("\nBCE:")  # BCE header
print("- Penalizes uncertainty in correct answers")  # BCE feature
print("- Harshly punishes confident mistakes")  # BCE feature
print("- Random predictions ≈ 0.693 (ln(2))")  # Mathematical fact