import numpy as np                                                                              # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                 # Import matplotlib for visualization

# ===== ACTIVATION FUNCTIONS =====
def sigmoid(x):                                                                                 # Sigmoid activation function
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))                                             # Clip to prevent overflow

def sigmoid_derivative(x):                                                                      # Sigmoid derivative
    s = sigmoid(x)                                                                              # Calculate sigmoid
    return s * (1 - s)                                                                          # Return derivative

# ===== XOR DATASET =====
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])                                                  # XOR inputs
y = np.array([[0], [1], [1], [0]])                                                              # XOR outputs

# ===== HYPERPARAMETERS =====
learning_rate = 0.1                                                                             # Learning rate
epochs = 1000                                                                                   # Number of training epochs
beta1, beta2 = 0.9, 0.999                                                                       # Adam momentum parameters
epsilon = 1e-8                                                                                  # Small constant for numerical stability

# ===== INITIALIZE NETWORK =====
np.random.seed(42)                                                                              # Set random seed
W1 = np.random.randn(2, 2) * 0.5                                                                # Input to hidden weights
b1 = np.zeros((1, 2))                                                                           # Hidden layer bias
W2 = np.random.randn(2, 1) * 0.5                                                                # Hidden to output weights
b2 = np.zeros((1, 1))                                                                           # Output layer bias

# ===== INITIALIZE ADAM PARAMETERS =====
mW1 = np.zeros_like(W1); vW1 = np.zeros_like(W1)                                                # First and second moment for W1
mb1 = np.zeros_like(b1); vb1 = np.zeros_like(b1)                                                # First and second moment for b1
mW2 = np.zeros_like(W2); vW2 = np.zeros_like(W2)                                                # First and second moment for W2
mb2 = np.zeros_like(b2); vb2 = np.zeros_like(b2)                                                # First and second moment for b2

losses = []                                                                                     # Store loss history

# ===== INITIAL LOSS =====
Z1_init = np.dot(X, W1) + b1                                                                    # Initial hidden layer
A1_init = sigmoid(Z1_init)                                                                      # Initial hidden activation
Z2_init = np.dot(A1_init, W2) + b2                                                              # Initial output layer
A2_init = sigmoid(Z2_init)                                                                      # Initial prediction
initial_loss = np.mean((A2_init - y) ** 2)                                                      # Initial loss

print(f"Initial Loss: {initial_loss:.4f}")                                                      # Print initial loss

# ===== TRAINING LOOP =====
for t in range(1, epochs + 1):                                                                  # Loop through epochs
    # Forward propagation
    Z1 = np.dot(X, W1) + b1                                                                     # Hidden layer weighted sum
    A1 = sigmoid(Z1)                                                                            # Hidden layer activation
    Z2 = np.dot(A1, W2) + b2                                                                    # Output layer weighted sum
    A2 = sigmoid(Z2)                                                                            # Output prediction
    
    # Calculate loss
    loss = np.mean((A2 - y) ** 2)                                                               # Mean squared error
    losses.append(loss)                                                                         # Store loss
    
    # Backpropagation
    dA2 = 2 * (A2 - y) / len(y)                                                                 # Loss gradient wrt A2
    dZ2 = dA2 * sigmoid_derivative(Z2)                                                          # Gradient wrt Z2
    dW2 = np.dot(A1.T, dZ2)                                                                     # Gradient wrt W2
    db2 = np.sum(dZ2, axis=0, keepdims=True)                                                    # Gradient wrt b2
    
    dA1 = np.dot(dZ2, W2.T)                                                                     # Gradient wrt A1
    dZ1 = dA1 * sigmoid_derivative(Z1)                                                          # Gradient wrt Z1
    dW1 = np.dot(X.T, dZ1)                                                                      # Gradient wrt W1
    db1 = np.sum(dZ1, axis=0, keepdims=True)                                                    # Gradient wrt b1
    
    # Adam optimizer update
    params = [W1, b1, W2, b2]                                                                   # List of parameters
    grads = [dW1, db1, dW2, db2]                                                                # List of gradients
    moments_m = [mW1, mb1, mW2, mb2]                                                            # First moments
    moments_v = [vW1, vb1, vW2, vb2]                                                            # Second moments
    
    for param, grad, m, v in zip(params, grads, moments_m, moments_v):                          # Update each parameter
        m[:] = beta1 * m + (1 - beta1) * grad                                                   # Update first moment
        v[:] = beta2 * v + (1 - beta2) * (grad ** 2)                                            # Update second moment
        m_hat = m / (1 - beta1 ** t)                                                            # Bias-corrected first moment
        v_hat = v / (1 - beta2 ** t)                                                            # Bias-corrected second moment
        param -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)                             # Update parameter


# ===== RESULTS =====
# Final predictions
A1_final = sigmoid(np.dot(X, W1) + b1) # Final hidden activations
A2_final = sigmoid(np.dot(A1_final, W2) + b2) # Final predictions
final_loss = np.mean((A2_final - y) ** 2) # Final loss

print(f"Final Loss: {final_loss:.4f}") # Print final loss
print("XOR Learning Results:") # Results header
for i in range(len(X)): # Loop through samples
    print(f"Input: {X[i]} → Prediction: {A2_final[i,0]:.3f}, Target: {y[i,0]}") # Print results

# ===== VISUALIZATION =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4)) # Create subplots

# Loss curve
axes[0].plot(losses, 'b-', linewidth=2) # Plot loss
axes[0].set_xlabel('Epoch') # Set x-label
axes[0].set_ylabel('Loss') # Set y-label
axes[0].set_title('Training Loss (Adam Optimizer)') # Set title
axes[0].grid(True, alpha=0.3) # Add grid

# Decision boundary
x_min, x_max = -0.5, 1.5 # X-axis range
y_min, y_max = -0.5, 1.5 # Y-axis range
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), # Create grid
                     np.linspace(y_min, y_max, 100))
grid_input = np.c_[xx.ravel(), yy.ravel()] # Flatten grid
Z_grid = sigmoid(np.dot(sigmoid(np.dot(grid_input, W1) + b1), W2) + b2) # Predictions for grid
Z_grid = Z_grid.reshape(xx.shape) # Reshape to grid

axes[1].contourf(xx, yy, Z_grid, levels=20, cmap='RdBu', alpha=0.6) # Plot decision boundary
axes[1].scatter(X[y.flatten() == 0, 0], X[y.flatten() == 0, 1], # Plot class 0
                c='red', marker='x', s=200, linewidths=3, label='Output 0')
axes[1].scatter(X[y.flatten() == 1, 0], X[y.flatten() == 1, 1], # Plot class 1
                c='blue', marker='o', s=200, linewidths=3, label='Output 1')
axes[1].set_xlabel('Input 1') # Set x-label
axes[1].set_ylabel('Input 2') # Set y-label
axes[1].set_title('XOR Decision Boundary') # Set title
axes[1].legend() # Add legend
axes[1].grid(True, alpha=0.3) # Add grid

plt.tight_layout() # Adjust layout
plt.show() # Display plot