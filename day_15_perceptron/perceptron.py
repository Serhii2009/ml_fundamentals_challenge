import numpy as np                                                                             # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                # Import matplotlib for visualization
from sklearn.datasets import make_classification                                               # Import function to generate dataset
from sklearn.linear_model import Perceptron                                                    # Import sklearn Perceptron for comparison

def step_function(z):                                                                          # Define step activation function
    """Step activation function: returns 1 if z > 0, else 0"""                                 # Function description
    return np.where(z > 0, 1, 0)                                                               # Apply step function element-wise

class CustomPerceptron:                                                                        # Custom perceptron implementation
    def __init__(self, learning_rate=0.1, max_epochs=100):                                     # Initialize perceptron parameters
        self.learning_rate = learning_rate                                                     # Learning rate for weight updates
        self.max_epochs = max_epochs                                                           # Maximum training epochs
        self.weights = None                                                                    # Weights for features
        self.bias = None                                                                       # Bias term
    
    def fit(self, X, y):                                                                       # Train the perceptron
        """Train perceptron using perceptron learning rule"""                                  # Training method description
        n_samples, n_features = X.shape                                                        # Get dataset dimensions
        
        self.weights = np.zeros(n_features)                                                    # Initialize weights to zero
        self.bias = 0                                                                          # Initialize bias to zero
        
        for epoch in range(self.max_epochs):                                                   # Loop through epochs
            errors = 0                                                                         # Count errors in this epoch
            
            for i in range(n_samples):                                                         # Loop through samples
                z = np.dot(X[i], self.weights) + self.bias                                     # Calculate linear combination
                prediction = step_function(z)                                                  # Apply activation function
                error = y[i] - prediction                                                      # Calculate prediction error
                
                if error != 0:                                                                 # Only update if error exists
                    self.weights += self.learning_rate * error * X[i]                          # Update weights using learning rule
                    self.bias += self.learning_rate * error                                    # Update bias using learning rule
                    errors += 1                                                                # Count this error
            
            if errors == 0:                                                                    # Check if converged
                break                                                                          # Exit training loop
        
        return self                                                                            # Return self for method chaining
    
    def predict(self, X):                                                                      # Make predictions
        """Make predictions on new data"""                                                     # Prediction method description
        z = np.dot(X, self.weights) + self.bias                                                # Calculate weighted sum for all samples
        return step_function(z)                                                                # Apply step function



# ===== MANUAL PERCEPTRON STEP =====
print("Manual Perceptron Learning Step:")                                                      # Section header

# Dataset
X = np.array([[2, 3], [1, 4], [3, 1]])                                                         # Input features
y_true = np.array([1, 0, 1])                                                                   # True labels

# Initial parameters
w = np.array([0.5, -0.2])                                                                      # Initial weights
b = 0.1                                                                                        # Initial bias
eta = 0.1                                                                                      # Learning rate

def step(z):                                                                                   # Simple step function
    return 1 if z > 0 else 0                                                                   # Return 1 if positive, 0 otherwise

# Single prediction and weight update
x = X[0]                                                                                       # Take first sample
z = np.dot(w, x) + b                                                                           # Calculate weighted sum
y_pred = step(z)                                                                               # Make prediction

print(f"Input: {x}, True: {y_true[0]}, Predicted: {y_pred}")                                   # Show prediction

# Update weights
w = w + eta * (y_true[0] - y_pred) * x                                                         # Update weights
b = b + eta * (y_true[0] - y_pred)                                                             # Update bias

print(f"Updated weights: {w}")                                                                 # Print new weights
print(f"Updated bias: {b:.1f}")                                                                # Print new bias
print()                                                                                        # Empty line


# ===== FULL PERCEPTRON TRAINING =====
print("Full Perceptron Training:")                                                             # Section header

# Generate dataset
X_data, y_data = make_classification(n_samples=100, n_features=2, n_informative=2,             # Generate classification dataset
                                   n_redundant=0, n_clusters_per_class=1, random_state=42)

# Train custom perceptron
custom_perceptron = CustomPerceptron(learning_rate=0.1, max_epochs=100)                        # Create custom perceptron
custom_perceptron.fit(X_data, y_data)                                                          # Train perceptron

# Train sklearn perceptron
sklearn_perceptron = Perceptron(max_iter=100, random_state=42)                                 # Create sklearn perceptron
sklearn_perceptron.fit(X_data, y_data)                                                         # Train sklearn perceptron

# Compare results
custom_acc = np.mean(custom_perceptron.predict(X_data) == y_data)                              # Calculate custom accuracy
sklearn_acc = np.mean(sklearn_perceptron.predict(X_data) == y_data)                            # Calculate sklearn accuracy

print(f"Custom Perceptron - Weights: {custom_perceptron.weights}, Bias: {custom_perceptron.bias:.3f}, Accuracy: {custom_acc:.3f}") # Custom results
print(f"Sklearn Perceptron - Weights: {sklearn_perceptron.coef_[0]}, Bias: {sklearn_perceptron.intercept_[0]:.3f}, Accuracy: {sklearn_acc:.3f}") # Sklearn results
print() # Empty line



# ===== VISUALIZATION =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4)) # Create subplots

ax1.scatter(X_data[y_data==0, 0], X_data[y_data==0, 1], c='red', marker='x', label='Class 0') # Plot class 0
ax1.scatter(X_data[y_data==1, 0], X_data[y_data==1, 1], c='blue', marker='o', label='Class 1') # Plot class 1

if custom_perceptron.weights[1] != 0: # Check if slope exists
    x_range = np.linspace(X_data[:, 0].min()-1, X_data[:, 0].max()+1, 100) # Create x range
    y_boundary = -(custom_perceptron.weights[0] * x_range + custom_perceptron.bias) / custom_perceptron.weights[1] # Calculate boundary
    ax1.plot(x_range, y_boundary, 'k-', linewidth=2, label='Decision Boundary') # Plot boundary

ax1.set_title('Custom Perceptron') # Set title
ax1.legend() # Add legend
ax1.grid(True, alpha=0.3) # Add grid

# Plot sklearn perceptron
ax2.scatter(X_data[y_data==0, 0], X_data[y_data==0, 1], c='red', marker='x', label='Class 0') # Plot class 0
ax2.scatter(X_data[y_data==1, 0], X_data[y_data==1, 1], c='blue', marker='o', label='Class 1') # Plot class 1

if sklearn_perceptron.coef_[0][1] != 0: # Check if slope exists
    y_boundary = -(sklearn_perceptron.coef_[0][0] * x_range + sklearn_perceptron.intercept_[0]) / sklearn_perceptron.coef_[0][1] # Calculate boundary
    ax2.plot(x_range, y_boundary, 'k-', linewidth=2, label='Decision Boundary') # Plot boundary

ax2.set_title('Sklearn Perceptron') # Set title
ax2.legend() # Add legend
ax2.grid(True, alpha=0.3) # Add grid

plt.tight_layout() # Adjust layout
plt.show() # Display plots