import numpy as np                                                                  # Import NumPy library for mathematical operations
import matplotlib.pyplot as plt                                                     # Import matplotlib for plotting and visualization
from sklearn.datasets import make_classification                                    # Import function to generate synthetic classification dataset
from sklearn.linear_model import LogisticRegression                                 # Import sklearn's LogisticRegression for comparison

def sigmoid(z):                                                                     # Define sigmoid activation function
    return 1 / (1 + np.exp(-z))                                                     # Return sigmoid output: maps any real number to (0,1)

def compute_loss(y, y_hat):                                                         # Define cross-entropy loss function
    m = len(y)                                                                      # Get number of training examples
    return -(1/m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))         # Return binary cross-entropy loss

# ===== CUSTOM LOGISTIC REGRESSION CLASS =====
class MyLogisticRegression:                                                         # Define custom logistic regression class
    def __init__(self, learning_rate=0.01, num_iterations=1000):                    # Initialize with hyperparameters
        self.learning_rate = learning_rate                                          # Store learning rate (step size for gradient descent)
        self.num_iterations = num_iterations                                        # Store number of training iterations

    def fit(self, X, y):                                                            # Training method to learn parameters
        m, n = X.shape                                                              # Get number of examples (m) and features (n)
        self.weights = np.zeros(n)                                                  # Initialize weights to zero vector
        self.bias = 0                                                               # Initialize bias term to zero
        
        for iteration in range(self.num_iterations):                                # Loop through training iterations
            z = np.dot(X, self.weights) + self.bias                                 # Calculate linear combination z = X*w + b
            y_hat = sigmoid(z)                                                      # Apply sigmoid to get probabilities
            
            dw = (1/m) * np.dot(X.T, (y_hat - y))                                   # Calculate gradient with respect to weights
            db = (1/m) * np.sum(y_hat - y)                                          # Calculate gradient with respect to bias
            
            self.weights -= self.learning_rate * dw                                 # Update weights: w = w - η * dw
            self.bias -= self.learning_rate * db                                    # Update bias: b = b - η * db
            
            if iteration % 200 == 0:                                                # Check if iteration is divisible by 200
                loss = compute_loss(y, y_hat)                                       # Calculate current loss
                print(f"Iteration {iteration:4d}: Loss = {loss:.6f}")               # Print iteration and loss value

    def predict_proba(self, X):                                                     # Method to predict probabilities
        return sigmoid(np.dot(X, self.weights) + self.bias)                         # Return probability predictions using learned parameters

    def predict(self, X, threshold=0.5):                                            # Method to make binary predictions
        return (self.predict_proba(X) >= threshold).astype(int)                     # Convert probabilities to binary predictions using threshold

X, y = make_classification(n_samples=200, n_features=2, n_informative=2,            # Generate 200 samples with 2 features
                          n_redundant=0, random_state=42)                           # Set random seed for reproducibility

print("Training custom logistic regression...")                                     # Training start message
model = MyLogisticRegression(learning_rate=0.1, num_iterations=1000)                # Create custom model instance
model.fit(X, y)                                                                     # Train the model on generated data

clf = LogisticRegression()                                                          # Create sklearn LogisticRegression instance
clf.fit(X, y)                                                                       # Train sklearn model on same data


# ===== RESULTS COMPARISON =====
y_pred_custom = model.predict(X) # Make predictions using custom model
y_pred_sklearn = clf.predict(X) # Make predictions using sklearn model

accuracy_custom = np.mean(y_pred_custom == y) # Calculate accuracy for custom model
accuracy_sklearn = np.mean(y_pred_sklearn == y) # Calculate accuracy for sklearn model

print(f"\nFinal Results:") # Results header
print(f"Custom Model  - Accuracy: {accuracy_custom:.3f}") # Print custom model accuracy
print(f"Sklearn Model - Accuracy: {accuracy_sklearn:.3f}") # Print sklearn model accuracy

# ===== VISUALIZATION =====
plt.figure(figsize=(10, 6)) # Create figure with specified size
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', alpha=0.7) # Scatter plot of data points colored by class

# Plot decision boundary
x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100) # Create range of x1 values for boundary line
x2_boundary = -(model.weights[0] * x1_range + model.bias) / model.weights[1] # Calculate corresponding x2 values for decision boundary

plt.plot(x1_range, x2_boundary, 'k-', linewidth=2, label="Decision Boundary") # Plot decision boundary as black line
plt.xlabel("Feature 1") # Set x-axis label
plt.ylabel("Feature 2") # Set y-axis label
plt.title("Logistic Regression Classification") # Set plot title
plt.legend() # Add legend
plt.grid(True, alpha=0.3) # Add grid for better readability
plt.show() # Display the plot