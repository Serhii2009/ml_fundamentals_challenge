import numpy as np                                                                               # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                  # Import matplotlib for visualization
from sklearn.datasets import make_classification                                                 # Import dataset generator
from sklearn.neural_network import MLPClassifier                                                 # Import neural network classifier

# ===== ACTIVATION FUNCTIONS =====
def sigmoid(x):                                                                                  # Sigmoid activation function
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))                                              # Clip to prevent overflow

def tanh_func(x):                                                                                # Tanh activation function
    return np.tanh(x)                                                                            # Use numpy tanh

def relu(x):                                                                                     # ReLU activation function
    return np.maximum(0, x)                                                                      # Return max of 0 and x

def leaky_relu(x, alpha=0.01):                                                                   # Leaky ReLU activation function
    return np.where(x > 0, x, alpha * x)                                                         # Return x if positive, alpha*x if negative

# ===== SINGLE NEURON DEMONSTRATION =====
# Test single neuron with different activations
inputs = np.array([2.0, -1.5, 0.0, -3.0, 1.0])                                                   # Test inputs
weights = np.array([0.5, -0.3])                                                                  # Neuron weights
bias = 0.1                                                                                       # Neuron bias

# Calculate outputs for sample input
sample_input = np.array([2, 3])                                                                  # Sample input
z = np.dot(weights, sample_input) + bias                                                         # Weighted sum

print(f"Input: {sample_input}, Weighted sum: {z:.3f}") # Show calculation
print(f"Sigmoid: {sigmoid(z):.3f}, Tanh: {tanh_func(z):.3f}, ReLU: {relu(z):.3f}, Leaky: {leaky_relu(z):.3f}") # Show outputs

# ===== NETWORK COMPARISON =====
# Generate dataset and compare activation functions
X, y = make_classification(n_samples=500, n_features=2, n_informative=2,                         # Create classification dataset
                          n_redundant=0, n_clusters_per_class=1, random_state=42)

activations = ['logistic', 'tanh', 'relu']                                                       # Available sklearn activations
for activation in activations:                                                                   # Test each activation
    model = MLPClassifier(hidden_layer_sizes=(20,), activation=activation,                       # Create neural network
                         max_iter=500, random_state=42)
    model.fit(X, y)                                                                              # Train model
    accuracy = model.score(X, y)                                                                 # Calculate accuracy
    print(f"{activation.capitalize()}: {accuracy:.3f}")                                          # Print result

# ===== DEAD NEURON PROBLEM =====
negative_input = -5.0                                                                            # Large negative input
relu_output = relu(negative_input)                                                               # ReLU output (will be 0)
leaky_output = leaky_relu(negative_input)                                                        # Leaky ReLU output
print(f"Dead neuron test (input={negative_input}): ReLU={relu_output}, Leaky={leaky_output:.3f}") # Show difference



# ===== VISUALIZATION =====
x_range = np.linspace(-4, 4, 1000) # Input range for plotting

# Create visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8)) # Create subplots

# Plot activation functions
ax1.plot(x_range, sigmoid(x_range), 'b-', label='Sigmoid', linewidth=2) # Plot sigmoid
ax1.plot(x_range, tanh_func(x_range), 'r-', label='Tanh', linewidth=2) # Plot tanh
ax1.plot(x_range, relu(x_range), 'g-', label='ReLU', linewidth=2) # Plot ReLU
ax1.plot(x_range, leaky_relu(x_range), 'orange', label='Leaky ReLU', linewidth=2) # Plot leaky ReLU
ax1.set_title('Activation Functions') # Set title
ax1.legend() # Add legend
ax1.grid(True, alpha=0.3) # Add grid

# Plot derivatives
sigmoid_grad = sigmoid(x_range) * (1 - sigmoid(x_range)) # Sigmoid gradient
tanh_grad = 1 - tanh_func(x_range)**2 # Tanh gradient
relu_grad = np.where(x_range > 0, 1, 0) # ReLU gradient
leaky_grad = np.where(x_range > 0, 1, 0.01) # Leaky ReLU gradient

ax2.plot(x_range, sigmoid_grad, 'b-', label='Sigmoid', linewidth=2) # Plot sigmoid gradient
ax2.plot(x_range, tanh_grad, 'r-', label='Tanh', linewidth=2) # Plot tanh gradient
ax2.plot(x_range, relu_grad, 'g-', label='ReLU', linewidth=2) # Plot ReLU gradient
ax2.plot(x_range, leaky_grad, 'orange', label='Leaky ReLU', linewidth=2) # Plot leaky ReLU gradient
ax2.set_title('Gradients') # Set title
ax2.legend() # Add legend
ax2.grid(True, alpha=0.3) # Add grid

# Focus on critical region
focus_range = np.linspace(-2, 2, 100) # Focused input range
ax3.plot(focus_range, sigmoid(focus_range), 'b-', label='Sigmoid', linewidth=2) # Plot sigmoid
ax3.plot(focus_range, tanh_func(focus_range), 'r-', label='Tanh', linewidth=2) # Plot tanh
ax3.plot(focus_range, relu(focus_range), 'g-', label='ReLU', linewidth=2) # Plot ReLU
ax3.plot(focus_range, leaky_relu(focus_range), 'orange', label='Leaky ReLU', linewidth=2) # Plot leaky ReLU
ax3.set_title('Active Region (-2 to 2)') # Set title
ax3.legend() # Add legend
ax3.grid(True, alpha=0.3) # Add grid

# Dead neuron comparison
negative_inputs = np.array([-3, -2, -1, -0.5]) # Test negative inputs
relu_outputs = relu(negative_inputs) # ReLU outputs
leaky_outputs = leaky_relu(negative_inputs) # Leaky ReLU outputs

x_pos = np.arange(len(negative_inputs)) # Bar positions
width = 0.35 # Bar width
ax4.bar(x_pos - width/2, relu_outputs, width, label='ReLU', alpha=0.7) # ReLU bars
ax4.bar(x_pos + width/2, leaky_outputs, width, label='Leaky ReLU', alpha=0.7) # Leaky ReLU bars
ax4.set_title('Dead Neuron Problem') # Set title
ax4.legend() # Add legend
ax4.set_xticks(x_pos) # Set x ticks
ax4.set_xticklabels([f'{inp}' for inp in negative_inputs]) # Set x labels

plt.tight_layout() # Adjust layout
plt.show() # Display plots