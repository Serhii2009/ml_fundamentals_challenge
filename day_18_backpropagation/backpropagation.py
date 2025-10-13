import numpy as np                                                                              # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                 # Import matplotlib for visualization

# ===== ACTIVATION FUNCTION =====
def sigmoid(x):                                                                                 # Sigmoid activation function
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))                                             # Clip to prevent overflow

# ===== INPUT DATA =====
X = np.array([[2, 3]])                                                                          # Input data (1 sample, 2 features)
y = np.array([[1]])                                                                             # Target output

# ===== INITIALIZE PARAMETERS =====
np.random.seed(42)                                                                              # Set random seed for reproducibility
w1 = np.random.randn(2, 2) * 0.5                                                                # Weights input → hidden (2x2)
b1 = np.zeros((1, 2))                                                                           # Bias for hidden layer
w2 = np.random.randn(2, 1) * 0.5                                                                # Weights hidden → output (2x1)
b2 = np.zeros((1, 1))                                                                           # Bias for output layer

# ===== FORWARD PROPAGATION =====
z1 = np.dot(X, w1) + b1                                                                         # Hidden layer weighted sum
a1 = sigmoid(z1)                                                                                # Hidden layer activation
z2 = np.dot(a1, w2) + b2                                                                        # Output layer weighted sum
a2 = sigmoid(z2)                                                                                # Output prediction

print("Forward Propagation:") # Header
print(f"Input: {X[0]}") # Print input
print(f"Hidden activations: {a1[0]}") # Print hidden layer
print(f"Output: {a2[0,0]:.4f}, Target: {y[0,0]}") # Print output vs target
print() # Empty line

# ===== NETWORK DIAGRAM =====
fig, ax = plt.subplots(figsize=(10, 6)) # Create figure

# Neuron positions
input_y = [0.7, 0.3] # Input neuron positions
hidden_y = [0.65, 0.35] # Hidden neuron positions
output_y = [0.5] # Output neuron position

# Draw input neurons
for i, y_pos in enumerate(input_y): # Loop through input neurons
    circle = plt.Circle((0.2, y_pos), 0.05, color='blue', alpha=0.7) # Create circle
    ax.add_patch(circle) # Add to plot
    ax.text(0.1, y_pos, f'{X[0,i]:.1f}', ha='right', fontsize=12) # Add value label

# Draw hidden neurons
for i, y_pos in enumerate(hidden_y): # Loop through hidden neurons
    circle = plt.Circle((0.5, y_pos), 0.05, color='green', alpha=0.7) # Create circle
    ax.add_patch(circle) # Add to plot
    ax.text(0.6, y_pos, f'{a1[0,i]:.2f}', ha='left', fontsize=12) # Add activation label

# Draw output neuron
circle = plt.Circle((0.8, output_y[0]), 0.05, color='red', alpha=0.7) # Create circle
ax.add_patch(circle) # Add to plot
ax.text(0.9, output_y[0], f'{a2[0,0]:.2f}', ha='left', fontsize=12) # Add output label

# Draw connections
for y1 in input_y: # Input to hidden connections
    for y2 in hidden_y: # For each hidden neuron
        ax.plot([0.2, 0.5], [y1, y2], 'gray', alpha=0.3, linewidth=1) # Draw line

for y1 in hidden_y: # Hidden to output connections
    for y2 in output_y: # For each output neuron
        ax.plot([0.5, 0.8], [y1, y2], 'gray', alpha=0.3, linewidth=1) # Draw line

# Layer labels
ax.text(0.2, 0.95, 'Input', ha='center', fontsize=14, fontweight='bold') # Input layer label
ax.text(0.5, 0.95, 'Hidden', ha='center', fontsize=14, fontweight='bold') # Hidden layer label
ax.text(0.8, 0.95, 'Output', ha='center', fontsize=14, fontweight='bold') # Output layer label

ax.set_xlim([0, 1]) # Set x-limits
ax.set_ylim([0, 1]) # Set y-limits
ax.axis('off') # Turn off axis
ax.set_title('Forward Propagation Flow', fontsize=16) # Set title

plt.tight_layout() # Adjust layout
plt.show() # Display plot

# ===== VISUALIZATION =====
fig, axes = plt.subplots(1, 3, figsize=(12, 4)) # Create subplots

# Input layer
axes[0].bar([0, 1], X[0], color='blue', alpha=0.7) # Plot input values
axes[0].set_title('Input Layer') # Set title
axes[0].set_ylabel('Value') # Set y-label
axes[0].set_xticks([0, 1]) # Set x-ticks
axes[0].set_xticklabels(['x1', 'x2']) # Set x-tick labels

# Hidden layer
axes[1].bar([0, 1], a1[0], color='green', alpha=0.7) # Plot hidden activations
axes[1].set_title('Hidden Layer') # Set title
axes[1].set_ylabel('Activation') # Set y-label
axes[1].set_ylim([0, 1]) # Set y-limits
axes[1].set_xticks([0, 1]) # Set x-ticks
axes[1].set_xticklabels(['h1', 'h2']) # Set x-tick labels

# Output layer
axes[2].bar([0, 1], [a2[0,0], y[0,0]], color=['red', 'gray'], alpha=0.7) # Plot output and target
axes[2].set_title('Output Layer') # Set title
axes[2].set_ylabel('Value') # Set y-label
axes[2].set_ylim([0, 1]) # Set y-limits
axes[2].set_xticks([0, 1]) # Set x-ticks
axes[2].set_xticklabels(['Output', 'Target']) # Set x-tick labels

plt.tight_layout() # Adjust layout
plt.show() # Display plot