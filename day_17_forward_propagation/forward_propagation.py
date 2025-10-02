import numpy as np                                                                              # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                 # Import matplotlib for visualization

# ===== ACTIVATION FUNCTIONS =====
def sigmoid(z):                                                                                 # Sigmoid activation function
    return 1 / (1 + np.exp(-np.clip(z, -250, 250)))                                             # Clip to prevent overflow

def relu(z):                                                                                    # ReLU activation function
    return np.maximum(0, z)                                                                     # Return max of 0 and z

# ===== SINGLE NEURON FORWARD PASS =====
def single_neuron_demo():                                                                       # Demonstrate single neuron forward pass
    x = np.array([2, 3])                                                                        # Input features
    w = np.array([0.5, -0.2])                                                                   # Weights
    b = 1                                                                                       # Bias

    z = np.dot(w, x) + b                                                                        # Calculate weighted sum
    a = sigmoid(z)                                                                              # Apply activation

    print("Single Neuron Forward Pass:")                                                        # Header
    print(f"Input: {x}")                                                                        # Print input
    print(f"Weights: {w}, Bias: {b}")                                                           # Print weights and bias
    print(f"Weighted sum (z): {z:.3f}")                                                         # Print weighted sum
    print(f"Output (sigmoid): {a:.3f}")                                                         # Print output
    print()                                                                                     # Empty line

# ===== SIMPLE NEURAL NETWORK CLASS =====
class SimpleNN:                                                                                 # Simple neural network implementation
    def __init__(self, input_size, hidden_size, output_size):                                   # Initialize network
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5                                # Input to hidden weights
        self.b1 = np.zeros(hidden_size)                                                         # Hidden layer bias
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5                               # Hidden to output weights
        self.b2 = np.zeros(output_size)                                                         # Output layer bias
    
    def forward(self, X):                                                                       # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1                                                  # Hidden layer weighted sum
        self.a1 = sigmoid(self.z1)                                                              # Hidden layer activation
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2                                            # Output layer weighted sum
        self.a2 = sigmoid(self.z2)                                                              # Output layer activation
        
        return self.a2                                                                          # Return final output
    
    def print_forward_details(self, X):                                                         # Print detailed forward pass
        output = self.forward(X)                                                                # Run forward pass
        
        print("Network Forward Pass:")                                                          # Header
        print(f"Input: {X}")                                                                    # Print input
        print(f"Hidden z: {self.z1}")                                                           # Print hidden weighted sum
        print(f"Hidden a: {self.a1}")                                                           # Print hidden activation
        print(f"Output z: {self.z2}")                                                           # Print output weighted sum
        print(f"Output: {output}")                                                              # Print final output
        return output                                                                           # Return output

# ===== MULTI-LAYER MANUAL CALCULATION =====
def manual_forward_pass():                                                                      # Manual forward pass calculation
    print("Manual Multi-Layer Calculation:")                                                    # Header
    
    # Input
    x = np.array([2, 3])                                                                        # Input vector
    print(f"Input: {x}")                                                                        # Print input
    
    # Hidden layer weights
    W1 = np.array([[0.5, -0.2], [0.8, 0.4]])                                                    # Hidden layer weights matrix
    b1 = np.array([1, -1])                                                                      # Hidden layer bias
    
    # Calculate hidden layer
    z1 = np.dot(x, W1) + b1                                                                     # Hidden weighted sum
    a1 = sigmoid(z1)                                                                            # Hidden activation
    print(f"Hidden layer z: {z1}")                                                              # Print hidden z
    print(f"Hidden layer a: {a1}")                                                              # Print hidden activation
    
    # Output layer weights
    W2 = np.array([[1.0], [-1.0]])                                                              # Output layer weights
    b2 = np.array([0.5])                                                                        # Output layer bias
    
    # Calculate output
    z2 = np.dot(a1, W2) + b2                                                                    # Output weighted sum
    a2 = sigmoid(z2)                                                                            # Output activation
    print(f"Output z: {z2}")                                                                    # Print output z
    print(f"Output: {a2}")                                                                      # Print final output
    print()                                                                                     # Empty line

# ===== BATCH PROCESSING =====
def batch_forward_demo():                                                                       # Demonstrate batch processing
    print("Batch Forward Pass:")                                                                # Header
    
    # Create batch of inputs
    X_batch = np.array([[2, 3], [1, 4], [3, 1]])                                                # Batch of 3 inputs
    
    # Create simple network
    net = SimpleNN(input_size=2, hidden_size=3, output_size=1)                                  # Initialize network
    
    # Process batch
    outputs = net.forward(X_batch)                                                              # Forward pass for batch
    
    print(f"Input batch shape: {X_batch.shape}")                                                # Print batch shape
    print(f"Outputs: {outputs.flatten()}")                                                      # Print outputs
    print()                                                                                     # Empty line

# ===== VISUALIZATION =====
def visualize_forward_pass(): # Visualize forward propagation
    # Create network
    net = SimpleNN(input_size=2, hidden_size=3, output_size=1) # Initialize network
    
    # Sample input
    X = np.array([[2, 3]]) # Sample input
    output = net.forward(X) # Run forward pass
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4)) # Create subplots
    
    # Input layer
    axes[0].bar([0, 1], X[0], color='blue', alpha=0.7) # Plot input values
    axes[0].set_title('Input Layer') # Set title
    axes[0].set_ylabel('Value') # Set y-label
    axes[0].set_xticks([0, 1]) # Set x-ticks
    axes[0].set_xticklabels(['x1', 'x2']) # Set x-tick labels

    # Hidden layer
    axes[1].bar(range(3), net.a1[0], color='green', alpha=0.7) # Plot hidden activations
    axes[1].set_title('Hidden Layer') # Set title
    axes[1].set_ylabel('Activation') # Set y-label
    axes[1].set_ylim([0, 1]) # Set y-limits
    axes[1].set_xticks([0, 1, 2]) # Set x-ticks
    axes[1].set_xticklabels(['h1', 'h2', 'h3']) # Set x-tick labels

    # Output layer
    axes[2].bar([0], output[0], color='red', alpha=0.7) # Plot output
    axes[2].set_title('Output Layer') # Set title
    axes[2].set_ylabel('Output') # Set y-label
    axes[2].set_ylim([0, 1]) # Set y-limits
    axes[2].set_xticks([0]) # Set x-ticks
    axes[2].set_xticklabels(['y']) # Set x-tick label

    plt.tight_layout() # Adjust layout
    plt.show() # Display plot

# ===== INFORMATION FLOW DIAGRAM =====
def plot_information_flow(): # Plot information flow through network
    # Create network and run forward pass
    net = SimpleNN(input_size=2, hidden_size=3, output_size=1) # Initialize network
    X = np.array([[2, 3]]) # Sample input
    net.forward(X) # Run forward pass

    # Create flow diagram
    fig, ax = plt.subplots(figsize=(10, 6)) # Create figure
    
    # Layer positions
    input_y = [0.7, 0.3] # Input neuron y-positions
    hidden_y = [0.8, 0.5, 0.2] # Hidden neuron y-positions
    output_y = [0.5] # Output neuron y-position

    # Draw neurons
    for i, y in enumerate(input_y): # Draw input neurons
        circle = plt.Circle((0.2, y), 0.05, color='blue', alpha=0.7) # Create circle
        ax.add_patch(circle) # Add to plot
        ax.text(0.1, y, f'x{i+1}={X[0,i]:.1f}', ha='right') # Add label

    for i, y in enumerate(hidden_y): # Draw hidden neurons
        circle = plt.Circle((0.5, y), 0.05, color='green', alpha=0.7) # Create circle
        ax.add_patch(circle) # Add to plot
        ax.text(0.6, y, f'h{i+1}={net.a1[0,i]:.2f}', ha='left') # Add label

    for i, y in enumerate(output_y): # Draw output neurons
        circle = plt.Circle((0.8, y), 0.05, color='red', alpha=0.7) # Create circle
        ax.add_patch(circle) # Add to plot
        ax.text(0.9, y, f'y={net.a2[0,0]:.2f}', ha='left') # Add label

    # Draw connections
    for i, y1 in enumerate(input_y): # Draw input to hidden connections
        for j, y2 in enumerate(hidden_y): # For each hidden neuron
            ax.plot([0.2, 0.5], [y1, y2], 'gray', alpha=0.3, linewidth=1) # Draw connection line

    for i, y1 in enumerate(hidden_y): # Draw hidden to output connections
        for j, y2 in enumerate(output_y): # For each output neuron
            ax.plot([0.5, 0.8], [y1, y2], 'gray', alpha=0.3, linewidth=1) # Draw connection line

    # Labels
    ax.text(0.2, 0.95, 'Input Layer', ha='center', fontsize=12, fontweight='bold') # Input layer label
    ax.text(0.5, 0.95, 'Hidden Layer', ha='center', fontsize=12, fontweight='bold') # Hidden layer label
    ax.text(0.8, 0.95, 'Output Layer', ha='center', fontsize=12, fontweight='bold') # Output layer label

    ax.set_xlim([0, 1]) # Set x-limits
    ax.set_ylim([0, 1]) # Set y-limits
    ax.axis('off') # Turn off axis
    ax.set_title('Forward Propagation: Information Flow') # Set title

    plt.tight_layout() # Adjust layout
    plt.show() # Display plot

# ===== MAIN EXECUTION =====
single_neuron_demo() # Run single neuron demo
manual_forward_pass() # Run manual calculation
batch_forward_demo() # Run batch demo
visualize_forward_pass() # Visualize activations
plot_information_flow() # Plot network flow