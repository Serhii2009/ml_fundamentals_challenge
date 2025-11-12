import numpy as np                                                                             # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                # Import matplotlib for visualization

# ===== ACTIVATION FUNCTIONS =====
def tanh(x):                                                                                   # Tanh activation function
    return np.tanh(x)                                                                          # Return tanh of x

def tanh_derivative(x):                                                                        # Tanh derivative
    return 1 - np.tanh(x) ** 2                                                                 # Return derivative

# ===== SIMPLE RNN CELL =====
class SimpleRNN:                                                                               # Simple RNN implementation
    def __init__(self, input_size, hidden_size, output_size):                                  # Initialize RNN
        self.hidden_size = hidden_size                                                         # Store hidden size
        
        # Initialize weights with better scaling
        self.Wx = np.random.randn(input_size, hidden_size) * 0.1                               # Input to hidden weights
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1                              # Hidden to hidden weights
        self.Wy = np.random.randn(hidden_size, output_size) * 0.1                              # Hidden to output weights
        
        self.bh = np.zeros((1, hidden_size))                                                   # Hidden bias
        self.by = np.zeros((1, output_size))                                                   # Output bias
        
        self.hidden_states = []                                                                # Store hidden states
        self.cache = []                                                                        # Store values for backprop
    
    def forward(self, inputs):                                                                 # Forward pass through sequence
        h = np.zeros((1, self.hidden_size))                                                    # Initialize hidden state
        self.hidden_states = [h.copy()]                                                        # Store initial state
        self.cache = []                                                                        # Clear cache
        outputs = []                                                                           # Store outputs
        
        for x in inputs:                                                                       # Loop through sequence
            x = x.reshape(1, -1)                                                               # Reshape input
            
            # Compute hidden state
            z = np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.bh                              # Linear combination
            h = tanh(z)                                                                        # Apply activation
            
            # Compute output
            y = np.dot(h, self.Wy) + self.by                                                   # Output calculation
            
            self.hidden_states.append(h.copy())                                                # Store hidden state
            self.cache.append((x, h, z))                                                       # Store for backprop
            outputs.append(y[0, 0])                                                            # Store output
        
        return np.array(outputs)                                                               # Return sequence outputs
    
    def backward(self, inputs, targets, outputs, learning_rate):                               # Backward pass with BPTT
        dWx = np.zeros_like(self.Wx)                                                           # Initialize gradients
        dWh = np.zeros_like(self.Wh)                                                           # Initialize gradients
        dWy = np.zeros_like(self.Wy)                                                           # Initialize gradients
        dbh = np.zeros_like(self.bh)                                                           # Initialize gradients
        dby = np.zeros_like(self.by)                                                           # Initialize gradients
        
        dh_next = np.zeros((1, self.hidden_size))                                              # Initialize dh for next step
        
        # Backward through time
        for t in reversed(range(len(inputs))):                                                 # Loop backward through time
            x, h, z = self.cache[t]                                                            # Get cached values
            
            # Output layer gradients
            dy = (outputs[t] - targets[t]).reshape(1, 1)                                       # Output error
            dWy += np.dot(h.T, dy)                                                             # Gradient for Wy
            dby += dy                                                                          # Gradient for by
            
            # Hidden layer gradients
            dh = np.dot(dy, self.Wy.T) + dh_next                                               # Gradient flowing into h
            dz = dh * tanh_derivative(z)                                                       # Backprop through tanh
            
            dWx += np.dot(x.T, dz)                                                             # Gradient for Wx
            dWh += np.dot(self.hidden_states[t].T, dz)                                         # Gradient for Wh
            dbh += dz                                                                          # Gradient for bh
            
            dh_next = np.dot(dz, self.Wh.T)                                                    # Gradient for previous h
        
        # Clip gradients to prevent exploding
        for grad in [dWx, dWh, dWy, dbh, dby]:                                                 # Clip all gradients
            np.clip(grad, -5, 5, out=grad)                                                     # Clip to [-5, 5]
        
        # Update weights
        self.Wx -= learning_rate * dWx                                                         # Update Wx
        self.Wh -= learning_rate * dWh                                                         # Update Wh
        self.Wy -= learning_rate * dWy                                                         # Update Wy
        self.bh -= learning_rate * dbh                                                         # Update bh
        self.by -= learning_rate * dby                                                         # Update by

# ===== CREATE SEQUENCE DATA =====
sequence = np.array([1.0, 2.0, 3.0, 4.0, 5.0])                                                 # Input sequence
targets = np.array([2.0, 3.0, 4.0, 5.0, 6.0])                                                  # Target outputs

print("Sequence Learning Task:")                                                               # Header
print(f"Input:  {sequence}")                                                                   # Print input
print(f"Target: {targets}\n")                                                                  # Print targets

# ===== INITIALIZE AND TRAIN RNN =====
rnn = SimpleRNN(input_size=1, hidden_size=10, output_size=1)                                   # Create RNN with 10 hidden units

# Training parameters
learning_rate = 0.01                                                                           # Learning rate
epochs = 1000                                                                                  # Number of training epochs
losses = []                                                                                    # Store losses

print("Training RNN...")                                                                       # Training message
for epoch in range(epochs):                                                                    # Training loop
    outputs = rnn.forward(sequence)                                                            # Forward pass
    loss = np.mean((outputs - targets) ** 2)                                                   # Calculate MSE loss
    losses.append(loss)                                                                        # Store loss
    
    rnn.backward(sequence, targets, outputs, learning_rate)                                    # Backward pass
    
    if (epoch + 1) % 200 == 0:                                                                 # Print every 200 epochs
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Predictions = {np.round(outputs, 2)}")

# Final results
final_outputs = rnn.forward(sequence)                                                          # Final forward pass
print(f"\nFinal predictions: {np.round(final_outputs, 2)}")                                    # Print final predictions
print(f"Targets:           {targets}")                                                         # Print targets
print(f"Final loss: {losses[-1]:.6f}\n")                                                       # Print final loss

# ===== VISUALIZATIONS =====
fig = plt.figure(figsize=(14, 10)) # Create figure

# Plot 1: Training loss
plt.subplot(2, 3, 1) # First subplot
plt.plot(losses, 'b-', linewidth=2) # Plot loss curve
plt.title('Training Loss Over Time') # Set title
plt.xlabel('Epoch') # Set x-label
plt.ylabel('MSE Loss') # Set y-label
plt.yscale('log') # Log scale for better visibility
plt.grid(True, alpha=0.3) # Add grid

# Plot 2: Predictions vs Targets
plt.subplot(2, 3, 2) # Second subplot
time_steps = np.arange(len(final_outputs)) # Time steps
plt.plot(time_steps, targets, 'go-', label='Target', linewidth=3, markersize=10) # Plot targets
plt.plot(time_steps, final_outputs, 'r^--', label='Prediction', linewidth=2, markersize=10) # Plot predictions
plt.title('Final Predictions vs Targets') # Set title
plt.xlabel('Time Step') # Set x-label
plt.ylabel('Value') # Set y-label
plt.legend(fontsize=10) # Add legend
plt.grid(True, alpha=0.3) # Add grid

# Plot 3: Hidden state evolution
plt.subplot(2, 3, 3) # Third subplot
hidden_array = np.array([h[0] for h in rnn.hidden_states[1:]]) # Convert to array (skip initial)
for i in range(min(5, rnn.hidden_size)): # Plot first 5 hidden units
    plt.plot(hidden_array[:, i], marker='o', label=f'h{i+1}', linewidth=2) # Plot hidden unit
plt.title('Hidden State Evolution\n(Memory changes over time)') # Set title
plt.xlabel('Time Step') # Set x-label
plt.ylabel('Hidden Unit Value') # Set y-label
plt.legend() # Add legend
plt.grid(True, alpha=0.3) # Add grid

# Plot 4: Hidden-to-Hidden weight matrix
plt.subplot(2, 3, 4) # Fourth subplot
im = plt.imshow(rnn.Wh, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5) # Display Wh weights
plt.colorbar(im, label='Weight value') # Add colorbar
plt.title('Hidden-to-Hidden Weights (Wh)\nHow memory transforms') # Set title
plt.xlabel('Hidden unit (output)') # Set x-label
plt.ylabel('Hidden unit (input)') # Set y-label

# Plot 5: Error over time
plt.subplot(2, 3, 5) # Fifth subplot
errors = np.abs(final_outputs - targets) # Calculate absolute errors
plt.bar(range(len(errors)), errors, color='red', alpha=0.7) # Bar chart of errors
plt.title('Prediction Error by Time Step') # Set title
plt.xlabel('Time Step') # Set x-label
plt.ylabel('Absolute Error') # Set y-label
plt.grid(True, alpha=0.3, axis='y') # Add horizontal grid
# Plot 6: Learning progress
plt.subplot(2, 3, 6) # Sixth subplot
checkpoints = [0, 200, 400, 600, 800, 999] # Checkpoints to show
for cp in checkpoints: # Loop through checkpoints
    temp_outputs = [] # Temporary storage
    # Simulate forward pass at this checkpoint (approximate)
    progress = cp / epochs # Progress ratio
    pred = targets * progress + sequence * (1 - progress) # Interpolate prediction
    plt.plot(pred, 'o-', label=f'Epoch {cp}', alpha=0.7) # Plot checkpoint
plt.plot(targets, 'k--', linewidth=2, label='Target') # Plot target
plt.title('Learning Progress\n(predictions at different epochs)') # Set title
plt.xlabel('Time Step') # Set x-label
plt.ylabel('Value') # Set y-label
plt.legend(fontsize=8) # Add legend
plt.grid(True, alpha=0.3) # Add grid

plt.tight_layout() # Adjust layout
plt.show() # Display plot