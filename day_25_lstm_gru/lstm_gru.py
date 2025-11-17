import numpy as np                                                                             # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                # Import matplotlib for visualization

# ===== ACTIVATION FUNCTIONS =====
def sigmoid(x):                                                                                # Sigmoid activation (0 to 1)
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))                                            # Clip to prevent overflow

def tanh(x):                                                                                   # Tanh activation (-1 to 1)
    return np.tanh(np.clip(x, -500, 500))                                                      # Clip to prevent overflow

def tanh_derivative(x):                                                                        # Tanh derivative
    return 1 - np.tanh(x) ** 2                                                                 # Derivative formula

# ===== LSTM CELL IMPLEMENTATION =====
class LSTMCell:                                                                                # LSTM cell with gates
    def __init__(self, input_size, hidden_size):                                               # Initialize LSTM
        self.hidden_size = hidden_size                                                         # Store hidden dimension
        
        # Forget gate weights
        self.Wf = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Forget gate weights
        self.bf = np.zeros((1, hidden_size))                                                   # Forget gate bias
        
        # Input gate weights
        self.Wi = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Input gate weights
        self.bi = np.zeros((1, hidden_size))                                                   # Input gate bias
        
        # Cell candidate weights
        self.Wc = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Cell candidate weights
        self.bc = np.zeros((1, hidden_size))                                                   # Cell candidate bias
        
        # Output gate weights
        self.Wo = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Output gate weights
        self.bo = np.zeros((1, hidden_size))                                                   # Output gate bias
        
        self.cache = []                                                                        # Store values for backprop
    
    def forward(self, x, h_prev, c_prev):                                                      # Forward pass through LSTM
        x = x.reshape(1, -1)                                                                   # Reshape input
        combined = np.hstack([x, h_prev])                                                      # Concatenate input and hidden
        
        # Forget gate: decides what to forget from cell state
        f = sigmoid(np.dot(combined, self.Wf) + self.bf)                                       # Compute forget gate
        
        # Input gate: decides what new info to add
        i = sigmoid(np.dot(combined, self.Wi) + self.bi)                                       # Compute input gate
        
        # Cell candidate: new potential values
        c_tilde = tanh(np.dot(combined, self.Wc) + self.bc)                                    # Compute cell candidate
        
        # Update cell state
        c = f * c_prev + i * c_tilde                                                           # New cell state (forget + input)
        
        # Output gate: decides what to output
        o = sigmoid(np.dot(combined, self.Wo) + self.bo)                                       # Compute output gate
        
        # New hidden state
        h = o * tanh(c)                                                                        # New hidden state
        
        self.cache.append((x, h_prev, c_prev, f, i, c_tilde, c, o, h))                         # Store for visualization
        
        return h, c, f, i, o                                                                   # Return states and gates

# ===== GRU CELL IMPLEMENTATION =====
class GRUCell:                                                                                 # GRU cell with gates
    def __init__(self, input_size, hidden_size):                                               # Initialize GRU
        self.hidden_size = hidden_size                                                         # Store hidden dimension
        
        # Reset gate weights
        self.Wr = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Reset gate weights
        self.br = np.zeros((1, hidden_size))                                                   # Reset gate bias
        
        # Update gate weights
        self.Wz = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Update gate weights
        self.bz = np.zeros((1, hidden_size))                                                   # Update gate bias
        
        # Candidate hidden state weights
        self.Wh = np.random.randn(input_size + hidden_size, hidden_size) * 0.1                 # Hidden candidate weights
        self.bh = np.zeros((1, hidden_size))                                                   # Hidden candidate bias
        
        self.cache = []                                                                        # Store values for visualization
    
    def forward(self, x, h_prev):                                                              # Forward pass through GRU
        x = x.reshape(1, -1)                                                                   # Reshape input
        combined = np.hstack([x, h_prev])                                                      # Concatenate input and hidden
        
        # Reset gate: decides how much past to use
        r = sigmoid(np.dot(combined, self.Wr) + self.br)                                       # Compute reset gate
        
        # Update gate: decides balance between old and new
        z = sigmoid(np.dot(combined, self.Wz) + self.bz)                                       # Compute update gate
        
        # Candidate hidden state (uses reset gate)
        combined_reset = np.hstack([x, r * h_prev])                                            # Reset applied to h_prev
        h_tilde = tanh(np.dot(combined_reset, self.Wh) + self.bh)                              # Candidate hidden state
        
        # New hidden state (interpolation between old and candidate)
        h = (1 - z) * h_prev + z * h_tilde                                                     # Update hidden state
        
        self.cache.append((x, h_prev, r, z, h_tilde, h))                                       # Store for visualization
        
        return h, r, z                                                                         # Return state and gates

# ===== GENERATE SEQUENCE DATA =====
def generate_long_sequence(length=20, pattern='sine'):                                         # Generate sequence with pattern
    """Create sequence that requires long-term memory"""
    if pattern == 'sine':                                                                      # Sine wave pattern
        x = np.linspace(0, 4*np.pi, length)                                                    # Time steps
        sequence = np.sin(x) + np.sin(0.5*x)                                                   # Combined sine waves
    elif pattern == 'sawtooth':                                                                # Sawtooth pattern
        sequence = np.mod(np.arange(length), 5) / 5.0                                          # Repeating ramp
    elif pattern == 'steps':                                                                   # Step pattern
        sequence = np.array([1 if i % 4 < 2 else -1 for i in range(length)], dtype=float)      # Square wave
    
    targets = np.roll(sequence, -1)                                                            # Shift for prediction
    targets[-1] = sequence[-1]                                                                 # Last target
    
    return sequence, targets                                                                   # Return sequence and targets

# ===== WRAPPER MODELS FOR TRAINING =====
class LSTMModel:                                                                               # LSTM model wrapper
    def __init__(self, input_size, hidden_size, output_size):                                  # Initialize model
        self.lstm = LSTMCell(input_size, hidden_size)                                          # Create LSTM cell
        self.Wy = np.random.randn(hidden_size, output_size) * 0.1                              # Output weights
        self.by = np.zeros((1, output_size))                                                   # Output bias
        self.hidden_size = hidden_size                                                         # Store hidden size
        
    def forward(self, inputs):                                                                 # Forward through sequence
        h = np.zeros((1, self.hidden_size))                                                    # Initialize hidden state
        c = np.zeros((1, self.hidden_size))                                                    # Initialize cell state
        
        self.lstm.cache = []                                                                   # Clear cache
        outputs = []                                                                           # Store outputs
        gate_history = {'forget': [], 'input': [], 'output': []}                               # Store gate activations
        
        for x in inputs:                                                                       # Loop through sequence
            h, c, f, i, o = self.lstm.forward(x, h, c)                                         # LSTM forward pass
            y = np.dot(h, self.Wy) + self.by                                                   # Compute output
            outputs.append(y[0, 0])                                                            # Store output
            
            gate_history['forget'].append(f[0, 0])                                             # Store forget gate
            gate_history['input'].append(i[0, 0])                                              # Store input gate
            gate_history['output'].append(o[0, 0])                                             # Store output gate
        
        return np.array(outputs), gate_history                                                 # Return outputs and gates

class GRUModel:                                                                                # GRU model wrapper
    def __init__(self, input_size, hidden_size, output_size):                                  # Initialize model
        self.gru = GRUCell(input_size, hidden_size)                                            # Create GRU cell
        self.Wy = np.random.randn(hidden_size, output_size) * 0.1                              # Output weights
        self.by = np.zeros((1, output_size))                                                   # Output bias
        self.hidden_size = hidden_size                                                         # Store hidden size
        
    def forward(self, inputs):                                                                 # Forward through sequence
        h = np.zeros((1, self.hidden_size))                                                    # Initialize hidden state
        
        self.gru.cache = []                                                                    # Clear cache
        outputs = []                                                                           # Store outputs
        gate_history = {'reset': [], 'update': []}                                             # Store gate activations
        
        for x in inputs:                                                                       # Loop through sequence
            h, r, z = self.gru.forward(x, h)                                                   # GRU forward pass
            y = np.dot(h, self.Wy) + self.by                                                   # Compute output
            outputs.append(y[0, 0])                                                            # Store output
            
            gate_history['reset'].append(r[0, 0])                                              # Store reset gate
            gate_history['update'].append(z[0, 0])                                             # Store update gate
        
        return np.array(outputs), gate_history                                                 # Return outputs and gates

# ===== CREATE AND TEST SEQUENCES =====
print("=" * 70)                                                                                # Header separator
print("LSTM vs GRU: Long-Term Memory Demonstration")                                           # Main title
print("=" * 70)                                                                                # Header separator

# Generate test sequence
sequence, targets = generate_long_sequence(length=20, pattern='sine')                          # Create sine wave sequence

print("\nSequence Pattern: Sine Wave (tests long-term memory)")                                # Pattern description
print(f"Sequence length: {len(sequence)} steps")                                               # Length info
print(f"First 5 values:  {np.round(sequence[:5], 3)}")                                         # Show first values
print(f"Target 5 values: {np.round(targets[:5], 3)}")                                          # Show targets
print(f"Task: Predict next value in sequence\n")                                               # Task description

# ===== INITIALIZE MODELS =====
lstm_model = LSTMModel(input_size=1, hidden_size=8, output_size=1)                             # Create LSTM with 8 units
gru_model = GRUModel(input_size=1, hidden_size=8, output_size=1)                               # Create GRU with 8 units

# ===== FORWARD PASS (NO TRAINING - JUST DEMONSTRATION) =====
print("Running forward pass through both models...")                                           # Processing message
lstm_outputs, lstm_gates = lstm_model.forward(sequence)                                        # LSTM forward pass
gru_outputs, gru_gates = gru_model.forward(sequence)                                           # GRU forward pass

# Calculate errors
lstm_error = np.mean(np.abs(lstm_outputs - targets))                                           # LSTM mean error
gru_error = np.mean(np.abs(gru_outputs - targets))                                             # GRU mean error

print(f"\nLSTM Mean Absolute Error: {lstm_error:.4f}")                                         # Print LSTM error
print(f"GRU Mean Absolute Error:  {gru_error:.4f}")                                            # Print GRU error
print(f"\nNote: Models are randomly initialized (not trained)")                                # Explanation note
print("The visualization shows how gates control information flow\n")                          # Gate explanation

# ===== VISUALIZATIONS =====
fig = plt.figure(figsize=(16, 12)) # Create large figure
fig.suptitle('LSTM vs GRU: Internal Mechanism Comparison', fontsize=14, fontweight='bold') # Main title

# Plot 1: Input Sequence Pattern
plt.subplot(3, 3, 1) # First subplot
time_steps = np.arange(len(sequence)) # Time axis
plt.plot(time_steps, sequence, 'b-o', linewidth=2, markersize=6, label='Input') # Plot sequence
plt.plot(time_steps, targets, 'r--^', linewidth=2, markersize=5, label='Target', alpha=0.7) # Plot targets
plt.title('Input Sequence Pattern\n(Sine wave - requires memory)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Value') # Y-label
plt.legend(loc='upper right') # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 2: LSTM Predictions
plt.subplot(3, 3, 2) # Second subplot
plt.plot(time_steps, targets, 'go-', linewidth=3, markersize=8, label='Target', alpha=0.7) # Plot targets
plt.plot(time_steps, lstm_outputs, 'bs--', linewidth=2, markersize=6, label='LSTM') # Plot LSTM output
plt.title('LSTM Predictions\n(3 gates control)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Value') # Y-label
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 3: GRU Predictions
plt.subplot(3, 3, 3) # Third subplot
plt.plot(time_steps, targets, 'go-', linewidth=3, markersize=8, label='Target', alpha=0.7) # Plot targets
plt.plot(time_steps, gru_outputs, 'r^--', linewidth=2, markersize=6, label='GRU') # Plot GRU output
plt.title('GRU Predictions\n(2 gates control)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Value') # Y-label
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 4: LSTM Forget Gate
plt.subplot(3, 3, 4) # Fourth subplot
plt.plot(time_steps, lstm_gates['forget'], 'r-o', linewidth=2, markersize=5) # Plot forget gate
plt.title('LSTM: Forget Gate (What to erase from memory)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Gate Activation (0-1)') # Y-label
plt.ylim([0, 1.1]) # Y-axis limits
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold') # Reference line
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 5: LSTM Input Gate
plt.subplot(3, 3, 5) # Fifth subplot
plt.plot(time_steps, lstm_gates['input'], 'g-o', linewidth=2, markersize=5) # Plot input gate
plt.title('LSTM: Input Gate (What new info to add)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Gate Activation (0-1)') # Y-label
plt.ylim([0, 1.1]) # Y-axis limits
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold') # Reference line
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 6: LSTM Output Gate
plt.subplot(3, 3, 6) # Sixth subplot
plt.plot(time_steps, lstm_gates['output'], 'b-o', linewidth=2, markersize=5) # Plot output gate
plt.title('LSTM: Output Gate (What to expose)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Gate Activation (0-1)') # Y-label
plt.ylim([0, 1.1]) # Y-axis limits
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold') # Reference line
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 7: GRU Reset Gate
plt.subplot(3, 3, 7) # Seventh subplot
plt.plot(time_steps, gru_gates['reset'], 'orange', marker='o', linewidth=2, markersize=5) # Plot reset gate
plt.title('GRU: Reset Gate (How much past to use)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Gate Activation (0-1)') # Y-label
plt.ylim([0, 1.1]) # Y-axis limits
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold') # Reference line
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 8: GRU Update Gate
plt.subplot(3, 3, 8) # Eighth subplot
plt.plot(time_steps, gru_gates['update'], 'purple', marker='o', linewidth=2, markersize=5) # Plot update gate
plt.title('GRU: Update Gate (Balance old vs new)', fontsize=11) # Title
plt.xlabel('Time Step') # X-label
plt.ylabel('Gate Activation (0-1)') # Y-label
plt.ylim([0, 1.1]) # Y-axis limits
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold') # Reference line
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 9: Architecture Comparison
plt.subplot(3, 3, 9) # Ninth subplot
models = ['LSTM', 'GRU'] # Model names
num_gates = [3, 2] # Number of gates
colors = ['#3498db', '#e74c3c'] # Bar colors
bars = plt.bar(models, num_gates, color=colors, alpha=0.7, edgecolor='black', linewidth=2) # Create bars
plt.title('Architecture Complexity\n(Number of Gates)', fontsize=11) # Title
plt.ylabel('Number of Gates') # Y-label
plt.ylim([0, 4]) # Y-axis limits
for i, (bar, count) in enumerate(zip(bars, num_gates)): # Annotate bars
    height = bar.get_height() # Get bar height
    plt.text(bar.get_x() + bar.get_width()/2., height, # Position text
             f'{count} gates', ha='center', va='bottom', fontsize=12, fontweight='bold') # Gate count text
plt.grid(True, alpha=0.3, axis='y') # Grid
plt.tight_layout() # Adjust spacing
plt.show() # Display plot