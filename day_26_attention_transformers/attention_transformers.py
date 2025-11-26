import numpy as np                                                                             # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                # Import matplotlib for visualization
import seaborn as sns                                                                          # Import seaborn for advanced visualization

# ===== ACTIVATION FUNCTIONS =====
def softmax(x):                                                                                # Softmax for attention weights
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))                                      # Stability: subtract max
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)                                       # Normalize to probabilities

def sigmoid(x):                                                                                # Sigmoid activation (0 to 1)
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))                                            # Clip to prevent overflow

# ===== SIMPLE RNN FOR COMPARISON =====
class SimpleRNN:                                                                               # Basic RNN implementation
    def __init__(self, input_size, hidden_size):                                               # Initialize RNN
        self.hidden_size = hidden_size                                                         # Store hidden dimension
        self.Wx = np.random.randn(input_size, hidden_size) * 0.1                               # Input-to-hidden weights
        self.Wh = np.random.randn(hidden_size, hidden_size) * 0.1                              # Hidden-to-hidden weights
        self.b = np.zeros((1, hidden_size))                                                    # Bias term
        
    def forward(self, inputs):                                                                 # Forward pass through sequence
        h = np.zeros((1, self.hidden_size))                                                    # Initialize hidden state
        hidden_states = []                                                                     # Store all hidden states
        
        for x in inputs:                                                                       # Process each token
            x = x.reshape(1, -1)                                                               # Reshape input
            h = np.tanh(np.dot(x, self.Wx) + np.dot(h, self.Wh) + self.b)                      # Update hidden state
            hidden_states.append(h.copy())                                                     # Store hidden state
        
        return hidden_states                                                                   # Return all states

# ===== SELF-ATTENTION MECHANISM =====
class SelfAttention:                                                                           # Self-Attention implementation
    def __init__(self, d_model):                                                               # Initialize attention
        self.d_model = d_model                                                                 # Model dimension
        self.W_q = np.random.randn(d_model, d_model) * 0.1                                     # Query projection matrix
        self.W_k = np.random.randn(d_model, d_model) * 0.1                                     # Key projection matrix
        self.W_v = np.random.randn(d_model, d_model) * 0.1                                     # Value projection matrix
        
    def forward(self, X):                                                                      # Forward pass
        # Step 1: Create Q, K, V
        Q = np.dot(X, self.W_q)                                                                # Query: what I'm looking for
        K = np.dot(X, self.W_k)                                                                # Key: what I can offer
        V = np.dot(X, self.W_v)                                                                # Value: actual information
        
        # Step 2: Calculate attention scores
        scores = np.dot(Q, K.T)                                                                # QK^T: similarity matrix
        
        # Step 3: Scale scores
        scores = scores / np.sqrt(self.d_model)                                                # Scale by sqrt(d_k)
        
        # Step 4: Apply softmax
        attention_weights = softmax(scores)                                                    # Convert to probabilities
        
        # Step 5: Apply attention to values
        output = np.dot(attention_weights, V)                                                  # Weighted sum of values
        
        return output, attention_weights, Q, K, V                                              # Return all components

# ===== MULTI-HEAD ATTENTION =====
class MultiHeadAttention:                                                                      # Multi-head attention
    def __init__(self, d_model, num_heads):                                                    # Initialize multi-head
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"              # Check divisibility
        
        self.d_model = d_model                                                                 # Model dimension
        self.num_heads = num_heads                                                             # Number of heads
        self.d_k = d_model // num_heads                                                        # Dimension per head
        
        # Create separate attention for each head
        self.heads = [SelfAttention(self.d_k) for _ in range(num_heads)]                       # Independent heads
        self.W_o = np.random.randn(d_model, d_model) * 0.1                                     # Output projection
        
    def forward(self, X):                                                                      # Forward pass
        batch_size, seq_len, _ = X.shape                                                       # Get dimensions
        
        # Split input for each head
        X_split = np.array_split(X, self.num_heads, axis=-1)                                   # Split along features
        
        # Apply attention in each head
        head_outputs = []                                                                      # Store head outputs
        head_weights = []                                                                      # Store attention weights
        
        for i, head in enumerate(self.heads):                                                  # Process each head
            output, weights, _, _, _ = head.forward(X_split[i][0])                             # Single-head attention
            head_outputs.append(output)                                                        # Store output
            head_weights.append(weights)                                                       # Store weights
        
        # Concatenate heads
        concat_output = np.concatenate(head_outputs, axis=-1)                                  # Merge all heads
        
        # Final projection
        final_output = np.dot(concat_output, self.W_o)                                         # Linear transformation
        
        return final_output, head_weights                                                      # Return output and weights

# ===== TEST SENTENCE ENCODING =====
def encode_sentence(sentence, vocab, embedding_dim=8):                                         # Convert sentence to embeddings
    """Convert sentence to numerical embeddings"""
    tokens = sentence.lower().split()                                                          # Tokenize sentence
    embeddings = []                                                                            # Store embeddings
    
    for token in tokens:                                                                       # Process each word
        if token in vocab:                                                                     # Check if in vocabulary
            embeddings.append(vocab[token])                                                    # Get embedding
        else:                                                                                  # Unknown word
            embeddings.append(np.random.randn(embedding_dim) * 0.1)                            # Random embedding
    
    return np.array(embeddings), tokens                                                        # Return embeddings and tokens

# ===== DEMONSTRATE ATTENTION VS RNN =====
print("=" * 80)                                                                                # Header separator
print("ATTENTION MECHANISM vs RNN: Information Flow Comparison")                               # Main title
print("=" * 80)                                                                                # Header separator

# Create simple vocabulary with embeddings
embedding_dim = 8                                                                              # Embedding dimension
vocab = {                                                                                      # Simple vocabulary
    'the': np.array([0.1, 0.2, 0.3, 0.1, 0.5, 0.2, 0.1, 0.3]),                                 # Article
    'cat': np.array([0.8, 0.1, 0.2, 0.7, 0.1, 0.6, 0.3, 0.2]),                                 # Noun (animal)
    'sat': np.array([0.2, 0.7, 0.1, 0.3, 0.8, 0.1, 0.5, 0.4]),                                 # Verb (action)
    'on': np.array([0.3, 0.1, 0.6, 0.2, 0.3, 0.4, 0.2, 0.1]),                                  # Preposition
    'mat': np.array([0.5, 0.3, 0.2, 0.6, 0.2, 0.7, 0.1, 0.5]),                                 # Noun (object)
}

# Test sentence
test_sentence = "the cat sat on the mat"                                                       # Example sentence
embeddings, tokens = encode_sentence(test_sentence, vocab, embedding_dim)                      # Get embeddings
print(f"\nTest Sentence: '{test_sentence}'")                                                   # Display sentence
print(f"Tokens: {tokens}")                                                                     # Display tokens
print(f"Sequence Length: {len(tokens)} words")                                                 # Length
print(f"Embedding Dimension: {embedding_dim}")                                                 # Dimension

# ===== RNN PROCESSING =====
print("\n" + "=" * 80)                                                                         # Section separator
print("RNN PROCESSING (Sequential Information Flow)")                                          # RNN section title
print("=" * 80)                                                                                # Section separator

rnn = SimpleRNN(input_size=embedding_dim, hidden_size=8)                                       # Create RNN
rnn_hidden_states = rnn.forward(embeddings)                                                    # Process sequence

print("\nRNN Hidden States (information fades over time):")                                    # RNN states title
for i, (token, state) in enumerate(zip(tokens, rnn_hidden_states)):                            # Show each state
    print(f"  Step {i+1} ('{token}'): {np.round(state[0][:4], 3)}... (showing first 4 dims)")  # First 4 dimensions

# Calculate information retention (correlation with first state)
first_state = rnn_hidden_states[0]                                                             # Initial state
correlations = []                                                                              # Store correlations
for state in rnn_hidden_states:                                                                # Check each state
    corr = np.corrcoef(first_state[0], state[0])[0, 1]                                         # Correlation coefficient
    correlations.append(corr)                                                                  # Store result

print(f"\nInformation Retention (correlation with initial state):")                            # Retention title
for i, (token, corr) in enumerate(zip(tokens, correlations)):                                  # Show retention
    print(f"  '{token}': {corr:.3f} {'⚠️ FADING' if corr < 0.5 and i > 0 else '✓'}")          # Indicate fading

# ===== ATTENTION PROCESSING =====
print("\n" + "=" * 80)                                                                         # Section separator
print("SELF-ATTENTION PROCESSING (Direct Information Access)")                                 # Attention section
print("=" * 80)                                                                                # Section separator

# Add batch dimension for attention
embeddings_batch = embeddings.reshape(1, len(tokens), embedding_dim)                           # Add batch dimension

attention = SelfAttention(d_model=embedding_dim)                                               # Create attention
output, attn_weights, Q, K, V = attention.forward(embeddings_batch[0])                         # Apply attention

print("\nAttention Weights Matrix (who looks at whom):")                                       # Weights title
print("Rows: Query words | Columns: Key words")                                                # Matrix explanation
print(np.round(attn_weights, 3))                                                               # Show weights matrix

print("\n" + "-" * 80)                                                                         # Separator
print("Interpretation: Each row shows where that word 'pays attention'")                       # Explanation
print("-" * 80)                                                                                # Separator

# Analyze specific attention patterns
for i, query_token in enumerate(tokens):                                                       # Analyze each word
    weights = attn_weights[i]                                                                  # Get attention weights
    max_idx = np.argmax(weights)                                                               # Find most attended
    max_weight = weights[max_idx]                                                              # Get max weight
    
    print(f"\n'{query_token}' (position {i+1}) attends most to:")                              # Query word
    print(f"  → '{tokens[max_idx]}' (weight: {max_weight:.3f})")                               # Most attended word
    
    # Show top 3 attended words
    top3_indices = np.argsort(weights)[-3:][::-1]                                              # Top 3 indices
    print(f"  Top 3 attended words:")                                                          # Top 3 header
    for idx in top3_indices:                                                                   # Show top 3
        print(f"    - '{tokens[idx]}': {weights[idx]:.3f}")                                    # Word and weight

# ===== MULTI-HEAD ATTENTION =====
print("\n" + "=" * 80)                                                                         # Section separator
print("MULTI-HEAD ATTENTION (Multiple Perspectives)")                                          # Multi-head section
print("=" * 80)                                                                                # Section separator

num_heads = 4                                                                                  # Number of heads
multi_head = MultiHeadAttention(d_model=embedding_dim, num_heads=num_heads)                    # Create multi-head
mh_output, mh_weights = multi_head.forward(embeddings_batch)                                   # Apply multi-head

print(f"\nNumber of Heads: {num_heads}")                                                       # Number of heads
print(f"Dimension per Head: {embedding_dim // num_heads}")                                     # Dimension per head

print("\nEach head learns different relationships:")                                           # Heads explanation
for head_idx, head_weights in enumerate(mh_weights):                                           # Show each head
    print(f"\n--- Head {head_idx + 1} ---")                                                    # Head number
    
    # Find strongest attention pattern in this head
    max_pos = np.unravel_index(np.argmax(head_weights), head_weights.shape)                    # Strongest attention
    from_token = tokens[max_pos[0]]                                                            # Source word
    to_token = tokens[max_pos[1]]                                                              # Target word
    weight = head_weights[max_pos]                                                             # Weight value
    
    print(f"Strongest pattern: '{from_token}' → '{to_token}' (weight: {weight:.3f})")          # Show pattern
    
    # Show this head's focus
    avg_weights = np.mean(head_weights, axis=0)                                                # Average attention
    most_attended_idx = np.argmax(avg_weights)                                                 # Most attended word
    print(f"Most attended word overall: '{tokens[most_attended_idx]}'")                        # Show most attended



# ===== VISUALIZATIONS =====
fig = plt.figure(figsize=(18, 12)) # Create large figure
fig.suptitle('Attention Mechanism vs RNN: Complete Comparison', fontsize=16, fontweight='bold') # Main title

# Plot 1: RNN Information Fading
plt.subplot(3, 3, 1) # First subplot
steps = np.arange(len(tokens)) # Time steps
plt.plot(steps, correlations, 'r-o', linewidth=3, markersize=8) # Plot correlation decay
plt.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='50% retention') # Reference line
plt.title('RNN: Information Fading\n(correlation with initial state)', fontsize=11) # Title
plt.xlabel('Sequence Position') # X-label
plt.ylabel('Correlation') # Y-label
plt.xticks(steps, tokens, rotation=45) # Token labels
plt.ylim([0, 1.1]) # Y-axis limits
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 2: Attention Weight Heatmap
plt.subplot(3, 3, 2) # Second subplot
sns.heatmap(attn_weights, annot=True, fmt='.2f', cmap='YlOrRd', # Attention heatmap
            xticklabels=tokens, yticklabels=tokens, cbar_kws={'label': 'Attention Weight'}) # Labels
plt.title('Self-Attention Weights\n(who attends to whom)', fontsize=11) # Title
plt.xlabel('Key (attended to)') # X-label
plt.ylabel('Query (attending from)') # Y-label

# Plot 3: Multi-Head Attention Overview
plt.subplot(3, 3, 3) # Third subplot
head_names = [f'Head {i+1}' for i in range(num_heads)] # Head labels
head_diversity = [np.std(weights) for weights in mh_weights] # Weight diversity
colors = plt.cm.viridis(np.linspace(0, 1, num_heads)) # Colors for bars
bars = plt.bar(head_names, head_diversity, color=colors, alpha=0.7, edgecolor='black', linewidth=2) # Bar plot
plt.title('Multi-Head Attention Diversity\n(standard deviation of weights)', fontsize=11) # Title
plt.ylabel('Weight Diversity') # Y-label
plt.xticks(rotation=45) # Rotate labels
plt.grid(True, alpha=0.3, axis='y') # Grid

# Plot 4-7: Individual Head Patterns
for head_idx in range(min(4, num_heads)): # First 4 heads
    plt.subplot(3, 3, 4 + head_idx) # Subplot position
    sns.heatmap(mh_weights[head_idx], annot=True, fmt='.2f', cmap='Blues', # Head heatmap
                xticklabels=tokens, yticklabels=tokens, cbar=False) # No colorbar
    plt.title(f'Head {head_idx + 1} Attention Pattern', fontsize=10) # Title
    plt.xlabel('Key') # X-label
    plt.ylabel('Query') # Y-label

# Plot 8: Query-Key-Value Dimensions
plt.subplot(3, 3, 8) # Eighth subplot
components = ['Query (Q)', 'Key (K)', 'Value (V)'] # Component names
q_norm = np.mean([np.linalg.norm(Q[i]) for i in range(len(Q))]) # Average Q norm
k_norm = np.mean([np.linalg.norm(K[i]) for i in range(len(K))]) # Average K norm
v_norm = np.mean([np.linalg.norm(V[i]) for i in range(len(V))]) # Average V norm
norms = [q_norm, k_norm, v_norm] # All norms
colors_qkv = ['#e74c3c', '#3498db', '#2ecc71'] # Colors
bars = plt.bar(components, norms, color=colors_qkv, alpha=0.7, edgecolor='black', linewidth=2) # Bar plot
plt.title('Q-K-V Component Magnitudes\n(average vector norms)', fontsize=11) # Title
plt.ylabel('Average Norm') # Y-label
plt.grid(True, alpha=0.3, axis='y') # Grid
for bar, norm in zip(bars, norms): # Annotate bars
    height = bar.get_height() # Bar height
    plt.text(bar.get_x() + bar.get_width()/2., height, # Position text
             f'{norm:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold') # Norm value

# Plot 9: Architecture Comparison
plt.subplot(3, 3, 9) # Ninth subplot
models = ['RNN\n(Sequential)', 'Self-Attention\n(Parallel)', 'Multi-Head\n(4 heads)'] # Model names
speeds = [1, 10, 8] # Relative speeds
memory_range = [3, 10, 10] # Memory capability

x = np.arange(len(models)) # X positions
width = 0.35 # Bar width

bars1 = plt.bar(x - width/2, speeds, width, label='Relative Speed', color='#3498db', alpha=0.7) # Speed bars
bars2 = plt.bar(x + width/2, memory_range, width, label='Memory Range', color='#e74c3c', alpha=0.7) # Memory bars

plt.title('Architecture Comparison\n(Speed vs Memory)', fontsize=11) # Title
plt.ylabel('Score (arbitrary units)') # Y-label
plt.xticks(x, models) # Model labels
plt.legend() # Legend
plt.grid(True, alpha=0.3, axis='y') # Grid

# Annotate bars
for bars in [bars1, bars2]: # Both bar sets
    for bar in bars: # Each bar
        height = bar.get_height() # Bar height
        plt.text(bar.get_x() + bar.get_width()/2., height, # Position
                 f'{height:.0f}', ha='center', va='bottom', fontsize=9) # Value

plt.tight_layout() # Adjust spacing
plt.show() # Display plot