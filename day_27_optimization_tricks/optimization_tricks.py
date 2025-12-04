import numpy as np                                                                             # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                # Import matplotlib for visualization
import seaborn as sns                                                                          # Import seaborn for advanced visualization
from matplotlib.patches import Rectangle                                                       # Import matplotlib for visualization

# ===== SIMULATED NEURAL NETWORK LAYER =====
class NeuralLayer:                                                                             # Simple layer for demonstration
    def __init__(self, input_size, output_size):                                               # Initialize layer
        self.weights = np.random.randn(input_size, output_size) * 0.5                          # Random weights
        self.bias = np.zeros(output_size)                                                      # Zero bias
        
    def forward(self, x):                                                                      # Forward pass
        return np.dot(x, self.weights) + self.bias                                             # Linear transformation

# ===== BATCH NORMALIZATION =====
class BatchNormalization:                                                                      # Batch normalization implementation
    def __init__(self, num_features, epsilon=1e-5):                                            # Initialize BatchNorm
        self.epsilon = epsilon                                                                 # Numerical stability constant
        self.gamma = np.ones(num_features)                                                     # Learnable scale (initialized to 1)
        self.beta = np.zeros(num_features)                                                     # Learnable shift (initialized to 0)
        
    def forward(self, x, training=True):                                                       # Forward pass
        if training:                                                                           # Training mode
            # Calculate batch statistics
            batch_mean = np.mean(x, axis=0)                                                    # Mean across batch
            batch_var = np.var(x, axis=0)                                                      # Variance across batch
            
            # Normalize
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)                # Standardize
            
            # Scale and shift
            output = self.gamma * x_normalized + self.beta                                     # Apply learnable parameters
            
            return output, batch_mean, batch_var, x_normalized                                 # Return all for visualization
        else:                                                                                  # Inference mode (use running stats)
            return self.gamma * x + self.beta                                                  # Simplified for demo

# ===== DROPOUT =====
class Dropout:                                                                                 # Dropout implementation
    def __init__(self, dropout_rate=0.5):                                                      # Initialize dropout
        self.dropout_rate = dropout_rate                                                       # Probability of dropping
        
    def forward(self, x, training=True):                                                       # Forward pass
        if training:                                                                           # Training mode: apply dropout
            # Create dropout mask
            mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)                  # Binary mask (0 or 1)
            
            # Apply mask and scale
            output = x * mask / (1 - self.dropout_rate)                                        # Drop and scale remaining
            
            return output, mask                                                                # Return output and mask
        else:                                                                                  # Inference mode: no dropout
            return x, np.ones_like(x)                                                          # All neurons active

# ===== LEARNING RATE SCHEDULERS =====
class LearningRateScheduler:                                                                   # LR scheduler base class
    def __init__(self, initial_lr):                                                            # Initialize with starting LR
        self.initial_lr = initial_lr                                                           # Store initial LR
        
    def step_decay(self, epoch, step_size=10, decay_rate=0.5):                                 # Step decay schedule
        """Drop LR by decay_rate every step_size epochs"""
        lr = self.initial_lr * (decay_rate ** (epoch // step_size))                            # Calculate LR
        return lr                                                                              # Return current LR
    
    def exponential_decay(self, epoch, decay_rate=0.95):                                       # Exponential decay schedule
        """Smooth exponential reduction"""
        lr = self.initial_lr * (decay_rate ** epoch)                                           # Exponential formula
        return lr                                                                              # Return current LR
    
    def cosine_annealing(self, epoch, T_max=50, eta_min=1e-5):                                 # Cosine annealing schedule
        """Cosine-based smooth reduction"""
        lr = eta_min + (self.initial_lr - eta_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2   # Cosine formula
        return lr                                                                              # Return current LR

# ===== GRADIENT CLIPPING =====
class GradientClipper:                                                                         # Gradient clipping utilities
    @staticmethod
    def clip_by_norm(gradients, max_norm=1.0):                                                 # Clip by total norm
        """Clip gradient norm to max_norm"""
        total_norm = np.linalg.norm(gradients)                                                 # Calculate gradient norm
        
        if total_norm > max_norm:                                                              # If exceeds threshold
            clipped = gradients * (max_norm / total_norm)                                      # Scale down proportionally
            return clipped, total_norm, True                                                   # Return clipped, norm, clipped flag
        else:                                                                                  # Within bounds
            return gradients, total_norm, False                                                # Return original, norm, not clipped
    
    @staticmethod
    def clip_by_value(gradients, clip_value=0.5):                                              # Clip by value
        """Clip each gradient value to [-clip_value, clip_value]"""
        clipped = np.clip(gradients, -clip_value, clip_value)                                  # Element-wise clipping
        was_clipped = not np.allclose(gradients, clipped)                                      # Check if any value was clipped
        return clipped, was_clipped                                                            # Return clipped gradients and flag

# ===== DEMONSTRATION SETUP =====
print("=" * 90)                                                                                # Header separator
print("OPTIMIZATION TRICKS FOR NEURAL NETWORKS: Complete Demonstration")                       # Main title
print("=" * 90)                                                                                # Header separator

# ===== 1. BATCH NORMALIZATION DEMONSTRATION =====
print("\n" + "=" * 90)                                                                         # Section separator
print("1. BATCH NORMALIZATION: Stabilizing Activations")                                       # Section title
print("=" * 90)                                                                                # Section separator

# Generate sample batch data (simulating layer activations)
batch_size = 32                                                                                # Number of samples
num_features = 4                                                                               # Number of features
np.random.seed(42)                                                                             # Reproducibility

# Create data with different scales (simulating unstable activations)
data_unstable = np.concatenate([                                                               # Combine different scales
    np.random.randn(batch_size, 1) * 0.1,      # Feature 1: small scale                        # Feature with small values
    np.random.randn(batch_size, 1) * 10,       # Feature 2: large scale                        # Feature with large values
    np.random.randn(batch_size, 1) * 1,        # Feature 3: medium scale                       # Feature with medium values
    np.random.randn(batch_size, 1) * 100,      # Feature 4: very large scale                   # Feature with huge values
], axis=1)

print("\nBefore BatchNorm:") # Before normalization
print(f"  Data shape: {data_unstable.shape}") # Shape info
print(f"  Feature means: {np.round(np.mean(data_unstable, axis=0), 3)}") # Mean per feature
print(f"  Feature stds:  {np.round(np.std(data_unstable, axis=0), 3)}") # Std per feature
print(f"  Value range:   [{data_unstable.min():.2f}, {data_unstable.max():.2f}]") # Overall range

# Apply Batch Normalization
batchnorm = BatchNormalization(num_features) # Create BatchNorm layer
data_normalized, bn_mean, bn_var, bn_normalized = batchnorm.forward(data_unstable, training=True) # Apply normalization

print("\nAfter BatchNorm:") # After normalization
print(f"  Normalized means: {np.round(np.mean(data_normalized, axis=0), 6)}") # Should be near 0
print(f"  Normalized stds:  {np.round(np.std(data_normalized, axis=0), 6)}") # Should be near 1
print(f"  Value range:      [{data_normalized.min():.2f}, {data_normalized.max():.2f}]") # Normalized range

print("\n✅ Effect: All features now have similar scales (mean≈0, std≈1)") # Key benefit
print("   This stabilizes gradient flow and speeds up training!") # Explanation

# ===== 2. DROPOUT DEMONSTRATION =====
print("\n" + "=" * 90)                                                                         # Section separator
print("2. DROPOUT: Preventing Overfitting")                                                    # Section title
print("=" * 90)                                                                                # Section separator

# Create sample activations
activations = np.random.randn(8, 10)                                                           # 8 samples, 10 neurons

# Apply dropout with different rates
dropout_rates = [0.0, 0.3, 0.5, 0.7]                                                           # Test different rates
dropout_results = {}                                                                           # Store results

print("\nDropout Effect on Network Activations:")                                              # Title
print("-" * 90)                                                                                # Separator

for rate in dropout_rates:                                                                     # Test each rate
    dropout = Dropout(dropout_rate=rate)                                                       # Create dropout with rate
    dropped_activations, mask = dropout.forward(activations, training=True)                    # Apply dropout
    
    num_active = np.sum(mask) / mask.size                                                      # Fraction of active neurons
    dropout_results[rate] = (dropped_activations, mask, num_active)                            # Store results
    
    print(f"\nDropout Rate: {rate:.1f}") # Display rate
    print(f"  Active neurons: {num_active*100:.1f}%") # Percentage active
    print(f"  Original mean:  {np.mean(activations):.4f}") # Original mean
    print(f"  Dropped mean:   {np.mean(dropped_activations):.4f}") # After dropout mean
    print(f"  Scaling factor: {1/(1-rate) if rate < 1 else 'N/A'}") # Compensation factor

print("\n✅ Effect: Random neurons dropped → forces network to learn robust features") # Key benefit
print("   During inference, all neurons active → ensemble effect!") # Inference behavior

# ===== 3. LEARNING RATE SCHEDULES =====
print("\n" + "=" * 90)                                                                         # Section separator
print("3. LEARNING RATE SCHEDULES: Dynamic Speed Control")                                     # Section title
print("=" * 90)                                                                                # Section separator

scheduler = LearningRateScheduler(initial_lr=0.1)                                              # Create scheduler
num_epochs = 50                                                                                # Number of epochs to simulate

# Calculate LR for each schedule
epochs = np.arange(num_epochs)                                                                 # Epoch array
lr_step = [scheduler.step_decay(e) for e in epochs]                                            # Step decay values
lr_exp = [scheduler.exponential_decay(e) for e in epochs]                                      # Exponential decay values
lr_cos = [scheduler.cosine_annealing(e) for e in epochs]                                       # Cosine annealing values

print("\nLearning Rate Evolution:") # Title
print("-" * 90) # Separator
print(f"Initial LR: {scheduler.initial_lr}") # Starting LR

# Show values at key epochs
key_epochs = [0, 10, 25, 49] # Epochs to display
print("\nLR at key epochs:") # Key epochs header
print(f"{'Epoch':<10} {'Step Decay':<15} {'Exponential':<15} {'Cosine':<15}") # Column headers

for epoch in key_epochs: # Display each key epoch
    print(f"{epoch:<10} {lr_step[epoch]:<15.6f} {lr_exp[epoch]:<15.6f} {lr_cos[epoch]:<15.6f}") # LR values

print("\n✅ Effect: High LR initially → fast progress") # Initial phase
print("           Low LR later → fine-tuning and convergence") # Final phase

# ===== 4. GRADIENT CLIPPING =====
print("\n" + "=" * 90)                                                                         # Section separator
print("4. GRADIENT CLIPPING: Taming Exploding Gradients")                                      # Section title
print("=" * 90)                                                                                # Section separator

# Simulate gradient scenarios
gradient_normal = np.random.randn(100) * 0.5                                                   # Normal gradients
gradient_exploding = np.random.randn(100) * 10                                                 # Exploding gradients

print("\nGradient Clipping Demonstration:") # Title
print("-" * 90) # Separator

# Normal gradients
grad_clip_norm, norm_normal, was_clipped_norm = GradientClipper.clip_by_norm(gradient_normal, max_norm=1.0) # Clip by norm
print(f"\nNormal Gradients:") # Normal case
print(f"  Original norm: {norm_normal:.4f}") # Original norm
print(f"  Was clipped:   {was_clipped_norm}") # Clipping status
print(f"  Max value:     {np.abs(gradient_normal).max():.4f}") # Max value

# Exploding gradients
grad_clip_exp, norm_exp, was_clipped_exp = GradientClipper.clip_by_norm(gradient_exploding, max_norm=1.0) # Clip exploding
print(f"\nExploding Gradients:") # Exploding case
print(f"  Original norm: {norm_exp:.4f} ⚠️ TOO LARGE!") # Original norm (large)
print(f"  Was clipped:   {was_clipped_exp}") # Clipping status
print(f"  Clipped norm:  {np.linalg.norm(grad_clip_exp):.4f} ✓") # After clipping
print(f"  Original max:  {np.abs(gradient_exploding).max():.4f}") # Original max
print(f"  Clipped max:   {np.abs(grad_clip_exp).max():.4f}") # After clipping

print("\n✅ Effect: Prevents gradient explosions in RNNs and deep networks") # Key benefit
print("           Stabilizes training without changing learning dynamics") # Explanation

# ===== 5. COMBINED OPTIMIZATION PIPELINE =====
print("\n" + "=" * 90) # Section separator
print("5. COMPLETE OPTIMIZATION PIPELINE") # Section title
print("=" * 90) # Section separator

print("\n🎯 Typical Training Pipeline with All Tricks:") # Pipeline description
print("-" * 90) # Separator
print("""
Step 1: Forward Pass
  → Input Data
  → Conv/LSTM Layer
  → BatchNorm        ✓ (stabilize activations)
  → Activation (ReLU)
  → Dropout          ✓ (prevent overfitting)
  → Output Layer

Step 2: Compute Loss
  → Loss Function
  → Add Weight Decay ✓ (L2 regularization)

Step 3: Backward Pass
  → Compute Gradients
  → Gradient Clipping ✓ (prevent explosions)
  → Optimizer Step with Current LR ✓ (from schedule)

Step 4: Update Learning Rate
  → LR Scheduler Step ✓

Step 5: Validation & Early Stopping
  → Check Validation Loss
  → Early Stop if no improvement ✓
""")

# Simulate training metrics with and without optimization tricks
np.random.seed(42)                                                                             # Reproducibility
epochs_sim = 30                                                                                # Simulation epochs

# Without optimization tricks (unstable)
loss_without = 2.5 * np.exp(-0.05 * np.arange(epochs_sim)) + np.random.randn(epochs_sim) * 0.3 # Noisy convergence
loss_without = np.clip(loss_without, 0.5, 3.0)                                                 # Realistic bounds

# With optimization tricks (stable)
loss_with = 2.5 * np.exp(-0.15 * np.arange(epochs_sim)) + np.random.randn(epochs_sim) * 0.05   # Smooth convergence
loss_with = np.clip(loss_with, 0.1, 2.5)                                                       # Realistic bounds

print("\nTraining Comparison (simulated over 30 epochs):") # Comparison title
print("-" * 90) # Separator
print(f"{'Epoch':<10} {'Without Tricks':<20} {'With Tricks':<20} {'Improvement':<20}") # Column headers

for epoch in [0, 5, 10, 15, 20, 29]:                                                           # Key epochs
    improvement = ((loss_without[epoch] - loss_with[epoch]) / loss_without[epoch]) * 100       # Percentage improvement
    print(f"{epoch:<10} {loss_without[epoch]:<20.4f} {loss_with[epoch]:<20.4f} {improvement:<20.1f}%") # Results

print(f"\nFinal Loss Reduction: {((loss_without[-1] - loss_with[-1]) / loss_without[-1]) * 100:.1f}%") # Total improvement

# ===== VISUALIZATIONS =====
fig = plt.figure(figsize=(20, 14)) # Create large figure
fig.suptitle('Neural Network Optimization Tricks: Complete Visual Guide', fontsize=16, fontweight='bold') # Main title

# Plot 1: Batch Normalization Effect
plt.subplot(3, 4, 1) # First subplot
x_pos = np.arange(num_features) # Feature positions
before_means = np.mean(data_unstable, axis=0) # Means before
before_stds = np.std(data_unstable, axis=0) # Stds before
plt.bar(x_pos - 0.2, before_means, 0.4, label='Before BN', color='#e74c3c', alpha=0.7) # Before bars
plt.bar(x_pos + 0.2, np.mean(data_normalized, axis=0), 0.4, label='After BN', color='#2ecc71', alpha=0.7) # After bars
plt.title('BatchNorm: Feature Means', fontsize=11, fontweight='bold') # Title
plt.xlabel('Feature Index') # X-label
plt.ylabel('Mean Value') # Y-label
plt.legend() # Legend
plt.xticks(x_pos, [f'F{i+1}' for i in range(num_features)]) # Feature labels
plt.grid(True, alpha=0.3, axis='y') # Grid

# Plot 2: Batch Normalization Variance
plt.subplot(3, 4, 2) # Second subplot
plt.bar(x_pos - 0.2, before_stds, 0.4, label='Before BN', color='#e74c3c', alpha=0.7) # Before std bars
plt.bar(x_pos + 0.2, np.std(data_normalized, axis=0), 0.4, label='After BN', color='#2ecc71', alpha=0.7) # After std bars
plt.title('BatchNorm: Feature Std Deviations', fontsize=11, fontweight='bold') # Title
plt.xlabel('Feature Index') # X-label
plt.ylabel('Std Deviation') # Y-label
plt.legend() # Legend
plt.xticks(x_pos, [f'F{i+1}' for i in range(num_features)]) # Feature labels
plt.grid(True, alpha=0.3, axis='y') # Grid

# Plot 3: Dropout Masks Visualization
plt.subplot(3, 4, 3) # Third subplot
_, mask_50, _ = dropout_results[0.5] # Get 50% dropout mask
plt.imshow(mask_50.T, cmap='RdYlGn', aspect='auto', interpolation='nearest') # Visualize mask
plt.colorbar(label='Active (1) / Dropped (0)') # Colorbar
plt.title('Dropout Mask (50% rate)\nWhite=Active, Dark=Dropped', fontsize=11, fontweight='bold') # Title
plt.xlabel('Sample Index') # X-label
plt.ylabel('Neuron Index') # Y-label

# Plot 4: Dropout Active Neurons
plt.subplot(3, 4, 4) # Fourth subplot
rates = list(dropout_results.keys()) # Dropout rates
active_percentages = [dropout_results[r][2] * 100 for r in rates] # Active percentages
colors_dropout = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'] # Color gradient
bars = plt.bar(range(len(rates)), active_percentages, color=colors_dropout, alpha=0.7, edgecolor='black', linewidth=2) # Bars
plt.title('Dropout: Active Neurons %', fontsize=11, fontweight='bold') # Title
plt.xlabel('Dropout Rate') # X-label
plt.ylabel('Active Neurons (%)') # Y-label
plt.xticks(range(len(rates)), [f'{r:.1f}' for r in rates]) # Rate labels
plt.ylim([0, 105]) # Y-axis limits
for bar, pct in zip(bars, active_percentages): # Annotate bars
    height = bar.get_height() # Bar height
    plt.text(bar.get_x() + bar.get_width()/2., height, f'{pct:.0f}%', # Position text
             ha='center', va='bottom', fontsize=10, fontweight='bold') # Format text
plt.grid(True, alpha=0.3, axis='y') # Grid

# Plot 5: Learning Rate - Step Decay
plt.subplot(3, 4, 5) # Fifth subplot
plt.plot(epochs, lr_step, 'b-', linewidth=3, label='Step Decay') # Step decay curve
plt.title('LR Schedule: Step Decay\n(drops every 10 epochs)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Epoch') # X-label
plt.ylabel('Learning Rate') # Y-label
plt.yscale('log') # Log scale for clarity
plt.grid(True, alpha=0.3) # Grid
plt.legend() # Legend

# Plot 6: Learning Rate - Exponential Decay
plt.subplot(3, 4, 6) # Sixth subplot
plt.plot(epochs, lr_exp, 'g-', linewidth=3, label='Exponential Decay') # Exponential curve
plt.title('LR Schedule: Exponential Decay\n(smooth reduction)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Epoch') # X-label
plt.ylabel('Learning Rate') # Y-label
plt.yscale('log') # Log scale
plt.grid(True, alpha=0.3) # Grid
plt.legend() # Legend

# Plot 7: Learning Rate - Cosine Annealing
plt.subplot(3, 4, 7) # Seventh subplot
plt.plot(epochs, lr_cos, 'r-', linewidth=3, label='Cosine Annealing') # Cosine curve
plt.title('LR Schedule: Cosine Annealing\n(wave-like reduction)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Epoch') # X-label
plt.ylabel('Learning Rate') # Y-label
plt.yscale('log') # Log scale
plt.grid(True, alpha=0.3) # Grid
plt.legend() # Legend

# Plot 8: All LR Schedules Comparison
plt.subplot(3, 4, 8) # Eighth subplot
plt.plot(epochs, lr_step, 'b-', linewidth=2, label='Step', alpha=0.7) # Step
plt.plot(epochs, lr_exp, 'g-', linewidth=2, label='Exponential', alpha=0.7) # Exponential
plt.plot(epochs, lr_cos, 'r-', linewidth=2, label='Cosine', alpha=0.7) # Cosine
plt.title('All LR Schedules Comparison', fontsize=11, fontweight='bold') # Title
plt.xlabel('Epoch') # X-label
plt.ylabel('Learning Rate') # Y-label
plt.yscale('log') # Log scale
plt.grid(True, alpha=0.3) # Grid
plt.legend() # Legend

# Plot 9: Gradient Clipping - Normal vs Exploding
plt.subplot(3, 4, 9) # Ninth subplot
plt.hist(gradient_normal, bins=30, alpha=0.7, color='#2ecc71', label='Normal', edgecolor='black') # Normal histogram
plt.hist(gradient_exploding, bins=30, alpha=0.5, color='#e74c3c', label='Exploding', edgecolor='black') # Exploding histogram
plt.axvline(x=1.0, color='orange', linestyle='--', linewidth=2, label='Clip Threshold')  # Threshold line
plt.axvline(x=-1.0, color='orange', linestyle='--', linewidth=2) # Threshold line
plt.title('Gradient Distribution\n(before clipping)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Gradient Value') # X-label
plt.ylabel('Frequency') # Y-label
plt.legend() # Legend
plt.grid(True, alpha=0.3, axis='y') # Grid

# Plot 10: Gradient Clipping Effect
plt.subplot(3, 4, 10) # Tenth subplot
sample_indices = np.arange(20) # Sample gradients to show
plt.plot(sample_indices, gradient_exploding[:20], 'ro-', linewidth=2, markersize=8, # Before clipping
         label='Before Clipping', alpha=0.7) # Label
plt.plot(sample_indices, grad_clip_exp[:20], 'g^-', linewidth=2, markersize=8, # After clipping
         label='After Clipping', alpha=0.7) # Label
plt.axhline(y=1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7) # Upper bound
plt.axhline(y=-1.0, color='orange', linestyle='--', linewidth=2, alpha=0.7) # Lower bound
plt.title('Gradient Clipping Effect\n(sample of 20 gradients)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Gradient Index') # X-label
plt.ylabel('Gradient Value') # Y-label
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 11: Training Loss Comparison
plt.subplot(3, 4, 11) # Eleventh subplot
plt.plot(np.arange(epochs_sim), loss_without, 'r-o', linewidth=2, markersize=4, # Without tricks
         label='Without Optimization', alpha=0.7) # Label
plt.plot(np.arange(epochs_sim), loss_with, 'g-^', linewidth=2, markersize=4, # With tricks
         label='With Optimization', alpha=0.7) # Label
plt.title('Training Loss Comparison\n(simulated training)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Epoch') # X-label
plt.ylabel('Training Loss') # Y-label
plt.legend() # Legend
plt.grid(True, alpha=0.3) # Grid

# Plot 12: Optimization Techniques Summary
plt.subplot(3, 4, 12) # Twelfth subplot
techniques = ['BatchNorm', 'Dropout', 'LR Schedule', 'Grad Clip', 'Weight Decay'] # Technique names
benefits = [9, 8, 9, 10, 7] # Relative importance (1-10)
colors_tech = ['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'] # Colors
bars = plt.barh(techniques, benefits, color=colors_tech, alpha=0.7, edgecolor='black', linewidth=2) # Horizontal bars
plt.title('Optimization Techniques\nImportance Score (1-10)', fontsize=11, fontweight='bold') # Title
plt.xlabel('Importance Score') # X-label
plt.xlim([0, 11]) # X-axis limits
for i, (bar, score) in enumerate(zip(bars, benefits)): # Annotate bars
    width = bar.get_width() # Bar width
    plt.text(width, bar.get_y() + bar.get_height()/2., f'{score}/10', # Position text
             ha='left', va='center', fontsize=10, fontweight='bold')  # Format text
plt.grid(True, alpha=0.3, axis='x') # Grid

plt.tight_layout() # Adjust spacing
plt.show() # Display plot