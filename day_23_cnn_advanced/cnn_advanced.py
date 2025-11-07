import numpy as np                                                                              # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                 # Import matplotlib for visualization

# ===== ACTIVATION & HELPER FUNCTIONS =====
def relu(x):                                                                                    # ReLU activation function
    return np.maximum(0, x)                                                                     # Return max of 0 and x

def batch_norm(x):                                                                              # Simplified batch normalization
    """Normalize to mean=0, std=1"""                                                            # Function description
    mean = np.mean(x)                                                                           # Calculate mean
    std = np.std(x) + 1e-8                                                                      # Calculate std (add epsilon)
    return (x - mean) / std                                                                     # Return normalized values

def dropout(x, rate=0.5, training=True):                                                        # Dropout function
    """Randomly drop neurons during training"""                                                 # Function description
    if not training:                                                                            # If in inference mode
        return x                                                                                # Return unchanged
    mask = np.random.binomial(1, 1-rate, size=x.shape)                                          # Create binary mask
    return x * mask / (1 - rate)                                                                # Apply mask and scale

def convolution_2d(image, kernel):                                                              # 2D convolution operation
    """Perform 2D convolution"""                                                                # Function description
    h, w = image.shape                                                                          # Get image dimensions
    kh, kw = kernel.shape                                                                       # Get kernel dimensions
    out_h, out_w = h - kh + 1, w - kw + 1                                                       # Calculate output size
    output = np.zeros((out_h, out_w))                                                           # Initialize output
    
    for i in range(out_h):                                                                      # Loop through rows
        for j in range(out_w):                                                                  # Loop through columns
            output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)                               # Compute convolution
    
    return output                                                                               # Return feature map

def max_pooling(x, size=2):                                                                     # Max pooling operation
    """2x2 max pooling"""                                                                       # Function description
    h, w = x.shape                                                                              # Get input dimensions
    out_h, out_w = h // size, w // size                                                         # Calculate output size
    output = np.zeros((out_h, out_w))                                                           # Initialize output
    
    for i in range(out_h):                                                                      # Loop through rows
        for j in range(out_w):                                                                  # Loop through columns
            output[i, j] = np.max(x[i*size:(i+1)*size, j*size:(j+1)*size])                      # Take max value
    
    return output                                                                               # Return pooled output

# ===== CREATE SYNTHETIC IMAGE =====
np.random.seed(42)                                                                              # Set random seed
image = np.random.rand(12, 12)                                                                  # Create 12x12 random image
image[3:9, 3:9] += 0.5                                                                          # Add bright square pattern
image = np.clip(image, 0, 1)                                                                    # Clip values to [0,1]

print(f"Original image shape: {image.shape}")                                                   # Print original size

# ===== DEFINE FILTERS =====
edge_filter = np.array([                                                                        # Edge detection filter
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# ===== REGULAR CNN PATH =====
print("\n=== Regular CNN Path ===") # Regular path header

# Conv1
conv1 = convolution_2d(image, edge_filter) # Apply convolution
conv1_norm = batch_norm(conv1) # Apply batch normalization
conv1_relu = relu(conv1_norm) # Apply ReLU
print(f"After Conv+BN+ReLU: {conv1_relu.shape}") # Print shape

# Pooling
pooled1 = max_pooling(conv1_relu) # Apply max pooling
print(f"After MaxPooling: {pooled1.shape}") # Print shape

# Dropout
pooled1_drop = dropout(pooled1, rate=0.3, training=True) # Apply dropout
print(f"After Dropout: {pooled1_drop.shape}") # Print shape

# ===== RESIDUAL CONNECTION PATH =====
print("\n=== Residual Path (with skip connection) ===") # Residual path header

# Main path
conv2 = convolution_2d(image, edge_filter) # Apply convolution
conv2_norm = batch_norm(conv2) # Apply batch normalization
conv2_relu = relu(conv2_norm) # Apply ReLU

# Skip connection (resize to match)
skip = image[1:11, 1:11] # Crop to match conv output
residual = conv2_relu + skip # Add skip connection
residual_relu = relu(residual) # Apply ReLU after addition

print(f"Main path: {conv2_relu.shape}") # Print main path shape
print(f"Skip connection: {skip.shape}") # Print skip shape
print(f"After residual: {residual_relu.shape}") # Print residual shape

# ===== STRIDE & PADDING DEMONSTRATION =====
print("\n=== Stride & Padding Effects ===") # Stride section header

# Original size
print(f"Input size: {image.shape[0]}x{image.shape[1]}") # Print input size

# Stride=1, no padding
out_s1 = convolution_2d(image, edge_filter) # Stride 1
print(f"Stride=1, no padding: {out_s1.shape[0]}x{out_s1.shape[1]}") # Print output size

# With padding (simulate by adding border)
padded = np.pad(image, 1, mode='constant') # Add padding
out_padded = convolution_2d(padded, edge_filter) # Convolve padded image
print(f"Stride=1, padding=1: {out_padded.shape[0]}x{out_padded.shape[1]}") # Print output size

# ===== VISUALIZATION =====
fig = plt.figure(figsize=(14, 10)) # Create figure

# Row 1: Regular CNN path
plt.subplot(3, 4, 1) # Original image
plt.imshow(image, cmap='gray') # Display image
plt.title('Original\n12x12') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 2) # After convolution
plt.imshow(conv1, cmap='viridis') # Display conv result
plt.title('Conv\n10x10') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 3) # After batch norm + ReLU
plt.imshow(conv1_relu, cmap='viridis') # Display after ReLU
plt.title('BN + ReLU\n10x10') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 4) # After pooling
plt.imshow(pooled1, cmap='viridis') # Display pooled
plt.title('Max Pool\n5x5') # Set title
plt.axis('off') # Turn off axis

# Row 2: Residual connection
plt.subplot(3, 4, 5) # Main path
plt.imshow(conv2_relu, cmap='viridis') # Display main path
plt.title('Main Path\n10x10') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 6) # Skip connection
plt.imshow(skip, cmap='gray') # Display skip
plt.title('Skip\n10x10') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 7) # After addition
plt.imshow(residual, cmap='viridis') # Display residual
plt.title('Main + Skip\n10x10') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 8) # After final ReLU
plt.imshow(residual_relu, cmap='viridis') # Display final result
plt.title('Final ReLU\n10x10') # Set title
plt.axis('off') # Turn off axis

# Row 3: Dropout effect
plt.subplot(3, 4, 9) # Before dropout
plt.imshow(pooled1, cmap='viridis') # Display before dropout
plt.title('Before Dropout\n5x5') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 10) # After dropout
plt.imshow(pooled1_drop, cmap='viridis') # Display after dropout
plt.title('After Dropout\n5x5') # Set title
plt.axis('off') # Turn off axis

# Padding comparison
plt.subplot(3, 4, 11) # No padding
plt.imshow(out_s1, cmap='viridis') # Display no padding
plt.title('No Padding\n10x10') # Set title
plt.axis('off') # Turn off axis

plt.subplot(3, 4, 12) # With padding
plt.imshow(out_padded, cmap='viridis') # Display with padding
plt.title('With Padding\n12x12') # Set title
plt.axis('off') # Turn off axis

plt.suptitle('Advanced CNN: Regularization & Skip Connections', fontsize=14) # Main title
plt.tight_layout() # Adjust layout
plt.show() # Display plot

# ===== ARCHITECTURE DIAGRAM =====
fig, ax = plt.subplots(figsize=(12, 6)) # Create figure
ax.axis('off') # Turn off axis

# Regular path
y_regular = 0.7 # Y position for regular path
stages_regular = ['Input\n12x12', 'Conv+BN\n10x10', 'ReLU\n10x10', 
                  'MaxPool\n5x5', 'Dropout\n5x5'] # Regular stages
x_positions = np.linspace(0.1, 0.9, len(stages_regular)) # X positions

for i, (stage, x) in enumerate(zip(stages_regular, x_positions)): # Draw regular path
    ax.text(x, y_regular, stage, ha='center', va='center', # Add text
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            fontsize=10)
    if i < len(stages_regular) - 1: # Draw arrows
        ax.annotate('', xy=(x_positions[i+1]-0.05, y_regular), 
                   xytext=(x+0.05, y_regular),
                   arrowprops=dict(arrowstyle='->', lw=2))

ax.text(0.05, y_regular, 'Regular:', ha='right', fontsize=12, fontweight='bold') # Label

# Residual path
y_residual = 0.3 # Y position for residual path
stages_residual = ['Input\n12x12', 'Conv+BN\n10x10', 'ReLU\n10x10', 
                   'Add Skip\n10x10', 'Output\n10x10'] # Residual stages

for i, (stage, x) in enumerate(zip(stages_residual, x_positions)): # Draw residual path
    ax.text(x, y_residual, stage, ha='center', va='center', # Add text
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
            fontsize=10)
    if i < len(stages_residual) - 1: # Draw arrows
        ax.annotate('', xy=(x_positions[i+1]-0.05, y_residual), 
                   xytext=(x+0.05, y_residual),
                   arrowprops=dict(arrowstyle='->', lw=2))

# Skip connection arrow
ax.annotate('', xy=(x_positions[3], y_residual+0.05), 
           xytext=(x_positions[0], y_residual+0.05),
           arrowprops=dict(arrowstyle='->', lw=2, color='red', 
                          linestyle='--')) # Draw skip connection
ax.text(0.5, y_residual+0.15, 'Skip Connection', ha='center', 
       color='red', fontsize=10) # Label skip

ax.text(0.05, y_residual, 'Residual:', ha='right', fontsize=12, fontweight='bold') # Label

ax.set_xlim([0, 1]) # Set x limits
ax.set_ylim([0, 1]) # Set y limits
ax.set_title('CNN Architecture Comparison', fontsize=14, fontweight='bold') # Set title

plt.tight_layout() # Adjust layout
plt.show() # Display plot