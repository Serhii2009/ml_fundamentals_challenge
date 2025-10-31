import numpy as np                                                                             # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                # Import matplotlib for visualization

# ===== ACTIVATION FUNCTIONS =====
def relu(x):                                                                                   # ReLU activation function
    return np.maximum(0, x)                                                                    # Return max of 0 and x

def softmax(x):                                                                                # Softmax activation function
    exp_x = np.exp(x - np.max(x))                                                              # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)                                                       # Return normalized probabilities

# ===== CONVOLUTION OPERATION =====
def convolution_2d(image, kernel):                                                             # 2D convolution operation
    """Perform 2D convolution on image with kernel"""                                          # Function description
    img_h, img_w = image.shape                                                                 # Get image dimensions
    ker_h, ker_w = kernel.shape                                                                # Get kernel dimensions
    
    out_h = img_h - ker_h + 1                                                                  # Calculate output height
    out_w = img_w - ker_w + 1                                                                  # Calculate output width
    output = np.zeros((out_h, out_w))                                                          # Initialize output

    for i in range(out_h):                                                                     # Loop through output rows
        for j in range(out_w):                                                                 # Loop through output columns
            region = image[i:i+ker_h, j:j+ker_w]                                               # Extract region
            output[i, j] = np.sum(region * kernel)                                             # Compute convolution
    
    return output                                                                              # Return feature map

# ===== MAX POOLING =====
def max_pooling(feature_map, pool_size=2):                                                     # Max pooling operation
    """Perform 2x2 max pooling"""                                                              # Function description
    h, w = feature_map.shape                                                                   # Get feature map dimensions
    out_h, out_w = h // pool_size, w // pool_size                                              # Calculate output dimensions
    output = np.zeros((out_h, out_w))                                                          # Initialize output
    
    for i in range(out_h):                                                                     # Loop through output rows
        for j in range(out_w):                                                                 # Loop through output columns
            region = feature_map[i*pool_size:(i+1)*pool_size,                                  # Extract pooling region
                                j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(region)                                                      # Take maximum value
    
    return output                                                                              # Return pooled output

# ===== CREATE SAMPLE IMAGE =====
# Create a simple 8x8 image with a pattern
image = np.array([                                                                             # Sample input image
    [0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [1, 1, 0, 0, 0, 0, 1, 1],
    [0, 1, 1, 0, 0, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0]
], dtype=np.float32)

# ===== DEFINE FILTERS =====
# Vertical edge detection filter
vertical_filter = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=np.float32)

# Horizontal edge detection filter
horizontal_filter = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1]
], dtype=np.float32)

# Diagonal edge detection filter
diagonal_filter = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=np.float32)

# ===== APPLY CONVOLUTIONS =====
print("Applying convolution filters...")                                                       # Status message
vertical_features = convolution_2d(image, vertical_filter)                                     # Apply vertical filter
horizontal_features = convolution_2d(image, horizontal_filter)                                 # Apply horizontal filter
diagonal_features = convolution_2d(image, diagonal_filter)                                     # Apply diagonal filter

# ===== APPLY RELU =====
vertical_relu = relu(vertical_features)                                                        # Apply ReLU to vertical features
horizontal_relu = relu(horizontal_features)                                                    # Apply ReLU to horizontal features
diagonal_relu = relu(diagonal_features)                                                        # Apply ReLU to diagonal features

# ===== APPLY MAX POOLING =====
vertical_pooled = max_pooling(vertical_relu)                                                   # Pool vertical features
horizontal_pooled = max_pooling(horizontal_relu)                                               # Pool horizontal features
diagonal_pooled = max_pooling(diagonal_relu)                                                   # Pool diagonal features

print(f"Original image size: {image.shape}") # Print original size
print(f"After convolution: {vertical_features.shape}") # Print after convolution
print(f"After pooling: {vertical_pooled.shape}") # Print after pooling

# ===== VISUALIZATION =====
fig = plt.figure(figsize=(14, 10)) # Create figure

# Original image
plt.subplot(4, 4, 1) # First subplot
plt.imshow(image, cmap='gray') # Display image
plt.title('Original Image\n(8x8)') # Set title
plt.axis('off') # Turn off axis

# Filters
plt.subplot(4, 4, 2) # Vertical filter
plt.imshow(vertical_filter, cmap='RdBu', vmin=-1, vmax=1) # Display filter
plt.title('Vertical Filter\n(3x3)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 3) # Horizontal filter
plt.imshow(horizontal_filter, cmap='RdBu', vmin=-1, vmax=1) # Display filter
plt.title('Horizontal Filter\n(3x3)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 4) # Diagonal filter
plt.imshow(diagonal_filter, cmap='RdBu', vmin=-1, vmax=1) # Display filter
plt.title('Diagonal Filter\n(3x3)') # Set title
plt.axis('off') # Turn off axis

# Feature maps after convolution
plt.subplot(4, 4, 6) # Vertical features
plt.imshow(vertical_features, cmap='viridis') # Display features
plt.title('Vertical Features\n(6x6)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 7) # Horizontal features
plt.imshow(horizontal_features, cmap='viridis') # Display features
plt.title('Horizontal Features\n(6x6)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 8) # Diagonal features
plt.imshow(diagonal_features, cmap='viridis') # Display features
plt.title('Diagonal Features\n(6x6)') # Set title
plt.axis('off') # Turn off axis

# After ReLU
plt.subplot(4, 4, 10) # Vertical ReLU
plt.imshow(vertical_relu, cmap='viridis') # Display after ReLU
plt.title('After ReLU\n(6x6)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 11) # Horizontal ReLU
plt.imshow(horizontal_relu, cmap='viridis') # Display after ReLU
plt.title('After ReLU\n(6x6)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 12) # Diagonal ReLU
plt.imshow(diagonal_relu, cmap='viridis') # Display after ReLU
plt.title('After ReLU\n(6x6)') # Set title
plt.axis('off') # Turn off axis

# After Max Pooling
plt.subplot(4, 4, 14) # Vertical pooled
plt.imshow(vertical_pooled, cmap='viridis') # Display after pooling
plt.title('After Pooling\n(3x3)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 15) # Horizontal pooled
plt.imshow(horizontal_pooled, cmap='viridis') # Display after pooling
plt.title('After Pooling\n(3x3)') # Set title
plt.axis('off') # Turn off axis

plt.subplot(4, 4, 16) # Diagonal pooled
plt.imshow(diagonal_pooled, cmap='viridis') # Display after pooling
plt.title('After Pooling\n(3x3)') # Set title
plt.axis('off') # Turn off axis

plt.suptitle('CNN Pipeline: Image → Convolution → ReLU → Max Pooling', fontsize=14) # Main title
plt.tight_layout() # Adjust layout
plt.show() # Display plot

# ===== COMPLETE CNN SIMULATION =====
# Flatten pooled features for classification
flattened = np.concatenate([                                                                   # Concatenate all pooled features
    vertical_pooled.flatten(),
    horizontal_pooled.flatten(),
    diagonal_pooled.flatten()
])

print(f"\nFlattened features size: {flattened.shape[0]}") # Print flattened size

# Simple fully connected layer simulation
num_classes = 3                                                                                # Number of output classes
W_fc = np.random.randn(flattened.shape[0], num_classes) * 0.1                                  # Random weights for FC layer
b_fc = np.zeros(num_classes)                                                                   # Bias for FC layer

# Forward pass through FC layer
logits = np.dot(flattened, W_fc) + b_fc                                                        # Calculate logits
probabilities = softmax(logits)                                                                # Apply softmax

print("\nFinal Classification Probabilities:") # Header
for i, prob in enumerate(probabilities): # Loop through classes
    print(f"  Class {i}: {prob:.3f}") # Print probability

# ===== ARCHITECTURE DIAGRAM =====
fig, ax = plt.subplots(figsize=(12, 6)) # Create figure
ax.axis('off') # Turn off axis

# Draw architecture flow
stages = ['Input\n8x8x1', 'Conv\n6x6x3', 'ReLU\n6x6x3', 'Pool\n3x3x3', 'Flatten\n27', 'Dense\n3'] # Stage labels
x_positions = np.linspace(0.1, 0.9, len(stages)) # X positions for stages

for i, (stage, x) in enumerate(zip(stages, x_positions)): # Draw each stage
    ax.text(x, 0.5, stage, ha='center', va='center', # Add text
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            fontsize=12, fontweight='bold')
    
    if i < len(stages) - 1: # Draw arrows between stages
        ax.annotate('', xy=(x_positions[i+1]-0.05, 0.5), xytext=(x+0.05, 0.5), # Add arrow
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_title('CNN Architecture Flow', fontsize=16, fontweight='bold') # Set title
plt.tight_layout() # Adjust layout
plt.show() # Display plot