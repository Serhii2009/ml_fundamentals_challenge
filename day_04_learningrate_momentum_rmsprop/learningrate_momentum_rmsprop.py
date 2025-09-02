import numpy as np                                                                 # Import NumPy library for mathematical operations
import matplotlib.pyplot as plt                                                    # Import matplotlib for plotting and visualization

def f(x, y):                                                                       # Define objective function f(x,y) = (x-3)² + (y+2)²
    return (x - 3)**2 + (y + 2)**2                                                 # Return function value at point (x,y)

def grad(x, y):                                                                    # Define gradient function (partial derivatives)
    return np.array([2*(x-3), 2*(y+2)])                                            # Return gradient vector [df/dx, df/dy]

eta = 0.1                                                                          # Learning rate (step size)
epochs = 50                                                                        # Number of optimization iterations

# ===== REGULAR GRADIENT DESCENT =====
print("="*60)                                                                      # Separator line for formatting
print("OPTIMIZATION METHODS COMPARISON")                                           # Section header
print("="*60)                                                                      # Separator line for formatting

x, y = 0.0, 0.0                                                                    # Initialize starting point for regular GD
trajectory_gd = [(x, y)]                                                           # Store trajectory points for visualization

for epoch in range(epochs):                                                        # Loop through optimization epochs
    g = grad(x, y)                                                                 # Calculate gradient at current point
    x, y = (x - eta*g[0], y - eta*g[1])                                            # Update parameters using gradient descent
    trajectory_gd.append((x, y))                                                   # Store new point in trajectory

print(f"Regular GD final point: ({x:.4f}, {y:.4f})")                               # Print final convergence point

# ===== MOMENTUM GRADIENT DESCENT =====
x, y = 0.0, 0.0                                                                    # Reset starting point for momentum
v = np.array([0.0, 0.0])                                                           # Initialize velocity vector (momentum)
beta = 0.9                                                                         # Momentum decay factor
trajectory_momentum = [(x, y)]                                                     # Store trajectory points for momentum

for epoch in range(epochs):                                                        # Loop through optimization epochs
    g = grad(x, y)                                                                 # Calculate gradient at current point
    v = beta*v + (1-beta)*g                                                        # Update momentum using exponential moving average
    x, y = (x - eta*v[0], y - eta*v[1])                                            # Update parameters using momentum
    trajectory_momentum.append((x, y))                                             # Store new point in trajectory

print(f"Momentum GD final point: ({x:.4f}, {y:.4f})")                              # Print final convergence point

# ===== RMSPROP OPTIMIZATION =====
x, y = 0.0, 0.0                                                                    # Reset starting point for RMSProp
s = np.array([0.0, 0.0])                                                           # Initialize squared gradient accumulator
beta = 0.9                                                                         # Decay factor for squared gradients
eps = 1e-8                                                                         # Small epsilon to prevent division by zero
trajectory_rmsprop = [(x, y)]                                                      # Store trajectory points for RMSProp

for epoch in range(epochs):                                                        # Loop through optimization epochs
    g = grad(x, y)                                                                 # Calculate gradient at current point
    s = beta*s + (1-beta)*(g**2)                                                   # Update squared gradient moving average
    x, y = (x - eta/np.sqrt(s[0]+eps)*g[0],                                        # Update x with adaptive learning rate
            y - eta/np.sqrt(s[1]+eps)*g[1])                                        # Update y with adaptive learning rate
    trajectory_rmsprop.append((x, y))                                              # Store new point in trajectory

print(f"RMSProp final point: ({x:.4f}, {y:.4f})")                                  # Print final convergence point


# ===== CONVERGENCE ANALYSIS =====
print("\n" + "="*60) # New line and separator
print("CONVERGENCE ANALYSIS") # Analysis section header
print("="*60) # Separator line for formatting

# Calculate final distances from true minimum (3, -2)
true_minimum = np.array([3, -2]) # True minimum coordinates
final_gd = np.array(trajectory_gd[-1]) # Final point for regular GD
final_momentum = np.array(trajectory_momentum[-1]) # Final point for momentum
final_rmsprop = np.array(trajectory_rmsprop[-1]) # Final point for RMSProp

distance_gd = np.linalg.norm(final_gd - true_minimum) # Calculate distance for regular GD
distance_momentum = np.linalg.norm(final_momentum - true_minimum) # Calculate distance for momentum
distance_rmsprop = np.linalg.norm(final_rmsprop - true_minimum) # Calculate distance for RMSProp

print("Distance from true minimum:") # Distance comparison header
print(f"  - Regular GD: {distance_gd:.6f}") # Print regular GD distance
print(f"  - Momentum:   {distance_momentum:.6f}") # Print momentum distance
print(f"  - RMSProp:    {distance_rmsprop:.6f}") # Print RMSProp distance


# ===== VISUALIZATION SETUP =====
def plot_trajectory(traj, label): # Function to plot optimization trajectory
    xs, ys = zip(*traj) # Separate x and y coordinates from trajectory
    plt.plot(xs, ys, marker="o", label=label) # Plot trajectory with markers and label

# ===== CREATE COMPARISON PLOT =====
plt.figure(figsize=(10, 8)) # Create figure with specified size
plot_trajectory(trajectory_gd, "Regular GD") # Plot regular gradient descent trajectory
plot_trajectory(trajectory_momentum, "Momentum") # Plot momentum gradient descent trajectory
plot_trajectory(trajectory_rmsprop, "RMSProp") # Plot RMSProp optimization trajectory
plt.scatter([3], [-2], color="red", marker="*", s=200, label="True Minimum") # Mark the true minimum point
plt.xlabel("X coordinate") # Set x-axis label
plt.ylabel("Y coordinate") # Set y-axis label
plt.title("Optimization Trajectories Comparison") # Set plot title
plt.legend() # Add legend to distinguish methods
plt.grid(True, alpha=0.3) # Add grid for better readability
plt.show() # Display the plot