import numpy as np                                                                  # Import NumPy for numerical computations
import matplotlib.pyplot as plt                                                     # Import Matplotlib for plotting

def f(x):                                                                           # Function f(x) = (x - 3)^2
    return (x - 3) ** 2                                                             # Quadratic function

def df(x):                                                                          # Derivative f'(x) = 2(x - 3)
    return 2 * (x - 3)                                                              # Gradient formula

start = 0                                                                           # Initial guess for x
learning_rate = 0.1                                                                 # Step size (learning rate)
history = []                                                                        # To store all x values during iterations

for i in range(20):                                                                 # Perform 20 iterations
    grad = df(start)                                                                # Compute gradient at current x
    start = start - learning_rate * grad                                            # Update x using gradient descent rule
    history.append(start)                                                           # Store current x
    print(f"Iteration {i+1:2d}: x = {start:.5f}, f(x) = {f(start):.5f}")            # Show progress

print("\nFinal optimized value of x:", round(start, 5))                             # Print final result
print("Minimum of f(x) is near x = 3")                                              # Theoretical minimum


# ===== VISUALIZATION =====

xs = np.linspace(-6, 6, 200) # Generate points for smooth curve
ys = f(xs) # Compute function values

plt.plot(xs, ys, label="f(x)") # Plot f(x)
plt.plot(history, [f(x) for x in history], "ro-", label="Gradient Descent Path") # Plot optimization path
plt.xlabel("x") # X-axis label
plt.ylabel("f(x)") # Y-axis label
plt.title("Gradient Descent Progress") # Title of plot
plt.legend() # Show legend
plt.grid(True, linestyle="--", alpha=0.6) # Add grid for clarity
plt.show() # Display plot