import numpy as np                                                                  # Import NumPy library for mathematical operations
import matplotlib.pyplot as plt                                                     # Import Matplotlib for data visualization
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet         # Import regression models from scikit-learn
from sklearn.model_selection import train_test_split                                # Import train-test split function
from sklearn.metrics import mean_squared_error                                      # Import MSE metric for model evaluation

print("="*80)                                                                       # Separator line for formatting (extended width)
print("REGULARIZATION COMPARISON: LINEAR vs RIDGE vs LASSO vs ELASTICNET")          # Section header with extended width
print("="*80)                                                                       # Separator line for formatting (extended width)

# ===== ADVANCED DATA GENERATION =====
np.random.seed(42)                                                                  # Set random seed for reproducibility
n_samples = 100                                                                     # Define number of data samples
n_features = 10                                                                     # Define number of features (increased complexity)
X = np.random.rand(n_samples, n_features) * 10                                      # Create 100x10 feature matrix with values 0-10
true_coefs = np.array([3, -2, 1.5] + [0]*7)                                         # True coefficients: [3, -2, 1.5, 0, 0, 0, 0, 0, 0, 0]
y = X @ true_coefs + np.random.randn(n_samples) * 5                                 # y = X * true_coefs + noise (std=5)

print("Dataset Information:")                                                       # Dataset information header
print(f"  - Number of samples: {X.shape[0]}")                                       # Display total number of samples
print(f"  - Number of features: {X.shape[1]}")                                      # Display total number of features
print(f"  - True coefficients: {true_coefs}")                                       # Display true coefficient values
print(f"  - Meaningful features: 3 (first 3 coefficients)")                         # Display number of meaningful features
print(f"  - Noise features: 7 (last 7 coefficients are zero)")                      # Display number of noise features
print(f"  - Noise standard deviation: 5.0")                                         # Display noise level

# ===== TRAIN-TEST SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # Split data: 70% train, 30% test

print(f"\nData Split Information:")                                                 # Data split information header
print(f"  - Training samples: {X_train.shape[0]}")                                  # Display training set size
print(f"  - Testing samples: {X_test.shape[0]}")                                    # Display testing set size
print(f"  - Training features shape: {X_train.shape}")                              # Display training features shape
print(f"  - Testing features shape: {X_test.shape}")                                # Display testing features shape

# ===== ADVANCED MODEL DEFINITIONS =====
models = {                                                                          # Dictionary of regularization models to compare
    "Linear": LinearRegression(),                                                   # Standard linear regression (no regularization)
    "Ridge (L2)": Ridge(alpha=1.0),                                                 # Ridge regression with L2 regularization (alpha=1.0)
    "Lasso (L1)": Lasso(alpha=0.5),                                                 # Lasso regression with L1 regularization (alpha=0.5)
    "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5)                               # ElasticNet with 50% L1 and 50% L2 regularization
}

print(f"\nModels Configuration:")                                                   # Models configuration header
for name, model in models.items():                                                  # Loop through each model
    if hasattr(model, 'alpha'):                                                     # Check if model has alpha parameter
        if hasattr(model, 'l1_ratio'):                                              # Check if model has l1_ratio parameter (ElasticNet)
            print(f"  - {name}: {type(model).__name__} (alpha={model.alpha}, l1_ratio={model.l1_ratio})")  # Display ElasticNet config
        else:                                                                       # Ridge or Lasso model
            print(f"  - {name}: {type(model).__name__} (alpha={model.alpha})")      # Display model with alpha parameter
    else:                                                                           # Linear regression (no regularization)
        print(f"  - {name}: {type(model).__name__} (no regularization)")            # Display linear regression config

# ===== MODEL TRAINING AND EVALUATION =====
results = {}                                                                        # Dictionary to store model results
plt.figure(figsize=(12,6))                                                          # Create larger matplotlib figure for better visibility
 
print(f"\nTraining Progress:")                                                      # Training progress header
 
for name, model in models.items():                                                  # Loop through each model for training
    print(f"  - Training {name}...")                                                # Display current model being trained
     
    model.fit(X_train, y_train)                                                     # Train model on training data
    y_pred = model.predict(X_test)                                                  # Make predictions on test data
    mse = mean_squared_error(y_test, y_pred)                                        # Calculate mean squared error on test set
    
    # Store comprehensive results
    results[name] = (model.coef_, model.intercept_, mse)                            # Store coefficients array, intercept, and MSE
    print(f"    ✓ {name} training completed (Test MSE: {mse:.4f})")



# ===== DETAILED RESULTS ANALYSIS =====
print("\n" + "="*80) # New line and extended separator
print("DETAILED RESULTS ANALYSIS") # Detailed results analysis header
print("="*80) # Extended separator line for formatting

print(f"{'Model':<15} {'Coefficients':<50} {'Intercept':<10} {'MSE':<8}") # Results table header with extended width
print("-" * 100) # Extended table separator line

for name, (coef, intercept, mse) in results.items(): # Loop through all model results
    coef_str = np.array2string(coef, precision=2, separator=',') # Format coefficient array as string with 2 decimal precision
    print(f"{name:<15} {coef_str:<50} {intercept:<10.2f} {mse:<8.2f}") # Display formatted results in table format

# ===== COEFFICIENT COMPARISON ANALYSIS =====
print("\n" + "="*80) # New line and extended separator
print("COEFFICIENT COMPARISON WITH TRUE VALUES") # Coefficient comparison header
print("="*80) # Extended separator line for formatting

print(f"True coefficients:     {true_coefs}") # Display true coefficient values for reference

for name, (coef, _, _) in results.items(): # Loop through model coefficients
    coef_diff = np.abs(coef - true_coefs) # Calculate absolute differences from true values
    total_error = np.sum(coef_diff) # Calculate total coefficient error
    print(f"{name:<15}: {coef} (Total Error: {total_error:.4f})") # Display coefficients with total error

# ===== FEATURE SELECTION ANALYSIS =====
print("\n" + "="*80) # New line and extended separator
print("FEATURE SELECTION ANALYSIS") # Feature selection analysis header
print("="*80) # Extended separator line for formatting

for name, (coef, _, _) in results.items(): # Loop through model results
    zero_coefs = np.sum(np.abs(coef) < 0.01) # Count coefficients close to zero (threshold=0.01)
    nonzero_coefs = n_features - zero_coefs # Count non-zero coefficients
    print(f"{name:<15}: {nonzero_coefs} active features, {zero_coefs} zeroed out") # Display feature selection results

# ===== MSE COMPARISON AND RANKING =====
print("\n" + "="*80) # New line and extended separator
print("MODEL PERFORMANCE RANKING") # Model performance ranking header
print("="*80) # Extended separator line for formatting

# Sort models by MSE (ascending order - best to worst)
sorted_results = sorted(results.items(), key=lambda x: x[1][2]) # Sort models by MSE value

print("Performance Ranking (by Test MSE):") # Performance ranking header
for rank, (name, (_, _, mse)) in enumerate(sorted_results, 1): # Loop through sorted results with ranking
    print(f"  {rank}. {name}: MSE = {mse:.4f}") # Display rank, model name, and MSE

best_model = sorted_results[0] # Get best performing model
print(f"\n🏆 Best Model: {best_model[0]} (MSE: {best_model[1][2]:.4f})") # Display best model with trophy emoji

# ===== REGULARIZATION EFFECTIVENESS ANALYSIS =====
print("\n" + "="*80) # New line and extended separator
print("REGULARIZATION EFFECTIVENESS ANALYSIS") # Regularization effectiveness header
print("="*80) # Extended separator line for formatting

linear_mse = results["Linear"][2] # Get Linear regression MSE for comparison
print("Improvement over Linear Regression:") # Improvement analysis header

for name, (_, _, mse) in results.items(): # Loop through all model results
    if name != "Linear": # Skip linear regression (baseline)
        improvement = ((linear_mse - mse) / linear_mse) * 100 # Calculate percentage improvement
        status = "✓ Better" if mse < linear_mse else "✗ Worse" # Determine if model is better or worse
        print(f"  - {name:<12}: {improvement:+.2f}% {status}") # Display improvement percentage and status


# ===== ADVANCED COEFFICIENTS VISUALIZATION =====
plt.figure(figsize=(12,6)) # Create figure for coefficient comparison chart
width = 0.2 # Set bar width for grouped bar chart
x = np.arange(n_features) # Create x-axis positions for features

# Create grouped bar chart for coefficient comparison
plt.bar(x - width*1.5, results["Linear"][0], width, label="Linear", alpha=0.8) # Plot Linear regression coefficients
plt.bar(x - width*0.5, results["Ridge (L2)"][0], width, label="Ridge (L2)", alpha=0.8) # Plot Ridge regression coefficients
plt.bar(x + width*0.5, results["Lasso (L1)"][0], width, label="Lasso (L1)", alpha=0.8) # Plot Lasso regression coefficients
plt.bar(x + width*1.5, results["ElasticNet"][0], width, label="ElasticNet", alpha=0.8) # Plot ElasticNet coefficients

# Add true coefficients as reference line
plt.plot(x, true_coefs, 'ro--', linewidth=2, markersize=8, label="True Coefficients") # Plot true coefficients as red line with markers

plt.xticks(x, [f"x{i+1}" for i in range(n_features)]) # Set x-axis labels as feature names
plt.xlabel("Features") # Set x-axis label
plt.ylabel("Coefficient Value") # Set y-axis label
plt.title("Coefficient Comparison: Linear vs Ridge vs Lasso vs ElasticNet") # Set chart title
plt.grid(True, alpha=0.3) # Add grid for better readability
plt.legend() # Add legend to identify different models
plt.tight_layout() # Adjust layout for better appearance
plt.show() # Display the coefficient comparison chart