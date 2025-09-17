import numpy as np                                                                               # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                  # Import matplotlib for visualization
from sklearn.datasets import load_iris                                                           # Import iris dataset
from sklearn.model_selection import train_test_split                                             # Import train-test split function
from sklearn.ensemble import (                                                                   # Import ensemble methods
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, BaggingClassifier
)
from sklearn.tree import DecisionTreeClassifier                                                  # Import decision tree for comparison
from sklearn.metrics import accuracy_score, classification_report                                # Import evaluation metrics


iris = load_iris()                                                                               # Load iris dataset
X, y = iris.data, iris.target                                                                    # Extract features and targets

print("Dataset Info:")                                                                           # Dataset information header
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")                                          # Print dataset dimensions
print(f"Classes: {iris.target_names}")                                                           # Print class names
print(f"Feature names: {iris.feature_names}")                                                    # Print feature names
print()                                                                                          # Empty line


X_train, X_test, y_train, y_test = train_test_split(                                             # Split data
    X, y,                                                                                        # Features and targets
    test_size=0.2,                                                                               # 20% for testing
    random_state=42,                                                                             # Set random seed
    stratify=y                                                                                   # Maintain class distribution
)

print(f"Training samples: {X_train.shape[0]}")                                                   # Print training set size
print(f"Testing samples: {X_test.shape[0]}")                                                     # Print testing set size
print()                                                                                          # Empty line

# ===== BASELINE: SINGLE DECISION TREE =====
single_tree = DecisionTreeClassifier(random_state=42)                                            # Create single decision tree
single_tree.fit(X_train, y_train)                                                                # Train single tree
single_pred = single_tree.predict(X_test)                                                        # Make predictions
single_accuracy = accuracy_score(y_test, single_pred)                                            # Calculate accuracy

print(f"Single Decision Tree Accuracy: {single_accuracy:.3f}")                                   # Print baseline accuracy
print()                                                                                          # Empty line


# ===== ENSEMBLE METHOD 1: RANDOM FOREST =====
rf_classifier = RandomForestClassifier(                                                          # Create Random Forest
    n_estimators=200,                                                                             # Enough trees for stability
    max_depth=5,                                                                                  # Limit depth to avoid overfitting
    max_features="sqrt",                                                                          # Random subset of features per split
    bootstrap=True,                                                                               # Classic bootstrap sampling
    random_state=42                                                                               # Reproducibility
)

rf_classifier.fit(X_train, y_train)                                                              # Train Random Forest
rf_predictions = rf_classifier.predict(X_test)                                                   # Make predictions
rf_accuracy = accuracy_score(y_test, rf_predictions)                                             # Calculate accuracy

print(f"Random Forest Accuracy: {rf_accuracy:.3f}")                                              # Print RF accuracy
print(f"Number of trees: {rf_classifier.n_estimators}")                                          # Print number of trees

print("\nFeature Importance (Random Forest):")                                                   # Feature importance header
for i, importance in enumerate(rf_classifier.feature_importances_):                              # Iterate through importances
    print(f"  {iris.feature_names[i]}: {importance:.3f}")                                        # Print feature importance

print()                                                                                          # Empty line


# ===== ENSEMBLE METHOD 2: GRADIENT BOOSTING =====
gb_classifier = GradientBoostingClassifier(                                                      # Create Gradient Boosting
    n_estimators=100,                                                                             # Fewer boosting stages (prevent overfitting)
    learning_rate=0.1,                                                                            # Standard learning rate
    max_depth=3,                                                                                  # Each tree depth
    subsample=1.0,                                                                                # Use full dataset each stage
    random_state=42                                                                               # Reproducibility
)

gb_classifier.fit(X_train, y_train)                                                              # Train Gradient Boosting
gb_predictions = gb_classifier.predict(X_test)                                                   # Make predictions
gb_accuracy = accuracy_score(y_test, gb_predictions)                                             # Calculate accuracy

print(f"Gradient Boosting Accuracy: {gb_accuracy:.3f}")                                          # Print GB accuracy
print(f"Learning rate: {gb_classifier.learning_rate}")                                           # Print learning rate

print("\nFeature Importance (Gradient Boosting):")                                               # Feature importance header
for i, importance in enumerate(gb_classifier.feature_importances_):                              # Iterate through importances
    print(f"  {iris.feature_names[i]}: {importance:.3f}")                                        # Print feature importance

print()                                                                                          # Empty line



# ===== RESULTS COMPARISON =====
print("=" * 50) # Section separator
print("RESULTS COMPARISON") # Results section header
print("=" * 50) # Section separator

results = [ # Create results list
    ("Single Decision Tree", single_accuracy), # Single tree result
    ("Random Forest", rf_accuracy), # Random Forest result
    ("Gradient Boosting", gb_accuracy), # Gradient Boosting result
]

print("Method                 Accuracy") # Table header
print("-" * 35) # Table separator
for method, accuracy in results: # Iterate through results
    print(f"{method:<20} {accuracy:.3f}") # Print method and accuracy

best_method = max(results, key=lambda x: x[1]) # Find best method
print(f"\nBest method: {best_method[0]} ({best_method[1]:.3f})") # Print best method

print() # Empty line

# ===== ENSEMBLE VISUALIZATION =====
methods = [result[0] for result in results] # Extract method names
accuracies = [result[1] for result in results] # Extract accuracies

plt.figure(figsize=(10, 6)) # Create figure
bars = plt.bar(methods, accuracies, color=['red', 'blue', 'green', 'orange', 'purple']) # Create bar plot
plt.title('Ensemble Methods Comparison on Iris Dataset', fontsize=14) # Set title
plt.xlabel('Method', fontsize=12) # Set x-axis label
plt.ylabel('Accuracy', fontsize=12) # Set y-axis label
plt.ylim(0.8, 1.0) # Set y-axis limits
plt.xticks(rotation=45) # Rotate x-axis labels

for bar, accuracy in zip(bars, accuracies): # Iterate through bars and accuracies
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, # Position text
             f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold') # Add text with accuracy

plt.tight_layout() # Adjust layout
plt.grid(True, alpha=0.3) # Add grid
plt.show() # Display plot