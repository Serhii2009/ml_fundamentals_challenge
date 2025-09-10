import numpy as np                                                                      # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                         # Import matplotlib for visualization
from sklearn.datasets import make_classification                                        # Import function to generate synthetic dataset
from sklearn.linear_model import LogisticRegression                                     # Import sklearn's LogisticRegression
from sklearn.metrics import (                                                           # Import metrics for model evaluation
    confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve
)

X, y = make_classification(                                                             # Generate classification dataset
    n_samples=1000,                                                                     # Number of samples
    n_features=10,                                                                      # Number of features  
    weights=[0.9, 0.1],                                                                 # Create imbalanced classes (90% class 0, 10% class 1)
    random_state=42                                                                     # Random seed for reproducibility
)

print("Dataset Info:")                                                                  # Dataset information header
print(f"Samples: {len(X)}, Features: {X.shape[1]}")                                     # Print dataset dimensions
print(f"Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")                                # Print class distribution
print()                                                                                 # Empty line for spacing

# ===== MODEL TRAINING =====
model = LogisticRegression(random_state=42)                                             # Create logistic regression model
model.fit(X, y)                                                                         # Train the model on data

y_pred = model.predict(X)                                                               # Get binary predictions
y_proba = model.predict_proba(X)[:, 1]                                                  # Get probabilities for positive class

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y, y_pred)                                                        # Calculate confusion matrix
print("Confusion Matrix:")                                                              # Matrix header
print(f"    Predicted")                                                                 # Column header
print(f"     0    1")                                                                   # Column labels
print(f"0  {cm[0,0]:3d}  {cm[0,1]:3d}")                                                 # Row 0: TN, FP
print(f"1  {cm[1,0]:3d}  {cm[1,1]:3d}")                                                 # Row 1: FN, TP
print()                                                                                 # Empty line

# ===== MANUAL METRICS CALCULATION =====
TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]                                     # Extract confusion matrix values

accuracy = (TP + TN) / (TP + TN + FP + FN)                                              # Calculate accuracy manually
precision = TP / (TP + FP) if (TP + FP) > 0 else 0                                      # Calculate precision manually
recall = TP / (TP + FN) if (TP + FN) > 0 else 0                                         # Calculate recall manually
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0   # Calculate F1 manually

print("Manual Calculations:")                                                           # Manual calculations header
print(f"Accuracy:  {accuracy:.3f}")                                                     # Print manual accuracy
print(f"Precision: {precision:.3f}")                                                    # Print manual precision
print(f"Recall:    {recall:.3f}")                                                       # Print manual recall
print(f"F1-Score:  {f1:.3f}")                                                           # Print manual F1
print()                                                                                 # Empty line

# ===== SKLEARN VERIFICATION =====
print("Sklearn Verification:") # Sklearn verification header
print(f"Accuracy:  {accuracy_score(y, y_pred):.3f}") # Sklearn accuracy
print(f"Precision: {precision_score(y, y_pred):.3f}") # Sklearn precision
print(f"Recall:    {recall_score(y, y_pred):.3f}") # Sklearn recall
print(f"F1-Score:  {f1_score(y, y_pred):.3f}") # Sklearn F1
print(f"ROC-AUC:   {roc_auc_score(y, y_proba):.3f}") # Sklearn ROC-AUC
print() # Empty line

# ===== ROC CURVE VISUALIZATION =====
fpr, tpr, thresholds = roc_curve(y, y_proba) # Calculate ROC curve points
auc_score = roc_auc_score(y, y_proba) # Get AUC score

plt.figure(figsize=(8, 6)) # Create figure
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})') # Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random') # Plot diagonal line
plt.xlabel('False Positive Rate') # X-axis label
plt.ylabel('True Positive Rate') # Y-axis label
plt.title('ROC Curve') # Plot title
plt.legend() # Show legend
plt.grid(True, alpha=0.3) # Add grid
plt.show() # Display plot