import numpy as np                                                                       # Import NumPy for numerical operations
import pandas as pd                                                                      # Import pandas for data manipulation
from math import log2                                                                    # Import log2 function for entropy calculations
from collections import Counter                                                          # Import Counter for counting occurrences
from sklearn.tree import DecisionTreeClassifier, export_text                             # Import sklearn decision tree
from sklearn.metrics import accuracy_score                                               # Import metrics for evaluation
from sklearn.preprocessing import LabelEncoder                                           # Import label encoder

def calculate_entropy(y):                                                                # Function to calculate entropy of a dataset
    """Calculate entropy: H(S) = -Σ(p_i * log2(p_i))"""                                  # Entropy formula explanation
    if len(y) == 0:                                                                      # Check for empty dataset
        return 0                                                                         # Return 0 entropy for empty set
    
    counts = Counter(y)                                                                  # Count occurrences of each class
    total = len(y)                                                                       # Total number of samples
    entropy = 0                                                                          # Initialize entropy
    
    for count in counts.values():                                                        # Iterate through class counts
        p = count / total                                                                # Calculate probability of each class
        if p > 0:                                                                        # Avoid log(0)
            entropy -= p * log2(p)                                                       # Add entropy contribution
    
    return entropy                                                                       # Return calculated entropy

# ===== TENNIS DATASET =====
def create_tennis_dataset():                                                             # Function to create tennis dataset
    """Create the classic tennis dataset from decision tree theory"""                    # Dataset creation description

    data = [                                                                             # Tennis dataset - 20 examples
        ["Sunny", "Hot", "High", "Weak", "No"],                                          # Day 1
        ["Sunny", "Hot", "High", "Strong", "No"],                                        # Day 2  
        ["Overcast", "Hot", "High", "Weak", "Yes"],                                      # Day 3
        ["Rain", "Mild", "High", "Weak", "Yes"],                                         # Day 4
        ["Rain", "Cool", "Normal", "Weak", "Yes"],                                       # Day 5
        ["Rain", "Cool", "Normal", "Strong", "No"],                                      # Day 6
        ["Overcast", "Cool", "Normal", "Strong", "Yes"],                                 # Day 7
        ["Sunny", "Mild", "High", "Weak", "No"],                                         # Day 8
        ["Sunny", "Cool", "Normal", "Weak", "Yes"],                                      # Day 9
        ["Rain", "Mild", "Normal", "Weak", "Yes"],                                       # Day 10
        ["Sunny", "Mild", "Normal", "Strong", "Yes"],                                    # Day 11
        ["Overcast", "Mild", "High", "Strong", "Yes"],                                   # Day 12
        ["Overcast", "Hot", "Normal", "Weak", "Yes"],                                    # Day 13
        ["Rain", "Mild", "High", "Strong", "No"],                                        # Day 14
        ["Sunny", "Hot", "Normal", "Weak", "Yes"],                                       # Day 15
        ["Overcast", "Cool", "High", "Weak", "Yes"],                                     # Day 16
        ["Rain", "Hot", "Normal", "Strong", "No"],                                       # Day 17
        ["Sunny", "Cool", "High", "Strong", "No"],                                       # Day 18
        ["Overcast", "Mild", "Normal", "Weak", "Yes"],                                   # Day 19
        ["Rain", "Cool", "High", "Weak", "Yes"]                                          # Day 20
    ]
    
    columns = ["Outlook", "Temperature", "Humidity", "Wind", "Play"]                     # Column names
    df = pd.DataFrame(data, columns=columns)                                             # Create DataFrame
    
    X = df[["Outlook", "Temperature", "Humidity", "Wind"]].values                        # Features
    y = df["Play"].values                                                                # Target labels
    feature_names = ["Outlook", "Temperature", "Humidity", "Wind"]                       # Feature names
    
    return X, y, feature_names, df                                                       # Return dataset components

# ===== MAIN ANALYSIS =====
print("=" * 60) # Print header separator
print("DECISION TREE ANALYSIS: TENNIS DATASET (20 EXAMPLES)") # Main header
print("=" * 60) # Print header separator

X, y, feature_names, df = create_tennis_dataset() # Create tennis dataset

print("Dataset Overview:") # Dataset overview header
print(f"Total examples: {len(X)}") # Total number of examples
print(f"Features: {feature_names}") # Feature names
print(f"Class distribution: Yes={list(y).count('Yes')}, No={list(y).count('No')}") # Class distribution
print() # Empty line

print("Sample data (first 10 rows):") # Sample data header
print(df.head(10)) # Show first 10 rows
print() # Empty line

# ===== ENTROPY ANALYSIS =====
root_entropy = calculate_entropy(y) # Calculate root entropy
print(f"Root entropy: {root_entropy:.3f}") # Print root entropy

# Calculate information gain for each feature
print("\nInformation Gain by Feature:") # Information gain header
for i, feature in enumerate(feature_names): # Iterate through features
    unique_values = set(X[:, i]) # Get unique values for feature
    weighted_entropy = 0 # Initialize weighted entropy

    for value in unique_values: # For each unique value
        mask = X[:, i] == value # Create mask for value
        subset_y = y[mask] # Get subset labels
        weight = len(subset_y) / len(y) # Calculate weight
        weighted_entropy += weight * calculate_entropy(subset_y) # Add weighted entropy

    info_gain = root_entropy - weighted_entropy # Calculate information gain
    print(f"  {feature}: {info_gain:.3f}") # Print information gain

print() # Empty line

# ===== SKLEARN DECISION TREE =====
print("=" * 50) # Section separator
print("SKLEARN DECISION TREE") # Sklearn section header
print("=" * 50) # Section separator

X_encoded = X.copy()                                                                     # Copy original features
encoders = {}                                                                            # Dictionary to store encoders

for i, feature in enumerate(feature_names):                                              # Encode each feature
    le = LabelEncoder()                                                                  # Create label encoder
    X_encoded[:, i] = le.fit_transform(X[:, i])                                          # Encode feature values
    encoders[feature] = le                                                               # Store encoder

tree = DecisionTreeClassifier(                                                           # Create sklearn tree
    criterion='entropy',                                                                 # Use entropy criterion
    max_depth=4,                                                                         # Maximum depth
    min_samples_split=2,                                                                 # Minimum samples to split
    random_state=42                                                                      # Set random state
)

tree.fit(X_encoded.astype(int), y)                                                       # Train sklearn tree
predictions = tree.predict(X_encoded.astype(int))                                        # Make predictions
accuracy = accuracy_score(y, predictions)                                                # Calculate accuracy

print(f"Decision Tree Accuracy: {accuracy:.3f}") # Print accuracy
print() # Empty line

print("Tree Structure:") # Tree structure header
tree_rules = export_text(tree, feature_names=feature_names) # Export tree rules
print(tree_rules) # Print tree rules

# ===== PREDICTION EXAMPLES =====
print("=" * 50) # Section separator
print("NEW PREDICTIONS") # Prediction example header
print("=" * 50) # Section separator

# Example 1: Should predict "No"
new_day_1 = ["Sunny", "Mild", "High", "Weak"]                                            # New example to predict
print(f"Day 1 conditions: {dict(zip(feature_names, new_day_1))}")                        # Print new day conditions

new_day_1_encoded = []                                                                   # Initialize encoded example
for i, feature in enumerate(feature_names):                                              # Encode each feature
    if new_day_1[i] in encoders[feature].classes_:                                       # Check if value seen before
        encoded_value = encoders[feature].transform([new_day_1[i]])[0]                   # Encode value
    else:                                                                                # If new value
        encoded_value = 0                                                                # Use default encoding
    new_day_1_encoded.append(encoded_value)                                              # Add to encoded list

prediction_1 = tree.predict([new_day_1_encoded])                                         # Make prediction
print(f"Prediction: {prediction_1[0]}")                                                  # Print prediction
print()                                                                                  # Empty line

# Example 2: Should predict "Yes"
new_day_2 = ["Overcast", "Cool", "Normal", "Strong"]                                     # New example that should predict Yes
print(f"Day 2 conditions: {dict(zip(feature_names, new_day_2))}")                        # Print new day conditions

new_day_2_encoded = []                                                                   # Initialize encoded example
for i, feature in enumerate(feature_names):                                              # Encode each feature
    if new_day_2[i] in encoders[feature].classes_:                                       # Check if value seen before
        encoded_value = encoders[feature].transform([new_day_2[i]])[0]                   # Encode value
    else:                                                                                # If new value
        encoded_value = 0                                                                # Use default encoding
    new_day_2_encoded.append(encoded_value)                                              # Add to encoded list

prediction_2 = tree.predict([new_day_2_encoded])                                         # Make prediction
print(f"Prediction: {prediction_2[0]}")                                                  # Print prediction