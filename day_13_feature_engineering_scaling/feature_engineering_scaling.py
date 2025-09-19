import pandas as pd                                                                                  # Import pandas for data manipulation
import numpy as np                                                                                   # Import NumPy for numerical operations
import matplotlib.pyplot as plt                                                                      # Import matplotlib for visualization
from sklearn.preprocessing import (                                                                  # Import preprocessing tools
    MinMaxScaler, StandardScaler, RobustScaler, 
    OneHotEncoder, LabelEncoder
)

# ===== DATASET CREATION =====
data = pd.DataFrame({                                                                                # Create sample dataset
    "Age": [18, 25, 35, 45, 55, 65, 22, 28, 38, 48],                                                 # Numerical feature - age
    "Income": [20000, 35000, 50000, 75000, 95000, 120000, 28000, 42000, 68000, 85000],               # Numerical feature - income  
    "City": ["Kiev", "Lviv", "Kiev", "Odessa", "Kiev", "Lviv", "Odessa", "Kiev", "Lviv", "Odessa"],  # Categorical feature
    "Education": ["High School", "Bachelor", "Master", "Bachelor", "PhD", "Master", "High School",   # Categorical feature
                  "Bachelor", "PhD", "Master"]                                                       # Categorical feature
})

print("Original Dataset:")                                                                           # Original dataset header
print(data)                                                                                          # Display original data
print(f"\nDataset shape: {data.shape}")                                                              # Print dataset dimensions
print(f"Data types:\n{data.dtypes}")                                                                 # Print data types
print()                                                                                              # Empty line

print("Numerical Features Statistics:")                                                              # Statistics header
print(data[["Age", "Income"]].describe())                                                            # Display statistics
print()                                                                                              # Empty line



# ===== SCALING METHODS COMPARISON =====
numerical_data = data[["Age", "Income"]].copy()                                                      # Copy numerical features

print("Original Numerical Data:")                                                                    # Original data header
print(f"Age - Min: {numerical_data['Age'].min()}, Max: {numerical_data['Age'].max()}, Mean: {numerical_data['Age'].mean():.2f}") # Age statistics
print(f"Income - Min: {numerical_data['Income'].min()}, Max: {numerical_data['Income'].max()}, Mean: {numerical_data['Income'].mean():.2f}") # Income statistics
print()                                                                                              # Empty line

# Method 1: MinMax Normalization (0-1 range)
minmax_scaler = MinMaxScaler()                                                                       # Create MinMax scaler
data_minmax = minmax_scaler.fit_transform(numerical_data)                                            # Apply MinMax scaling
df_minmax = pd.DataFrame(data_minmax, columns=["Age_MinMax", "Income_MinMax"])                       # Create DataFrame

print("After MinMax Scaling (0-1 range):")                                                           # MinMax results header
print(df_minmax.describe())                                                                          # Display MinMax statistics
print()                                                                                              # Empty line

# Method 2: Standard Scaling (mean=0, std=1)
standard_scaler = StandardScaler()                                                                   # Create Standard scaler
data_standard = standard_scaler.fit_transform(numerical_data)                                        # Apply Standard scaling
df_standard = pd.DataFrame(data_standard, columns=["Age_Standard", "Income_Standard"])               # Create DataFrame

print("After Standard Scaling (mean=0, std=1):")                                                     # Standard results header
print(df_standard.describe())                                                                        # Display Standard statistics
print()                                                                                              # Empty line

# Method 3: Robust Scaling (median and IQR)
robust_scaler = RobustScaler()                                                                       # Create Robust scaler
data_robust = robust_scaler.fit_transform(numerical_data)                                            # Apply Robust scaling
df_robust = pd.DataFrame(data_robust, columns=["Age_Robust", "Income_Robust"])                       # Create DataFrame

print("After Robust Scaling (median=0, IQR=1):")                                                     # Robust results header
print(df_robust.describe())                                                                          # Display Robust statistics
print()                                                                                              # Empty line



# ===== CATEGORICAL ENCODING =====
print("=" * 50)                                                                                      # Section separator
print("CATEGORICAL ENCODING METHODS")                                                                # Encoding section header
print("=" * 50)                                                                                      # Section separator

print("Original Categorical Data:")                                                                  # Original categorical data header
print("City values:", data["City"].unique())                                                         # Display unique city values
print("Education values:", data["Education"].unique())                                               # Display unique education values
print()                                                                                              # Empty line

# Method 1: Label Encoding (ordinal numbers)
label_encoder_city = LabelEncoder()                                                                  # Create label encoder for city
label_encoder_education = LabelEncoder()                                                             # Create label encoder for education

data["City_Label"] = label_encoder_city.fit_transform(data["City"])                                  # Apply label encoding to city
data["Education_Label"] = label_encoder_education.fit_transform(data["Education"])                   # Apply label encoding to education

print("After Label Encoding:")                                                                       # Label encoding results header
print("City Label Mapping:")                                                                         # City mapping header
for i, city in enumerate(label_encoder_city.classes_):                                               # Iterate through city classes
    print(f"  {city}: {i}")                                                                          # Print city to number mapping

print("Education Label Mapping:")                                                                    # Education mapping header
for i, education in enumerate(label_encoder_education.classes_):                                     # Iterate through education classes
    print(f"  {education}: {i}")                                                                     # Print education to number mapping
print()                                                                                              # Empty line

# Method 2: One-Hot Encoding (binary columns)
onehot_encoder = OneHotEncoder(sparse_output=False)                                                  # Create OneHot encoder
city_onehot = onehot_encoder.fit_transform(data[["City"]])                                           # Apply OneHot encoding to city
city_onehot_df = pd.DataFrame(                                                                       # Create DataFrame for OneHot
    city_onehot, 
    columns=onehot_encoder.get_feature_names_out(["City"])
)

print("After One-Hot Encoding (City):")                                                              # OneHot results header
print(city_onehot_df.head())                                                                         # Display OneHot results
print()                                                                                              # Empty line



# ===== COMPLETE PREPROCESSED DATASET =====
print("=" * 50) # Section separator
print("COMPLETE PREPROCESSED DATASET") # Complete dataset section header
print("=" * 50) # Section separator

# Combine all preprocessing results
final_dataset = pd.concat([ # Combine all processed data
    data[["Age", "Income"]], # Original numerical features
    df_minmax, # MinMax scaled features
    df_standard, # Standard scaled features
    data[["City_Label", "Education_Label"]], # Label encoded features
    city_onehot_df # OneHot encoded features
], axis=1)

print("Final Preprocessed Dataset:") # Final dataset header
print(final_dataset.head()) # Display final dataset
print(f"\nFinal dataset shape: {final_dataset.shape}") # Print final dimensions
print() # Empty line



# ===== VISUALIZATION =====
fig, axes = plt.subplots(2, 2, figsize=(12, 8)) # Create subplot grid
fig.suptitle('Scaling Methods Comparison', fontsize=16) # Set main title

# Original data
axes[0, 0].scatter(numerical_data["Age"], numerical_data["Income"], alpha=0.7) # Plot original data
axes[0, 0].set_title("Original Data") # Set title
axes[0, 0].set_xlabel("Age") # Set x-label
axes[0, 0].set_ylabel("Income") # Set y-label

# MinMax scaled data  
axes[0, 1].scatter(df_minmax["Age_MinMax"], df_minmax["Income_MinMax"], alpha=0.7, color='red') # Plot MinMax data
axes[0, 1].set_title("MinMax Scaled (0-1)") # Set title
axes[0, 1].set_xlabel("Age (MinMax)") # Set x-label
axes[0, 1].set_ylabel("Income (MinMax)") # Set y-label

# Standard scaled data
axes[1, 0].scatter(df_standard["Age_Standard"], df_standard["Income_Standard"], alpha=0.7, color='green') # Plot Standard data
axes[1, 0].set_title("Standard Scaled (mean=0, std=1)") # Set title
axes[1, 0].set_xlabel("Age (Standard)") # Set x-label
axes[1, 0].set_ylabel("Income (Standard)") # Set y-label

# Robust scaled data
axes[1, 1].scatter(df_robust["Age_Robust"], df_robust["Income_Robust"], alpha=0.7, color='purple') # Plot Robust data
axes[1, 1].set_title("Robust Scaled (median=0)") # Set title
axes[1, 1].set_xlabel("Age (Robust)") # Set x-label
axes[1, 1].set_ylabel("Income (Robust)") # Set y-label

plt.tight_layout() # Adjust layout
plt.show() # Display plot