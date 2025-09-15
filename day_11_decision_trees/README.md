# 🌳 LESSON 11: DECISION TREES

## 1. Theory: What Are Decision Trees?

### 🔹 The Simple Idea

A decision tree is like playing "20 Questions" or "Guess the Animal" game. You ask yes/no questions, and each answer narrows down the possibilities until you reach the final answer.

**Decision trees work with simple "if → then" rules:**

```
If weather is sunny AND humidity is high → Don't play tennis
If weather is rainy AND wind is strong → Don't play tennis
If weather is overcast → Play tennis
```

The tree learns these rules by looking at historical examples (your training data) and finding the best questions to ask that separate examples into clean groups.

### 🔹 Why Use Decision Trees?

**Main advantage:** **Interpretability** - You can easily read and understand the rules, unlike black-box models.

**Perfect for:** Classification tasks where you need to explain WHY the model made a decision.

📌 **Example:** A bank using a decision tree can explain to customers: "Your loan was rejected because your income is below $50k AND you have more than 3 credit cards."

### ✅ Quick Check:

How is a decision tree similar to the "Guess the Animal" game?

---

## 2. Theory: How Trees Choose Questions - Entropy

### 🔹 What is Entropy?

**Formula:**

```
H(S) = -Σ pᵢ × log₂(pᵢ)
```

**Simple explanation:** Entropy measures how "mixed up" or "messy" a group is.

- **Entropy = 0** → Perfect! All examples belong to the same class
- **Higher entropy** → More mixed up, harder to predict

### 🔹 Entropy Examples

| Group Content | Entropy | Interpretation             |
| ------------- | ------- | -------------------------- |
| 10 Yes, 0 No  | 0.0     | Perfect - all same class   |
| 5 Yes, 5 No   | 1.0     | Maximum mess - 50/50 split |
| 7 Yes, 3 No   | 0.88    | Somewhat messy             |

📌 **Analogy:** Think of entropy like measuring how difficult it would be to guess the class of a random example from the group. Clean groups (low entropy) are easy to predict!

### ✅ Quick Check:

What happens to entropy when all objects in a group belong to the same class?

---

## 3. Theory: Information Gain - Choosing the Best Split

### 🔹 Information Gain Formula

```
IG(S,A) = H(S) - Σ (|Sᵥ|/|S|) × H(Sᵥ)
```

**What this means:** How much did we reduce the "messiness" after splitting the data by feature A?

- **Higher Information Gain** → Better split
- **IG = 0** → Split didn't help at all

### 🔹 Step-by-Step Process

1. Calculate entropy of the original group
2. Split the group by each possible feature
3. Calculate weighted average entropy of the subgroups
4. **Information Gain = Original entropy - New weighted entropy**
5. Choose the feature with highest Information Gain

📌 **Intuition:** We want splits that create the "cleanest" subgroups possible.

### ✅ Quick Check:

What does it mean if Information Gain = 0 for a particular feature?

---

## 4. Theory: Alternative Measure - Gini Index

### 🔹 Gini Formula

```
Gini(S) = 1 - Σ pᵢ²
```

**Purpose:** Another way to measure "impurity" of a group.

- **Gini = 0** → Perfect purity (all same class)
- **Higher Gini** → More mixed up

### 🔹 Entropy vs Gini

| Aspect          | Entropy                 | Gini                    |
| --------------- | ----------------------- | ----------------------- |
| **Range**       | 0 to log₂(classes)      | 0 to 0.5                |
| **Used by**     | ID3 algorithm           | CART algorithm          |
| **Computation** | Slightly slower         | Faster                  |
| **Performance** | Usually similar results | Usually similar results |

📌 **Practical note:** Both work well in practice. Gini is slightly faster to compute, so it's often preferred.

### ✅ Quick Check:

What does a smaller Gini value indicate?

---

## 5. Real Example: Play Tennis Dataset

### 🔹 The Dataset (14 Examples)

Here's our classic dataset where each row represents one day:

```
Day | Outlook  | Temp | Humidity | Wind   | Play?
----|----------|------|----------|--------|-------
1   | Sunny    | Hot  | High     | Weak   | No
2   | Sunny    | Hot  | High     | Strong | No
3   | Overcast | Hot  | High     | Weak   | Yes
4   | Rain     | Mild | High     | Weak   | Yes
5   | Rain     | Cool | Normal   | Weak   | Yes
... | ...      | ...  | ...      | ...    | ...
14  | Rain     | Mild | High     | Strong | No
```

**Total:** 9 Yes, 5 No

### 🔹 Step 1: Calculate Root Entropy

```
p_yes = 9/14 ≈ 0.64
p_no = 5/14 ≈ 0.36

H(root) = -(0.64 × log₂(0.64)) - (0.36 × log₂(0.36))
H(root) ≈ 0.94
```

**Interpretation:** 0.94 is fairly "messy" - we need to split the data.

### 🔹 Step 2: Calculate Information Gain for Each Feature

**A) Outlook (Sunny, Overcast, Rain):**

- **Sunny:** 5 examples → 2 Yes, 3 No → H ≈ 0.97
- **Overcast:** 4 examples → 4 Yes, 0 No → H = 0.0 (perfect!)
- **Rain:** 5 examples → 3 Yes, 2 No → H ≈ 0.97

Weighted entropy = (5/14 × 0.97) + (4/14 × 0.0) + (5/14 × 0.97) ≈ 0.69

**IG(Outlook) = 0.94 - 0.69 = 0.25** ← Best split!

**B) Other features:**

- IG(Humidity) ≈ 0.15
- IG(Wind) ≈ 0.05
- IG(Temperature) ≈ 0.03

### 🔹 Step 3: Build the Tree

**Root:** Split by Outlook (highest IG)

```
Outlook?
├─ Overcast → Play = Yes (pure group!)
├─ Sunny    → Still mixed (2 Yes, 3 No) - need more splits
└─ Rain     → Still mixed (3 Yes, 2 No) - need more splits
```

**Continue recursively** for Sunny and Rain branches until all leaves are pure or stopping criteria are met.

### ✅ Quick Check:

Why did we choose Outlook as the root split instead of Temperature?

---

## 6. The Complete Tree Solution

### 🔹 Final Tree Structure

```
Outlook?
├─ Overcast → Play = Yes
├─ Sunny
│   └─ Humidity?
│       ├─ High   → Play = No
│       └─ Normal → Play = Yes
└─ Rain
    └─ Wind?
        ├─ Weak   → Play = Yes
        └─ Strong → Play = No
```

### 🔹 How to Use the Tree

**Example prediction:** New day with Outlook=Sunny, Humidity=High

1. Start at root: Outlook? → Go to Sunny branch
2. At Sunny node: Humidity? → Go to High branch
3. At High leaf: **Prediction = No**

📌 **That's it!** Simple rules that anyone can follow and understand.

### ✅ Quick Check:

What would be the prediction for: Outlook=Rain, Wind=Weak?

---

## 7. The Overfitting Problem

### 🔹 Why Trees Overfit

**The problem:** If we let the tree grow too deep, it will create very specific rules for each training example, including noise and outliers.

**Result:**

- **Training accuracy:** 100% (perfect memory)
- **Test accuracy:** Poor (can't generalize)

### 🔹 How to Prevent Overfitting

**1. Set Stopping Criteria:**

- `max_depth`: Limit tree depth
- `min_samples_split`: Minimum examples needed to split a node
- `min_samples_leaf`: Minimum examples in each leaf

**2. Pruning:**

- Grow full tree, then cut weak branches

**3. Use Ensembles:**

- Random Forest (many trees voting)
- Gradient Boosting (sequential improvement)

📌 **Key insight:** Don't pursue perfect purity on training data - leave some "messiness" for better generalization.

### ✅ Quick Check:

Why shouldn't we grow a tree until every leaf is 100% pure?

---

## 8. Python Implementation

### 8.1 Helper Functions

```python
import numpy as np
from math import log2
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text

# Calculate entropy
def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))  # +1e-10 to avoid log(0)

# Calculate Gini index
def gini(y):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return 1 - np.sum(probs**2)
```

### 8.2 Test with Small Dataset

```python
# Mini dataset for demonstration
X = [
    ["Sunny", "High"],
    ["Sunny", "High"],
    ["Rain", "High"],
    ["Rain", "Normal"],
    ["Sunny", "Normal"]
]
y = ["No", "No", "Yes", "Yes", "Yes"]

print(f"Dataset entropy: {entropy(y):.2f}")
print(f"Dataset Gini: {gini(y):.2f}")
```

**Output:**

```
Dataset entropy: 0.97
Dataset Gini: 0.48
```

### 8.3 Real Example with Iris Dataset

```python
# Load Iris dataset
iris = load_iris()

# Create decision tree
clf = DecisionTreeClassifier(
    criterion="entropy",    # Use entropy for splitting
    max_depth=3,           # Prevent overfitting
    min_samples_split=5    # Need at least 5 samples to split
)

# Train the tree
clf.fit(iris.data, iris.target)

# Print tree structure
print(export_text(clf, feature_names=iris.feature_names))

# Make predictions
print(f"Accuracy: {clf.score(iris.data, iris.target):.3f}")
```

### 8.4 Understanding Feature Importance

```python
# See which features are most important
feature_importance = clf.feature_importances_
feature_names = iris.feature_names

for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.3f}")
```

📌 **This shows which features contributed most to the tree's decisions.**

### ✅ Quick Check:

What do the numbers in `clf.feature_importances_` represent?

---

## 9. Continuous Features: Finding the Best Threshold

### 🔹 How Trees Handle Numbers

For continuous features (like age, income, temperature), the tree needs to find the best **threshold** to split on.

**Process:**

1. Sort all values of the feature
2. Try splitting between each pair of adjacent values
3. Calculate Information Gain for each possible threshold
4. Choose the threshold with highest IG

### 🔹 Example

If we have ages: [25, 30, 35, 40, 45], the tree will try thresholds:

- age ≤ 27.5 (between 25 and 30)
- age ≤ 32.5 (between 30 and 35)
- age ≤ 37.5 (between 35 and 40)
- age ≤ 42.5 (between 40 and 45)

The threshold that creates the cleanest split wins!

📌 **That's why you see rules like "petal width ≤ 0.80" in sklearn output.**

### ✅ Quick Check:

Why does the tree try thresholds between adjacent values rather than at the exact values?

---

## 10. Practical Tips and Best Practices

### 🔹 Hyperparameter Tuning

**Important parameters to tune:**

- `max_depth`: Start with 3-7, increase if underfitting
- `min_samples_split`: Try 2, 5, 10
- `min_samples_leaf`: Try 1, 2, 5
- `criterion`: Try both "entropy" and "gini"

### 🔹 When to Use Decision Trees

**Great for:**

- Need explainable predictions
- Mixed data types (categorical + numerical)
- Non-linear relationships
- Feature interactions matter

**Not ideal for:**

- Very high-dimensional data
- When small changes in data cause big changes in tree structure
- Need probabilistic outputs

### 🔹 Real-World Applications

- **Medical diagnosis:** "If symptom A AND test B > threshold → likely disease X"
- **Credit approval:** Clear rules for loan decisions
- **Marketing:** Customer segmentation with interpretable rules
- **Fraud detection:** Transparent rule-based alerts

📌 **Remember:** Decision trees are often used as building blocks in more powerful ensemble methods like Random Forest and XGBoost!

### ✅ Quick Check:

Why might a bank prefer decision trees over neural networks for loan approval?

---

## 12. Practice Questions

### 🎤 Test Your Understanding:

1. **Why does entropy equal 0 when all examples belong to the same class?**
2. **What would happen if we calculated Information Gain for a feature that has the same value for all examples?**
3. **In the Play Tennis example, why didn't we need to split the Overcast branch further?**
4. **How would you explain to a non-technical person why decision trees can overfit?**
5. **When might you choose Gini over entropy as your splitting criterion?**
6. **What's the difference between `min_samples_split` and `min_samples_leaf` parameters?**

These questions will help solidify your understanding of decision trees! 🌳

---

### 🔹 Key Takeaways

- **Decision trees** create interpretable "if-then" rules from data
- **Entropy/Gini** measure how "mixed" a group of examples is
- **Information Gain** helps choose the best feature to split on
- **Overfitting** happens when trees memorize training data instead of learning patterns
- **Hyperparameters** like max_depth help control overfitting
- **Trees** work well for both categorical and numerical features
- **Interpretability** is the main advantage over more complex models

_Happy Learning! 🚀_
