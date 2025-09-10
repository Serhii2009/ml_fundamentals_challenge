# 🔥 30-Day Machine Learning & Deep Learning Challenge

Welcome to my **30-Day Machine Learning & Deep Learning Challenge** repository! This repository is a structured learning journey designed to take you from foundational concepts to advanced deep learning architectures in **just 30 days**. The goal is not only to understand the theory but also to implement every concept practically in **Python**, and be able to explain it clearly—even to a beginner.

---

## 🎯 Purpose

The purpose of this repository is to provide a **complete, hands-on guide** to Machine Learning (ML) and Deep Learning (DL) for learners of all levels. By following this challenge, you will:

- Understand the **fundamental mathematics** behind ML/DL concepts.
- Learn how to implement **core algorithms** from scratch in Python.
- Build a strong intuition for concepts through **analogies and examples**.
- Practice explaining topics in your own words to ensure **deep comprehension**.
- Have a reference that can be revisited, modified, and shared with others for educational purposes.

This repository is designed with **volunteering and learning in mind**, so anyone can follow along, experiment, and improve their ML/DL skills.

---

## 📚 Repository Structure

The repository is organized into **30 folders**, one for each day of the challenge. Each folder contains:

- `day_<number>_<topic>`  
  Example: `day_01_loss_function`

Inside each folder, you will find:

1. **Python script (`.py`)**
   - Contains **full working examples** of the topic.
   - Includes step-by-step implementation, code comments, and outputs.
2. **ReadMe.md (optional for each day)**
   - Explains the topic in **plain English**.
   - Includes **analogies, mathematical explanations, and mini-exercises**.
   - Designed so you could explain the concept to a 10-year-old after studying.

---

## 🗂 Example Folder Structure

```
ml_fundamentals_challenge/
├── day_01_loss_function/
│   ├── loss_function.py
│   └── README.md
├── day_02_gradient_descent/
│   ├── gradient_descent.py
│   └── README.md
├── day_03_regularization/
│   ├── regularization.py
│   └── README.md
...
├── day_30_final_project/
│   ├── final_project.py
│   └── README.md
└── README.md
```

---

## 🔑 Learning Approach

Each day follows a **clear and structured path**:

1. **Theory**
   - Full explanation of the concept with formulas.
   - Analogies to make abstract ideas intuitive.
2. **Practical Implementation**
   - Python code with detailed comments.
   - Example outputs to see results in action.
3. **Verification & Reflection**
   - Mini exercises to reinforce learning.
   - Encouragement to explain the topic to others to **solidify understanding**.

---

## 📈 30-Day Learning Roadmap

| Week       | Days       | Focus Area                                        | Key Topics                                                                          |
| ---------- | ---------- | ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **Week 1** | Days 1-7   | **Deep Understanding of Loss & Gradient Descent** | Loss Functions, Gradient Descent, Learning Rate, Momentum, Regularization           |
| **Week 2** | Days 8-14  | **ML Foundations**                                | Linear/Logistic Regression, Metrics, Decision Trees, Ensembles, Feature Engineering |
| **Week 3** | Days 15-21 | **Deep Learning Fundamentals**                    | Perceptron, Neural Networks, Forward/Backward Propagation, Optimization             |
| **Week 4** | Days 22-30 | **Advanced Deep Learning**                        | CNNs, RNNs, LSTM, Transformers, Final Project                                       |

### Detailed Day-by-Day Plan

## Week 1: Deep Understanding of Loss & Gradient Descent

**Goal:** Understand how models learn from the inside — loss, gradients, optimization steps.

| Day       | Topic                                       | Theory                                     | Practice                                         | Goal                                       | 1-Minute LinkedIn Video                                                                                                                |
| --------- | ------------------------------------------- | ------------------------------------------ | ------------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Day 1** | Loss Function Mathematics                   | MSE, Cross-Entropy, formulas, meaning      | Implement MSE and Cross-Entropy with numpy       | Explain loss functions to a 10-year-old    | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-machinelearning-activity-7362854560731734016-kAfl                     |
| **Day 2** | Introduction to Gradient Descent            | Derivative as direction of smallest change | Implement gradient descent for one variable      | Understand "rolling down the hill" analogy | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-machinelearning-activity-7367575410915565568-gL_X                     |
| **Day 3** | Multidimensional Gradient Descent           | Gradients for vectors and matrices         | Implement gradient descent for linear regression | Calculate gradient step manually           | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-dl-activity-7368267543695740930-x_OQ                                  |
| **Day 4** | Learning Rate, Momentum, RMSProp            | Why step size regulation matters           | Add momentum to gradient descent                 | Master optimization techniques             | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-machinelearning-activity-7368644528934797313-VADw                     |
| **Day 5** | Regularization                              | L1, L2, Elastic Net, Dropout               | Add L2 regularization to linear regression       | Understand "penalty for complexity"        | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-machinelearning-activity-7366103411445927937-rST6                     |
| **Day 6** | Practice: Gradient Descent + Regularization | Combine concepts                           | Build model on synthetic data                    | Experiment with hyperparameters            | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-artificialintelligence-activity-7369390120652808194-ZZG1              |
| **Day 7** | Week 1 Explanation                          | Review and solidify                        | Explain all concepts in your own words           | Deep comprehension check                   | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-artificialintelligence-machinelearning-activity-7369739519358672896-pjad |

## Week 2: ML Foundations

**Goal:** Build foundation for classical algorithms.

| Day        | Topic                                       | Theory                                             | Practice                               | Goal                             | 1-Minute LinkedIn Video                                                                                                          |
| ---------- | ------------------------------------------- | -------------------------------------------------- | -------------------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Day 8**  | Linear Regression                           | Formulas, MSE, gradient descent vs normal equation | Linear regression on Boston dataset    | Master linear relationships      | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-machinelearning-deeplearning-activity-7370093692860395520-sY4g     |
| **Day 9**  | Logistic Regression                         | Sigmoid, cross-entropy, gradient descent           | Implement from scratch in Python       | Understand classification basics | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-machinelearning-deeplearning-activity-7371156573056028672-GUj_/    |
| **Day 10** | Classification Metrics                      | Accuracy, Precision, Recall, F1-score, ROC-AUC     | Apply sklearn on simple classification | Evaluate model performance       | https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_happy-day-10-of-our-ml-dl-challenge-activity-7371543490503053312-AlWw |
| **Day 11** | Decision Trees                              | Space partitioning, entropy, Gini                  | Build decision tree with sklearn       | Understand tree-based decisions  | Coming Soon                                                                                                                      |
| **Day 12** | Ensembles: Random Forest, Gradient Boosting | Bagging vs Boosting concepts                       | Apply Random Forest on dataset         | Master ensemble methods          | Coming Soon                                                                                                                      |
| **Day 13** | Feature Engineering & Scaling               | Normalization, standardization, one-hot encoding   | Preprocess real dataset                | Prepare data for models          | Coming Soon                                                                                                                      |
| **Day 14** | Week 2 Explanation                          | Review ML algorithms and preprocessing             | Explain all concepts in your own words | Solidify ML fundamentals         | Coming Soon                                                                                                                      |

## Week 3: Deep Learning Fundamentals

**Goal:** Understand neural network structure and backpropagation.

| Day        | Topic                 | Theory                                  | Practice                                   | Goal                                      | 1-Minute LinkedIn Video |
| ---------- | --------------------- | --------------------------------------- | ------------------------------------------ | ----------------------------------------- | ----------------------- |
| **Day 15** | Perceptron            | Formula, linear combination, activation | Implement single perceptron                | Understand "decision with weighted brain" | Coming Soon             |
| **Day 16** | Activation Functions  | Sigmoid, ReLU, Tanh, LeakyReLU          | Visualize functions on graphs              | Master non-linearity                      | Coming Soon             |
| **Day 17** | Forward Propagation   | Signal flow from input to output        | Implement forward propagation with numpy   | Understand information flow               | Coming Soon             |
| **Day 18** | Backpropagation       | Chain rule, gradients for each layer    | Calculate gradients manually for one layer | Master "backward wave for corrections"    | Coming Soon             |
| **Day 19** | Optimization          | SGD, Momentum, Adam                     | Implement Adam for simple NN               | Advanced optimization techniques          | Coming Soon             |
| **Day 20** | Practice: NN on MNIST | Combine all concepts                    | Build 1-2 layer NN with numpy              | End-to-end neural network                 | Coming Soon             |
| **Day 21** | Week 3 Explanation    | Review NN, forward/backward propagation | Explain all concepts in your own words     | Deep learning comprehension               | Coming Soon             |

## Week 4: Advanced DL & Modern Architectures

**Goal:** Understanding CNN, RNN, LSTM, Transformer, Attention.

| Day        | Topic                            | Theory                                              | Practice                                       | Goal                                    | 1-Minute LinkedIn Video |
| ---------- | -------------------------------- | --------------------------------------------------- | ---------------------------------------------- | --------------------------------------- | ----------------------- |
| **Day 22** | CNN Basics                       | Convolution, Pooling, Flatten                       | Simple CNN on MNIST with PyTorch/Keras         | Understand spatial processing           | Coming Soon             |
| **Day 23** | Advanced CNN                     | Padding, stride, filter size                        | Visualize feature maps                         | Master convolutional operations         | Coming Soon             |
| **Day 24** | RNN Basics                       | Sequential data, hidden state                       | Simple RNN on synthetic sequence               | Process sequential information          | Coming Soon             |
| **Day 25** | LSTM/GRU                         | Gates, memory cell                                  | Implement LSTM with PyTorch/Keras              | Handle long-term dependencies           | Coming Soon             |
| **Day 26** | Attention & Transformer          | Self-attention, query/key/value                     | Simple transformer scheme                      | "Each word looks at others for context" | Coming Soon             |
| **Day 27** | Optimization Tricks              | BatchNorm, Dropout, LR scheduler, Gradient clipping | Add to CNN/RNN                                 | Master training techniques              | Coming Soon             |
| **Day 28** | Complex Practice                 | Mini-project combining concepts                     | Image classification + metrics + visualization | Apply all knowledge                     | Coming Soon             |
| **Day 29** | Final DL Explanation             | Review all DL concepts                              | Explain CNN, RNN, LSTM, Transformer            | Complete understanding                  | Coming Soon             |
| **Day 30** | Mini-Project + Final Explanation | End-to-end project                                  | MNIST/FashionMNIST with NN/CNN                 | Master complete ML pipeline             | Coming Soon             |

---

## 🧩 How to Use This Repository

1. **Clone the repository:**

```bash
git clone https://github.com/Serhii2009/ml_fundamentals_challenge
cd ml_fundamentals_challenge
```

2. **Go through each folder day by day:**

```bash
cd day_01_loss_function
# Open Python script and study it
python loss_function.py
```

3. **Read the README.md** in each folder for explanations, analogies, and exercises.

4. **Practice** by modifying the code, experimenting with parameters, and solving exercises.

5. **Explain each topic** in your own words (even to a 10-year-old!)—this is a key step for deep understanding.

---

## 💡 Notes

- All code is written in **Python 3**, using NumPy, pandas, scikit-learn, and PyTorch/Keras for deep learning examples.
- Each day builds on the previous, so it's recommended to follow the sequence from **Day 1 to Day 30**.
- This repository is designed for **self-learning, teaching, and collaboration**.

---

## 🎓 Outcome

By completing this 30-day challenge:

- You will have a **solid understanding** of ML and DL fundamentals.
- You will be able to **implement algorithms from scratch** and understand their inner workings.
- You will gain **confidence to explain concepts clearly** to others.
- You will have a **structured portfolio** of practical ML/DL projects.

**Learning by doing, reflecting, and teaching is the fastest path to mastering Machine Learning and Deep Learning.**

If you follow this repository day by day, and truly practice each topic, you will understand the math, the code, and the intuition behind every core concept.

---

## 📝 License

This repository is licensed under the **MIT License** – see the [LICENSE](./LICENSE) file for details.
