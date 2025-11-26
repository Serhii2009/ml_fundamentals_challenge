# 📘 LESSON 26: ATTENTION & TRANSFORMERS

## 1. Introduction: Why Attention Mechanism Exists

### 🔹 The Fundamental Problem RNN/LSTM Can't Solve

Imagine translating a long sentence:

```
English: "The agreement that we signed last year with the European partners..."
French:  "L'accord que nous avons signé l'année dernière avec les partenaires européens..."
```

**RNN/LSTM problem:**

- Must compress entire English sentence into fixed-size hidden state
- By the time it reaches "partners", information about "agreement" is faded
- Important words are far apart → information degraded

**Visual:**

```
"The agreement ... partners" (10+ words apart)
     ↓              ↓
RNN: [h1] → [h2] → ... → [h10]
            Information fades through chain!
```

### 🔹 What Attention Solves

**Core idea:** Instead of compressing everything into one state, let the model **directly look at any part** of the input when needed.

```
Translating "partners":
✅ Attention: Look back at "agreement", "European", "signed"
❌ RNN: Only has faded memory h10
```

**The Search Engine Analogy:**

```
Question (Query): "Who won the 1998 World Cup?"
Documents (Keys):  [Doc1: "1994 World Cup..."]
                   [Doc2: "1998 World Cup France..."] ← HIGH MATCH
                   [Doc3: "2002 World Cup..."]
Content (Values):  Extract answer from Doc2

Attention works the same:
- Query: What info do I need?
- Keys: What info is available?
- Values: The actual information
```

### 🔹 How Attention Changed Everything

**Before (2014):** RNN → Encoder compresses → Decoder generates
**After (2014):** Encoder → **Attention** → Decoder can look back anywhere

**Impact:**

- Translation quality jumped 20%+
- Enabled handling 1000+ token sequences
- Led to Transformers → GPT, BERT, ChatGPT

### ✅ Quick Check:

Why is Attention useful for long sentences compared to RNN?

---

## 2. Self-Attention: How Words Look at Each Other

### 🔹 The Core Concept

**Self-Attention:** Each word in a sentence looks at **all other words** (including itself) to understand context.

```
Sentence: "The cat sat on the mat"

Word "sat":
- Looks at "cat" → Who is sitting? (HIGH attention)
- Looks at "mat" → Where sitting? (HIGH attention)
- Looks at "The" → Not relevant (LOW attention)
```

### 🔹 Why It Works Mathematically

**Step 1:** Convert words to vectors (embeddings)

```
"cat" → [0.2, 0.8, 0.1, 0.5]
"sat" → [0.6, 0.3, 0.7, 0.2]
```

**Step 2:** Measure similarity between vectors

```
similarity("cat", "sat") = dot_product([0.2,0.8,0.1,0.5], [0.6,0.3,0.7,0.2])
                         = 0.2×0.6 + 0.8×0.3 + 0.1×0.7 + 0.5×0.2
                         = 0.12 + 0.24 + 0.07 + 0.10 = 0.53
```

**Step 3:** High similarity → High attention weight

### 🔹 Real Example with Pronouns

```
Sentence: "John gave Mary a book. She thanked him."

Word "She":
- Attention to "John": 0.1 (male, unlikely)
- Attention to "Mary": 0.8 (female, recent subject)
- Attention to "book": 0.05 (inanimate)
- Attention to "gave": 0.05 (verb)

Result: "She" representation includes 80% of "Mary" info
```

### 🔹 Self-Attention vs Cross-Attention

**Self-Attention:** Look within same sequence

```
English sentence looking at itself:
"The" → looks at → ["The", "cat", "sat", ...]
```

**Cross-Attention:** Look at different sequence

```
French translation looking at English source:
"Le" → looks at → ["The", "cat", "sat", ...]
```

### ✅ Quick Check:

How does self-attention help with long-range dependencies?

---

## 3. Query-Key-Value (Q-K-V): The Attention Mechanism

### 🔹 The Three Components

Think of a library search system:

**Query (Q):** What you're looking for

```
"I need books about machine learning"
```

**Key (K):** Labels on each shelf/book

```
Shelf 1: "Mathematics textbooks"
Shelf 2: "Machine Learning" ← MATCH!
Shelf 3: "History books"
```

**Value (V):** The actual content

```
Shelf 2 contains: [Book1, Book2, Book3] ← RETRIEVE
```

### 🔹 How Q-K-V Work Together

**For word "learning" in sentence "machine learning":**

```
Step 1: Create Query
Q_learning = W_q × embedding_learning
"What context do I need?"

Step 2: Create Keys for all words
K_machine = W_k × embedding_machine
K_learning = W_k × embedding_learning
"What information do I have?"

Step 3: Compare Query with Keys
score_1 = Q_learning · K_machine  = 0.9 (HIGH - related)
score_2 = Q_learning · K_learning = 0.6 (MEDIUM - self)

Step 4: Normalize scores (softmax)
weight_1 = 0.7  (focus 70% on "machine")
weight_2 = 0.3  (focus 30% on itself)

Step 5: Get Values
V_machine = W_v × embedding_machine
V_learning = W_v × embedding_learning

Step 6: Weighted sum
output_learning = 0.7 × V_machine + 0.3 × V_learning
```

### 🔹 Why Separate K and V?

**Key:** Used for **matching** (finding relevance)
**Value:** Used for **retrieving** (getting information)

**Example:**

```
Key might encode: "This is a noun related to animals"
Value might encode: "Detailed semantic meaning of 'cat'"

Separating them gives flexibility:
- Match on grammatical features (Key)
- Retrieve semantic content (Value)
```

### 🔹 Mathematical Flow

```
1. Q = X × W_q    # (seq_len, d_k)
2. K = X × W_k    # (seq_len, d_k)
3. V = X × W_v    # (seq_len, d_v)

4. scores = Q × K^T / sqrt(d_k)  # (seq_len, seq_len)
5. weights = softmax(scores)      # (seq_len, seq_len)
6. output = weights × V           # (seq_len, d_v)
```

### ✅ Quick Check:

What role does the Key play in attention?

---

## 4. Complete Attention Flow: Step-by-Step

### 🔹 Step-by-Step Breakdown

**Input:** Sentence "I love AI"

```
Step 1: Word Embeddings
─────────────────────
"I"    → [0.1, 0.5, 0.2, 0.8]
"love" → [0.6, 0.2, 0.9, 0.1]
"AI"   → [0.3, 0.7, 0.4, 0.6]

Why: Convert words to numbers
Result: X = 3×4 matrix
```

```
Step 2: Create Q, K, V
─────────────────────
Q = X × W_q  # Query: "What do I need?"
K = X × W_k  # Key: "What do I offer?"
V = X × W_v  # Value: "Here's my info"

Why: Three different perspectives
Result: Each word has Q, K, V vectors
```

```
Step 3: Calculate Attention Scores
──────────────────────────────────
For "love" (row 2):
score["love" looking at "I"]    = Q_love · K_I
score["love" looking at "love"] = Q_love · K_love
score["love" looking at "AI"]   = Q_love · K_AI

Why: Measure relevance between words
Result: Score matrix (3×3)
```

```
Step 4: Scale Scores
───────────────────
scores = scores / sqrt(d_k)

Why: Prevent very large values that make softmax too sharp
Result: Stabilized scores
```

```
Step 5: Softmax Normalization
─────────────────────────────
weights = softmax(scores, axis=1)

Example for "love":
[0.2, 0.3, 0.5] → sums to 1.0

Why: Convert scores to probabilities
Result: Attention weights (interpretable)
```

```
Step 6: Weighted Sum of Values
──────────────────────────────
For "love":
output_love = 0.2×V_I + 0.3×V_love + 0.5×V_AI

Why: Gather information from relevant words
Result: New representation of "love" with context
```

### 🔹 Why It's Fully Parallel

**RNN (Sequential):**

```
Step 1: Process "I"    → h1  [Must wait]
Step 2: Process "love" → h2  [Must wait]
Step 3: Process "AI"   → h3  [Must wait]
```

**Attention (Parallel):**

```
All words processed simultaneously:
["I", "love", "AI"] → All Q, K, V computed at once
                   → All attention scores at once
                   → All outputs at once
```

**GPU loves this!** Matrix operations run in parallel → 10-100× faster

### ✅ Quick Check:

Why is attention faster than RNN?

---

## 5. Multi-Head Attention: Multiple Perspectives

### 🔹 What Is a "Head"?

One independent attention mechanism with smaller dimensions.

**Single Head:**

```
Full dimension: 512
One attention computation
```

**Multi-Head (8 heads):**

```
Each head: 512 / 8 = 64 dimensions
8 independent attention computations
Concatenate results
```

### 🔹 Why Multiple Heads?

**Different heads learn different relationships:**

```
Sentence: "The quick brown fox jumps"

Head 1: Focuses on grammar
- "The" → "fox" (determiner-noun)
- "quick" → "fox" (adjective-noun)

Head 2: Focuses on actions
- "fox" → "jumps" (subject-verb)

Head 3: Focuses on descriptions
- "brown" → "fox" (attribute-entity)

Head 4: Focuses on long-range
- "The" → "jumps" (sentence structure)
```

### 🔹 The Multiple Camera Analogy

**Single head = One camera angle**

```
You see the scene from one perspective
Miss details from other angles
```

**Multi-head = Multiple cameras**

```
Camera 1: Close-up on faces
Camera 2: Wide shot for context
Camera 3: Bird's eye view
Camera 4: Detail shots

Director combines all angles → Full understanding
```

### 🔹 Implementation

```python
# Multi-Head Attention
num_heads = 8
d_model = 512
d_k = d_model // num_heads = 64

For each head h:
    Q_h = X × W_q_h  # (seq_len, 64)
    K_h = X × W_k_h  # (seq_len, 64)
    V_h = X × W_v_h  # (seq_len, 64)

    head_h = Attention(Q_h, K_h, V_h)

# Concatenate all heads
output = Concat(head_1, head_2, ..., head_8)  # (seq_len, 512)

# Final linear projection
output = output × W_o
```

### ✅ Quick Check:

Why use 8 small heads instead of 1 large head?

---

## 6. Transformers: The Complete Architecture

### 🔹 How Attention Fits Into Transformers

**Transformer = Encoder + Decoder**

**Encoder Stack:**

```
Input Embedding + Positional Encoding
    ↓
Multi-Head Self-Attention
    ↓
Add & Normalize
    ↓
Feed-Forward Network
    ↓
Add & Normalize
    ↓
(Repeat 6-12 times)
```

**Decoder Stack:**

```
Output Embedding + Positional Encoding
    ↓
Masked Multi-Head Self-Attention  (can't see future)
    ↓
Add & Normalize
    ↓
Multi-Head Cross-Attention  (look at encoder)
    ↓
Add & Normalize
    ↓
Feed-Forward Network
    ↓
Add & Normalize
    ↓
(Repeat 6-12 times)
```

### 🔹 Why No Recurrence?

**RNN:** Must process sequentially

```
word1 → h1 → word2 → h2 → word3 → h3
[Can't parallelize]
```

**Transformer:** Process all at once

```
[word1, word2, word3] → All attention simultaneously
[Fully parallelizable]
```

### 🔹 Positional Encoding

**Problem:** Attention has no sense of order!

```
"Dog bites man" vs "Man bites dog"
Without position info → same attention weights!
```

**Solution:** Add position information

```
embedding = word_embedding + position_embedding

Position 0: [0.0, 1.0, 0.0, 1.0, ...]
Position 1: [0.84, 0.54, 0.91, 0.41, ...]
Position 2: [0.91, -0.42, 0.99, -0.99, ...]
```

**Using sine/cosine:**

```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 🔹 Encoder Data Flow

```
Input: "I love machine learning"
    ↓
Tokenize: [101, 3142, 2599, 5067, 102]
    ↓
Embedding: Each token → 512-dim vector
    ↓
+ Positional Encoding
    ↓
Layer 1: Self-Attention (words look at each other)
    ↓
Layer 2: Feed-Forward (process each position)
    ↓
... (repeat 6 times)
    ↓
Final encoder output: Contextualized representations
```

### 🔹 Decoder Data Flow

```
Input: "<start> J'aime l'apprentissage"
    ↓
Masked Self-Attention:
- "J'aime" can see: "<start>", "J'aime"
- "l'apprentissage" can see: "<start>", "J'aime", "l'apprentissage"
[Prevents looking at future words during training]
    ↓
Cross-Attention:
- French word looks at ALL English words
- "J'aime" attends to "love"
- "l'apprentissage" attends to "machine learning"
    ↓
Feed-Forward
    ↓
Output: Next word probabilities
```

### 🔹 Why Transformers Won

**vs RNN/LSTM:**

- ✅ 10-100× faster (parallelization)
- ✅ Better long-range dependencies (direct attention)
- ✅ Easier to scale to billions of parameters

**vs CNN:**

- ✅ Better for sequences (not fixed receptive field)
- ✅ Captures global context (not just local)

### ✅ Quick Check:

Why does the decoder use masked self-attention?

---

## 7. Practical Code: Building Attention

### 7.1 Simple Attention (NumPy)

```python
import numpy as np

# Input: 3 words, embedding size = 4
X = np.array([
    [1.0, 0.0, 1.0, 0.0],  # word 1
    [0.0, 2.0, 0.0, 0.5],  # word 2
    [1.0, 1.0, 0.0, 1.0],  # word 3
])  # shape (3, 4)

# Random projection matrices (normally learned)
W_q = np.random.randn(4, 4) * 0.1
W_k = np.random.randn(4, 4) * 0.1
W_v = np.random.randn(4, 4) * 0.1

# Step 1: Create Q, K, V
Q = X.dot(W_q)  # (3, 4)
K = X.dot(W_k)  # (3, 4)
V = X.dot(W_v)  # (3, 4)

print("Query matrix shape:", Q.shape)
print("Key matrix shape:", K.shape)
print("Value matrix shape:", V.shape)

# Step 2: Calculate attention scores
scores = Q.dot(K.T)  # (3, 3) - each word to every word
print("\nAttention scores:")
print(scores)

# Step 3: Scale (for stability)
d_k = K.shape[1]
scores = scores / np.sqrt(d_k)

# Step 4: Softmax to get weights
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

weights = softmax(scores)
print("\nAttention weights (after softmax):")
print(weights)
print("Row sums:", weights.sum(axis=1))  # Should be [1, 1, 1]

# Step 5: Weighted sum of values
output = weights.dot(V)
print("\nOutput (contextualized representations):")
print(output)

# Interpretation
print("\n=== Interpretation ===")
for i in range(3):
    print(f"\nWord {i+1} attention distribution:")
    for j in range(3):
        print(f"  Attention to word {j+1}: {weights[i,j]:.3f}")
```

### 7.2 PyTorch Self-Attention

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, X):
        """
        X: (batch_size, seq_len, d_model)
        Returns: (batch_size, seq_len, d_model)
        """
        # Step 1: Create Q, K, V
        Q = self.W_q(X)  # (batch, seq_len, d_model)
        K = self.W_k(X)
        V = self.W_v(X)

        # Step 2: Calculate attention scores
        # Q × K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)

        # Step 3: Scale
        scores = scores / (self.d_model ** 0.5)

        # Step 4: Softmax
        weights = F.softmax(scores, dim=-1)

        # Step 5: Apply attention to values
        output = torch.matmul(weights, V)  # (batch, seq_len, d_model)

        return output, weights

# Example usage
d_model = 64
seq_len = 5
batch_size = 2

# Random input
X = torch.randn(batch_size, seq_len, d_model)

# Create attention layer
attention = SelfAttention(d_model)

# Forward pass
output, weights = attention(X)

print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"\nAttention weights for first sample:")
print(weights[0].detach().numpy())
```

### 7.3 Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """
        Split last dimension into (num_heads, d_k)
        (batch, seq_len, d_model) → (batch, num_heads, seq_len, d_k)
        """
        x = x.view(batch_size, -1, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def forward(self, X):
        batch_size = X.size(0)

        # Linear projections
        Q = self.W_q(X)  # (batch, seq_len, d_model)
        K = self.W_k(X)
        V = self.W_v(X)

        # Split into multiple heads
        Q = self.split_heads(Q, batch_size)  # (batch, num_heads, seq_len, d_k)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(weights, V)

        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(attention_output)

        return output, weights

# Example
d_model = 512
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

X = torch.randn(2, 10, d_model)  # (batch=2, seq_len=10, d_model=512)
output, weights = mha(X)

print(f"Input: {X.shape}")
print(f"Output: {output.shape}")
print(f"Weights: {weights.shape}")  # (batch, num_heads, seq_len, seq_len)
```

### ✅ Quick Check:

What does the attention weight matrix represent?

---

## 8. Real-World Applications

### 🔹 Machine Translation

**Task:** English → French

```
Input: "The cat sits on the mat"
Encoder: Processes all English words with self-attention
Decoder: Generates French word-by-word
  - "Le" (looks at "The")
  - "chat" (looks at "cat")
  - "s'assied" (looks at "sits")
  - "sur" (looks at "on")
  - "le tapis" (looks at "the mat")
```

**Why Attention Helps:**

- Direct connection to relevant source words
- No information bottleneck
- Handles different word orders

### 🔹 Text Generation (GPT)

**Task:** Complete "Once upon a time"

```
Model uses self-attention to:
1. Look at all previous words
2. Predict next word based on full context
3. Repeat

"Once upon a time" → "there"
"Once upon a time there" → "was"
"Once upon a time there was" → "a"
...
```

### 🔹 Question Answering

**Task:** Answer based on document

```
Document: "Paris is the capital of France. It has 2.1M people."
Question: "What is the capital of France?"

Cross-Attention:
- "capital" attends to "Paris" (HIGH)
- "France" attends to "France" (HIGH)
Answer: "Paris"
```

### 🔹 Sentiment Analysis

```
Review: "The movie was great but the ending was disappointing"

Self-Attention learns:
- "great" → positive context
- "disappointing" → negative context
- "but" → contrast signal

Output: Mixed sentiment (3/5 stars)
```

### 🔹 Code Generation (Copilot)

```
Comment: "Function to sort list in descending order"
Model uses attention to:
- Understand "sort", "descending"
- Generate appropriate Python code
- Maintain syntax correctness

Output:
def sort_desc(lst):
    return sorted(lst, reverse=True)
```

### ✅ Quick Check:

Why is Transformer better than RNN for translation?

---

## 9. Summary: The Attention Revolution

### 🔹 What You Now Know

✅ **Understand** why attention solves RNN's limitations
✅ **Explain** Query-Key-Value mechanism
✅ **Calculate** attention weights step-by-step
✅ **Implement** self-attention from scratch
✅ **Recognize** multi-head attention benefits
✅ **Apply** attention to real-world NLP tasks
✅ **Understand** Transformer architecture

### 🔹 Key Concepts Recap

**Attention Formula:**

```
Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
```

**Q-K-V Roles:**

- **Query:** What I'm looking for
- **Key:** What I can offer
- **Value:** The actual information

**Why It Works:**

- Direct connections (no fading)
- Parallel computation (fast)
- Flexible context (task-specific)

### 🔹 Attention vs Previous Methods

| Method        | Dependency | Parallelization | Long-Range |
| ------------- | ---------- | --------------- | ---------- |
| **RNN**       | Sequential | ❌ No           | ❌ Poor    |
| **LSTM**      | Sequential | ❌ No           | ⚠️ Better  |
| **CNN**       | Local      | ✅ Yes          | ⚠️ Limited |
| **Attention** | Global     | ✅ Yes          | ✅ Perfect |

### 🔹 When to Use Attention

**✅ Use Attention when:**

- Need long-range dependencies
- Have sufficient training data
- Need interpretable model (can visualize attention)
- Speed is important (parallel processing)

**❌ Consider alternatives when:**

- Very small dataset (attention needs data)
- Simple sequential patterns (RNN might suffice)
- Extremely long sequences (use sparse attention variants)

### 🔹 The Transformer Family Tree

```
2017: Transformer (original)
    ↓
2018: BERT (encoder-only)
    ↓
2018: GPT (decoder-only)
    ↓
2019: GPT-2, RoBERTa, ALBERT
    ↓
2020: GPT-3 (175B parameters)
    ↓
2022: ChatGPT (GPT-3.5)
    ↓
2023: GPT-4
```

---

_Ready to pay attention and transform everything! 🚀_
