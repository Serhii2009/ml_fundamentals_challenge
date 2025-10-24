# 🎯 DAY 20: PRACTICE — BUILDING A NEURAL NETWORK FROM SCRATCH (MNIST)

Today is **100% practical**. No long theory lessons — just real code, real neural networks, and real results.

Instead of repeating what's already been done, I'm taking you directly to a **complete, working project** that I've built and shared:

**👉 A fully functional Neural Network for handwritten digit recognition (MNIST dataset)**  
**👉 Built entirely with NumPy — no TensorFlow, no PyTorch, no Keras**  
**👉 Interactive GUI for testing in real-time**

This project demonstrates everything we've been learning:

- Forward propagation
- Backpropagation
- Gradient descent optimization
- Weight initialization
- Training loops
- Model evaluation
- Real-world application

---

## 🚀 Your Mission Today

Your goal is simple but powerful:

### 1️⃣ **Explore the Project**

Visit the repositories below and study how a neural network is built from mathematical fundamentals:

🔗 **GitHub Repository:**  
[https://github.com/Serhii2009/mnist-neural-network](https://github.com/Serhii2009/mnist-neural-network)

🔗 **Kaggle Notebook:**  
[https://www.kaggle.com/code/serhiikravchenko2009/mnist-neural-network](https://www.kaggle.com/code/serhiikravchenko2009/mnist-neural-network)

### 2️⃣ **Clone and Run It Locally**

```bash
git clone https://github.com/Serhii2009/mnist-neural-network.git
cd mnist-neural-network
python main.py
```

### 3️⃣ **Experiment**

This is where real learning happens. Try modifying:

**Architecture:**

- Change hidden layer sizes (try `[784, 32, 32, 10]` or `[784, 256, 128, 10]`)
- Add more layers
- Remove a layer

**Hyperparameters:**

- Learning rate: `0.1`, `0.25`, `0.5`
- Batch size: `16`, `32`, `64`
- Epochs: `20`, `50`, `100`

**Activation functions:**

- Experiment with ReLU instead of sigmoid
- Try different initialization strategies

**Training:**

- Run `python main.py --train` to train from scratch
- Compare training times and final accuracy

### 4️⃣ **Test the GUI**

The project includes an interactive drawing interface:

```bash
python main.py
```

Draw digits with your mouse and watch the neural network predict in real-time!

### 5️⃣ **Support the Project**

If you find this helpful:

- ⭐ **Star the GitHub repo** (top right corner)
- 📤 **Fork it** and build your own version
- 💬 **Share feedback** — what worked? What confused you?

---

## 🧠 What You'll Learn

By exploring and experimenting with this project, you'll understand:

### **Neural Network Fundamentals:**

- How layers connect mathematically
- Matrix operations in forward propagation
- Gradient calculation in backpropagation
- Weight updates via optimization

### **Training Process:**

- Data preprocessing (normalization, one-hot encoding)
- Mini-batch gradient descent
- Loss tracking across epochs
- Model evaluation

### **Practical Implementation:**

- Pure NumPy implementation (no frameworks)
- Clean, readable code structure
- Real-time inference
- GUI integration with Tkinter

### **Debugging & Optimization:**

- What happens when learning rate is wrong
- How batch size affects convergence
- Why deeper networks aren't always better
- Performance trade-offs

---

## 📺 Watch the Walkthrough

I've created a video explaining the project architecture, code structure, and results:

🎥 **[[My LinkedIn video](https://www.linkedin.com/posts/serhii-kravchenko-b941272a6_ai-ml-machinelearning-activity-7363910272433987584-UG62)]**

In this video, I cover:

- Project structure and file organization
- Neural network architecture design
- Training process and hyperparameter choices

---

## 💡 Why This Approach?

Instead of creating yet another tutorial from scratch, I'm showing you **production-quality code** that:

✅ Actually works in the real world  
✅ Includes proper error handling  
✅ Has clean documentation  
✅ Demonstrates best practices  
✅ Includes interactive visualization

This is the kind of project that **employers notice** and **demonstrates real understanding**.

---

## 📝 Recommended Study Path

Follow this sequence to maximize learning:

### **Step 1: Read the README**

Start with the [GitHub README](https://github.com/Serhii2009/mnist-neural-network/blob/main/README.md) — it explains the architecture, hyperparameters, and structure.

### **Step 2: Examine `network.py`**

This file contains the core neural network implementation:

- `__init__`: Weight initialization
- `forward`: Forward propagation logic
- `backprop`: Backpropagation algorithm
- `update_mini_batch`: SGD with mini-batches

### **Step 3: Study `main.py`**

See how everything connects:

- Data loading from MNIST files
- Training loop
- Testing/evaluation
- GUI integration

### **Step 4: Run Experiments**

Clone the repo and modify hyperparameters:

```python
# In main.py, modify these lines:
nn = NeuralNetwork([784, 128, 64, 10])  # Architecture
nn.train(training_data,
         epochs=30,           # Try different values
         mini_batch_size=32,  # Experiment here
         eta=0.25)           # Learning rate
```

### **Step 5: Visualize Results**

Train your modified network and compare:

- Training time
- Final accuracy
- Loss curves
- Real-world testing with GUI

---

## 🎓 Challenge Yourself

Once you're comfortable with the basic project, try these extensions:

### **Beginner:**

- Modify learning rate and observe convergence speed
- Change network size (more/fewer neurons)
- Test different batch sizes

### **Intermediate:**

- Implement ReLU activation function
- Add learning rate decay
- Track and plot loss curves

### **Advanced:**

- Implement momentum optimization
- Add dropout regularization
- Try different weight initialization strategies
- Extend to other datasets (Fashion-MNIST)

---

## 🏆 What Success Looks Like

By the end of today, you should:

✅ Understand how neural networks work **without frameworks**  
✅ Be able to read and modify neural network code  
✅ Know how hyperparameters affect training  
✅ Have hands-on experience with MNIST classification  
✅ Feel confident explaining backpropagation to someone else

---

## 🤝 Share Your Progress

After exploring the project:

1. **Take screenshots** of your experiments
2. **Note what worked** and what didn't
3. **Share on LinkedIn and Tag me** so I can see your progress!

---

## 📚 Additional Resources

### **Project Links:**

- [GitHub Repository](https://github.com/Serhii2009/mnist-neural-network) — Full source code
- [Kaggle Notebook](https://www.kaggle.com/code/serhiikravchenko2009/mnist-neural-network) — Run in browser with GPU

### **Learning Resources:**

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) — Original dataset source
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — Michael Nielsen's free book
- [3Blue1Brown Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) — Visual explanations

---

## ✨ Final Thoughts

This isn't just another coding exercise — it's a **real, working neural network** that you can:

- Run on your machine
- Modify and experiment with
- Show to employers
- Build upon for your own projects

The best way to learn neural networks is to **build them**, **break them**, and **fix them**. This project gives you the perfect playground.

---

## 📬 Questions or Feedback?

If you:

- Have questions about the code
- Found interesting results from experiments
- Want to discuss neural network concepts
- Built something cool on top of this

**Reach out!** I'm here to help and learn together.

---

**Ready to dive in? Let's build neural networks! 🚀**

_Remember: The best learning happens when you get your hands dirty with code._
