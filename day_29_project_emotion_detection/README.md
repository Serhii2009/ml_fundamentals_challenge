# 🎭 DAY 29: PRACTICE — REAL-TIME EMOTION DETECTION WITH CNN 🎯

Today is **100% practical**. No lengthy theory — just real code, real neural networks, and real emotion recognition in action.

Instead of building from scratch what's already been perfected, I'm taking you directly to a **complete, production-ready project** that I've built and deployed:

**👉 A full-stack emotion detection system with CNN**  
**👉 Real-time facial emotion recognition through webcam**  
**👉 Interactive web interface with React + Flask backend**

This project demonstrates everything we've learned about CNNs:

- Convolutional layers and feature extraction
- Batch normalization and dropout regularization
- Spatial attention mechanisms
- Model training with data augmentation
- Real-world deployment with REST API
- Frontend integration with live predictions

---

## 🚀 Your Mission Today

Your goal is powerful and practical:

### 1️⃣ **Explore the Project**

Visit the repositories below and study how a production-ready emotion detection system is built:

🔗 **GitHub Repository:**  
[https://github.com/Serhii2009/emotion-detection](https://github.com/Serhii2009/emotion-detection)

🔗 **Kaggle Notebook:**  
[https://www.kaggle.com/code/serhiikravchenko2009/facial-emotion-detection](https://www.kaggle.com/code/serhiikravchenko2009/facial-emotion-detection)

### 2️⃣ **Clone and Run It Locally**

```bash
# Clone repository
git clone https://github.com/Serhii2009/emotion-detection.git
cd emotion-detection

# Setup backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py

# Setup frontend (new terminal)
cd frontend
npm install
npm run dev
```

### 3️⃣ **Experiment with the Live System**

Open `http://localhost:5173` in your browser:

**Test the real-time detection:**

- Allow webcam access when prompted
- Watch as the model detects your emotions every 1.5 seconds
- Try different facial expressions: smile, frown, surprise
- Observe confidence scores for predictions

**Understanding what you see:**

- **Emoji overlay:** Visual representation of detected emotion
- **Confidence:** How certain the model is (0-100%)
- **Probabilities:** Distribution across all 7 emotion classes

### 4️⃣ **Experiment with Architecture**

The power of this project is in experimentation:

**Architecture modifications:**

```python
# In the Kaggle notebook, try:

# Deeper network - add more convolutional blocks
# Original: Block 1 → Block 2 → Block 3 → Block 4
# Try: Add Block 5 with 1024 filters

# Different filter sizes
# Original: All 3×3 filters
# Try: Mix 3×3 and 5×5 filters

# Attention mechanisms
# Original: Spatial attention in Block 3
# Try: Add attention to all blocks
```

**Training hyperparameters:**

```python
# Modify in the training code:

BATCH_SIZE = 64  # Try: 16, 32, 64
LEARNING_RATE = 1e-3  # Try: 1e-4, 3e-4, 1e-3
EPOCHS = 100  # Try: 50, 150

# Data augmentation
rotation_range = 25  # Try: 15, 30, 45
zoom_range = 0.2  # Try: 0.1, 0.3
```

**Regularization techniques:**

```python
# Experiment with dropout rates
SpatialDropout2D(0.1)  # Try: 0.0, 0.2, 0.3
Dropout(0.3)  # Try: 0.2, 0.5

# Try removing/adding BatchNormalization
# Compare training speed and accuracy
```

### 5️⃣ **Support the Project**

If you find this helpful:

- ⭐ **Star the GitHub repo** (top right corner)
- 🍴 **Fork it** and build your own version
- 💬 **Share results** — what accuracy did you achieve?

---

## 🧠 What You'll Learn

By exploring and experimenting with this project, you'll understand:

### **CNN Architecture for Images:**

- How convolutional filters detect edges, textures, and patterns
- Batch normalization for stable training
- Spatial attention to focus on important regions (eyes, mouth)
- Global pooling to convert feature maps to predictions

### **Training Process:**

- Data augmentation (rotation, shift, flip, zoom)
- Focal Loss for handling class imbalance
- Learning rate scheduling (ReduceLROnPlateau)
- Early stopping to prevent overfitting

### **Full-Stack Integration:**

- Flask REST API for model inference
- React frontend with webcam access
- Real-time image processing with OpenCV
- Base64 encoding for image transfer

### **Debugging & Optimization:**

- What happens when learning rate is too high/low
- How batch size affects training stability
- Why deeper networks aren't always better
- Performance optimization for real-time inference (50-100ms)

---

## 📺 Watch It in Action

The system detects 7 emotions in real-time:

- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😄 **Happy**
- 😐 **Neutral**
- 😢 **Sad**
- 😮 **Surprise**

**Performance metrics:**

- Test accuracy: 68.43%
- Macro F1 Score: 0.6575
- Inference time: 50-100ms per image
- Training time: 30-40 minutes on Kaggle T4 GPU

---

## 💡 Why This Approach?

Instead of creating another tutorial from scratch, I'm showing you **production-quality code** that:

✅ Works in the real world with live webcam  
✅ Includes proper error handling and CORS  
✅ Has clean documentation  
✅ Demonstrates industry best practices  
✅ Includes interactive visualization

This is the kind of project that **employers notice** and **demonstrates real understanding** of deep learning deployment.

---

## 📝 Recommended Study Path

Follow this sequence to maximize learning:

### **Step 1: Read the GitHub README**

Start with the [GitHub README](https://github.com/Serhii2009/emotion-detection/blob/main/README.md) — it explains:

- Tech stack (React + Flask + TensorFlow)
- Installation instructions
- API endpoints
- Model architecture overview

### **Step 2: Examine the Kaggle Notebook**

This contains the complete CNN training pipeline:

**Data preparation:**

```python
# FER2013 dataset: 35,887 images (48×48 grayscale)
# Split: 80% train, 10% val, 10% test
# 7 emotion classes
```

**Model architecture:**

```python
# 4 convolutional blocks with increasing filters: 64→128→256→512
# Batch normalization after each conv layer
# Spatial attention in Block 3
# Global pooling (average + max)
# Dense layers: 512→256→7
```

**Training configuration:**

```python
# Focal Loss for class imbalance
# Adam optimizer (lr=3e-4)
# ReduceLROnPlateau scheduler
# Early stopping (patience=20)
# Data augmentation
```

### **Step 3: Study `backend/app.py`**

See how the model integrates with Flask:

```python
# Load trained model
model = load_model('best_emotion_model.h5')

# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    # Receive base64 image
    # Decode and preprocess
    # Face detection with Haar Cascade
    # Model prediction
    # Return emotion + confidence
```

### **Step 4: Study `frontend/src/App.jsx`**

See how React handles webcam and predictions:

```javascript
// Capture image every 1.5 seconds
// Convert to base64
// Send to Flask API
// Display emotion with animated emoji
// Show confidence scores
```

### **Step 5: Run Your Own Experiments**

Clone the repo and modify:

```python
# In Kaggle notebook:

# Experiment 1: Add more blocks
x = create_conv_block(x, 1024, "Block5")  # New block

# Experiment 2: Remove attention
# Comment out the attention mechanism
# Compare accuracy

# Experiment 3: Change augmentation
rotation_range=45,  # More aggressive
horizontal_flip=False,  # No flipping

# Experiment 4: Different architecture
# Try VGG-style: smaller filters, deeper network
# Try ResNet-style: add skip connections
```

---

## 🎓 Challenge Yourself

Once you're comfortable with the basic project, try these extensions:

### **Beginner:**

- Modify `CAPTURE_INTERVAL` in frontend (1s, 2s, 3s)
- Change batch size and observe training speed
- Test different learning rates (1e-4, 5e-4, 1e-3)
- Add new emotions to the emoji mapping

### **Intermediate:**

- Implement skip connections (ResNet-style)
- Add attention to all convolutional blocks
- Experiment with different pooling strategies
- Track and plot loss curves in real-time
- Add confusion matrix visualization to frontend

### **Advanced:**

- Extend to video emotion tracking (emotion over time)
- Add multi-face detection (detect all faces in frame)
- Implement ensemble predictions (multiple models)
- Fine-tune on custom dataset (your own labeled images)
- Deploy to cloud (AWS, GCP, Heroku)
- Add emotion intensity estimation (not just category)

---

## 🏆 What Success Looks Like

By the end of today, you should:

✅ Understand CNN architecture for image classification  
✅ Know how attention mechanisms improve focus  
✅ Be able to train models with class imbalance (Focal Loss)  
✅ Understand full-stack ML deployment (backend + frontend)  
✅ Have hands-on experience with real-time inference  
✅ Feel confident deploying your own CNN projects

---

## 🧠 Deep Dive: Understanding the CNN Architecture

Let's break down exactly how this emotion detection CNN works, layer by layer.

### **Overview: The Data Flow**

We have a dataset of facial images showing different emotions. Each image is 48×48 pixels in grayscale (not RGB, just one channel representing shades from black to white). Our input is a matrix of 48×48×1, where 1 means one channel (grayscale).

The entire dataset is divided into **batches**. A batch is a group of images processed simultaneously through the network. With a batch size of 32, the model processes 32 images in parallel each time. There are 763 batches in the training set, meaning the entire training dataset is divided into 763 groups of 32 images (the last batch may be slightly smaller).

When we process all 763 batches once, that's **one epoch**. An epoch is a complete pass through the entire dataset. Training for 100 epochs means the model sees the entire dataset 100 times, with weights gradually improving each time.

### **GPU Parallelism**

You correctly understand that GPUs enable parallel processing. Without a GPU, this would be extremely slow because the CPU would process images sequentially. A GPU has thousands of small computational cores that can simultaneously multiply matrices, apply filters, and calculate activations — all in parallel for all 32 images in a batch.

When we say a batch of 32 images passes through a convolutional layer, it means the GPU simultaneously applies all filters to all 32 images. This is very fast. If a CPU without parallelism did this, each image would be processed sequentially, taking hours.

### **Block 1: Initial Feature Extraction**

**First Convolutional Layer (conv2d):**

- 64 filters, each 3×3
- Each filter is a small 3×3 matrix with weights
- Initially, weights are random (using 'he_normal' initialization for good starting values)

**What does a filter do?**
It takes a 3×3 square on the image, multiplies each pixel by the corresponding weight, sums all products, and adds a bias. The result is a single number showing how "similar" this square is to the pattern the filter is searching for.

The filter slides across the entire image from left to right, top to bottom, with stride=1. Because we use `padding='same'`, the image size doesn't change. The output is a feature map of size 48×48. Since there are 64 filters, we get 64 feature maps, each 48×48.

**Key insight:** Each pixel on a feature map contains information about 9 pixels from the original image (the 3×3 square). Different filters create different feature maps because they have different weights. One filter might detect vertical lines, another horizontal lines, a third diagonals, and so on.

**Batch Normalization:**
After the convolutional layer, we have 64 feature maps, and pixel values can vary greatly — some very large, some very small. Batch normalization brings all these values to a more uniform scale.

**How it works:** For each feature map, it calculates the mean and standard deviation across the entire batch (all 32 images). Then each pixel is normalized: subtract the mean and divide by standard deviation. After that, values are scaled and shifted using two learned parameters (gamma and beta), so the model can decide what scale it needs.

**Why?** This speeds up training and makes it more stable. Without batch normalization, weights can "run away" — some become very large, others very small, and gradients become unstable. With batch normalization, everything is more controlled.

**Activation (ReLU):**
After batch normalization, we apply the ReLU (Rectified Linear Unit) activation function. It's very simple: all negative values are replaced with zeros, positive values remain as they are.

**Why?** First, negative values aren't very useful here — they might mean the filter "didn't find" its pattern in this location. Zeros mean "nothing here," and positive values mean "something important here." Second, ReLU introduces non-linearity. Without activation, the entire network would be linear, and it couldn't learn complex non-linear dependencies (like a smile being when mouth corners go up and eyes narrow).

After the first activation, we have 64 feature maps of size 48×48. Each pixel on these feature maps contains information about a 3×3 square from the original image.

**Second Convolutional Layer (conv2d_1):**
Now these 64 feature maps go into the next convolutional layer. Here again we have 64 filters of 3×3, but they now work with feature maps, not the original image. Each pixel on a feature map already contains information about 9 pixels from the original image. Now the filter takes a 3×3 square of such "generalized" pixels and processes them. This means it's effectively analyzing a larger zone of the original image.

**Key concept:** With each subsequent convolutional layer, we "see" more and more of the original image. If the first layer saw 3×3 squares, the second already sees something like 5×5 or 7×7 squares (exact calculation is complex, but the idea is that the receptive field expands).

After the second convolutional layer, followed by batch normalization and ReLU, we again have 64 feature maps of size 48×48. Now each pixel on these feature maps contains information about a larger zone of the original image — roughly 81 pixels (3×3×9).

**MaxPooling:**
After two convolutional layers, we perform MaxPooling. This reduces the size of feature maps. We take a 2×2 square on the feature map and select the maximum value from it. This maximum becomes one pixel of the new, smaller feature map.

For example, if it was 48×48, after MaxPooling 2×2 it becomes 24×24. The size is halved.

**Why?** First, it reduces computations — fewer pixels, faster model. Second, it helps generalize information. We select the brightest (largest) value from each 2×2 square, preserving the most important information. Third, it reduces overfitting — the model can't "memorize" every pixel too precisely, it's forced to generalize.

**SpatialDropout:**
After MaxPooling, we apply SpatialDropout. This is dropout, but not for individual neurons (as in regular dropout), but for entire feature maps.

During training, SpatialDropout randomly turns off some feature maps completely. For example, out of 64 feature maps, it might turn off 6-7 (depending on rate=0.1, i.e., 10%). This means these feature maps become zeros and don't participate in calculations for this batch.

**Why?** This protects against overfitting. If the model relies too much on specific feature maps, it might "memorize" the training data. By randomly turning off some feature maps, the model is forced to learn to make conclusions based on different combinations of feature maps, making it more robust.

SpatialDropout turns off entire feature maps, not individual pixels. This is important for convolutional layers because pixels on one feature map strongly correlate (they were all obtained using one filter), so it's better to turn off the entire feature map at once.

### **Block 2: Deeper Feature Learning**

After Block 1, we have 64 feature maps of size 24×24. They go into Block 2.

**Why increase filters to 128?** Because the deeper into the network, the more complex patterns it can detect. Early layers detect simple things (lines, corners), and deeper layers detect more complex combinations (like parts of a face — eyes, nose, mouth).

Now we have 128 filters of 3×3. Each filter works with all 64 input feature maps. So one filter has 3×3×64 weights (plus bias). After this layer, we have 128 feature maps of size 24×24.

Again batch normalization, activation, then another convolutional layer with 128 filters, again batch normalization, activation. After this, MaxPooling — size reduces from 24×24 to 12×12. And SpatialDropout.

Now each pixel on these feature maps contains information about a very large zone of the original image. If before it was a 3×3 or 9×9 square, now it's something like a 20×20 square or more. This means the model can already "see" not just lines, but parts of a face — like a mouth corner or eye contour.

### **Block 3: Spatial Attention Mechanism**

After Block 2, we have 128 feature maps of size 12×12. They go into Block 3, where the number of filters increases to 256.

**Two convolutional layers:**
With 256 filters, batch normalization, activation after each. After this, we have 256 feature maps of size 12×12.

**Spatial Attention:**
Here's an interesting detail — Spatial Attention. After the second convolutional layer of Block 3, we apply one more convolutional layer with one filter of size 1×1. This filter creates one "attention map" of size 12×12. Then we apply sigmoid activation to this map, which converts all values to the range from 0 to 1.

This attention map shows which parts of the feature maps are most important. Then we multiply our 256 feature maps by this attention map (Multiply operation). This means pixels in important places remain almost unchanged, while those in unimportant places are reduced or zeroed out.

**Why?** This helps the model focus on the most important parts of the image. For example, for emotion recognition, the most important zones are the eyes and mouth. Spatial Attention learns to automatically identify these zones and pay more attention to them.

The Multiply operation performs element-wise multiplication of feature maps by the attention map to amplify important parts and weaken unimportant ones. It's not concatenation (joining), but actual element-wise multiplication.

After Multiply comes MaxPooling — size reduces from 12×12 to 6×6. And SpatialDropout.

### **Block 4: High-Level Features**

Now we have 256 feature maps of size 6×6. They go into Block 4, where the number of filters increases to 512.

Two convolutional layers with 512 filters, batch normalization, activation after each. After this, MaxPooling — size reduces from 6×6 to 3×3. And SpatialDropout.

Now we have 512 feature maps of size 3×3. What does this mean? Each pixel on these feature maps contains information about a very large part of the original image. In fact, one pixel might contain information about almost the entire face or a significant part of it. The model no longer sees individual lines or corners — it sees holistic patterns, like "smiling face" or "frowning eyebrows."

You correctly guessed that here we have very global generalization. The receptive field (the zone of the original image that one feature map pixel "sees") is very large — almost the entire image.

### **Global Pooling: From Feature Maps to Vector**

After Block 4, we have 512 feature maps of size 3×3. This is still quite a lot of data — 512×3×3=4,608 numbers for one image.

To transition to Dense layers (fully connected layers), we need to convert these feature maps into a vector. How?

Two types of pooling are used:

**GlobalAveragePooling2D:**
This layer takes each feature map (size 3×3) and calculates the average value of all 9 pixels. The result is one number for each feature map. Since there are 512 feature maps, the output is a vector of 512 numbers.

**GlobalMaxPooling2D:**
This layer takes each feature map (size 3×3) and selects the maximum value from all 9 pixels. The result is also one number for each feature map. Also a vector of 512 numbers.

**Concatenate:**
These two vectors (512 numbers from GlobalAveragePooling and 512 numbers from GlobalMaxPooling) are combined into one vector of size 1,024.

**Why use both types of pooling?** GlobalAveragePooling gives general information about the feature map — how "active" it is on average. GlobalMaxPooling gives information about the brightest point on the feature map — where there's the strongest activation. Together they give a more complete picture.

### **Dense Layers: Making the Final Decision**

After concatenate, we have a vector of 1,024 numbers. Now this vector goes into Dense layers (fully connected layers).

**First Dense layer:**
512 neurons. Each neuron is connected to all 1,024 inputs. So each neuron has 1,024 weights (plus bias). Total here: 1,024×512=524,288 parameters (plus 512 bias = 524,800).

After Dense comes batch normalization, activation (ReLU), then dropout. Dropout here is regular dropout, which randomly turns off some neurons (you have rate=0.3, i.e., 30% of neurons).

**Second Dense layer:**
256 neurons. Each connected to 512 outputs from the previous layer. Again batch normalization, activation, dropout (rate=0.25, i.e., 25%).

**Output layer:**
7 neurons, because you have 7 emotion classes. Each neuron is responsible for one emotion. Here we use activation='softmax', which converts outputs into probabilities. For example, the model might say: "this face is 80% happy, 15% neutral, 5% sad."

**Why these Dense layers?** They're needed to combine all information from feature maps and make the final decision about emotion. Convolutional layers detect patterns, and Dense layers "think" about which emotion is most likely based on these patterns.

### **Training One Batch**

Let's go through the training process. Take one batch — 32 images.

**Forward Pass (forward propagation):**

1. Batch enters Input layer: 32 images of size 48×48×1
2. Passes through all convolutional blocks: Block 1, Block 2, Block 3, Block 4
   - At each step: filters, batch normalization, activation, pooling, dropout
   - All happens in parallel for all 32 images thanks to GPU
3. Global Pooling: Feature maps converted to vectors of size 1,024
4. Dense layers: Vectors pass through fully connected layers
   - Output: 7 probabilities for each image
5. Calculate loss: Model compares predictions with correct answers (ground truth)
   - Using Focal Loss (works better with imbalanced data)
   - Loss shows how much the model was wrong
   - Lower loss = better

**Backward Pass (backpropagation):**

After calculating loss, backpropagation begins. This is the process where the model "understands" which weights need to be changed to reduce loss.

1. **Calculate gradients:** Model calculates how loss will change if we slightly change each weight. This is called gradient. Gradient shows the direction to change the weight to reduce loss.

2. **Gradients go backward through network:** First gradients are calculated for output layer, then for Dense layers, then for convolutional layers. Done using chain rule from calculus.

3. **Update weights:** After gradients are calculated for all weights, weights are updated. You use Adam optimizer with learning rate 0.0003. Adam is a smart optimizer that automatically chooses how much to change each weight. Learning rate 0.0003 is the step size for weight changes. If learning rate is too large, model might "overshoot" the optimum. If too small, training will be very slow.

**Which weights are updated?** All weights in the model: filter weights in all convolutional layers, batch normalization parameters (gamma and beta), Dense layer weights, bias terms. Total: 5,347,784 trainable parameters.

**When does backpropagation occur?** After each batch, not after each epoch:

- Batch of 32 images → forward pass → calculate loss → backpropagation → update weights
- Next batch of 32 images → forward pass (already with new weights) → calculate loss → backpropagation → update weights
- And so on, until all 763 batches pass

After this, one epoch ends. Then the next epoch begins — everything repeats, but with updated weights.

### **Complete Training Cycle**

**Epoch 1:**

- Batch 1 (32 images): forward pass → loss → backpropagation → update weights
- Batch 2 (32 images): forward pass → loss → backpropagation → update weights
- ...
- Batch 763: forward pass → loss → backpropagation → update weights

After all 763 batches, model runs through validation dataset to check how well it works on data it hasn't seen during training. This gives val_loss and val_accuracy.

**Epoch 2:**
Same thing, but now model has better weights than at start. It has already learned a bit to recognize emotions.

**Epochs 3, 4, ..., 100:**
Process repeats. With each epoch, model gets better (if everything goes well). Loss decreases, accuracy increases.

### **Callbacks During Training**

You have callbacks that control training:

**EarlyStopping:**
If val_loss doesn't improve for 20 epochs, training stops automatically. This prevents overfitting and saves time.

**ModelCheckpoint:**
When val_accuracy improves, model is saved to file 'best_emotion_optimized.h5'. This way, at the end of training you get the best version of the model, not the last one. Because the last might be worse if model started overfitting.

**ReduceLROnPlateau:**
If val_loss doesn't improve for 6 epochs, learning rate is halved (factor=0.5). This helps model "fine-tune." Initially learning rate is large (0.0003) so model learns quickly. But when model is close to optimum, large learning rate can interfere — model "jumps" around minimum but can't hit it precisely. So reducing learning rate helps model "calm down" and find better weights.

Minimum learning rate is 1e-6 (0.000001). Learning rate doesn't go below this, otherwise training would be too slow.

---

## 🤝 Share Your Progress

After exploring the project:

1. **Take screenshots** of your experiments and results
2. **Note interesting findings** — which architecture worked best?
3. **Share on LinkedIn** and tag me to discuss!

**Questions to explore:**

- How does attention improve accuracy?
- What happens if you remove batch normalization?
- Can you beat 68% accuracy with your modifications?

---

## 📚 Additional Resources

### **Project Links:**

- [GitHub Repository](https://github.com/Serhii2009/emotion-detection) — Full source code
- [Kaggle Notebook](https://www.kaggle.com/code/serhiikravchenko2009/facial-emotion-detection) — Train with free GPU

### **Dataset:**

- [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) — 35,887 emotion images
- **Classes:** Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise
- **Challenge:** Imbalanced (Disgust only 436 images vs Happy 7,215)

### **Learning Resources:**

- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — Interactive visualization
- [TensorFlow CNN Tutorial](https://www.tensorflow.org/tutorials/images/cnn) — Official docs
- [Flask-CORS Documentation](https://flask-cors.readthedocs.io/) — API setup

### **Related Papers:**

- Facial Emotion Recognition using CNN (Goodfellow et al.)
- Challenges in Representation Learning (FER2013 paper)
- Focal Loss for Dense Object Detection (Lin et al.)

---

## ✨ Final Thoughts

This isn't just another coding exercise — it's a **real, deployed emotion detection system** that you can:

- Run on your machine with live webcam
- Modify and experiment with different architectures
- Show to employers as a portfolio project
- Build upon for your own applications (customer sentiment, mental health monitoring, game interactions)

**The best way to learn deep learning is to build it, deploy it, and see it work in real-time.** This project gives you the perfect playground.

**Key achievements:**

- ✅ Trained CNN with 68% accuracy on complex emotion dataset
- ✅ Implemented attention mechanism for focusing on important features
- ✅ Built full-stack application (React + Flask)
- ✅ Deployed real-time inference system (50-100ms latency)
- ✅ Handled class imbalance with Focal Loss

---

## 📬 Questions or Feedback?

If you:

- Have questions about the CNN architecture
- Found interesting results from experiments
- Want to discuss attention mechanisms or training tricks
- Built something cool on top of this system

**Reach out!** Let's learn and build together.

**Connect with me:**

- **GitHub:** [@Serhii2009](https://github.com/Serhii2009)
- **Kaggle:** [@serhiikravchenko2009](https://www.kaggle.com/serhiikravchenko2009)
- **LinkedIn:** [Serhii Kravchenko](https://www.linkedin.com/in/serhii-kravchenko-b941272a6/)

---

**Ready to detect emotions with CNNs? Let's build! 🚀**

_Remember: Real learning happens when you see your code working in the real world._
