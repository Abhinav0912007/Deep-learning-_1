# Deep-learning-_1
Started learning about Activation Functions
# README

## Project Overview

This project demonstrates:

1. Common **activation functions** used in neural networks
2. Training a **small TensorFlow/Keras model** on a small dataset
3. Model training using **100 epochs**

The goal is to provide a simple, beginner‑friendly example that is easy to understand and modify.

---

## Activation Functions

Activation functions introduce non‑linearity into neural networks, allowing them to learn complex patterns.

### 1. ReLU (Rectified Linear Unit)

```python
relu(x) = max(0, x)
```

**Pros:** Fast, simple, widely used
**Cons:** Can suffer from dying ReLU problem

### 2. Sigmoid

```python
sigmoid(x) = 1 / (1 + e^-x)
```

**Pros:** Outputs values between 0 and 1 (good for binary classification)
**Cons:** Vanishing gradient for deep networks

### 3. Tanh

```python
tanh(x) = (e^x - e^-x) / (e^x + e^-x)
```

**Pros:** Zero‑centered output
**Cons:** Still suffers from vanishing gradients

### 4. Softmax

Used in the output layer for multi‑class classification.

```python
softmax(x_i) = e^{x_i} / Σ e^{x_j}
```

---

## Dataset

* Small, toy dataset (e.g., from NumPy or scikit‑learn)
* Suitable for quick experiments and learning
* Features and labels are normalized before training

Example:

* Input features: 2–10 numerical values
* Output: Binary or multi‑class labels

---

## Model Architecture

A simple feedforward neural network built using TensorFlow Keras:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## Model Compilation

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

## Training the Model

The model is trained for **100 epochs** on a small dataset.

```python
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)
```

---

## Evaluation

```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_accuracy)
```

---

## Results

* Training accuracy improves steadily over epochs
* Suitable for demonstrating overfitting and underfitting
* Fast training due to small dataset size

---

## Requirements

* Python 3.8+
* TensorFlow 2.x
* NumPy
* scikit‑learn (optional)

Install dependencies:

```bash
pip install tensorflow numpy scikit-learn
```

---

## Notes

* You can switch activation functions to observe performance changes
* Increase dataset size or model depth for more complex experiments
* Ideal for educational and experimental purposes

---

## Author

Created for learning and demonstration of activation functions and small TensorFlow models
by Abhinav Singh
