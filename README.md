# AI-Powered Sign Language Translator

## 🔍 Overview

The AI-Powered Sign Language Translator is a deep learning-based system designed to convert sign language gestures into readable text and audible speech in real-time. This project bridges communication gaps between the Deaf and hearing communities by leveraging computer vision and convolutional neural networks (CNNs) to interpret hand gestures.


## 🛠 Features

* Converts American Sign Language (ASL) gestures to text and speech.
* High-accuracy deep learning model (CNN-based).
* Dynamic visualizations of model performance metrics using Plotly.
* Easily deployable with saved model weights.


## 🧠 Model Development & Refinement

### 1. Dataset and Preprocessing

* Merged and randomly re-split ASL image datasets to enhance randomness.
* Standardized preprocessing: image resizing, normalization.
* Used separate validation data to monitor model performance.

### 2. Model Architecture

* Convolutional Neural Network (CNN) tailored for image classification.
* Designed to automatically extract spatial features from sign language images.

### 3. Training Optimization

* **Early Stopping**: Implemented to halt training upon plateauing of validation loss.
* **Dynamic Visualizations**: Replaced static graphs with interactive Plotly charts.
* **Validation Strategy**: Avoided cross-validation due to computational cost; used hold-out validation split.

### 4. Evaluation Metrics

| Dataset    | Accuracy | Loss   |
| ---------- | -------- | ------ |
| Validation | 98.84%   | 0.0711 |
| Test       | 94.12%   | 0.1709 |


## 🚀 Installation

### Prerequisites

* Python 3.x
* pip (Python package manager)

### Setup Instructions

```bash
# Clone the repository
git clone https://github.com/edasaruhan/FTL_Ethiopia_ML1_Gr10.git
cd FTL_Ethiopia_ML1_Gr10

# Install dependencies
pip install -r requirements.txt
```


## ⚙️ Usage

### Run Inference with Trained Model

```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('asl_cnn_model.h5')

# Predict on test images
predictions = model.predict(test_images)
```

### Evaluate Model

```python
# Evaluate on test data
test_loss, test_accuracy = model.evaluate(test_images, test_classes_encoded)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
```

## 🧪 Model Training Snippet

```python
from tensorflow.keras.callbacks import EarlyStopping

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# Train the CNN model
history = cnn_model.fit(
    data_gen.flow(train_images, train_classes_encoded, batch_size=32),
    epochs=50,
    validation_data=(val_images, val_classes_encoded),
    callbacks=[early_stopping]
)
```


## 📈 Visualizations

* Interactive accuracy/loss plots using Plotly.
* Visual assessment of overfitting and training progression.
* Model interpretability enhanced with graph interactivity.


## 💡 Conclusion

This project successfully demonstrates the potential of AI in enabling accessible communication. The model generalizes well on unseen data and achieves strong accuracy with minimal overfitting, thanks to refinement techniques like early stopping, dynamic visualizations, and structured validation.

