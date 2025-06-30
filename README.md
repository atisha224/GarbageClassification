# Garbage Classification - Week 1

This repository contains the Week 1 progress of my Garbage Classification project using EfficientNetV2B2. The goal is to classify waste into categories like plastic, metal, glass, cardboard, paper, and trash — contributing towards smarter waste management using AI.

---

## Week 1 Tasks Completed

### 1. Dataset Download from Kaggle
- Downloaded the **Garbage Classification dataset** directly from [Kaggle](https://www.kaggle.com/).
- Used the Kaggle API to access and download the dataset securely into Google Colab.

### 2. Folder Structure Cleaning
- Unzipped the dataset in Colab and noticed duplicate nested folders.
- Cleaned and restructured the dataset to match the following format:
  /content/Garbage classification/
  ├── cardboard/
  ├── glass/
  ├── metal/
  ├── paper/
  ├── plastic/
  └── trash/
- Removed redundant and incorrectly nested folders to enable clean data loading.

### 3. Directory Setup for Classification
- Ensured each waste category was separated into individual subfolders.
- This format supports automatic labeling using libraries like `ImageDataGenerator` or PyTorch’s `ImageFolder`.

### 4. Image Preprocessing & Augmentation
- Resized all images to a standard shape of **224x224** pixels for EfficientNetV2B2 compatibility.
- Applied real-time data augmentation using techniques like:
  - Rotation
  - Zoom
  - Horizontal flipping
  - Rescaling

These augmentations help the model generalize better and improve performance on unseen data.

---

## Tech Stack
- Google Colab (Python 3 + GPU)
- Kaggle API
- TensorFlow
- NumPy, Matplotlib

## Author
**Atisha Jain**  
B.Tech CSE Student | AI & ML Enthusiast

---------------------------------------------------------------------------

# Garbage Classification - Week 2
This section documents the Week 2 progress of my Garbage Classification project. The focus this week was on building, training, and evaluating the EfficientNetV2B2 model with the cleaned dataset prepared in Week 1.

---

## Week 2 Tasks Completed

### 1. Model Building using EfficientNetV2B2
- Loaded EfficientNetV2B2 as the base model with pretrained ImageNet weights.
- Set include_top=False to remove the default classification head.
- Appended custom classification layers:
  - GlobalAveragePooling2D
  - Dropout(0.3) for regularization
  - Dense(6, activation='softmax') for six waste categories
- Initially froze the base model to use it as a feature extractor.

### 2. Compilation & Training
- Compiled the model with:
  - optimizer: Adam (learning rate = 0.001)
  - loss: sparse categorical crossentropy
  - metrics: accuracy
- Trained the model for 10 epochs using:
  - train_gen: generator with real-time augmented images
  - val_gen: validation set without augmentation

### 3. Training Outcome
- Observed slow learning and low accuracy (around 25%) in both training and validation sets.
- Noted that loss values did not decrease significantly, indicating underfitting or frozen feature extraction limitations.

### 4. Accuracy & Loss Curves
- The training and validation performance was tracked using the history object returned by model.fit().
- Code for displaying Accuracy and Loss graphs:
    import matplotlib.pyplot as plt
    # Accuracy Plot
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss Plot
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

---

## Tools & Libraries Used
  - TensorFlow / Keras
  - EfficientNetV2B2
  - ImageDataGenerator
  - Matplotlib (for plotting)
  - NumPy

## Author
**Atisha Jain**  
B.Tech CSE Student | AI & ML Enthusiast

---------------------------------------------------------------------------

