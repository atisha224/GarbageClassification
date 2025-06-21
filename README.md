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

