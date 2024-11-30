# Doodle
by Shaunik Musukula

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

## Table of Contents
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Benchmarks](#benchmarks)
- [Acknowledgments](#acknowledgments)

## Model Architecture

The model is a **Convolutional Neural Network (CNN)** designed for image classification tasks. It is structured with 3 convolutional blocks followed by fully connected layers. Each block consists of:

- **Convolutional Layers**: Two convolutional layers with 3x3 filters, ReLU activation, and L2 regularization to prevent overfitting.
- **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate training.
- **Max Pooling**: Reduces the spatial dimensions of the feature maps, helping to down-sample the input representation.
- **Dropout**: Used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

The final layers include:

- **Flatten Layer**: Converts the 2D matrix into a vector.
- **Dense Layers**: A fully connected layer with 512 units and ReLU activation, followed by a dropout layer.
- **Output Layer**: A dense layer with a softmax activation function to output probabilities for each class.

The model is compiled with the **Adam optimizer** and uses **sparse categorical cross-entropy** as the loss function, with **accuracy** as the evaluation metric.

---

## Benchmarks

### Confusion Matrix
![Confusion Matrix](model/cnn_benchmarks/confusion_matrix.png)

### Classification Report
![Classification Report](model/cnn_benchmarks/classification_report.png)

These benchmarks are saved as images in [`model/cnn_benchmarks`](model/cnn_benchmarks).

---

## Acknowledgments 🙏

This project uses the [Quick, Draw! Dataset](https://quickdraw.withgoogle.com/data) provided by Google. I went ahead and downloaded the dataset and compiled it into an `npz` file. You can access it [here](https://drive.google.com/drive/folders/1eCo87_mNv0MAS-3zTeKbxPg8cCcrVFNH).
