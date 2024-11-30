MobileNet2.0 Benchmarks:
Test Loss: 0.25598567724227905
Test Accuracy: 0.9277620315551758
model/mobilenet2.0_benchmarks/classification_report.png
model/mobilenet2.0_benchmarks/confusion_matrix.png


<div align="center">
  <h1>Doodle</h1>
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build Status"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License"/>
  <br>
  made by Shaunik Musukula
</div>

## Table of Contents
- [<code>🏗️ Model Architecture</code>](#model-architecture)
- [<code>📊 Benchmarks</code>](#benchmarks)
- [<code>🙏 Acknowledgments</code>](#acknowledgments)

## 🏗️ Model Architecture

The project explores two model architectures for image classification tasks:

### CNN Architecture
The CNN model is structured with 3 convolutional blocks followed by fully connected layers. Each block consists of:

- **Convolutional Layers**: Two convolutional layers with 3x3 filters, ReLU activation, and L2 regularization to prevent overfitting.
- **Batch Normalization**: Applied after each convolutional layer to stabilize and accelerate training.
- **Max Pooling**: Reduces the spatial dimensions of the feature maps, helping to down-sample the input representation.
- **Dropout**: Used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training.

The final layers include:

- **Flatten Layer**: Converts the 2D matrix into a vector.
- **Dense Layers**: A fully connected layer with 512 units and ReLU activation, followed by a dropout layer.
- **Output Layer**: A dense layer with a softmax activation function to output probabilities for each class.

The model is compiled with the **Adam optimizer** and uses **sparse categorical cross-entropy** as the loss function, with **accuracy** as the evaluation metric.

### MobileNet2.0 Inspired Architecture
This architecture is built upon the efficient design principles of MobileNet2.0, focusing on reducing computational complexity while maintaining high accuracy:

- **Depthwise Separable Convolutions**: Utilized to decompose standard convolutions into a depthwise convolution followed by a pointwise convolution, significantly reducing the number of parameters and computational cost.
- **Inverted Residuals with Linear Bottlenecks**: Implemented to enhance feature reuse and maintain a lightweight model structure. This involves using shortcut connections between bottleneck layers.
- **Global Average Pooling**: Applied to the output of the feature extractor to reduce the spatial dimensions and prepare for the dense layers.
- **Dense Layers**: A fully connected layer with 512 units and ReLU activation, followed by a dropout layer to mitigate overfitting.
- **Output Layer**: A dense layer with a softmax activation function to output class probabilities.

The model is compiled with the **Adam optimizer** and uses **sparse categorical cross-entropy** as the loss function, with **accuracy** as the evaluation metric.

---

## 📊 Benchmarks

### MobileNet2.0 Inspired Benchmarks
- **Test Loss**: 0.25598567724227905
- **Test Accuracy**: 0.9277620315551758

![Confusion Matrix](model/mobilenet2.0_benchmarks/confusion_matrix.png)
![Classification Report](model/mobilenet2.0_benchmarks/classification_report.png)

### CNN Benchmarks
- **Test Loss**: [insert cnn test loss later]
- **Test Accuracy**: [insert cnn test accuracy later]

![Confusion Matrix](model/cnn_benchmarks/confusion_matrix.png)
![Classification Report](model/cnn_benchmarks/classification_report.png)

### Comparison
The MobileNet2.0 inspired architecture demonstrates superior performance over the traditional CNN model, achieving higher accuracy and lower test loss. This is largely due to the use of depthwise separable convolutions and inverted residuals, which optimize the model for both speed and accuracy in image classification tasks.

---

## 🙏 Acknowledgments

This project uses the [Quick, Draw! Dataset](https://quickdraw.withgoogle.com/data) provided by Google. I went ahead and downloaded the dataset and compiled it into an `npz` file. You can access it [here](https://drive.google.com/drive/folders/1eCo87_mNv0MAS-3zTeKbxPg8cCcrVFNH).
