# Simple Neural Network Model for Iris Dataset Classification

## Overview

This project presents a simple neural network model designed to classify the Iris dataset into three distinct species of iris flowers. The model consists of three hidden layers and uses the ReLU activation function to introduce non-linearity. The aim is to demonstrate the effectiveness of a multi-layer perceptron on a well-known dataset.

## Dataset

The model is trained and evaluated on the Iris dataset, a classic dataset in the field of machine learning. The dataset contains 150 samples, each with four features: sepal length, sepal width, petal length, and petal width. The samples are divided into three classes, each representing a different species of iris flower.

- **Dataset Source:** [Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)

## Model Architecture

The neural network architecture comprises:

- **Input Layer:** 4 neurons (one for each feature)
- **Hidden Layer 1:** 8 neurons, ReLU activation
- **Hidden Layer 2:** 9 neurons, ReLU activation
- **Hidden Layer 3:** 10 neurons, ReLU activation
- **Output Layer:** 3 neurons (one for each class), no activation (used in conjunction with CrossEntropyLoss)

## Training and Evaluation

### Training

The model is trained using the Adam optimizer with a learning rate of 0.01. The loss function used is CrossEntropyLoss, suitable for multi-class classification problems. The training process runs for 150 epochs, and the loss is monitored and printed every 10 epochs.

### Evaluation

The model's performance is evaluated on a test set, which is 20% of the entire dataset. The accuracy of the model is calculated by comparing the predicted class labels to the true labels.

### Results

- **Accuracy:** The model achieved an accuracy of approximately **93.33%** on the test set, indicating a good level of performance given the simplicity of the network.

## Final Note

This project demonstrates the capability of a simple neural network to perform classification tasks effectively. The Iris dataset serves as a perfect starting point for those new to neural networks and machine learning, showcasing how relatively straightforward models can achieve high accuracy.

## Acknowledgments

- **Dataset:** [UCI Machine Learning Repository - Iris Dataset](https://archive.ics.uci.edu/dataset/53/iris)
- **Framework:** PyTorch


