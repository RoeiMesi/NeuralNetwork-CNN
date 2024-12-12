# Machine Learning

This project is an assignment designed to enhance your understanding of Neural Networks and their implementation using NumPy and PyTorch. It involves implementing and training models for handwritten digit recognition (using the MNIST dataset) and clothing classification (using the Fashion-MNIST dataset).

## Overview

The assignment is divided into two main parts:

### Part 1: Neural Network Using NumPy
1. **Dataset**: Use the MNIST dataset of handwritten digits.
2. **Objective**: Implement a neural network (multilayer perceptron) using only NumPy for operations.
3. **Key Steps**:
   - Preprocess the dataset with Min-Max normalization.
   - Split the data into training (80%) and testing (20%).
   - Implement activation functions (e.g., sigmoid, softmax) and loss functions.
   - Train the neural network using gradient descent.
   - Test and evaluate the model to ensure accuracy > 80%.

### Part 2: Neural Network Using PyTorch
1. **Dataset**: Use the Fashion-MNIST dataset for clothing classification.
2. **Objectives**:
   - Train a simple fully connected neural network.
   - Train a convolutional neural network (CNN) for better performance.
3. **Key Steps**:
   - Use PyTorch’s DataLoader to preprocess and load data.
   - Define and train network architectures.
   - Experiment with hyperparameters to improve validation accuracy (> 80%).
   - Save predictions for the test dataset to a file named `predictions.txt`.

## Environment

- **Programming Language**: Python 3.x
- **Libraries**: 
  - NumPy
  - PyTorch
  - Matplotlib
  - torchvision (for datasets)
- **System Requirements**: GPU support is recommended for Part 2 (PyTorch).

## Usage Instructions

### Part 1: Neural Network Using NumPy
1. **Run the Notebook**:
   - Preprocess the MNIST dataset.
   - Implement neural network layers (input, hidden, and output).
   - Train the network for 100 epochs using gradient descent.
2. **Test the Model**:
   - Evaluate accuracy on the test dataset.
   - Expected accuracy: > 80%.

### Part 2: Neural Network Using PyTorch
1. **Setup Environment**:
   - Use Google Colab or a local machine with GPU support.
   - Download the FashionMNIST dataset.
2. **Train the Models**:
   - Fully connected neural network.
   - Convolutional neural network (CNN).
3. **Experimentation**:
   - Modify hyperparameters, network layers, and other configurations to achieve the best validation accuracy (> 80%).
4. **Save Predictions**:
   - Load the provided test dataset.
   - Write predictions to a file named `predictions.txt`.

## Key Features

- **NumPy Implementation**: Provides a hands-on understanding of the inner workings of neural networks without relying on libraries.
- **PyTorch Implementation**: Demonstrates the use of a powerful deep learning framework for efficient model development.
- **Dataset Preprocessing**: Includes normalization, data splitting, and transformations.
- **Model Training**: Implements gradient descent and backpropagation for learning.
- **CNN Architecture**: Leverages convolutional layers, pooling, and dropout for robust performance.

## Results

- **NumPy Neural Network**:
  - Average Loss: 0.0066 (after 100 epochs)
  - Test Accuracy: > 96%
- **PyTorch Models**:
  - Fully Connected Network Accuracy: ~81%
  - CNN Validation Accuracy: > 92%

## Submission Instructions

1. Complete all sections of the notebook.
2. Save the notebook as `Ex5_Machine_Learning.ipynb`.
3. Save your predictions for the test dataset in `predictions.txt`.
4. Zip the files into `ex5.zip`.
5. Submit to the designated system under course number `89-2511`, group `01`, and assignment `ex5_ML`.

## Notes

- Ensure that you do not change any provided code.
- Follow the instructions for submission strictly.
- Use only the libraries specified in each part.

This project provides valuable insights into neural network implementation, preprocessing, and optimization techniques using both NumPy and PyTorch frameworks.
