MNIST Handwritten Digit Recognition
Overview

This project demonstrates a Machine Learning model that recognizes and classifies handwritten digits (0–9) using the MNIST dataset. The dataset consists of 70,000 grayscale images of handwritten digits, each sized 28x28 pixels.
The goal is to build, train, and evaluate a model that can accurately identify digits from unseen images.

Objectives

Preprocess the dataset for training and testing.

Build and train a neural network using TensorFlow and Keras.

Evaluate the model’s performance on test data.

Save and export the trained model for future predictions.

Dataset

Name: MNIST Handwritten Digits

Source: Yann LeCun’s MNIST Dataset

Details:

Training images: 60,000

Test images: 10,000

Image size: 28x28 pixels

Format: Grayscale

Labels: 0–9

Project Workflow
1. Data Loading

The MNIST dataset is directly loaded from TensorFlow’s Keras library:

from tensorflow.keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

2. Data Preprocessing

Normalized pixel values to a range of 0–1.

Reshaped data into a 4D tensor for CNN input.

One-hot encoded labels using to_categorical().

3. Model Architecture

A Convolutional Neural Network (CNN) is used for digit recognition:

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

4. Model Compilation

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metric: Accuracy

5. Model Training

Trained for 5–10 epochs with validation data:

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

6. Model Evaluation

Evaluated on test data to calculate overall performance:

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

Results

Accuracy: ~99% on test data

Loss: Minimal

The model effectively recognizes handwritten digits even with variations in style or thickness.

Model Saving and Download
model.save('/content/mnist_digit_model.h5')
from google.colab import files
files.download('/content/mnist_digit_model.h5')


The trained model is saved as mnist_digit_model.h5 and can be directly downloaded from Google Colab.

Tools and Technologies

Programming Language: Python

Frameworks/Libraries:

TensorFlow / Keras

NumPy

Matplotlib (for visualization)

Environment: Google Colab

Future Enhancements

Integrate with a Streamlit or Flask web app to upload handwritten images and get predictions.

Experiment with dropout layers or batch normalization for improved generalization.

Extend to real-world handwritten datasets for custom digit recognition.

Conclusion

This project successfully builds a deep learning model capable of recognizing handwritten digits with high accuracy. It demonstrates key concepts in computer vision, data preprocessing, and neural network design using TensorFlow and Keras.
