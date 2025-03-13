# Digit classification using Convolutional Neural Network (CNN)

This repository contains the code and resources for training a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

## Overview

This project utilizes TensorFlow and Keras to build and train a CNN model. The model is designed to classify grayscale images of handwritten digits (0-9). The code includes data preprocessing, model architecture definition, training, evaluation, and visualization of results.

## Prerequisites

Before running the code, ensure you have the following libraries installed:

-   Python 3.x
-   TensorFlow
-   Seaborn
-   NumPy
-   Pandas
-   pydot

## You can install these libraries using pip:

pip install tensorflow matplotlib seaborn numpy pandas pydot
(Replace [your_script_name.py] with the actual name of your Python file.)

## Code Description
[your_script_name.py]: This Python script performs the following tasks:
Loads the MNIST dataset.
Preprocesses the data (normalization, reshaping).
Defines the CNN model architecture.
Compiles and trains the model.
Evaluates the model's performance.
Visualizes the training history and predictions.
Saves and loads the trained model.
Displays sample predictions with color coded results.

## Model Architecture
The CNN model consists of the following layers:

Convolutional layers with ReLU activation.
Max pooling layers.
Flatten layer.
Dense layers with ReLU and softmax activation.
Dropout layer.

## Training and Evaluation
The model is trained using the Adam optimizer and the sparse categorical cross-entropy loss function. Training and validation accuracy and loss are tracked and plotted. The trained model is saved as an H5 file.

## Visualization
The script generates visualizations of:

Sample images from the MNIST dataset.
Training and validation loss and accuracy curves.
Predictions on the test set, with correct predictions in green and incorrect predictions in red.
The model architecture.
## Results
![Screenshot 2025-03-13 150807](https://github.com/user-attachments/assets/364b8a61-a808-4393-a276-e48bb2abd696)
![Screenshot 2025-03-13 150847](https://github.com/user-attachments/assets/c7f3e598-15f3-4994-a245-15d9ca05bf2f)
![Screenshot 2025-03-13 150930](https://github.com/user-attachments/assets/35d66bf5-c74c-4268-b65e-80e25b49ec0d)
![Screenshot 2025-03-13 151031](https://github.com/user-attachments/assets/d30a3c6d-6c79-4ebc-a62a-fa97e48baf97)
![Screenshot 2025-03-13 150911](https://github.com/user-attachments/assets/eb4bfdf3-5884-45cd-ac08-ea81debb2ca9)
![Screenshot 2025-03-13 143441](https://github.com/user-attachments/assets/76e51e0f-1868-4993-b9c2-018500b871e6)

## contacts
aryanghorpade60@gmail.com
ghorpade656
