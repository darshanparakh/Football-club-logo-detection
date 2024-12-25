# Football Club Logo Classification Project

![image](https://github.com/user-attachments/assets/fc775f34-a76d-4152-af42-2ec57c4eeec9)


## Overview
This project involves classifying football clubs based on their logos using various machine learning models. The models employed include Support Vector Machine (SVM), Random Forest, K-Nearest Neighbors (KNN), Naive Bayes, Artificial Neural Networks (ANN), and Convolutional Neural Networks (CNN). The dataset contains images of football club logos, each associated with a specific club.

## Requirements
To run this project, you need to install the following dependencies:

Python 3.x,
pandas,
numpy,
scikit-learn,
matplotlib,
seaborn,
opencv-python,
keras,
tensorflow,
torch,
torchvision,
PIL (Pillow),
sklearn,

You can install the dependencies via pip:

## Dataset

(https://www.kaggle.com/datasets/alexteboul/english-premier-league-logo-detection-20k-images)

## Details
1. Data Preprocessing and Visualization
The project loads football club logos and visualizes them in a grid, displaying the image and its corresponding team name. It also visualizes the class distribution of the dataset to show the number of images per football club.


2. Model Training and Evaluation
The project uses multiple machine learning models to classify the logos:

a. SVM (Support Vector Machine) Model

b. Random Forest Classifier

c. KNN (K-Nearest Neighbors) Classifier

d. Naive Bayes Classifier

e. Artificial Neural Network (ANN)

f. Convolutional Neural Network (CNN)


3. Confusion Matrix and Visualization

The project generates confusion matrices for each model's predictions and visualizes them using seaborn. It also displays incorrect predictions with the corresponding true and predicted labels.

![image](https://github.com/user-attachments/assets/2f65de46-9081-4f1a-a7b3-c45af388dac1)


4. False Predictions Visualization

For each model, the project visualizes images where the model made incorrect predictions. The true label and predicted label are displayed for each incorrect image.

![image](https://github.com/user-attachments/assets/cbf728d1-699d-4544-a58b-cd08096bdb04)

