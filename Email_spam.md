Email Spam Classification using Machine Learning
Overview

This project focuses on building a machine learning model capable of classifying emails as spam or not spam based on their word-frequency features. The model is trained using a labeled dataset containing thousands of email records, each represented as a vector of word occurrences.

The goal is to create a reliable, high-accuracy system that can automatically detect spam emails using traditional machine learning methods.

Project Objectives

Load and explore the given email dataset.

Preprocess the data to prepare it for machine learning.

Train a classification model to distinguish between spam and non-spam emails.

Evaluate model performance using accuracy, precision, recall, and F1-score.

Save the trained model for future use.

Dataset Information

Filename: emails.csv

Total Samples: 5172

Total Features: 3002

Label Column: Prediction

0 → Not Spam

1 → Spam

Each row represents an email, and each column (except identifiers and the label) represents a word’s frequency count in that email.

Example columns:

['the', 'to', 'ect', 'and', 'for', 'of', 'a', 'you', 'hou', ...]

Technologies Used

Programming Language: Python

Environment: Google Colab

Libraries:

pandas – Data manipulation

scikit-learn – Model training and evaluation

joblib – Saving and loading trained models

numpy – Numerical operations

Workflow
1. Data Loading and Inspection

The dataset was uploaded into Google Colab and explored to understand its structure, shape, and class distribution.

2. Data Preprocessing

Removed unnecessary columns like Email No.

Extracted feature columns and target label (Prediction)

Split the dataset into training (80%) and testing (20%) sets

3. Model Building

A Random Forest Classifier was used for training due to its robustness and high accuracy in tabular data classification tasks.

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

4. Model Training

The model was trained on the processed dataset using the training split.

model.fit(X_train, y_train)

5. Model Evaluation

The model was tested on unseen data to evaluate performance.

Results:

Metric	Score
Accuracy	96.43%
Precision	96%
Recall	96%
F1-Score	96%

Confusion Matrix:

[[715  20]
 [ 17 283]]


This shows that the model effectively differentiates between spam and non-spam emails with minimal misclassification.

6. Model Saving

After achieving strong results, the model was saved for future use:

import joblib
joblib.dump(model, "email_spam_model.pkl")

File Structure
├── emails.csv                # Dataset file
├── email_spam_model.pkl      # Trained model
├── spam_classifier.ipynb     # Colab notebook
└── README.md                 # Project documentation

How to Run

Open Google Colab

Upload the dataset (emails.csv)

Paste the training code into a Colab cell and run it

The model will train, evaluate, and save automatically

After training, download the model:

from google.colab import files
files.download('/content/email_spam_model.pkl')

Results and Discussion

The Random Forest model performed with an impressive 96% accuracy, making it suitable for detecting spam emails in similar datasets.

However, this model works best for datasets with preprocessed text converted into numeric features (like word frequency counts or TF-IDF).
Future improvements can include:

Using TF-IDF or Word Embeddings for richer feature extraction

Testing Gradient Boosting or Neural Networks

Building a web-based application to upload and analyze emails

Conclusion

This project demonstrates how classical machine learning can effectively identify spam emails using structured text frequency data. The resulting model achieves high accuracy and can serve as the foundation for more advanced email filtering systems or integrated web applications.
