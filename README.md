# Logistic Regression Pipeline

## 1. Project Overview

This project demonstrates a complete Logistic Regression pipeline using a real-world binary classification dataset. The goal is to classify breast tumors as **Malignant** or **Benign** using Machine Learning.

## 2. Dataset Description

The project uses the Breast Cancer Wisconsin Diagnostic dataset provided by Scikit-learn.

- **Total samples:** 569
- **Number of features:** 30
- **Total columns:** 31 (including the target column)
- **Target classes:**
  - `0`: Malignant
  - `1`: Benign

## 3. Data Exploration and Preprocessing

The following steps were performed:

- Loaded the dataset using Pandas.
- Explored the dataset using `head()`, `info()`, and `describe()`.
- Checked for missing values and duplicate records.
- Visualized the class distribution.
- Plotted distributions of selected features.
- Split the dataset into training and testing sets using an 80/20 ratio.
- Applied StandardScaler to standardize numerical features.
- Fitted the scaler only on the training data to avoid data leakage.

## 4. Model Training

A Logistic Regression model from Scikit-learn was trained using all 30 features.

- **Training samples:** 455
- **Testing samples:** 114
- **Maximum iterations:** 1000
- **Random state:** 42

## 5. Mathematical Functions

Custom Python implementations were created for the following mathematical functions:

- **Sigmoid Function:** Converts numerical scores into probabilities between 0 and 1.
- **Softmax Function:** Converts multiple class scores into a probability distribution, supporting multiclass classification.
- **Binary Cross-Entropy Loss:** Measures the difference between actual labels and predicted probabilities.

## 6. Visualizations

The notebook includes the following visualizations:

1. Dataset summary and feature distributions.
2. Class distribution.
3. Sigmoid function plot.
4. Binary Cross-Entropy Loss plot.
5. 2D Logistic Regression Decision Boundary using Mean Radius and Mean Texture.
6. Prediction Probability Distribution for the Benign class.

## 7. Key Takeaways

- Logistic Regression is suitable for binary classification tasks.
- Feature scaling helps prepare numerical features for model training.
- The Sigmoid function converts model scores into probabilities.
- Softmax extends probability prediction to multiclass classification.
- Cross-Entropy Loss measures prediction error.
- A 2D decision boundary helps visualize how Logistic Regression separates two classes.
- Prediction probability distributions provide insight into model predictions.
- Additional classification metrics are needed for a complete evaluation of model performance.

## 8. Project Structure

```text
logistic_regression_pipeline/
│
├── notebook_1.ipynb
└── README.md
```
