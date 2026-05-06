# Telco Customer Churn Prediction

## Project Overview
This project focuses on predicting customer churn for a telecom company using the Telco Customer Churn dataset. It is a binary classification problem where the goal is to identify whether a customer will churn or not. The dataset is highly imbalanced, making it a realistic and challenging machine learning problem commonly encountered in industry.

## Dataset
* Source: [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
* Contains customer demographic, account, and service-related features
* Target variable: Churn (binary classification)

## Workflow

### 1. Exploratory Data Analysis (EDA)
* Analyzed customer behavior patterns
* Identified key churn indicators
* Studied class imbalance distribution

### 2. Data Preprocessing
* Handled missing values
* Encoded categorical variables
* Scaled numerical features
* Built reproducible preprocessing pipeline

### 3. Model Selection
Before model implementation, multiple baseline models were evaluated using K-Fold Cross Validation to select the top 3 models. The following models were compared:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Random Forest
* XGBoost
* LightGBM

### 4. Model Building
In this stage, the selected models were trained and optimized using a complete machine learning pipeline. The process included handling missing values, feature engineering, encoding categorical variables, and scaling numerical features where required. All transformations were applied in a way that prevented data leakage by ensuring that preprocessing steps were fitted only on the training data and then applied to the test data. Implemented and compared the following models:
* Logistic Regression
* LightGBM
* XGBoost

### 6. Hyperparameter Tuning
Hyperparameter tuning was performed using Optuna to systematically search for the best model configurations. The objective was to maximize model performance while ensuring generalization on unseen data.

### 8. Evaluation Strategy
Due to class imbalance, the following metrics were used:
* F1 Macro Score
* Precision-Recall AUC (PR AUC)

## Best Model: XGBoost
The best-performing model was XGBoost achieving:
* Macro F1 Score: 0.7518
* PR AUC Score: 0.668
