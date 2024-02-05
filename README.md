# Credit-Card-Fraud-Detection

This code appears to be a step-by-step data science project that focuses on fraud detection. Here's a breakdown of what each section does:

Importing Necessary Libraries:

Importing the required Python packages, including pandas, numpy, matplotlib, seaborn, and scikit-learn.
Importing the Dataset:

Loading a dataset named 'card_transdata.csv' into a Pandas DataFrame called 'df'.
Displaying the first 5 rows of the DataFrame to inspect the data.
Data Preprocessing:

Checking the distribution of the 'fraud' column, which seems to be a binary target variable.
Handling Duplicates:

Identifying and removing duplicate rows in the dataset, keeping only the first occurrence of each duplicate.
Exploratory Data Analysis (EDA):

Checking data types, the number of unique values in each column, and generating statistical summaries for the numerical attributes.
Visualizing the data using various plots like pie charts and pair plots to explore relationships and patterns.
Data Balancing:

Balancing the dataset by randomly sampling an equal number of non-fraudulent (0) and fraudulent (1) transactions to create a balanced dataset.
Feature Scaling:

Standardizing the feature values using StandardScaler and MinMaxScaler to prepare the data for modeling.
Train-Test Split:

Splitting the data into training and testing sets (70% train, 30% test).
Model Selection and Evaluation:

Importing several machine learning models (Logistic Regression, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, and Naive Bayes).
Defining functions for model evaluation, including metrics such as accuracy, mean absolute error, mean squared error, R-squared, and generating classification reports and confusion matrices.
Training each model on the training data, evaluating its performance on both training and testing sets, and visualizing confusion matrices.
Selecting the best-performing model based on test accuracy.
Neural Network (ANN) Model:

Building and training an artificial neural network (ANN) using TensorFlow/Keras with two dense layers.
Evaluating the ANN model's performance using classification reports, confusion matrices, and accuracy.
Stratified K-Fold Cross-Validation (Optional):

Implementing stratified k-fold cross-validation to assess model performance more robustly using the area under the ROC curve (AUROC).
The project aims to detect fraud in credit card transactions using various machine learning models, and the best-performing model is the Decision Tree with an accuracy of 99.98%. Additionally, an artificial neural network (ANN) achieved an accuracy of 96% on the test set.

The code provides a comprehensive analysis of the dataset, model evaluation, and potential improvements such as cross-validation. However, further steps like hyperparameter tuning and feature engineering could be explored to enhance model performance.
