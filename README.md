## Assignment-6: Handling Missing Data and Model Evaluation
This project demonstrates various strategies for handling missing data in the UCI Credit Card Default Clients Dataset. It evaluates the impact of different imputation techniques on model performance using Logistic Regression.

# Dataset
The dataset used in this project is the UCI Credit Card Default Clients Dataset, which contains information about credit card clients in Taiwan. The target variable is default.payment.next.month, indicating whether a client will default on their payment.

# Project Structure
Load and Prepare Data: Load the dataset, rename columns for clarity, and introduce Missing At Random (MAR) values in selected columns.
Imputation Strategies:
Simple Imputation (Median): Fill missing values with the median of each column.
Linear Regression Imputation: Predict missing values using a linear regression model.
Non-Linear Regression Imputation: Predict missing values using a K-Nearest Neighbors (KNN) regression model.
Listwise Deletion: Remove rows with missing values.
Data Splitting: Split the datasets into training and testing sets.
Feature Standardization: Standardize features using StandardScaler.
Model Training: Train Logistic Regression models on each dataset.
Model Evaluation: Evaluate models using classification metrics (Accuracy, Precision, Recall, F1-score).
Results Comparison: Summarize and compare the performance of models trained on different datasets.

# Installation
To run this project, you need Python and the following libraries installed:

pandas
numpy
scikit-learn
Install the required libraries using:

# Usage
Clone the repository or download the notebook file.
Place the UCI_Credit_Card.csv dataset in the same directory as the notebook.
Open the notebook in Jupyter Notebook or VS Code.
Run the cells sequentially to execute the code.
# Results
The project compares the performance of Logistic Regression models trained on datasets with different imputation strategies. The results are summarized in a table showing the F1-scores for each model:

# Model	F1-Score
A (Median Imputation)	value
B (Linear Regression)	value
C (Non-Linear Regression)	value
D (Listwise Deletion)	value
# Discussion
Listwise Deletion: Removes rows with missing values, leading to data loss and potentially reduced model performance.
Median Imputation: Simple and robust to outliers but may not capture relationships between features.
Linear Regression Imputation: Assumes a linear relationship between features, which may not always hold.
Non-Linear Regression Imputation: Captures complex relationships between features, often resulting in better performance.
#Conclusion
The project demonstrates the trade-offs between different missing data handling strategies. Non-linear regression imputation is recommended for this scenario due to its balance of accuracy and robustness.

# License
This project is for educational purposes only.
