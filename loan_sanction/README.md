# Loan Sanction Prediction and Fairness Analysis

This repository contains a Jupyter Notebook (`loan_sanction_prediction.ipynb`) that explores a dataset related to loan sanction decisions. The notebook performs a thorough exploratory data analysis (EDA) to understand the factors influencing loan approvals. It then builds and evaluates various machine learning models to predict loan status, with a crucial focus on identifying and analyzing potential biases related to sensitive features like gender and marital status.

## Contents

*   `loan_sanction_prediction.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `loan_sanction_train.csv`: The training dataset for loan sanction prediction.
*   `loan_sanction_test.csv`: The test dataset (used only for initial loading, not in model training in the provided code snippet).
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `loan_sanction_train.csv` and inspects for missing values and overall structure.
*   **Missing Value Handling**: Drops rows with missing values for initial analysis.
*   **Target Variable Analysis**: Analyzes the distribution of `Loan_Status`.
*   **Feature-wise Analysis and Pivots**:
    *   Examines the relationship between `Gender`, `Married` status, and `Loan_Status` using pivot tables to show approval percentages.
    *   Groups `LoanAmount` into bins and analyzes its relationship with `Loan_Status`.
    *   Creates a `TotalIncome` feature (ApplicantIncome + CoapplicantIncome) and visualizes its relationship with `LoanAmount`.

### 2. Data Preprocessing for Machine Learning

*   **Categorical to Numerical Conversion**: Maps `Loan_Status` ('Y' to 1, 'N' to 0) and uses `pd.get_dummies` for one-hot encoding of categorical features.
*   **Advanced Preprocessing Pipeline**:
    *   Constructs a comprehensive `ColumnTransformer` with `Pipeline` steps for robust preprocessing of various feature types:
        *   **Categorical Features**: Imputation (most frequent) and One-Hot Encoding.
        *   **Numerical Features (General)**: Imputation (mean) and Normalization.
        *   **`LoanAmount`**: Imputation (mean) and Binning (`KBinsDiscretizer`).
    *   Integrates `SelectKBest` for feature selection (`f_classif`) and a `KNeighborsClassifier` into a single, end-to-end machine learning pipeline.

### 3. Machine Learning Model Training and Evaluation

*   **Data Splitting**: Splits the processed data into training and testing sets.
*   **Model Training and Comparison**:
    *   Trains and evaluates a **Decision Tree Classifier**, a **K-Nearest Neighbors Classifier**, and a **Dummy Classifier** (as a baseline).
    *   Compares their performance using `accuracy_score`, `confusion_matrix`, and `ConfusionMatrixDisplay`.
*   **Cross-Validation**: Performs 10-fold cross-validation for more robust evaluation of the Decision Tree and K-Nearest Neighbors models.
*   **Hyperparameter Tuning (K-Nearest Neighbors)**: Uses `GridSearchCV` to find optimal `n_neighbors` and `weights` for the K-Nearest Neighbors Classifier.
*   **Pipeline Evaluation**: Demonstrates how to train and evaluate the complete preprocessing and modeling pipeline (`pipe`) for seamless integration.

### 4. Fairness Analysis with `fairlearn`

*   **Demographic Parity**: Utilizes `fairlearn.metrics.demographic_parity_ratio` to assess fairness, specifically examining if loan approval rates are similar across different gender groups (`Gender_Male`).
*   **Impact of Feature Removal**: Explores the effect on accuracy when sensitive features like 'Gender' and 'Married' status are explicitly removed from the training data, highlighting potential trade-offs between fairness and overall model accuracy.

## Key Learnings

*   In-depth exploratory data analysis for understanding dataset characteristics and relationships.
*   Robust data preprocessing techniques, including handling missing values, encoding categorical features, and advanced data transformations using `ColumnTransformer` and `Pipeline`.
*   Training and comparing various classification models, including Decision Trees and K-Nearest Neighbors, against a baseline.
*   Applying hyperparameter tuning with `GridSearchCV` to optimize model performance.
*   Crucially, understanding and evaluating **model fairness** using `fairlearn`, demonstrating how to assess and analyze demographic parity with respect to sensitive attributes.
*   Building and evaluating end-to-end machine learning pipelines for streamlined workflows.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, preprocessing, classification models, and evaluation metrics)
*   `fairlearn` (for fairness metrics)

## Setup and Usage

To run this notebook locally, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Create and activate a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate
    ```
3.  **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```
    (Ensure you create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment after installing all the necessary libraries, especially `fairlearn`.)
4.  **Launch Jupyter**:
    ```bash
    jupyter lab
    ```
    or `jupyter notebook`
5.  Open `loan_sanction_prediction.ipynb` to view and execute the analysis.

## Datasets

*   `loan_sanction_train.csv` and `loan_sanction_test.csv` are datasets typically used in machine learning challenges for predicting loan approval status based on various applicant attributes.

---
