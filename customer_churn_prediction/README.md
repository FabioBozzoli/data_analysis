# Customer Churn Prediction and Fairness Analysis in Banking

This repository contains a Jupyter Notebook (`customer_churn_prediction.ipynb`) that conducts an extensive exploratory data analysis (EDA) and builds predictive models for customer churn in a banking context. The project aims to identify factors contributing to customers exiting a bank and to develop a machine learning model to predict churn, with a focus on assessing potential biases related to sensitive demographic features like gender and geography.

## Contents

*   `customer_churn_prediction.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `dataset.csv`: The dataset used in the analysis, containing anonymized bank customer data.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `dataset.csv` and checks for missing values and overall structure.
*   **Target Variable Analysis**: Analyzes the distribution of `Exited` customers (churned vs. retained).
*   **Feature Influence on Churn**:
    *   **Age**: Groups `Age` into bins and visualizes the churn rate across different age groups.
    *   **Balance**: Groups `Balance` into bins and, for customers aged 60 and above, analyzes the churn rate across balance groups.
    *   **EstimatedSalary**: Groups `EstimatedSalary` into bins and examines the average `CreditScore` for each `Gender` within these salary groups.
*   **Specific Segment Analysis**: Filters data for customers with credit cards, high balance (>100,000), and residing in France or Spain. It then compares the `CreditScore` means between these two geographical groups.

### 2. Data Preprocessing for Machine Learning

*   **Irrelevant Feature Removal**: Drops `RowNumber`, `CustomerId`, and `Surname` columns as they are unique identifiers and not useful for prediction.
*   **One-Hot Encoding**: Applies `pd.get_dummies` to convert all categorical features (e.g., `Geography`, `Gender`) into numerical representations for machine learning models.

### 3. Machine Learning Model Training and Evaluation

The notebook explores various classification models for predicting `Exited` status:

*   **Data Splitting**: Splits the dataset into training and testing sets, ensuring stratification based on the `Exited` target variable to maintain class balance.
*   **Model Training and Comparison**:
    *   **Decision Tree Classifier**: A Decision Tree model is trained and its performance is evaluated using `accuracy_score` and `f1_score`.
    *   **K-Nearest Neighbors Classifier**: A K-Nearest Neighbors model is trained and evaluated using similar metrics.
    *   **Dummy Classifier**: A `DummyClassifier` (using the "most frequent" strategy) is included as a baseline to compare against.
    *   Evaluates performance using `confusion_matrix` and `ConfusionMatrixDisplay` for each model to visualize prediction errors.
*   **Cross-Validation**: Performs 10-fold cross-validation for both Decision Tree and K-Nearest Neighbors models for robust accuracy estimates.
*   **Fairness Analysis (Gender Bias)**:
    *   Calculates and compares the churn rates and prediction accuracies for male and female customers separately. This highlights potential disparities in how the model performs for different demographic groups.
*   **Impact of Sensitive Feature Removal**:
    *   Retrains both Decision Tree and K-Nearest Neighbors models after explicitly dropping 'Gender' related columns (`Gender_Male`, `Gender_Female`).
    *   Re-evaluates `accuracy_score` and `f1_score` to observe the impact of removing sensitive attributes on overall model performance, providing insights into fairness-accuracy trade-offs.

## Key Learnings

*   Comprehensive exploratory data analysis to identify churn drivers in a banking context.
*   Effective data preprocessing for machine learning, including handling irrelevant features and one-hot encoding.
*   Training and evaluating common classification models (`DecisionTreeClassifier`, `KNeighborsClassifier`).
*   Utilizing `f1_score` and `confusion_matrix` for evaluating imbalanced classification tasks (churn often is).
*   Understanding and assessing **model fairness**, specifically examining performance disparities across gender groups.
*   Analyzing the impact and trade-offs of removing sensitive features on model accuracy and fairness.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, classification models, and evaluation metrics)

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
    (Ensure you create a `requirements.txt` file by running `pip freeze > requirements.txt` in your activated environment after installing all the necessary libraries.)
4.  **Launch Jupyter**:
    ```bash
    jupyter lab
    ```
    or `jupyter notebook`
5.  Open `customer_churn_prediction.ipynb` to view and execute the analysis.

## Dataset

*   `dataset.csv` is a dataset containing anonymized information about bank customers, including various attributes and a target variable indicating whether the customer has "Exited" (churned).

---
