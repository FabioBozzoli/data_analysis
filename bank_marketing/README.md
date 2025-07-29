# Bank Marketing Campaign Analysis and Prediction

This repository contains a Jupyter Notebook (`bank_marketing_analysis.ipynb`) that conducts an extensive exploratory data analysis (EDA) on a bank marketing campaign dataset. The primary goal is to understand the factors influencing a client's decision to subscribe to a term deposit. Furthermore, the notebook builds and evaluates machine learning models to predict subscription outcomes.

## Contents

*   `bank_marketing_analysis.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `BankMarketingDataSet.csv`: The dataset used in the analysis, containing client data and campaign results.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `BankMarketingDataSet.csv` (assuming it's semicolon-separated) and checks for missing values.
*   **Target Variable Analysis**: Analyzes the distribution of the target variable `y` (client subscribed to term deposit).
*   **Feature-wise Analysis and Pivots**:
    *   Examines the distribution of `age`.
    *   Investigates the influence of `marital` status on subscription rates using pivot tables and bar plots.
    *   Creates `age_grouped` bins and analyzes subscription rates across different age groups.
    *   Performs a detailed analysis of subscription percentages (`yes` vs `no`) for each individual age, visualizing these trends.

### 2. Data Preprocessing for Machine Learning

*   **Target Variable Mapping**: Converts the target variable `y` from 'yes'/'no' strings to 1/0 numerical values.
*   **Irrelevant Feature Removal**: Drops the `id` column.
*   **Handling Unknown Values**: Removes rows where several key categorical features (`housing`, `loan`, `job`, `marital`, `education`, `default`) have 'unknown' values, ensuring a cleaner dataset for modeling.
*   **One-Hot Encoding**: Applies `pd.get_dummies` to convert all remaining categorical features into numerical representations for machine learning models.

### 3. Machine Learning Model Training and Evaluation

*   **Data Splitting**: Splits the preprocessed data into training and testing sets, ensuring stratification based on the target variable `y`.
*   **Model Training and Comparison**:
    *   **Decision Tree Classifier**: A Decision Tree model is trained and its performance is evaluated using `accuracy_score` and `confusion_matrix`.
    *   **K-Nearest Neighbors Classifier**: A K-Nearest Neighbors model is trained and evaluated using similar metrics.
*   **Cross-Validation**: Performs 10-fold cross-validation for both Decision Tree and K-Nearest Neighbors models on the entire processed dataset to get a more robust estimate of performance.
*   **Numerical Features-Only Model**: Trains and evaluates the K-Nearest Neighbors model using only numerical columns from the original dataset, providing a baseline for the impact of categorical features.

## Key Learnings

*   Comprehensive exploratory data analysis, including value counts, histograms, pivot tables, and various plotting techniques, to derive insights from a marketing dataset.
*   Effective data cleaning strategies, specifically handling 'unknown' values in categorical features.
*   Preparation of data for machine learning, including target variable mapping and one-hot encoding of categorical features.
*   Implementation and evaluation of fundamental classification models like Decision Tree and K-Nearest Neighbors.
*   Application of cross-validation for robust model performance estimation.
*   Understanding the influence of different feature sets (e.g., only numerical) on model accuracy.

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
5.  Open `bank_marketing_analysis.ipynb` to view and execute the analysis.

## Dataset

*   `BankMarketingDataSet.csv` is a dataset commonly used for direct marketing campaigns of a Portuguese banking institution. The goal is to predict if a client will subscribe to a term deposit.

---
