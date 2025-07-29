# Student Performance Analysis and Prediction

This repository contains a Jupyter Notebook (`student_performance_analysis.ipynb`) that conducts an in-depth exploratory data analysis (EDA) and builds predictive models on a student performance dataset. The goal is to understand factors influencing student grades and to predict final grades (`G3`) based on various demographic, social, and academic attributes.

## Contents

*   `student_performance_analysis.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `bdastudents.csv`: The dataset used in the analysis, containing student information and grades.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `bdastudents.csv` (assuming it's semicolon-separated) and checks for missing values.
*   **Feature Exploration**:
    *   Analyzes the distribution of `school` and `sex` features.
    *   Calculates `GRate` (grade improvement from `G1` to `G2`) and visualizes its average change by `age`.
*   **Grade Distribution Analysis**: Pivots data to show the distribution of final grades (`G3`) by `sex`, and later by `school` and `sex`, calculating the mean `G3` score for each group.
*   **Parental Education Analysis**: Visualizes the distribution of mother's (`Medu`) and father's (`Fedu`) education levels using histograms, comparing their distributions.

### 2. Machine Learning Model Building and Evaluation

The notebook explores different feature sets for predicting the final grade (`G3`) using a `LogisticRegression` model, followed by `DecisionTreeClassifier` and cross-validation for more robust evaluation.

*   **Feature Set Definition**: Defines three main feature sets:
    *   `numeric`: All original numerical columns (excluding `G3`).
    *   `reduced`: Only `G1` and `G2` (previous grades).
    *   `less_reduced`: Numerical columns excluding `G1`, `G2`, and `G3`.
*   **Data Splitting**: Splits data into training and testing sets for each feature set.
*   **Model Training and Evaluation (Logistic Regression)**:
    *   Trains a `LogisticRegression` model on `less_reduced`, `reduced`, and `numeric` feature sets.
    *   Evaluates performance using `accuracy_score` and `confusion_matrix` for each.
*   **Cross-Validation (Decision Tree)**:
    *   Performs 10-fold cross-validation using `DecisionTreeClassifier` on all three feature sets (`less_reduced`, `reduced`, `numeric`) for more robust accuracy estimates.
    *   **Feature Engineering and Impact Analysis**:
        *   Discretizes `age` into 3 bins and checks the impact on Decision Tree accuracy.
        *   One-hot encodes the binned `age` feature and re-evaluates.
        *   Includes `Mjob` and `Fjob` (mother's and father's job) and one-hot encodes them to see their effect on accuracy.
*   **School-Specific Models**:
    *   Splits the numerical data by `school` (GP and MS).
    *   Trains separate `DecisionTreeClassifier` models for each school.
    *   Evaluates the accuracy of these school-specific models when predicting on a combined test set.

## Key Learnings

*   Insights into how demographic factors (age, gender, parental education) and previous grades influence final student performance.
*   Comparative analysis of different feature sets for predicting grades, highlighting the importance of feature selection.
*   Application of various data preprocessing techniques, including binning and one-hot encoding, and their impact on model performance.
*   Training and evaluating classification models (`LogisticRegression`, `DecisionTreeClassifier`).
*   Utilizing cross-validation for robust model assessment.
*   Exploring the concept of building separate models for different subgroups (e.g., by school) to potentially capture unique patterns.

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
5.  Open `student_performance_analysis.ipynb` to view and execute the analysis.

## Dataset

*   `bdastudents.csv` is a dataset containing student-related information (demographics, social, and school-related features) and their academic performance in two Portuguese schools. The grades (`G1`, `G2`, `G3`) represent period grades and the final grade.

---
