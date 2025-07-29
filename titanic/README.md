# Titanic Survival Prediction and Exploratory Data Analysis

This repository contains a Jupyter Notebook dedicated to analyzing the famous Titanic dataset. It performs comprehensive Exploratory Data Analysis (EDA) to uncover patterns and relationships within the data that might influence survival. Furthermore, it demonstrates a complete machine learning pipeline, including data preprocessing, feature engineering, model training, and evaluation, to predict passenger survival.

## Contents

*   `Titanic_EDA_and_ML_Pipeline.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `Titanic-Dataset.csv`: The dataset used in the analysis.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loading the `Titanic-Dataset.csv` and checking for missing values, data types, and overall structure.
*   **Target Variable Analysis**: Investigating the distribution of `Survived` passengers.
*   **Feature-wise Analysis**:
    *   **Age**: Distribution of passenger ages, mean age of survivors vs. non-survivors, and age group analysis.
    *   **Fare**: Mean fare prices for survivors and non-survivors.
    *   **Pclass (Passenger Class)**: Distribution of passenger classes and survival rates per class, highlighting the correlation between class and survival.
    *   **Embarked (Port of Embarkation)**: Distribution of embarkation ports for deceased passengers and survival rates per port.
*   **Feature Engineering**: Creation of a new feature `FamilySize` by combining `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard).

### 2. Data Preprocessing and Feature Engineering for Machine Learning

*   **Handling Missing Values**: Rows with missing values in key features are dropped for a clean dataset.
*   **Feature Transformation using `ColumnTransformer`**:
    *   **One-Hot Encoding**: `Sex` and `Embarked` are converted into numerical representations.
    *   **MinMaxScaler**: Numerical features like `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, and `FamilySize` are scaled to a common range.
    *   **KBinsDiscretizer**: The `Fare` feature is binned into discrete categories.
    *   **OrdinalEncoder**: `Age_grouped` (binned age categories) is converted into an ordinal numerical format.
*   **Feature Selection**: Irrelevant columns are dropped before model training.

### 3. Machine Learning Model Training and Evaluation

*   **Data Splitting**: The preprocessed data is split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).
*   **Model Training and Evaluation**:
    *   **Decision Tree Classifier**: A Decision Tree model is trained and evaluated using accuracy, confusion matrix, and precision-recall-f1 score.
    *   **Logistic Regression**: A Logistic Regression model is trained and evaluated using similar metrics for comparison.
*   **Hyperparameter Tuning (Decision Tree)**: `GridSearchCV` is used to find the optimal hyperparameters (`max_depth`, `criterion`) for the Decision Tree Classifier, demonstrating how to improve model performance.
*   **Machine Learning Pipeline**: Construction of a `Pipeline` incorporating `ColumnTransformer` (with `FeatureUnion` for combined transformations) and `DecisionTreeClassifier` to streamline the preprocessing and modeling steps.
*   **Cross-Validation**: Evaluation of the pipeline's performance using 5-fold cross-validation on the entire dataset.

## Key Learnings

*   Practical application of various EDA techniques to understand dataset characteristics.
*   Comprehensive data preprocessing methods using `sklearn.compose.ColumnTransformer` for mixed data types.
*   Feature engineering to create more informative predictors.
*   Training and evaluating common classification models like Decision Trees and Logistic Regression.
*   Implementing hyperparameter tuning with `GridSearchCV` to optimize model performance.
*   Building robust machine learning pipelines with `sklearn.pipeline.Pipeline` and `FeatureUnion` for efficient workflow management and cross-validation.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, preprocessing, classification models, and evaluation metrics)

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
5.  Open `Titanic_EDA_and_ML_Pipeline.ipynb` to view and execute the analysis.

## Dataset

The `Titanic-Dataset.csv` file contains historical passenger data from the Titanic voyage, including information on survival, age, class, sex, and more. This dataset is widely used for classification tasks in machine learning.

---
