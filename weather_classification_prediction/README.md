# Weather Classification and Predictive Modeling

This repository contains a Jupyter Notebook (`weather_classification_prediction.ipynb`) that performs a comprehensive analysis of weather data, focusing on classifying the main weather condition (`weather_main`). The notebook demonstrates advanced data preprocessing, feature engineering, and the application of various machine learning models and pipelines for multi-class classification.

## Contents

*   `weather_classification_prediction.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `weather_train.csv`: The training dataset of weather observations.
*   `weather_test.csv`: The test dataset of weather observations.
*   `class.csv`: (Implied or explicitly used for final evaluation in the notebook, containing true labels for the test set).
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Data Loading and Exploratory Data Analysis (EDA)

*   **Data Concatenation**: Combines `weather_train.csv` and `weather_test.csv` into a single DataFrame for unified processing.
*   **Missing Value Handling**: Drops rows with missing values.
*   **Data Cleaning**: Removes rows where `pressure` or `humidity` are zero (indicating potentially bad data).
*   **Temperature Analysis**:
    *   Examines the distribution of `temp`.
    *   Identifies cities with unusually high temperatures (top 5%).
    *   Calculates the mean temperature in Celsius and mean temperature range for snowy conditions.
    *   Analyzes weather conditions in cities experiencing high temperatures.
*   **Feature Engineering**: Creates a new feature `temp_range` (`temp_max` - `temp_min`).

### 2. Data Preprocessing and Feature Engineering for Machine Learning

*   **Feature Removal**: Drops irrelevant columns (`dt_iso`, `city_name`, `weather_description`, `weather_icon`, `weather_id`, `clouds_all`) from the dataset.
*   **Target Variable Mapping**: Maps the `weather_main` categorical target variable ('clouds', 'clear', 'rain') to numerical values (0, 1, 2).
*   **Normalization**: Applies L2 normalization to numerical features, demonstrating its impact on model performance.
*   **Dimensionality Reduction**: Utilizes `PCA` (Principal Component Analysis) for dimensionality reduction.
*   **Advanced Pipelines with `FeatureUnion`**: Combines `PCA` and `Normalizer` into a `FeatureUnion` within a `Pipeline` for flexible preprocessing.

### 3. Machine Learning Model Training and Evaluation

The notebook explores various classification models for predicting `weather_main`:

*   **Data Splitting**: Splits the processed data into training and testing sets, ensuring stratification based on the `weather_main` target.
*   **Model Training and Comparison**:
    *   **Decision Tree Classifier**: Trains and evaluates with a specified `max_depth`.
    *   **Logistic Regression**: Trains and evaluates with `solver='saga'` and increased `max_iter`.
    *   Evaluates performance using `accuracy_score` and `confusion_matrix` (with `ConfusionMatrixDisplay`) for both models.
*   **Cross-Validation**: Performs 10-fold cross-validation for both Decision Tree and Logistic Regression models for robust accuracy estimates.
*   **Hyperparameter Tuning**: Uses `GridSearchCV` to optimize the `Pipeline` by tuning `PCA` components and `DecisionTreeClassifier` criteria (`gini`, `entropy`).
*   **Final Prediction and Evaluation on Test Set**:
    *   Demonstrates predicting `weather_main` for the `weather_test.csv` dataset using the best trained pipeline.
    *   Evaluates the final accuracy against a ground truth `class.csv` (implied or provided separately).
*   **Regression for Classification (Approximation)**: Attempts to use `LinearRegression` for the multi-class classification task by training on numerical `weather_main` values and rounding predictions, assessing accuracy. This highlights an alternative, albeit approximate, approach.

## Key Learnings

*   Comprehensive data cleaning and exploratory data analysis of time-series/geographical weather data.
*   Advanced data preprocessing techniques like L2 normalization and `PCA`.
*   Building robust machine learning pipelines using `sklearn.pipeline.Pipeline` and `sklearn.pipeline.FeatureUnion` for combining different transformations.
*   Training and evaluating multi-class classification models (`DecisionTreeClassifier`, `LogisticRegression`).
*   Implementing hyperparameter tuning with `GridSearchCV` for optimal model performance.
*   Strategies for handling and predicting on separate test datasets, and evaluating predictions against true labels.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, preprocessing, dimensionality reduction, classification/regression models, and evaluation metrics)

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
5.  Open `weather_classification_prediction.ipynb` to view and execute the analysis.

## Datasets

*   `weather_train.csv` and `weather_test.csv` are datasets containing various weather observations for training and testing purposes, often used for weather forecasting or classification tasks.
*   `class.csv` (assumed) contains the true `weather_main` labels for the `weather_test.csv` data, used for final model evaluation.

---
