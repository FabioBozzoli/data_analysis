# Weather Prediction and Exploratory Data Analysis

This repository contains a Jupyter Notebook (`weather_prediction_analysis.ipynb`) that conducts an exploratory data analysis (EDA) and builds predictive models for weather patterns, specifically focusing on forecasting whether it will rain tomorrow. The notebook demonstrates various data processing techniques, feature engineering, and the application of machine learning models for classification.

## Contents

*   `weather_prediction_analysis.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `weather.csv`: The dataset used in the analysis, containing various weather observations.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `weather.csv` (assuming it's semicolon-separated) and inspects for missing values and overall structure.
*   **Target Variable Analysis**: Analyzes the distribution of `RainTomorrow`.
*   **Location Analysis**: Visualizes the distribution of observations by `Location`.
*   **Monthly Data Coverage**: Pivots data to show the number of observations per `Location` and `Month`, filling missing combinations with zeros.
*   **Humidity Analysis**:
    *   Calculates `min_humidity` and `max_humidity` from `Humidity9am` and `Humidity3pm`.
    *   Pivots data to find minimum and maximum humidity per `Location` and `Month`.
*   **Temperature Range Analysis**:
    *   Calculates `TemperatureRange` (`MaxTemp` - `MinTemp`).
    *   Pivots data to find the maximum `TemperatureRange` per `Month` and visualizes it.

### 2. Data Preprocessing and Feature Engineering

*   **Target and Feature Mapping**: Converts `RainToday` and `RainTomorrow` from 'Yes'/'No' to 1/0 numerical values.
*   **Feature Engineering**:
    *   Calculates `MeanTemp` from `MaxTemp` and `MinTemp`.
    *   Calculates `MeanWind` from `WindSpeed9am` and `WindSpeed3pm`.
    *   Drops original `MaxTemp`, `MinTemp`, `WindSpeed9am`, `WindSpeed3pm` columns.
*   **Handling Missing `Cloud3pm` Data**: Demonstrates a strategy to impute missing `Cloud3pm` values by training an `SVR` model on non-missing data and using it to predict the missing ones. This highlights a more advanced imputation technique.

### 3. Machine Learning Model Training and Evaluation

The notebook explores various classification models for predicting `RainTomorrow`:

*   **Data Splitting**: Splits the dataset into training and testing sets, ensuring stratification based on the `RainTomorrow` target variable.
*   **Model Training and Comparison**:
    *   **Decision Tree Classifier**: A Decision Tree model is trained and evaluated using `accuracy_score` and `ConfusionMatrixDisplay`.
    *   **Logistic Regression**: A Logistic Regression model is trained and evaluated using similar metrics.
*   **Cross-Validation**: Performs 5-fold cross-validation for both Decision Tree and Logistic Regression models on different feature sets (including and excluding engineered features) for robust accuracy estimates.
*   **SVR for Classification (Approximation)**: Demonstrates training an `SVR` (Support Vector Regressor) model for the classification task and then rounding its predictions to get binary outcomes, assessing its accuracy. This shows an alternative approach for binary classification using a regression model.

## Key Learnings

*   In-depth exploratory data analysis for understanding meteorological patterns and relationships.
*   Practical data preprocessing techniques, including mapping categorical values, handling missing data (including an advanced imputation method with SVR), and feature engineering.
*   Training and evaluating common classification models like Decision Trees and Logistic Regression.
*   Utilizing cross-validation for robust model assessment.
*   Exploring alternative modeling approaches, such as using a regressor (`SVR`) for a binary classification task.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, preprocessing, classification models, regression models, and evaluation metrics)

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
5.  Open `weather_prediction_analysis.ipynb` to view and execute the analysis.

## Dataset

*   `weather.csv` is a dataset containing daily weather observations from various locations, suitable for tasks like weather forecasting and pattern analysis.

---
