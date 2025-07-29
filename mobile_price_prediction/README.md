# Mobile Price Range Prediction and Feature Engineering

This repository contains a Jupyter Notebook (`mobile_price_prediction.ipynb`) that explores a dataset of mobile phone features and their corresponding price ranges. The project aims to understand how various hardware specifications and functionalities influence a phone's price bracket and to build machine learning models capable of predicting these price ranges. The notebook showcases comprehensive exploratory data analysis (EDA), advanced feature engineering techniques, and robust machine learning pipelines.

## Contents

*   `mobile_price_prediction.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `dataset.csv`: The dataset used in the analysis, detailing mobile phone specifications.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `dataset.csv` and checks for missing values and overall structure.
*   **Target Variable Analysis**: Analyzes the distribution of `price_range`.
*   **Feature Distribution Analysis**:
    *   Examines the distribution of `battery_power`.
    *   Creates a `battery_grouped` feature by binning `battery_power` and visualizes its distribution using a histogram.
*   **Feature Relationships**:
    *   Investigates the relationship between `ram`, `int_memory`, `four_g`, and `price_range` for phones with 4G and high RAM, using a scatter plot.
    *   Pivots data to analyze the count of phones within different `price_range` categories based on `touch_screen` presence and `int_memory_grouped` bins.

### 2. Machine Learning Model Training and Evaluation

The notebook explores various classification models and advanced data preprocessing techniques for predicting `price_range`:

*   **Data Splitting**: Splits the dataset into training and testing sets.
*   **Initial Model Comparison**:
    *   Trains and evaluates a **K-Nearest Neighbors Classifier** and a **Decision Tree Classifier** on the raw features.
    *   Compares their performance using `accuracy_score` and `confusion_matrix`.
    *   Includes a **Dummy Classifier** (most frequent strategy) as a baseline for performance comparison.
*   **Cross-Validation**: Performs 10-fold cross-validation for all three models (K-Nearest Neighbors, Decision Tree, Dummy) to obtain more robust accuracy estimates.
*   **Hyperparameter Tuning (Decision Tree)**: Uses `GridSearchCV` to optimize the Decision Tree Classifier's hyperparameters (`min_samples_leaf`, `criterion`).
*   **Feature Selection based on Correlation**: Identifies and uses the top 6 features most correlated with `price_range` to train models and assess their impact on accuracy.

### 3. Advanced Feature Engineering and Pipelines

The notebook demonstrates building sophisticated machine learning pipelines:

*   **Feature Transformation using `ColumnTransformer`**:
    *   Applies `StandardScaler` to features like `int_memory`, `ram`, `talk_time`.
    *   Applies `KBinsDiscretizer` to features like `mobile_wt` and `battery_power` for binning.
    *   Integrates these transformations into a `ColumnTransformer`.
*   **Pipelines for Preprocessing and Modeling**:
    *   Constructs a `Pipeline` combining `ColumnTransformer` with an optimized `DecisionTreeClassifier`.
    *   **Feature Selection within Pipeline**: Incorporates `SelectKBest` (using `f_classif`) within the pipeline and uses `GridSearchCV` to tune the number of best features (`kbest__k`) and bin sizes (`ct__kbins__n_bins`).
    *   **Dimensionality Reduction and Feature Combination**: Utilizes `FeatureUnion` to combine different processing paths:
        *   The `ColumnTransformer` output.
        *   `TruncatedSVD` for dimensionality reduction on the remaining features.
    *   Tunes the number of components for `TruncatedSVD` (`fu__svd__n_components`) within the pipeline using `GridSearchCV`.

## Key Learnings

*   Comprehensive exploratory data analysis for understanding the relationship between mobile features and price.
*   Effective data preprocessing, including feature binning, scaling, and handling mixed data types.
*   Training and comparing various classification models and setting appropriate baselines.
*   Advanced feature engineering techniques:
    *   Identifying top correlated features.
    *   Applying `ColumnTransformer` for tailored feature transformations.
    *   Integrating `SelectKBest` for feature selection.
    *   Utilizing `TruncatedSVD` for dimensionality reduction.
    *   Combining different processing paths with `FeatureUnion`.
*   Building robust machine learning pipelines with `sklearn.pipeline.Pipeline` and optimizing them using `GridSearchCV`.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, preprocessing, feature selection, dimensionality reduction, classification models, and evaluation metrics)

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
5.  Open `mobile_price_prediction.ipynb` to view and execute the analysis.

## Dataset

*   `dataset.csv` is a synthetic dataset commonly used for mobile price prediction, featuring various hardware specifications and a target `price_range` (0-3).

---
