# Housing Price Prediction and Feature Engineering

This repository contains a Jupyter Notebook (`housing_price_prediction.ipynb`) that conducts an extensive exploratory data analysis (EDA) and builds predictive regression models for housing prices. The project aims to understand the factors influencing house prices and to develop a robust machine learning pipeline for accurate predictions.

## Contents

*   `housing_price_prediction.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `dataset.csv`: The dataset used in the analysis, containing various housing features.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Exploratory Data Analysis (EDA)

*   **Data Loading and Initial Inspection**: Loads `dataset.csv` and checks for missing values and overall structure.
*   **Feature Influence Analysis**:
    *   Calculates the percentage difference in `price` for houses on a `mainroad` versus the overall average.
    *   Analyzes the average `price` based on `guestroom` and `basement` presence for houses on a `mainroad` using a pivot table.
*   **Area Analysis**:
    *   Examines the distribution of `area` using a histogram.
    *   Creates `area_grouped` bins and visualizes their distribution.
    *   Investigates the distribution of `bedrooms` within each `area_grouped` category.
*   **Feature Relationships Visualization**:
    *   Plots `price` vs `area` for houses with at least 2 bedrooms and 2 bathrooms, color-coded by `airconditioning` presence, to visualize price trends and feature impact.

### 2. Data Preprocessing for Machine Learning

*   **Binary Feature Mapping**: Converts 'yes'/'no' categorical features (`mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`) to 1/0 numerical values.
*   **One-Hot Encoding**: Applies `pd.get_dummies` to handle any other implicit categorical features (though in the provided code, most are already numerical or binary).

### 3. Machine Learning Model Training and Evaluation

The notebook explores various regression models and advanced data preprocessing techniques for predicting `price`:

*   **Data Splitting**: Splits the dataset into training and testing sets.
*   **Initial Model Comparison**:
    *   Trains and evaluates a **Linear Regression** model.
    *   Trains and evaluates a **SGDRegressor** model.
    *   Compares their performance using `mean_squared_log_error` (MSLE) and `r2_score`.
*   **Cross-Validation**: Performs 5-fold cross-validation for both `LinearRegression` and `SGDRegressor` for robust performance estimates.
*   **Hyperparameter Tuning (SGDRegressor)**: Uses `GridSearchCV` to optimize `SGDRegressor`'s hyperparameters (`loss`, `penalty`).
*   **Feature Selection based on Correlation**: Identifies and uses the top 6 features most correlated with `price` to train models and assess their impact on accuracy.

### 4. Advanced Feature Engineering and Pipelines

The notebook demonstrates building sophisticated machine learning pipelines:

*   **Custom Feature Engineering (`FunctionTransformer`)**: Defines a `add_area_bins` function to add a binned `area` feature, and integrates it into a pipeline using `FunctionTransformer`.
*   **Feature Transformation using `ColumnTransformer`**:
    *   Applies `StandardScaler` to numerical features.
    *   Applies `KBinsDiscretizer` to specific features for binning.
    *   Integrates these transformations into a `ColumnTransformer`.
*   **Pipelines for Preprocessing and Modeling**:
    *   Constructs a `Pipeline` combining custom feature engineering, `ColumnTransformer`, and `SGDRegressor`.
    *   **Alternative Feature Combination (`FeatureUnion`)**: Uses `FeatureUnion` to combine `Normalizer` and `KBinsDiscretizer` for `area` into a single pipeline step before `SGDRegressor`.
    *   **Feature Selection within Pipeline**: Incorporates `SelectKBest` (using `f_regression`) within the pipeline and uses `GridSearchCV` to tune the number of best features (`kbest__k`) and `add_bins` parameters (`add_bins__kw_args` for `n_bins`).
*   **Fine-tuned Binning in Pipeline**: Explores a pipeline with `KBinsDiscretizer` specifically on `bedrooms`, `bathrooms`, `stories` and `StandardScaler` on `area`.

## Key Learnings

*   Comprehensive exploratory data analysis for understanding the impact of various house features on price.
*   Effective data preprocessing, including binary feature mapping and handling mixed data types.
*   Training and comparing various regression models (`LinearRegression`, `SGDRegressor`).
*   Advanced feature engineering techniques:
    *   Creating custom binned features with `FunctionTransformer`.
    *   Tailored transformations using `ColumnTransformer`.
    *   Integrating feature selection (`SelectKBest`) into pipelines.
    *   Combining different preprocessing strategies with `FeatureUnion`.
*   Building robust machine learning pipelines with `sklearn.pipeline.Pipeline` and optimizing them using `GridSearchCV`.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn` (for data splitting, preprocessing, feature selection, regression models, and evaluation metrics)

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
5.  Open `housing_price_prediction.ipynb` to view and execute the analysis.

## Dataset

*   `dataset.csv` is a dataset containing various features of houses (e.g., area, number of bedrooms/bathrooms, amenities) and their corresponding prices, commonly used for regression tasks.

---
