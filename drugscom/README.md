# Drug Review Analysis and Prediction

This repository contains a Jupyter Notebook (`drug_review_analysis.ipynb`) that performs an in-depth analysis of drug reviews, focusing on patient ratings and conditions. The notebook demonstrates comprehensive data preprocessing, exploratory data analysis, and the implementation of machine learning models for both regression (predicting ratings) and classification (predicting conditions based on reviews and other features).

## Contents

*   `drug_review_analysis.ipynb`: The Jupyter Notebook containing all the Python code for data analysis and machine learning.
*   `drugsComTrain_raw.csv`: The training dataset of drug reviews.
*   `drugsComTest_raw.csv`: The test dataset of drug reviews.
*   `README.md`: This file, providing an overview of the project.

## Project Overview

The notebook covers the following key stages:

### 1. Data Loading and Initial Exploration

*   Loads raw training (`drugsComTrain_raw.csv`) and test (`drugsComTest_raw.csv`) datasets.
*   Inspects dataframes, checks for missing values, and identifies the overall structure.

### 2. Data Preprocessing and Feature Engineering

*   **Review Text Cleaning**: A custom function `remove_line_break` is applied to clean review text by removing line breaks.
*   **Review Length Analysis**: Calculates and visualizes the distribution of review lengths for both training and test sets.
*   **Handling Missing Values**: Drops rows with any missing values for cleaner analysis.
*   **Feature Removal**: Drops `uniqueID` and `date` columns as they are not used in the models.
*   **Exploratory Data Analysis (EDA) on Categorical Data**:
    *   Examines the count of unique conditions.
    *   Visualizes the top 10 most frequent conditions in both datasets using bar plots.
    *   Analyzes the mean rating for conditions with the most "useful" reviews, offering insights into highly-regarded conditions.

### 3. Machine Learning Model Implementation

The notebook implements two distinct machine learning tasks using `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer` for streamlined preprocessing:

#### 3.1. Rating Prediction (Regression Task)

*   **Objective**: Predict the `rating` based on drug information, review text, and `usefulCount`.
*   **Preprocessing Pipeline**:
    *   `OrdinalEncoder`: For `drugName` and `condition` (categorical features).
    *   `TfidfVectorizer`: For `review` text (text feature).
    *   `MinMaxScaler`: For `usefulCount` (numerical feature).
*   **Model**: `LinearRegression` is used to predict the numerical rating.
*   **Evaluation**: `r2_score` is used to assess the model's performance.

#### 3.2. Condition Prediction (Classification Task)

*   **Objective**: Predict the `condition` based on `drugName`, `review` text, and `usefulCount`.
*   **Preprocessing Pipeline**:
    *   `OrdinalEncoder`: For `drugName`.
    *   `TfidfVectorizer`: For `review` text.
    *   `MinMaxScaler`: For `usefulCount`.
*   **Model**: `LinearSVC` (Linear Support Vector Classifier) is used to classify the condition.
*   **Evaluation**: `accuracy_score` and `classification_report` are used to evaluate the classification model's performance, providing detailed metrics like precision, recall, and F1-score for each class.

## Key Learnings

*   Practical application of data loading, cleaning, and exploratory data analysis techniques for real-world datasets.
*   Effective use of `sklearn.compose.ColumnTransformer` to apply different preprocessing steps to different column types (categorical, text, numerical).
*   Implementation of `sklearn.pipeline.Pipeline` for building robust and reproducible machine learning workflows.
*   Demonstration of both regression and classification tasks within a single machine learning project.
*   Insight into text feature engineering (`TfidfVectorizer`) and its integration into a broader pipeline.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `scikit-learn` (for preprocessing, feature extraction, models, and metrics)

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
5.  Open `drug_review_analysis.ipynb` to view and execute the analysis.

## Datasets

*   `drugsComTrain_raw.csv` and `drugsComTest_raw.csv` are datasets containing patient drug reviews, along with associated conditions, ratings, and useful counts. These are commonly used datasets for natural language processing and machine learning tasks in healthcare.

---
