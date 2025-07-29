# Fair Machine Learning for Hiring Decisions

This repository contains a Jupyter Notebook that explores the application of machine learning to simulated hiring decision data, with a particular focus on fairness metrics. The notebook demonstrates a typical machine learning pipeline from data loading and preprocessing to model training, evaluation, and an analysis of demographic parity concerning a sensitive attribute (gender).

## Contents

*   `hiring_decisions.ipynb` (or `.py` if it's a script): The Jupyter Notebook containing the Python code for the analysis.
*   `hiring_decisions.csv`: The dataset used in the analysis (simulated data).
*   `README.md`: This file, providing an overview of the project.

## Project Overview

In this notebook, we:

1.  **Load and Explore Data**: Import the `hiring_decisions.csv` dataset, inspect its structure, check for missing values, and analyze the distribution of the target variable (`HiringDecision`).
2.  **Data Preprocessing**:
    *   Drop irrelevant columns (`id`).
    *   Apply `StandardScaler` to numerical features using `ColumnTransformer` for robust scaling.
3.  **Model Training and Evaluation (with Gender Feature)**:
    *   Split the data into training and testing sets.
    *   Train a `SGDClassifier` model on the preprocessed data, including the 'Gender' feature.
    *   Evaluate the model's performance using standard metrics like accuracy and F1-score.
    *   Analyze the confusion matrix to understand prediction errors.
4.  **Fairness Analysis**:
    *   Calculate and compare the hiring rates for different gender groups (women vs. men) to identify potential biases.
    *   Utilize `fairlearn`'s `demographic_parity_ratio` to quantify fairness, assessing whether the model's prediction rate is similar across different demographic groups.
5.  **Model Retraining (without Gender Feature)**:
    *   Re-train the `SGDClassifier` model after intentionally excluding the 'Gender' feature from the input data.
    *   Re-evaluate the model's F1-score to observe the impact of removing the sensitive attribute on overall performance.

## Key Learnings

*   Understanding a basic machine learning pipeline for classification.
*   The importance of analyzing model performance not just overall, but also across different subgroups.
*   Introduction to fairness metrics, specifically demographic parity, and how they can reveal biases in AI systems.
*   The trade-offs and considerations when dealing with sensitive attributes in machine learning models for ethical AI development.

## Technologies Used

*   Python
*   Jupyter Notebook
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `scikit-learn` (for model selection, preprocessing, classification, and metrics)
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
5.  Open `hiring_decisions.ipynb` (or the equivalent `.py` file) to view and execute the analysis.

## Dataset

The `hiring_decisions.csv` file is a simulated dataset. It contains various features related to job applicants and their corresponding hiring decisions. **Note: This dataset is synthetic and does not represent real-world hiring data.**

---
