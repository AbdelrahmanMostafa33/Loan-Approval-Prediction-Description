# Loan Approval Prediction: Binary Classification with Python

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.5%2B-green)](https://pandas.pydata.org/)

## Project Overview

This project implements a machine learning pipeline for predicting loan approval (Approved/Rejected) based on applicant financial and demographic data. It uses binary classification techniques, addressing key challenges like class imbalance (~86% Approved cases) through SMOTE oversampling. Models compared include Logistic Regression and Decision Tree Classifier, evaluated on precision, recall, F1-score (with focus on the minority "Rejected" class), ROC-AUC, and confusion matrices.

Key features:
- **Data Preprocessing**: Handles categorical encoding (LabelEncoder), numerical scaling (StandardScaler), and missing values (none in this dataset).
- **EDA**: Visualizations for distributions, correlations, and class imbalance.
- **Imbalance Handling**: SMOTE for balanced training data.
- **Evaluation**: Stratified train-test split (80/20), with model comparisons.

The script `main.py` replicates the full analysis from the original Jupyter notebook, generating plots and metrics directly.

## Dataset

- **Source**: [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset) on Kaggle.
- **Size**: ~4,269 samples, 12 features + 1 target.
- **Features**:
  - Numerical: `no_of_dependents`, `income_annum`, `loan_amount`, `loan_term`, `cibil_score`, `residential_assets_value`, `commercial_assets_value`, `luxury_assets_value`, `bank_asset_value`.
  - Categorical: `education`, `self_employed`.
- **Target**: `loan_status` (binary: Approved=1, Rejected=0).
- **File**: `../data/loan_approval_dataset.csv` (download from Kaggle and place in the `data/` folder).

**Note**: `loan_id` is dropped as it's non-predictive.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure the dataset CSV is in `../data/loan_approval_dataset.csv` relative to `main.py`.
2. Run the script:
   ```
   python main.py
   ```

This will:
- Load and explore the data.
- Perform EDA (plots for distributions, correlations, etc.).
- Preprocess features.
- Train/evaluate models (with/without SMOTE).
- Output metrics, confusion matrices, feature importance, and ROC curves.
- Display a comparison table of model performance.

**Expected Output**:
- Console: Dataset stats, class distributions, classification reports, ROC-AUC scores, and model comparison table.
- Plots: Histograms, heatmaps, countplots, confusion matrices, feature importance bar chart, and ROC curve.

## Results Summary

| Model            | Precision (Rejected) | Recall (Rejected) | F1 (Rejected) | ROC-AUC |
|------------------|----------------------|-------------------|---------------|---------|
| LR Original     | 0.90                | 0.87             | 0.88         | 0.973  |
| DT Original     | 0.95                | 0.97             | 0.96         | 0.988  |
| LR SMOTE        | 0.88                | 0.92             | 0.90         | 0.973  |
| DT SMOTE        | 0.95                | 0.97             | 0.96         | 0.994  |

- **Best Model**: Decision Tree with SMOTE (97% accuracy, high recall for Rejected class to minimize false approvals).
- **Key Insights**:
  - CIBIL Score is the top predictor (importance ~0.45).
  - Assets and income strongly influence approvals.

## Dependencies

See `requirements.txt` for exact versions:

```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
imbalanced-learn>=0.9.0
```

## Acknowledgments

- Dataset from Kaggle.
- Built with scikit-learn and inspired by standard ML workflows for imbalanced classification.
