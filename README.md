# ML Model Evaluation with Cross-Validation

Complete Python solution for model evaluation using scikit-learn with cross-validation and Random Forest hyperparameter tuning.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate sample data (optional):
```bash
python generate_sample_data.py
```

3. Run the evaluation:
```bash
python ml_evaluation.py
```

## Features

- **Data Loading & Preprocessing**: Handles missing values and categorical encoding
- **Cross-Validation Comparison**: K-Fold vs Stratified K-Fold
- **Multi-Model Evaluation**: Random Forest, SVM, Decision Tree
- **Hyperparameter Tuning**: GridSearchCV for Random Forest optimization
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score
- **Feature Importance**: Analysis of top contributing features

## Dataset Requirements

Your CSV file should have:
- A column named 'target' containing class labels
- Feature columns (numeric or categorical)
- Missing values are handled automatically

## Output

The script provides:
- Cross-validation scores for each model
- Best hyperparameters for Random Forest
- Final evaluation metrics on test set
- Feature importance rankings

Data loading and preprocessing (load_and_preprocess_data)

What it does:

Reads a CSV file into a DataFrame.

Fills missing values:

Numeric columns → median.

Categorical columns → most frequent value (mode).

Splits data into:

X = features (all columns except target)

y = target column

Converts categorical features into numeric using one-hot encoding (pd.get_dummies).

Assumptions:

Your dataset must contain a column named target.

This is a classification problem.

Risk:

Blindly filling missing values can bias the data.

One-hot encoding can explode feature count.

Compare K-Fold vs Stratified K-Fold (compare_cross_validation_methods)

What it does:

Trains a RandomForestClassifier using:

KFold cross-validation (5 splits)

StratifiedKFold cross-validation (5 splits)

Computes accuracy for both.

Prints mean accuracy and variability.

Purpose:

Demonstrates why StratifiedKFold is better for imbalanced datasets (keeps class proportions in each fold).

Evaluate multiple models with cross-validation (evaluate_models_with_cv)

Models evaluated:

Random Forest

SVM (with StandardScaler in a Pipeline)

Decision Tree

What it does:
For each model:

Runs 5-fold Stratified cross-validation.

Computes:

Accuracy

Precision (weighted)

Recall (weighted)

F1-score (weighted)

Prints mean and variance of each metric.

Why pipeline is used for SVM:

Scaling is mandatory for SVM.

Prevents data leakage by scaling inside CV folds.

Time complexity:

Expensive: 3 models × 4 metrics × 5 folds = 60 model trainings.

Hyperparameter tuning (hyperparameter_tuning_rf)

What it does:

Uses GridSearchCV to tune Random Forest parameters:

n_estimators

max_depth

min_samples_split

min_samples_leaf

Uses StratifiedKFold (5 folds).

Evaluates combinations using accuracy.

Finds the best hyperparameters.

This is brute-force search.
Worst-case models trained =
3×4×3×3 × 5 folds = 540 Random Forest trainings.

This is computationally heavy.

Final evaluation (final_evaluation)

What it does:

Splits dataset into:

80% training

20% test (stratified)

Trains best Random Forest model on training set.

Predicts on test set.

Computes:

Accuracy

Precision

Recall

F1-score

Prints top 10 most important features using feature_importances_.

Purpose:

Gives an unbiased estimate of final model performance.

Main function (main)

Execution flow:

Loads dataset.

Runs cross-validation comparison.

Evaluates three models.

Tunes Random Forest.

Evaluates best model on test data.

Handles:

File not found errors.

Generic runtime errors.

Overall purpose of the code

This is a complete machine learning pipeline for a classification problem that:

Preprocesses data

Compares CV strategies

Compares multiple models

Tunes hyperparameters

Evaluates final model

Explains feature importance

In short:
It is a demonstration of best practices in supervised ML evaluation using cross-validation and grid search.

Key limitations:

Assumes column name target.

Not suitable for very large datasets (GridSearchCV will be too slow).

No feature selection.

No imbalance handling (SMOTE, class weights).

Uses accuracy as tuning metric (bad choice for highly imbalanced data).

Conclusion:
This code builds, validates, tunes, and evaluates classification models using cross-validation and grid search in a structured, production-style workflow.