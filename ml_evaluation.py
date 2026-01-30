import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, 
    GridSearchCV, train_test_split
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(csv_file):
    """Load dataset and handle missing values"""
    # Load data
    df = pd.read_csv(csv_file)
    
    # Handle missing values - fill numeric with median, categorical with mode
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Convert categorical features to numeric if needed
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def compare_cross_validation_methods(X, y):
    """Compare K-Fold vs Stratified K-Fold"""
    print("=" * 60)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 60)
    
    # Simple Random Forest for comparison
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # K-Fold Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    kfold_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')
    
    # Stratified K-Fold Cross-Validation
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    stratified_scores = cross_val_score(rf, X, y, cv=stratified_kfold, scoring='accuracy')
    
    print(f"K-Fold CV Accuracy: {kfold_scores.mean():.4f} (+/- {kfold_scores.std() * 2:.4f})")
    print(f"Stratified K-Fold CV Accuracy: {stratified_scores.mean():.4f} (+/- {stratified_scores.std() * 2:.4f})")
    print("\nStratified K-Fold maintains class distribution in each fold,")
    print("making it more reliable for imbalanced datasets.\n")

def evaluate_models_with_cv(X, y):
    """Evaluate multiple models using cross-validation"""
    print("=" * 60)
    print("MODEL EVALUATION WITH CROSS-VALIDATION")
    print("=" * 60)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42))
        ]),
        'Decision Tree': DecisionTreeClassifier(random_state=42)
    }
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Evaluate each model
    results = {}
    for name, model in models.items():
        # Calculate multiple metrics
        accuracy = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        precision = cross_val_score(model, X, y, cv=cv, scoring='precision_weighted')
        recall = cross_val_score(model, X, y, cv=cv, scoring='recall_weighted')
        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy.mean():.4f} (+/- {accuracy.std() * 2:.4f})")
        print(f"  Precision: {precision.mean():.4f} (+/- {precision.std() * 2:.4f})")
        print(f"  Recall:    {recall.mean():.4f} (+/- {recall.std() * 2:.4f})")
        print(f"  F1-Score:  {f1.mean():.4f} (+/- {f1.std() * 2:.4f})")
    
    return results

def hyperparameter_tuning_rf(X, y):
    """Perform hyperparameter tuning for Random Forest"""
    print("=" * 60)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Setup GridSearchCV with StratifiedKFold
    rf = RandomForestClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit grid search
    print("Performing grid search... (this may take a moment)")
    grid_search.fit(X, y)
    
    # Print results
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def final_evaluation(best_rf, X, y):
    """Final evaluation on test set"""
    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)
    
    # Split data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train best model on training set
    best_rf.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = best_rf.predict(X_test)
    
    # Calculate final metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nFinal Test Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Feature importance
    if hasattr(best_rf, 'feature_importances_'):
        feature_names = X.columns if hasattr(X, 'columns') else [f'feature_{i}' for i in range(X.shape[1])]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Feature Importances:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")

def main():
    """Main execution function"""
    # Recommended Kaggle datasets:
    # 1. Titanic: https://www.kaggle.com/c/titanic/data (use train.csv, rename 'Survived' to 'target')
    # 2. Heart Disease: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
    # 3. Wine Quality: https://www.kaggle.com/datasets/uciml/red-wine-quality-datasets
    csv_file = 'your_dataset.csv'
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        X, y = load_and_preprocess_data(csv_file)
        print(f"Dataset shape: {X.shape}")
        print(f"Target classes: {np.unique(y)}")
        
        # Compare cross-validation methods
        compare_cross_validation_methods(X, y)
        
        # Evaluate models with cross-validation
        cv_results = evaluate_models_with_cv(X, y)
        
        # Hyperparameter tuning for Random Forest
        best_rf = hyperparameter_tuning_rf(X, y)
        
        # Final evaluation
        final_evaluation(best_rf, X, y)
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file}'")
        print("Please ensure the CSV file exists and update the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()