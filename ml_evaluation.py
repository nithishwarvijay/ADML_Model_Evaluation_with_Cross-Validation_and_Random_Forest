import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, KFold, 
    GridSearchCV, train_test_split, validation_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

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
    """Compare K-Fold vs Stratified K-Fold with visualization"""
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
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot comparison
    cv_data = pd.DataFrame({
        'K-Fold': kfold_scores,
        'Stratified K-Fold': stratified_scores
    })
    
    axes[0].boxplot([kfold_scores, stratified_scores], labels=['K-Fold', 'Stratified K-Fold'])
    axes[0].set_title('Cross-Validation Methods Comparison')
    axes[0].set_ylabel('Accuracy Score')
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot with error bars
    methods = ['K-Fold', 'Stratified K-Fold']
    means = [kfold_scores.mean(), stratified_scores.mean()]
    stds = [kfold_scores.std(), stratified_scores.std()]
    
    axes[1].bar(methods, means, yerr=stds, capsize=5, alpha=0.7, color=['skyblue', 'lightcoral'])
    axes[1].set_title('Mean Accuracy with Standard Deviation')
    axes[1].set_ylabel('Accuracy Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return kfold_scores, stratified_scores

def evaluate_models_with_cv(X, y):
    """Evaluate multiple models using cross-validation with visualization"""
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
    metrics_data = {'Model': [], 'Metric': [], 'Score': []}
    
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
        
        # Store data for plotting
        for metric_name, scores in [('Accuracy', accuracy), ('Precision', precision), 
                                   ('Recall', recall), ('F1-Score', f1)]:
            for score in scores:
                metrics_data['Model'].append(name)
                metrics_data['Metric'].append(metric_name)
                metrics_data['Score'].append(score)
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {accuracy.mean():.4f} (+/- {accuracy.std() * 2:.4f})")
        print(f"  Precision: {precision.mean():.4f} (+/- {precision.std() * 2:.4f})")
        print(f"  Recall:    {recall.mean():.4f} (+/- {recall.std() * 2:.4f})")
        print(f"  F1-Score:  {f1.mean():.4f} (+/- {f1.std() * 2:.4f})")
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Convert to DataFrame for easier plotting
    df_metrics = pd.DataFrame(metrics_data)
    
    # 1. Box plot for all metrics
    sns.boxplot(data=df_metrics, x='Model', y='Score', hue='Metric', ax=axes[0,0])
    axes[0,0].set_title('Model Performance Comparison (All Metrics)')
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 2. Heatmap of mean scores
    mean_scores = df_metrics.groupby(['Model', 'Metric'])['Score'].mean().unstack()
    sns.heatmap(mean_scores, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0,1])
    axes[0,1].set_title('Mean Performance Scores Heatmap')
    
    # 3. Bar plot focusing on accuracy
    accuracy_data = df_metrics[df_metrics['Metric'] == 'Accuracy']
    sns.barplot(data=accuracy_data, x='Model', y='Score', ax=axes[1,0], palette='viridis')
    axes[1,0].set_title('Accuracy Comparison')
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # 4. Violin plot for Random Forest (detailed view)
    rf_data = df_metrics[df_metrics['Model'] == 'Random Forest']
    sns.violinplot(data=rf_data, x='Metric', y='Score', ax=axes[1,1], palette='Set2')
    axes[1,1].set_title('Random Forest Performance Distribution')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return results

def hyperparameter_tuning_rf(X, y):
    """Perform hyperparameter tuning for Random Forest with visualization"""
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
    
    # Create validation curves for key parameters
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. n_estimators validation curve
    param_range = [10, 50, 100, 150, 200, 250]
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42), X, y,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    axes[0,0].plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
    axes[0,0].plot(param_range, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    axes[0,0].fill_between(param_range, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                          np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    axes[0,0].fill_between(param_range, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                          np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1)
    axes[0,0].set_xlabel('n_estimators')
    axes[0,0].set_ylabel('Accuracy Score')
    axes[0,0].set_title('Validation Curve: n_estimators')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. max_depth validation curve
    param_range = [5, 10, 15, 20, 25, 30]
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42), X, y,
        param_name='max_depth', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    axes[0,1].plot(param_range, np.mean(train_scores, axis=1), 'o-', label='Training score')
    axes[0,1].plot(param_range, np.mean(test_scores, axis=1), 'o-', label='Cross-validation score')
    axes[0,1].fill_between(param_range, np.mean(train_scores, axis=1) - np.std(train_scores, axis=1),
                          np.mean(train_scores, axis=1) + np.std(train_scores, axis=1), alpha=0.1)
    axes[0,1].fill_between(param_range, np.mean(test_scores, axis=1) - np.std(test_scores, axis=1),
                          np.mean(test_scores, axis=1) + np.std(test_scores, axis=1), alpha=0.1)
    axes[0,1].set_xlabel('max_depth')
    axes[0,1].set_ylabel('Accuracy Score')
    axes[0,1].set_title('Validation Curve: max_depth')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Grid search results heatmap (top combinations)
    results_df = pd.DataFrame(grid_search.cv_results_)
    top_results = results_df.nlargest(20, 'mean_test_score')[['params', 'mean_test_score']]
    
    # Extract parameter combinations for visualization
    param_combinations = []
    scores = []
    for idx, row in top_results.iterrows():
        params = row['params']
        param_str = f"n_est:{params['n_estimators']}\nmax_d:{params['max_depth']}\nmin_split:{params['min_samples_split']}"
        param_combinations.append(param_str)
        scores.append(row['mean_test_score'])
    
    axes[1,0].barh(range(len(param_combinations)), scores, color='lightgreen')
    axes[1,0].set_yticks(range(len(param_combinations)))
    axes[1,0].set_yticklabels(param_combinations, fontsize=8)
    axes[1,0].set_xlabel('Cross-Validation Score')
    axes[1,0].set_title('Top 20 Parameter Combinations')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Feature importance of best model
    best_rf = grid_search.best_estimator_
    best_rf.fit(X, y)
    
    if hasattr(X, 'columns'):
        feature_names = X.columns
        importances = best_rf.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        axes[1,1].bar(range(len(indices)), importances[indices])
        axes[1,1].set_xlabel('Features')
        axes[1,1].set_ylabel('Importance')
        axes[1,1].set_title('Top 15 Feature Importances (Best RF)')
        axes[1,1].set_xticks(range(len(indices)))
        axes[1,1].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return grid_search.best_estimator_

def final_evaluation(best_rf, X, y):
    """Final evaluation on test set with comprehensive visualization"""
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
    
    # Create comprehensive final evaluation plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
    axes[0,0].set_title('Confusion Matrix')
    axes[0,0].set_xlabel('Predicted')
    axes[0,0].set_ylabel('Actual')
    
    # 2. Feature Importance
    if hasattr(best_rf, 'feature_importances_') and hasattr(X, 'columns'):
        feature_names = X.columns
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(15)
        axes[0,1].barh(range(len(top_features)), top_features['importance'], color='lightcoral')
        axes[0,1].set_yticks(range(len(top_features)))
        axes[0,1].set_yticklabels(top_features['feature'])
        axes[0,1].set_xlabel('Importance')
        axes[0,1].set_title('Top 15 Feature Importances')
        axes[0,1].grid(True, alpha=0.3)
        
        print(f"\nTop 10 Feature Importances:")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # 3. Performance Metrics Bar Chart
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [accuracy, precision, recall, f1]
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    bars = axes[1,0].bar(metrics, values, color=colors, alpha=0.7)
    axes[1,0].set_title('Final Model Performance Metrics')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{value:.3f}', ha='center', va='bottom')
    
    # 4. Class Distribution
    unique_classes, class_counts = np.unique(y, return_counts=True)
    axes[1,1].pie(class_counts, labels=[f'Class {cls}' for cls in unique_classes], 
                  autopct='%1.1f%%', startangle=90)
    axes[1,1].set_title('Target Class Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_learning_curves(X, y):
    """Plot learning curves to analyze model performance vs training size"""
    print("=" * 60)
    print("LEARNING CURVES ANALYSIS")
    print("=" * 60)
    
    from sklearn.model_selection import learning_curve
    
    # Define models to compare
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))])
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (name, model) in enumerate(models.items()):
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        axes[idx].plot(train_sizes, train_mean, 'o-', label='Training score')
        axes[idx].plot(train_sizes, val_mean, 'o-', label='Cross-validation score')
        
        axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        axes[idx].set_xlabel('Training Set Size')
        axes[idx].set_ylabel('Accuracy Score')
        axes[idx].set_title(f'Learning Curve: {name}')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

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
        cv_scores = compare_cross_validation_methods(X, y)
        
        # Evaluate models with cross-validation
        cv_results = evaluate_models_with_cv(X, y)
        
        # Plot learning curves
        plot_learning_curves(X, y)
        
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