"""
Focused plotting script for Cross-Validation vs Random Forest comparison
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('your_dataset.csv')
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def create_comprehensive_comparison_plots():
    """Create comprehensive comparison plots"""
    X, y = load_data()
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Cross-Validation Methods Comparison (Top Left)
    ax1 = plt.subplot(3, 3, 1)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    kfold_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')
    stratified_scores = cross_val_score(rf, X, y, cv=stratified_kfold, scoring='accuracy')
    
    cv_comparison = pd.DataFrame({
        'K-Fold': kfold_scores,
        'Stratified K-Fold': stratified_scores
    })
    
    sns.boxplot(data=cv_comparison, ax=ax1)
    ax1.set_title('Cross-Validation Methods Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Model Performance Comparison (Top Center)
    ax2 = plt.subplot(3, 3, 2)
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))])
    }
    
    model_scores = []
    model_names = []
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
        model_scores.extend(scores)
        model_names.extend([name] * len(scores))
    
    model_df = pd.DataFrame({'Model': model_names, 'Accuracy': model_scores})
    sns.boxplot(data=model_df, x='Model', y='Accuracy', ax=ax2)
    ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Random Forest n_estimators Validation Curve (Top Right)
    ax3 = plt.subplot(3, 3, 3)
    param_range = [10, 50, 100, 150, 200, 250]
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42), X, y,
        param_name='n_estimators', param_range=param_range,
        cv=5, scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    ax3.plot(param_range, train_mean, 'o-', label='Training', linewidth=2)
    ax3.plot(param_range, test_mean, 'o-', label='Validation', linewidth=2)
    ax3.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.2)
    ax3.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.2)
    ax3.set_xlabel('n_estimators')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('RF: n_estimators Validation Curve', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Cross-Validation Fold Consistency (Middle Left)
    ax4 = plt.subplot(3, 3, 4)
    fold_data = []
    for i, (kf_score, sk_score) in enumerate(zip(kfold_scores, stratified_scores)):
        fold_data.append({'Fold': f'Fold {i+1}', 'K-Fold': kf_score, 'Stratified': sk_score})
    
    fold_df = pd.DataFrame(fold_data)
    x_pos = np.arange(len(fold_df))
    width = 0.35
    
    ax4.bar(x_pos - width/2, fold_df['K-Fold'], width, label='K-Fold', alpha=0.8)
    ax4.bar(x_pos + width/2, fold_df['Stratified'], width, label='Stratified K-Fold', alpha=0.8)
    ax4.set_xlabel('Cross-Validation Folds')
    ax4.set_ylabel('Accuracy Score')
    ax4.set_title('Fold-by-Fold Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(fold_df['Fold'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Random Forest Feature Importance (Middle Center)
    ax5 = plt.subplot(3, 3, 5)
    rf_best = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
    rf_best.fit(X, y)
    
    if hasattr(X, 'columns'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_best.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        sns.barplot(data=feature_importance, y='feature', x='importance', ax=ax5, palette='viridis')
        ax5.set_title('Top 10 RF Feature Importances', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Importance')
    
    # 6. Performance Metrics Radar Chart (Middle Right)
    ax6 = plt.subplot(3, 3, 6)
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_best.fit(X_train, y_train)
    y_pred = rf_best.predict(X_test)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    rf_values = [
        (y_test == y_pred).mean(),
        precision_score(y_test, y_pred, average='weighted'),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, average='weighted')
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = ax6.bar(metrics, rf_values, color=colors, alpha=0.7)
    ax6.set_title('Random Forest Performance Metrics', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Score')
    ax6.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, rf_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # 7. Learning Curve Comparison (Bottom Left)
    ax7 = plt.subplot(3, 3, 7)
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores_rf, val_scores_rf = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42), X, y,
        cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    
    train_sizes, train_scores_dt, val_scores_dt = learning_curve(
        DecisionTreeClassifier(random_state=42), X, y,
        cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    
    ax7.plot(train_sizes, np.mean(val_scores_rf, axis=1), 'o-', label='Random Forest', linewidth=2)
    ax7.plot(train_sizes, np.mean(val_scores_dt, axis=1), 'o-', label='Decision Tree', linewidth=2)
    ax7.fill_between(train_sizes, 
                    np.mean(val_scores_rf, axis=1) - np.std(val_scores_rf, axis=1),
                    np.mean(val_scores_rf, axis=1) + np.std(val_scores_rf, axis=1), alpha=0.2)
    ax7.set_xlabel('Training Set Size')
    ax7.set_ylabel('Validation Accuracy')
    ax7.set_title('Learning Curves Comparison', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Cross-Validation Score Distribution (Bottom Center)
    ax8 = plt.subplot(3, 3, 8)
    all_cv_scores = np.concatenate([kfold_scores, stratified_scores])
    cv_labels = ['K-Fold'] * len(kfold_scores) + ['Stratified'] * len(stratified_scores)
    
    for i, method in enumerate(['K-Fold', 'Stratified']):
        scores = kfold_scores if method == 'K-Fold' else stratified_scores
        ax8.hist(scores, alpha=0.7, label=method, bins=10, density=True)
    
    ax8.set_xlabel('Accuracy Score')
    ax8.set_ylabel('Density')
    ax8.set_title('CV Score Distributions', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Model Complexity vs Performance (Bottom Right)
    ax9 = plt.subplot(3, 3, 9)
    max_depths = [5, 10, 15, 20, 25, 30, None]
    depth_scores = []
    
    for depth in max_depths:
        if depth is None:
            rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            rf_temp = RandomForestClassifier(n_estimators=100, max_depth=depth, random_state=42)
        scores = cross_val_score(rf_temp, X, y, cv=5, scoring='accuracy')
        depth_scores.append(scores.mean())
    
    depth_labels = [str(d) if d is not None else 'None' for d in max_depths]
    ax9.plot(depth_labels, depth_scores, 'o-', linewidth=2, markersize=8)
    ax9.set_xlabel('Max Depth')
    ax9.set_ylabel('CV Accuracy')
    ax9.set_title('RF Complexity vs Performance', fontsize=14, fontweight='bold')
    ax9.tick_params(axis='x', rotation=45)
    ax9.grid(True, alpha=0.3)
    
    plt.suptitle('Comprehensive ML Model Evaluation: Cross-Validation vs Random Forest Analysis', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.show()
    
    # Print summary statistics
    print("=" * 80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Dataset Shape: {X.shape}")
    print(f"Target Classes: {np.unique(y)}")
    print(f"\nCross-Validation Comparison:")
    print(f"  K-Fold CV:        {kfold_scores.mean():.4f} ± {kfold_scores.std():.4f}")
    print(f"  Stratified K-Fold: {stratified_scores.mean():.4f} ± {stratified_scores.std():.4f}")
    print(f"\nRandom Forest Final Performance:")
    print(f"  Accuracy:  {rf_values[0]:.4f}")
    print(f"  Precision: {rf_values[1]:.4f}")
    print(f"  Recall:    {rf_values[2]:.4f}")
    print(f"  F1-Score:  {rf_values[3]:.4f}")

if __name__ == "__main__":
    create_comprehensive_comparison_plots()