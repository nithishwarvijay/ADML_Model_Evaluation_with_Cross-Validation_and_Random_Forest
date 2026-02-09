"""
Focused Cross-Validation vs Random Forest Comparison Plots
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and preprocess the dataset"""
    df = pd.read_csv('your_dataset.csv')
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    X = df.drop('target', axis=1)
    y = df['target']
    X = pd.get_dummies(X, drop_first=True)
    
    return X, y

def create_cv_vs_rf_plots():
    """Create focused comparison between CV methods and RF performance"""
    X, y = load_data()
    
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Validation vs Random Forest: Comprehensive Comparison', 
                 fontsize=16, fontweight='bold')
    
    # 1. Cross-Validation Methods Side-by-Side
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    kfold_scores = cross_val_score(rf, X, y, cv=kfold, scoring='accuracy')
    stratified_scores = cross_val_score(rf, X, y, cv=stratified_kfold, scoring='accuracy')
    
    # Plot 1: CV Methods Comparison
    cv_data = {
        'Method': ['K-Fold'] * 5 + ['Stratified K-Fold'] * 5,
        'Fold': list(range(1, 6)) * 2,
        'Accuracy': list(kfold_scores) + list(stratified_scores)
    }
    cv_df = pd.DataFrame(cv_data)
    
    sns.barplot(data=cv_df, x='Fold', y='Accuracy', hue='Method', ax=axes[0,0])
    axes[0,0].set_title('Cross-Validation Methods: Fold-by-Fold')
    axes[0,0].set_ylabel('Accuracy Score')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Statistical Comparison
    methods = ['K-Fold', 'Stratified K-Fold']
    means = [kfold_scores.mean(), stratified_scores.mean()]
    stds = [kfold_scores.std(), stratified_scores.std()]
    
    bars = axes[0,1].bar(methods, means, yerr=stds, capsize=10, 
                        color=['lightblue', 'lightcoral'], alpha=0.7)
    axes[0,1].set_title('CV Methods: Mean ± Std Dev')
    axes[0,1].set_ylabel('Accuracy Score')
    axes[0,1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                      f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Random Forest vs Other Models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42))])
    }
    
    model_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=stratified_kfold, scoring='accuracy')
        model_results[name] = scores
    
    # Create box plot
    model_data = []
    for name, scores in model_results.items():
        for score in scores:
            model_data.append({'Model': name, 'Accuracy': score})
    
    model_df = pd.DataFrame(model_data)
    sns.boxplot(data=model_df, x='Model', y='Accuracy', ax=axes[0,2])
    axes[0,2].set_title('Model Performance Comparison')
    axes[0,2].tick_params(axis='x', rotation=45)
    axes[0,2].grid(True, alpha=0.3)
    
    # Plot 4: Random Forest Performance Metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_best = RandomForestClassifier(n_estimators=200, max_depth=30, random_state=42)
    rf_best.fit(X_train, y_train)
    y_pred = rf_best.predict(X_test)
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [
        accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred, average='weighted'),
        recall_score(y_test, y_pred, average='weighted'),
        f1_score(y_test, y_pred, average='weighted')
    ]
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = axes[1,0].bar(metrics, values, color=colors, alpha=0.8)
    axes[1,0].set_title('Random Forest: All Metrics')
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_ylim(0, 1)
    axes[1,0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                      f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Cross-Validation Stability
    cv_stability = pd.DataFrame({
        'K-Fold': kfold_scores,
        'Stratified K-Fold': stratified_scores
    })
    
    # Calculate coefficient of variation (std/mean) for stability
    kfold_cv = kfold_scores.std() / kfold_scores.mean()
    stratified_cv = stratified_scores.std() / stratified_scores.mean()
    
    stability_data = pd.DataFrame({
        'Method': ['K-Fold', 'Stratified K-Fold'],
        'Coefficient of Variation': [kfold_cv, stratified_cv],
        'Standard Deviation': [kfold_scores.std(), stratified_scores.std()]
    })
    
    # Twin axis plot
    ax5_twin = axes[1,1].twinx()
    
    bars1 = axes[1,1].bar([0, 1], stability_data['Standard Deviation'], 
                         width=0.4, alpha=0.7, color='lightblue', label='Std Dev')
    bars2 = ax5_twin.bar([0.4, 1.4], stability_data['Coefficient of Variation'], 
                        width=0.4, alpha=0.7, color='lightcoral', label='CV')
    
    axes[1,1].set_title('Cross-Validation Stability')
    axes[1,1].set_ylabel('Standard Deviation', color='blue')
    ax5_twin.set_ylabel('Coefficient of Variation', color='red')
    axes[1,1].set_xticks([0.2, 1.2])
    axes[1,1].set_xticklabels(['K-Fold', 'Stratified'])
    
    # Plot 6: Performance Summary Heatmap
    summary_data = {
        'K-Fold CV': [kfold_scores.mean(), kfold_scores.std(), kfold_scores.min(), kfold_scores.max()],
        'Stratified CV': [stratified_scores.mean(), stratified_scores.std(), 
                         stratified_scores.min(), stratified_scores.max()],
        'Random Forest': [values[0], np.std([values[0]]), min(values), max(values)]
    }
    
    summary_df = pd.DataFrame(summary_data, 
                             index=['Mean', 'Std Dev', 'Min', 'Max'])
    
    sns.heatmap(summary_df, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                ax=axes[1,2], cbar_kws={'label': 'Score'})
    axes[1,2].set_title('Performance Summary Heatmap')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed comparison
    print("=" * 80)
    print("DETAILED CROSS-VALIDATION vs RANDOM FOREST COMPARISON")
    print("=" * 80)
    
    print(f"\n📊 Dataset Information:")
    print(f"   • Shape: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   • Target classes: {np.unique(y)}")
    print(f"   • Class distribution: {np.bincount(y)}")
    
    print(f"\n🔄 Cross-Validation Analysis:")
    print(f"   • K-Fold CV:")
    print(f"     - Mean Accuracy: {kfold_scores.mean():.4f}")
    print(f"     - Std Deviation: {kfold_scores.std():.4f}")
    print(f"     - Coefficient of Variation: {kfold_cv:.4f}")
    print(f"     - Range: [{kfold_scores.min():.4f}, {kfold_scores.max():.4f}]")
    
    print(f"   • Stratified K-Fold CV:")
    print(f"     - Mean Accuracy: {stratified_scores.mean():.4f}")
    print(f"     - Std Deviation: {stratified_scores.std():.4f}")
    print(f"     - Coefficient of Variation: {stratified_cv:.4f}")
    print(f"     - Range: [{stratified_scores.min():.4f}, {stratified_scores.max():.4f}]")
    
    print(f"\n🌲 Random Forest Performance:")
    print(f"   • Accuracy:  {values[0]:.4f}")
    print(f"   • Precision: {values[1]:.4f}")
    print(f"   • Recall:    {values[2]:.4f}")
    print(f"   • F1-Score:  {values[3]:.4f}")
    
    print(f"\n📈 Key Insights:")
    if stratified_scores.mean() > kfold_scores.mean():
        print(f"   • Stratified K-Fold shows {((stratified_scores.mean() - kfold_scores.mean()) * 100):.2f}% better accuracy")
    if stratified_scores.std() < kfold_scores.std():
        print(f"   • Stratified K-Fold is more stable (lower std dev)")
    print(f"   • Random Forest achieves {values[0]:.1%} accuracy on test set")
    
    # Feature importance summary
    if hasattr(X, 'columns'):
        top_features = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_best.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        
        print(f"\n🎯 Top 5 Most Important Features:")
        for idx, row in top_features.iterrows():
            print(f"   • {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    create_cv_vs_rf_plots()