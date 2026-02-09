"""
Generate sample dataset for testing the ML evaluation script
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_sample_dataset():
    """Generate a sample classification dataset"""
    # Create synthetic classification dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    # Introduce some missing values randomly
    np.random.seed(42)
    for col in feature_names[:5]:  # Add missing values to first 5 features
        missing_idx = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan
    
    # Save to CSV
    df.to_csv('your_dataset.csv', index=False)
    print(f"Sample dataset created: your_dataset.csv")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Missing values per column:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

if __name__ == "__main__":
    generate_sample_dataset()