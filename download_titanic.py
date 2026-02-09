"""
Download and prepare Titanic dataset for ML evaluation
Since Kaggle API requires authentication, this script creates the Titanic dataset
from the well-known structure and provides instructions for manual download.
"""
import pandas as pd
import numpy as np

def create_titanic_sample():
    """Create a sample Titanic-like dataset for testing"""
    np.random.seed(42)
    
    # Create sample data with Titanic structure
    n_samples = 891
    
    data = {
        'PassengerId': range(1, n_samples + 1),
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger_{i}' for i in range(1, n_samples + 1)],
        'Sex': np.random.choice(['male', 'female'], n_samples, p=[0.65, 0.35]),
        'Age': np.random.normal(29.7, 14.5, n_samples),
        'SibSp': np.random.choice([0, 1, 2, 3, 4, 5, 8], n_samples, p=[0.68, 0.23, 0.06, 0.02, 0.005, 0.003, 0.002]),
        'Parch': np.random.choice([0, 1, 2, 3, 4, 5, 6], n_samples, p=[0.76, 0.13, 0.08, 0.02, 0.004, 0.003, 0.003]),
        'Ticket': [f'TICKET_{i}' for i in range(1, n_samples + 1)],
        'Fare': np.random.lognormal(2.5, 1.2, n_samples),
        'Cabin': [f'C{i}' if np.random.random() > 0.77 else None for i in range(1, n_samples + 1)],
        'Embarked': np.random.choice(['C', 'Q', 'S'], n_samples, p=[0.19, 0.09, 0.72])
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values to match real Titanic dataset
    age_missing_idx = np.random.choice(df.index, size=177, replace=False)
    df.loc[age_missing_idx, 'Age'] = np.nan
    
    embarked_missing_idx = np.random.choice(df.index, size=2, replace=False)
    df.loc[embarked_missing_idx, 'Embarked'] = np.nan
    
    # Create survival target based on realistic patterns
    survival_prob = 0.3  # Base survival rate
    
    # Adjust probability based on features
    prob_adjustments = np.zeros(n_samples)
    prob_adjustments += (df['Sex'] == 'female') * 0.4  # Women more likely to survive
    prob_adjustments += (df['Pclass'] == 1) * 0.2      # First class more likely
    prob_adjustments += (df['Pclass'] == 2) * 0.1      # Second class somewhat more likely
    prob_adjustments -= (df['Age'] > 60) * 0.1         # Elderly less likely
    prob_adjustments += (df['Age'] < 16) * 0.2         # Children more likely
    
    final_probs = np.clip(survival_prob + prob_adjustments, 0, 1)
    df['target'] = np.random.binomial(1, final_probs)
    
    # Drop PassengerId for ML (not a feature)
    df = df.drop('PassengerId', axis=1)
    
    return df

def download_instructions():
    """Print instructions for manual Kaggle download"""
    print("=" * 60)
    print("KAGGLE TITANIC DATASET DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nTo get the real Titanic dataset:")
    print("1. Go to: https://www.kaggle.com/c/titanic/data")
    print("2. Create a Kaggle account if you don't have one")
    print("3. Click 'Download All' or download 'train.csv'")
    print("4. Save as 'titanic_train.csv' in this directory")
    print("5. Run: python prepare_titanic.py")
    print("\nAlternatively, I've created a sample dataset for testing...")

def prepare_real_titanic():
    """Prepare real Titanic dataset if available"""
    try:
        # Try to load real Titanic data
        df = pd.read_csv('titanic_train.csv')
        print("Found real Titanic dataset!")
        
        # Rename Survived to target
        if 'Survived' in df.columns:
            df = df.rename(columns={'Survived': 'target'})
        
        # Drop PassengerId (not a feature)
        if 'PassengerId' in df.columns:
            df = df.drop('PassengerId', axis=1)
        
        # Save prepared dataset
        df.to_csv('your_dataset.csv', index=False)
        print(f"Prepared real Titanic dataset: {df.shape}")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        return True
        
    except FileNotFoundError:
        return False

def main():
    """Main function to prepare Titanic dataset"""
    download_instructions()
    
    # Try to prepare real dataset first
    if prepare_real_titanic():
        print("\n✅ Real Titanic dataset prepared successfully!")
        return
    
    # Create sample dataset if real one not available
    print("\n📊 Creating sample Titanic-like dataset for testing...")
    df = create_titanic_sample()
    df.to_csv('your_dataset.csv', index=False)
    
    print(f"\n✅ Sample dataset created: your_dataset.csv")
    print(f"Shape: {df.shape}")
    print(f"Target distribution:\n{df['target'].value_counts()}")
    print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nFeatures: {list(df.columns)}")
    print("\n🚀 You can now run: python ml_evaluation.py")

if __name__ == "__main__":
    main()