"""
Prepare real Titanic dataset for ML evaluation
Run this after downloading train.csv from Kaggle
"""
import pandas as pd

def prepare_titanic_dataset():
    """Prepare Titanic dataset for ML evaluation"""
    try:
        # Try different possible filenames
        possible_files = ['train.csv', 'titanic_train.csv', 'titanic.csv']
        df = None
        
        for filename in possible_files:
            try:
                df = pd.read_csv(filename)
                print(f"✅ Found Titanic dataset: {filename}")
                break
            except FileNotFoundError:
                continue
        
        if df is None:
            raise FileNotFoundError("No Titanic dataset found. Please download train.csv from Kaggle.")
        
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Rename Survived to target (required by ml_evaluation.py)
        if 'Survived' in df.columns:
            df = df.rename(columns={'Survived': 'target'})
            print("✅ Renamed 'Survived' column to 'target'")
        
        # Drop PassengerId (not a useful feature for ML)
        if 'PassengerId' in df.columns:
            df = df.drop('PassengerId', axis=1)
            print("✅ Dropped 'PassengerId' column")
        
        # Save prepared dataset
        df.to_csv('your_dataset.csv', index=False)
        
        # Print dataset info
        print(f"\n📊 Prepared dataset saved as 'your_dataset.csv'")
        print(f"Final shape: {df.shape}")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        print(f"\nMissing values per column:")
        missing = df.isnull().sum()
        for col, count in missing[missing > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
        
        print(f"\nFeature types:")
        for col in df.columns:
            if col != 'target':
                dtype = 'Categorical' if df[col].dtype == 'object' else 'Numerical'
                print(f"  {col}: {dtype}")
        
        print(f"\n🚀 Ready to run: python ml_evaluation.py")
        return True
        
    except Exception as e:
        print(f"❌ Error preparing dataset: {str(e)}")
        return False

if __name__ == "__main__":
    prepare_titanic_dataset()