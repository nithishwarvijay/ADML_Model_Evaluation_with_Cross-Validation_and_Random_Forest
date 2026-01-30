"""
Setup script to help configure Kaggle API and download Titanic dataset
"""
import os
import json

def setup_kaggle_api():
    """Guide user through Kaggle API setup"""
    print("=" * 60)
    print("KAGGLE API SETUP GUIDE")
    print("=" * 60)
    
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")
    
    print(f"\n1. Go to: https://www.kaggle.com/account")
    print(f"2. Scroll down to 'API' section")
    print(f"3. Click 'Create New API Token'")
    print(f"4. Download the kaggle.json file")
    print(f"5. Place it at: {kaggle_json_path}")
    
    if not os.path.exists(kaggle_dir):
        os.makedirs(kaggle_dir)
        print(f"\n✅ Created directory: {kaggle_dir}")
    
    if os.path.exists(kaggle_json_path):
        print(f"\n✅ Kaggle API token found at: {kaggle_json_path}")
        return True
    else:
        print(f"\n❌ Kaggle API token not found at: {kaggle_json_path}")
        print(f"\nAfter placing kaggle.json, run:")
        print(f"  py -c \"import os; os.chmod('{kaggle_json_path}', 0o600)\"")
        return False

def download_titanic_with_api():
    """Download Titanic dataset using Kaggle API"""
    try:
        import kaggle
        print("\n📥 Downloading Titanic dataset...")
        kaggle.api.competition_download_files('titanic', path='.', quiet=False)
        
        # Extract if downloaded as zip
        import zipfile
        if os.path.exists('titanic.zip'):
            with zipfile.ZipFile('titanic.zip', 'r') as zip_ref:
                zip_ref.extractall('.')
            os.remove('titanic.zip')
            print("✅ Extracted Titanic dataset files")
        
        # Prepare the dataset
        if os.path.exists('train.csv'):
            os.system('py prepare_titanic.py')
            return True
        
    except Exception as e:
        print(f"❌ Error downloading: {str(e)}")
        return False

def main():
    """Main setup function"""
    print("🚀 Kaggle Titanic Dataset Setup")
    
    # Check if API is configured
    if setup_kaggle_api():
        # Try to download
        if download_titanic_with_api():
            print("\n🎉 Success! Real Titanic dataset is ready.")
            print("Run: py ml_evaluation.py")
        else:
            print("\n⚠️  API configured but download failed.")
            print("You can manually download from: https://www.kaggle.com/c/titanic/data")
    else:
        print("\n📋 Manual download option:")
        print("1. Go to: https://www.kaggle.com/c/titanic/data")
        print("2. Download 'train.csv'")
        print("3. Run: py prepare_titanic.py")
        print("4. Run: py ml_evaluation.py")
    
    print(f"\n💡 Current status: Sample dataset is ready for testing!")
    print(f"   Run: py ml_evaluation.py")

if __name__ == "__main__":
    main()