# Recommended Kaggle Datasets for ML Evaluation

## 1. Titanic Dataset (HIGHLY RECOMMENDED)
**URL**: https://www.kaggle.com/c/titanic/data
- **File**: Download `train.csv`
- **Target Column**: Rename `Survived` to `target` 
- **Size**: 891 rows, 12 columns
- **Task**: Binary classification (survived/died)
- **Features**: Mix of categorical (Sex, Embarked) and numerical (Age, Fare)
- **Missing Values**: Yes (Age, Cabin, Embarked)
- **Perfect for**: Testing all aspects of your ML pipeline

### Quick Setup for Titanic:
```python
# After downloading train.csv from Kaggle
import pandas as pd
df = pd.read_csv('train.csv')
df = df.rename(columns={'Survived': 'target'})
df.to_csv('your_dataset.csv', index=False)
```

## 2. Heart Disease Dataset
**URL**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Target Column**: Already named `target`
- **Size**: 303 rows, 14 columns
- **Task**: Binary classification (heart disease/no heart disease)
- **Features**: All numerical medical measurements
- **Missing Values**: None
- **Perfect for**: Clean dataset testing

## 3. Wine Quality Dataset
**URL**: https://www.kaggle.com/datasets/uciml/red-wine-quality-datasets
- **File**: `winequality-red.csv`
- **Target Column**: Rename `quality` to `target`
- **Size**: 1,599 rows, 12 columns
- **Task**: Multi-class classification (quality scores 3-8)
- **Features**: All numerical chemical properties
- **Missing Values**: None

### Convert to Binary Classification:
```python
# Convert quality to binary (good wine >= 6, bad wine < 6)
df['target'] = (df['quality'] >= 6).astype(int)
```

## 4. Iris Dataset (Quick Testing)
**URL**: https://www.kaggle.com/datasets/uciml/iris
- **Target Column**: Rename `Species` to `target`
- **Size**: 150 rows, 5 columns
- **Task**: Multi-class classification (3 flower species)
- **Features**: 4 numerical measurements
- **Missing Values**: None
- **Perfect for**: Quick algorithm testing

## Download Instructions:
1. Create a Kaggle account at kaggle.com
2. Go to the dataset URL
3. Click "Download" button
4. Extract CSV file to your project directory
5. Rename target column if needed
6. Update `csv_file` variable in `ml_evaluation.py`

## Dataset Characteristics Summary:
| Dataset | Rows | Features | Classes | Missing Values | Difficulty |
|---------|------|----------|---------|----------------|------------|
| Titanic | 891 | 11 | 2 | Yes | Beginner |
| Heart Disease | 303 | 13 | 2 | No | Beginner |
| Wine Quality | 1,599 | 11 | 6 (or 2) | No | Intermediate |
| Iris | 150 | 4 | 3 | No | Beginner |

**Start with Titanic** - it's the most comprehensive for testing your ML pipeline!