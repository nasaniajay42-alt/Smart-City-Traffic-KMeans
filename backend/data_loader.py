"""
Data Loader Module for Customer Segmentation.
Handles dataset loading, column normalization, validation, and summary statistics.
"""

import os
import pandas as pd
import numpy as np

# Standard expected column names
STANDARD_COLUMNS = {
    'customer_id': 'CustomerID',
    'gender': 'Gender',
    'age': 'Age',
    'annual_income': 'Annual Income (k$)',
    'spending_score': 'Spending Score (1-100)'
}

DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', 'datasets', 'Mall_Customers.csv')


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize common column variations to the standard Mall Customer format.
    """
    df = df.copy()
    rename_map = {}
    for col in df.columns:
        col_clean = col.strip().lower().replace('_', ' ').replace('-', ' ')
        if 'customer' in col_clean and 'id' in col_clean:
            rename_map[col] = STANDARD_COLUMNS['customer_id']
        elif 'gender' in col_clean or 'sex' in col_clean:
            rename_map[col] = STANDARD_COLUMNS['gender']
        elif 'age' in col_clean:
            rename_map[col] = STANDARD_COLUMNS['age']
        elif 'income' in col_clean:
            rename_map[col] = STANDARD_COLUMNS['annual_income']
        elif 'spending' in col_clean or 'score' in col_clean:
            rename_map[col] = STANDARD_COLUMNS['spending_score']
            
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def validate_dataset(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Validate that the dataframe has the essential numerical columns for segmentation.
    Returns (is_valid, message).
    """
    if df is None or df.empty:
        return False, "Dataset is empty."

    df_norm = normalize_column_names(df)
    
    # Required core features
    required = [STANDARD_COLUMNS['annual_income'], STANDARD_COLUMNS['spending_score']]
    missing = [c for c in required if c not in df_norm.columns]
    
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}. Please ensure income and spending score are present."
        
    # Check numeric types
    for col in required:
        if not pd.api.types.is_numeric_dtype(df_norm[col]):
            try:
                pd.to_numeric(df_norm[col])
            except Exception:
                return False, f"Column '{col}' contains non-numeric values that cannot be parsed."

    if len(df_norm) < 10:
        return False, f"Dataset has only {len(df_norm)} records. At least 10 records are required for K-Means clustering."

    return True, "Dataset is valid."


def load_dataset(file_path: str = None) -> pd.DataFrame:
    """
    Load dataset from specified path or default to datasets/Mall_Customers.csv.
    """
    path = file_path if file_path else DEFAULT_DATASET_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)
    df = normalize_column_names(df)
    return df


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """
    Extract comprehensive summary, missing values, data types, and statistics.
    """
    total_records = int(len(df))
    columns_info = []
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        null_count = int(df[col].isnull().sum())
        null_pct = round((null_count / total_records) * 100, 2) if total_records > 0 else 0
        columns_info.append({
            'name': col,
            'type': col_type,
            'null_count': null_count,
            'null_percentage': null_pct
        })
        
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    stats = {}
    if num_cols:
        desc = df[num_cols].describe().round(2).to_dict()
        stats = desc

    # High-level metrics
    income_col = STANDARD_COLUMNS['annual_income']
    spending_col = STANDARD_COLUMNS['spending_score']
    age_col = STANDARD_COLUMNS['age']
    gender_col = STANDARD_COLUMNS['gender']

    avg_income = round(float(df[income_col].mean()), 2) if income_col in df.columns else None
    avg_spending = round(float(df[spending_col].mean()), 2) if spending_col in df.columns else None
    avg_age = round(float(df[age_col].mean()), 2) if age_col in df.columns else None

    gender_distribution = {}
    if gender_col in df.columns:
        gender_distribution = df[gender_col].value_counts().to_dict()

    return {
        'total_records': total_records,
        'total_columns': int(len(df.columns)),
        'columns': columns_info,
        'statistics': stats,
        'kpi': {
            'avg_income': avg_income,
            'avg_spending': avg_spending,
            'avg_age': avg_age,
            'gender_distribution': gender_distribution
        }
    }
