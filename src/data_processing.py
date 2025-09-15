# src/data_processing.py
import pandas as pd
import numpy as np

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().reset_index(drop=True)
    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # Ensure label dtype if present
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(int)
    return df

def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    # Log transform of Amount
    if 'Amount' in df.columns and 'log_amount' not in df.columns:
        df['log_amount'] = np.log1p(df['Amount'])
    # Hour-of-day feature from Time (seconds since start)
    if 'Time' in df.columns and 'hour' not in df.columns:
        df['hour'] = (df['Time'] // 3600) % 24
    return df
