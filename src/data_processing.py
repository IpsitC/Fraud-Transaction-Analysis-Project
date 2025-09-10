# src/data_processing.py
import pandas as pd
import numpy as np

def load_csv(path):
    df = pd.read_csv(path)
    return df

def basic_clean(df):
    df = df.drop_duplicates().reset_index(drop=True)
    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    # If timestamps, convert
    if 'Time' in df.columns:
        try:
            df['Time'] = pd.to_datetime(df['Time'], unit='s')
        except Exception:
            pass
    return df

def feature_engineer(df):
    # Example: create log of amount if present
    if 'Amount' in df.columns:
        df['log_amount'] = np.log1p(df['Amount'])
    # Example: create hour of day if Time column exists
    if 'Time' in df.columns:
        df['hour'] = pd.to_datetime(df['Time']).dt.hour
    return df
