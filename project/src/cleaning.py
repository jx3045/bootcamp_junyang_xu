import pandas as pd
import numpy as np

def fill_missing_median(df: pd.DataFrame, cols: list) -> pd.DataFrame:
  
    for col in cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df


def drop_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    
    limit = int((1 - threshold) * len(df))
    return df.dropna(axis=1, thresh=limit)


def normalize_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Normalize numeric columns to 0–1 range (min-max scaling).
    
    """
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[col] = (df[col] - min_val) / (max_val - min_val)
    return df
