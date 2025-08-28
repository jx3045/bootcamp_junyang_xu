# src/utils.py
import pandas as pd

def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
 
    return df.describe()