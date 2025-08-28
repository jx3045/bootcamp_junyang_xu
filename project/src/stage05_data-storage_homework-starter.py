# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

import os, pathlib, datetime as dt
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
RAW = pathlib.Path(os.getenv('DATA_DIR_RAW', 'data/raw'))
PROC = pathlib.Path(os.getenv('DATA_DIR_PROCESSED', 'data/processed'))
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)
print('RAW ->', RAW.resolve())
print('PROC ->', PROC.resolve())

import numpy as np
dates = pd.date_range('2024-01-01', periods=20, freq='D')
df = pd.DataFrame({'date': dates, 'ticker': ['AAPL']*20, 'price': 150 + np.random.randn(20).cumsum()})
df.head()

def ts(): return dt.datetime.now().strftime('%Y%m%d-%H%M%S')

# TODO: Save CSV
csv_path = RAW / f"sample_{ts()}.csv"
df.to_csv(csv_path, index=False)
csv_path

# TODO: Save Parquet
pq_path = PROC / f"sample_{ts()}.parquet"
try:
    df.to_parquet(pq_path)
except Exception as e:
    print('Parquet engine not available. Install pyarrow or fastparquet to complete this step.')
    pq_path = None
pq_path

def validate_loaded(original, reloaded):
    checks = {
        'shape_equal': original.shape == reloaded.shape,
        'date_is_datetime': pd.api.types.is_datetime64_any_dtype(reloaded['date']) if 'date' in reloaded.columns else False,
        'price_is_numeric': pd.api.types.is_numeric_dtype(reloaded['price']) if 'price' in reloaded.columns else False,
    }
    return checks

df_csv = pd.read_csv(csv_path, parse_dates=['date'])
validate_loaded(df, df_csv)

if pq_path:
    try:
        df_pq = pd.read_parquet(pq_path)
        validate_loaded(df, df_pq)
    except Exception as e:
        print('Parquet read failed:', e)

import typing as t, pathlib

def detect_format(path: t.Union[str, pathlib.Path]):
    s = str(path).lower()
    if s.endswith('.csv'): return 'csv'
    if s.endswith('.parquet') or s.endswith('.pq') or s.endswith('.parq'): return 'parquet'
    raise ValueError('Unsupported format: ' + s)

def write_df(df: pd.DataFrame, path: t.Union[str, pathlib.Path]):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    fmt = detect_format(p)
    if fmt == 'csv':
        df.to_csv(p, index=False)
    else:
        try:
            df.to_parquet(p)
        except Exception as e:
            raise RuntimeError('Parquet engine not available. Install pyarrow or fastparquet.') from e
    return p

def read_df(path: t.Union[str, pathlib.Path]):
    p = pathlib.Path(path)
    fmt = detect_format(p)
    if fmt == 'csv':
        return pd.read_csv(p, parse_dates=['date']) if 'date' in pd.read_csv(p, nrows=0).columns else pd.read_csv(p)
    else:
        try:
            return pd.read_parquet(p)
        except Exception as e:
            raise RuntimeError('Parquet engine not available. Install pyarrow or fastparquet.') from e

# Demo
p_csv = RAW / f"util_{ts()}.csv"
p_pq  = PROC / f"util_{ts()}.parquet"
write_df(df, p_csv); read_df(p_csv).head()
try:
    write_df(df, p_pq)
    read_df(p_pq).head()
except RuntimeError as e:
    print('Skipping Parquet util demo:', e)

p_csv = RAW / f"util_{ts()}.csv"
write_df(df, p_csv)
df_csv = read_df(p_csv)
print(validate_loaded(df, df_csv))

p_pq = PROC / f"util_{ts()}.parquet"
write_df(df, p_pq)
df_pq = read_df(p_pq)
print(validate_loaded(df, df_pq))