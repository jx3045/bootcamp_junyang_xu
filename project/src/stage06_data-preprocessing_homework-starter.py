# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

import sys, pathlib
sys.path.append(str(pathlib.Path("..").resolve()))
import pandas as pd
from src import cleaning
import pathlib

RAW = pathlib.Path("../data/raw/sample_data.csv")
PROC = pathlib.Path("../data/processed/sample_data_cleaned.csv")
df = pd.read_csv(RAW)
df.head()

df_clean = cleaning.fill_missing_median(df, ['price'])   # suppose losses
df_clean = cleaning.drop_missing(df_clean, threshold=0.5)
df_clean = cleaning.normalize_data(df_clean, ['price'])

# df.to_csv('../data/processed/sample_data_cleaned.csv', index=False)

print("cleaned data：")
print(df_clean.info())
df_clean.head()

PROC.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(PROC, index=False)

print('cleaned_data_saved')

print("initial losses：")
print(df.isna().sum())

print("cleaned losses：")
print(df_clean.isna().sum())

print("comparassion：")
print("initial:", df['price'].describe())
print("cleaned:", df_clean['price'].describe())