# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

# Stage 09 — Feature Engineering Notebook

import pandas as pd
import numpy as np

# === Load dataset from Stage08 processed output ===
PROCESSED = "../data/processed/stage08_clean.csv"
df = pd.read_csv(PROCESSED)

print("Data loaded:", df.shape)
df.head()

df['spend_income_ratio'] = df['spend'] / df['income']

df[['income', 'spend', 'spend_income_ratio']].head()

# TODO: Add another feature
# Example: df['rolling_spend_mean'] = df['monthly_spend'].rolling(3).mean()

# Rolling 5-day average of spend
df['rolling_spend_mean'] = df['spend'].rolling(5).mean()

df[['spend', 'rolling_spend_mean']].head(10)

# Categorize age into bins
df['age_group'] = pd.cut(df['age'],
                         bins=[17, 29, 39, 49, 59, 70],
                         labels=['20s','30s','40s','50s','60+'])

df[['age','age_group']].head()

# Calculate region spend share
region_spend = df.groupby('region')['spend'].transform('sum')
df['region_spend_share'] = df['spend'] / region_spend

# Rolling 5-day variance in income (proxy for variability)
df['rolling_income_var'] = df['income'].rolling(5).var()

# Save to processed folder
FEATURED = "../data/processed/stage09_features.csv"
df.to_csv(FEATURED, index=False)
print(f"Feature-engineered dataset saved to {FEATURED}")