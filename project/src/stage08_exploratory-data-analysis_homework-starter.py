# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import skew, kurtosis
sns.set(context='talk', style='whitegrid')
np.random.seed(8)
pd.set_option('display.max_columns', 100)
import pathlib

n = 200
df = pd.DataFrame({
    'date': pd.date_range('2025-02-01', periods=n, freq='D'),
    'region': np.random.choice(['North','South','East','West'], size=n),
    'age': np.random.normal(40, 8, size=n).clip(18, 70).round(1),
    'income': np.random.lognormal(mean=10.6, sigma=0.3, size=n).round(2),
    'transactions': np.random.poisson(lam=3, size=n),
})
base = df['income'] * 0.002 + df['transactions']*20 + np.random.normal(0, 30, size=n)
df['spend'] = np.maximum(0, base).round(2)

# inject a bit of missingness and outliers
df.loc[np.random.choice(df.index, 5, replace=False), 'income'] = np.nan
df.loc[np.random.choice(df.index, 3, replace=False), 'spend'] = np.nan
df.loc[np.random.choice(df.index, 2, replace=False), 'transactions'] = df['transactions'].max()+15

RAW = pathlib.Path("../data/raw/sample_data.csv")
RAW.parent.mkdir(parents=True, exist_ok=True)  # make sure folder exists
df.to_csv(RAW, index=False)

print(f"✅ Sample data saved to: {RAW}")
df.head()

import pandas as pd

RAW = "../data/raw/sample_data.csv"
df = pd.read_csv(RAW)

df.head()
df.info(), df.isna().sum()

desc = df[['age','income','transactions','spend']].describe().T
desc['skew'] = [skew(df[c].dropna()) for c in desc.index]
desc['kurtosis'] = [kurtosis(df[c].dropna()) for c in desc.index]
desc

# Income distribution
sns.histplot(df['income'], kde=True)
plt.title('Income Distribution')
plt.show()

# Transactions boxplot
sns.boxplot(x=df['transactions'])
plt.title('Transactions (Outliers)')
plt.show()

# Spend distribution
sns.histplot(df['spend'], kde=True)
plt.title('Spend Distribution')
plt.show()

sns.scatterplot(data=df, x='income', y='spend', hue='region')
plt.title('Income vs Spend by Region')
plt.show()

sns.scatterplot(data=df, x='age', y='spend')
plt.title('Age vs Spend')
plt.show()

corr = df[['age','income','transactions','spend']].corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

corr

# Example: save processed dataset
PROCESSED = "../data/processed/stage08_clean.csv"

# Simple cleaning demo (drop rows with missing values)
df_clean = df.dropna()

df_clean.to_csv(PROCESSED, index=False)
print(f"Processed dataset saved to {PROCESSED}")