# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

import os
os.chdir('/Users/xujunyang/Desktop/bootcamp_junyang_xu') # shift to the root dir
print(os.getcwd())

import numpy as np
import time


arr = np.arange(1_000_000)


start = time.time()
squared_loop = [x**2 for x in arr]
end = time.time()
print("Loop time:", end-start)


start = time.time()
squared_vector = arr**2
end = time.time()
print("Vectorized time:", end-start)

import pandas as pd

df = pd.read_csv('data/starter_data.csv')
print(df.info())
print(df.head())

summary = df.describe()
print(summary)

# mean
grouped = df.groupby('category').mean(numeric_only=True)
print(grouped)

import os
os.makedirs('data/processed', exist_ok=True)
summary.to_csv('data/processed/summary.csv', index=True)

from src.utils import get_summary_stats

summary2 = get_summary_stats(df)
print(summary2)

import matplotlib.pyplot as plt


df['value'].hist()
plt.title("Histogram of value")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()