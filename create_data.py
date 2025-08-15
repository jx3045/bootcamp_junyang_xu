import pandas as pd
import numpy as np
import os

# make sure the dir exists
os.makedirs('data', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# generate random data
np.random.seed(42)
df = pd.DataFrame({
    'CustomerID': np.arange(1, 101),
    'CategoryColumn': np.random.choice(['A', 'B', 'C'], 100),
    'NumericColumn1': np.random.randn(100) * 10 + 50,
    'NumericColumn2': np.random.randint(0, 100, 100)
})

# save as starter_data.csv
df.to_csv('data/starter_data.csv', index=False)
print("done：data/starter_data.csv")
