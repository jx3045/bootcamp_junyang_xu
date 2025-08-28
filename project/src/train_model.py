

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.linear_model import LinearRegression


data_path = Path("../data/processed/train.csv")
if not data_path.exists():
    raise FileNotFoundError(f"{data_path} does not exist.")

df = pd.read_csv(data_path)



numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)


categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])


feature_cols = [c for c in numeric_cols if c != 'spend']
X = df[feature_cols].values
y = df['spend'].values


model = LinearRegression()
model.fit(X, y)
print("Model trained successfully.")

model_dir = Path("../model")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {model_path}")


with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

# Example test prediction (replace with actual feature values)
test_features = X[:2, :]  # take first 2 rows as example
preds = loaded_model.predict(test_features)
print("Test predictions:", preds)
