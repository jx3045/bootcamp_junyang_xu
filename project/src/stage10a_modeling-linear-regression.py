# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import scipy.stats as st

sns.set(style="whitegrid", context="talk")
np.random.seed(42)

# Load processed dataset from Stage09
DATA = "../data/processed/stage09_features.csv"
df = pd.read_csv(DATA)

df.head()

# Select predictors and target
features = ['age','income','transactions','spend_income_ratio','rolling_spend_mean','region_spend_share']
X = df[features]
y = df['spend']

X = df[features]
y = df['spend']

# Drop rows with missing values
mask = X.notnull().all(axis=1) & y.notnull()
X = X[mask]
y = y[mask]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Fit baseline linear regression
lr = LinearRegression().fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)
print(f"Baseline Model   R²={r2:.4f}   RMSE={rmse:.2f}")

# Residuals
resid = y_test - y_pred
fitted = y_pred

# Residuals vs Fitted
plt.figure(figsize=(6,4))
plt.scatter(fitted, resid, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(resid, bins=20, edgecolor="black")
plt.title("Residual Histogram")
plt.xlabel("Residuals")
plt.show()

# QQ Plot
plt.figure(figsize=(6,4))
st.probplot(resid, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()

# Residuals vs Age (key predictor check)
plt.figure(figsize=(6,4))
plt.scatter(X_test['age'], resid, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Age")
plt.ylabel("Residuals")
plt.title("Residuals vs Age")
plt.show()

# Residuals
resid = y_test - y_pred
fitted = y_pred

# Residuals vs Fitted
plt.figure(figsize=(6,4))
plt.scatter(fitted, resid, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

# Histogram
plt.figure(figsize=(6,4))
plt.hist(resid, bins=20, edgecolor="black")
plt.title("Residual Histogram")
plt.xlabel("Residuals")
plt.show()

# QQ Plot
plt.figure(figsize=(6,4))
st.probplot(resid, dist="norm", plot=plt)
plt.title("QQ Plot of Residuals")
plt.show()

# Residuals vs Age (key predictor check)
plt.figure(figsize=(6,4))
plt.scatter(X_test['age'], resid, alpha=0.7)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Age")
plt.ylabel("Residuals")
plt.title("Residuals vs Age")
plt.show()

# Add polynomial term
df['income_sq'] = df['income']**2
X2 = df[features + ['income_sq']]
mask = X2.notnull().all(axis=1) & y.notnull()
X2 = X2[mask]
y = y[mask]
X2_train, X2_test = X2.iloc[:len(X_train)], X2.iloc[len(X_train):]
lr2 = LinearRegression().fit(X2_train, y_train)
y_pred2 = lr2.predict(X2_test)

r2_2 = r2_score(y_test, y_pred2)
rmse_2 = mean_squared_error(y_test, y_pred2)
print(f"With income^2   R²={r2_2:.4f}   RMSE={rmse_2:.2f}")