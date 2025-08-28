# Auto-generated from Jupyter Notebook
# Only code cells preserved (markdown/outputs removed)

# Imports
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, TimeSeriesSplit
np.random.seed(7); sns.set(); plt.rcParams['figure.figsize']=(9,4)

# TODO: load your data
# df = pd.read_csv('path/to.csv', parse_dates=['Date'], index_col='Date')
RAW = "../data/raw/sample_data.csv"
df = pd.read_csv(RAW)


df['Date'] = pd.date_range("2021-01-01", periods=len(df), freq="D")
df = df.set_index("Date")


series = df['spend'].copy()


series = series.fillna(method='ffill').fillna(0)

df['spend'] = series

df['ret'] = df['spend'].pct_change().fillna(0.0)

# TODO: create at least two features
df['lag_1'] = df['ret'].shift(1)
df['roll_mean_5'] = df['ret'].rolling(5).mean().shift(1)
df['roll_std_10'] = df['ret'].rolling(10).std().shift(1)

# Target: next-step return & up/down
df['y_next_ret'] = df['ret'].shift(-1)
df['y_up'] = (df['y_next_ret']>0).astype(int)

df_feat = df.dropna().copy()

# Time-aware split
cut=int(len(df_feat)*0.8)
train, test = df_feat.iloc[:cut], df_feat.iloc[cut:]
features=['lag_1','roll_mean_5','roll_std_10']

X_tr, X_te = train[features], test[features]
y_tr_reg, y_te_reg = train['y_next_ret'], test['y_next_ret']
y_tr_clf, y_te_clf = train['y_up'], test['y_up']

# Track 1: Forecasting returns
reg = Pipeline([('scaler', StandardScaler()), ('linreg', LinearRegression())])
reg.fit(X_tr, y_tr_reg)
pred_reg = reg.predict(X_te)

mae = mean_absolute_error(y_te_reg, pred_reg)
rmse = mean_squared_error(y_te_reg, pred_reg)
print(f'Regression MAE={mae:.5f}, RMSE={rmse:.5f}')
plt.plot(y_te_reg.index, y_te_reg, label='True')
plt.plot(y_te_reg.index, pred_reg, label='Predicted')
plt.legend(); plt.title("Forecasting Returns (spend): Prediction vs Truth"); plt.show()

# Track 2: Classification (up/down)
clf = Pipeline([('scaler', StandardScaler()), ('logit', LogisticRegression(max_iter=1000))])
clf.fit(X_tr, y_tr_clf)
pred_clf = clf.predict(X_te)

print(classification_report(y_te_clf, pred_clf, digits=3))

cm = confusion_matrix(y_te_clf, pred_clf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.show()

OUT = "../data/processed/sample_data_timeseries.csv"
df_feat.to_csv(OUT, index=True)
print(f"Processed dataset saved to {OUT}")