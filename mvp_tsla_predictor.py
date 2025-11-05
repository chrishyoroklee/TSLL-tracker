"""
TSLA Next-Day Price Prediction MVP
----------------------------------
- Downloads TSLA historical daily prices
- Builds simple return & volatility features
- Trains linear regression model w/ walk-forward validation
- Predicts next-day return & price
"""

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit


# -------- 1) Download TSLA data --------
df = yf.download("TSLA", start="2015-01-01", auto_adjust=False, progress=False)
df = df[["Open", "High", "Low", "Close", "Volume"]]
df.dropna(inplace=True)

# -------- 2) Feature Engineering --------
df["return"] = np.log(df["Close"] / df["Close"].shift(1))
df["r3"] = df["return"].rolling(3).mean()
df["r5"] = df["return"].rolling(5).mean()
df["r10"] = df["return"].rolling(10).mean()
df["vol5"] = df["return"].rolling(5).std()
df["vol10"] = df["return"].rolling(10).std()
df["target"] = df["return"].shift(-1)  # Next-day log return
df.dropna(inplace=True)

features = ["r3", "r5", "r10", "vol5", "vol10"]
X = df[features]
y = df["target"]

# -------- 3) Train model w/ TimeSeriesSplit --------
tscv = TimeSeriesSplit(n_splits=5)
model = LinearRegression()

preds = []
actuals = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    preds.extend(y_pred)
    actuals.extend(y_test)

# Final training on all data
model.fit(X, y)

# -------- 4) Metrics --------
# squared=False was added in newer sklearn; compute RMSE manually for compatibility.
mse = mean_squared_error(actuals, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, preds)

print("---- Model Performance ----")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")

# -------- 5) Predict tomorrow --------
last_row = X.iloc[-1:].values
pred_return = model.predict(last_row)[0]

last_price = float(df["Close"].iloc[-1])
predicted_price = float(last_price * np.exp(pred_return))

print("\n---- Prediction ----")
print(f"Last Close Price: {last_price:.2f}")
print(f"Predicted Return: {pred_return:.6f}")
print(f"Predicted Next Close: {predicted_price:.2f}")
