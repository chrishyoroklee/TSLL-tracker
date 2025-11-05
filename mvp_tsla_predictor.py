"""
TSLA Next-Day Price Prediction MVP
----------------------------------
- Downloads TSLA historical daily prices
- Builds simple return, volatility, and microstructure features
- Trains a LightGBM regressor w/ walk-forward validation
- Predicts next-day return & price
"""

import re

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:  # pragma: no cover - guidance for missing dependency
    raise ImportError(
        "LightGBM is required. Install it with `pip install lightgbm` inside your environment."
    ) from exc


# -------- 1) Download TSLA data --------
df = yf.download("TSLA", start="2015-01-01", auto_adjust=False, progress=False)
df = df[["Open", "High", "Low", "Close", "Volume"]]
df.dropna(inplace=True)

# -------- 2) Feature Engineering --------
df["return"] = np.log(df["Close"] / df["Close"].shift(1))
df["return_lag1"] = df["return"].shift(1)
df["return_lag2"] = df["return"].shift(2)
df["r3"] = df["return"].rolling(3).mean()
df["r5"] = df["return"].rolling(5).mean()
df["r10"] = df["return"].rolling(10).mean()
df["r21"] = df["return"].rolling(21).mean()
df["vol5"] = df["return"].rolling(5).std()
df["vol10"] = df["return"].rolling(10).std()
df["vol21"] = df["return"].rolling(21).std()
df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(10).mean()
df["price_ma10_ratio"] = df["Close"] / df["Close"].rolling(10).mean()
df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
df["target"] = df["return"].shift(-1)  # Next-day log return
df.dropna(inplace=True)

features = [
    "return_lag1",
    "return_lag2",
    "r3",
    "r5",
    "r10",
    "r21",
    "vol5",
    "vol10",
    "vol21",
    "volume_ratio",
    "price_ma10_ratio",
    "intraday_range",
]


def _sanitize(name: str) -> str:
    """LightGBM disallows JSON meta characters; normalize to safe identifiers."""
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    return sanitized


sanitized_feature_names = [_sanitize(name) for name in features]

X = df[features].copy()
X.columns = sanitized_feature_names
y = df["target"]

# -------- 3) Train model w/ TimeSeriesSplit --------
tscv = TimeSeriesSplit(n_splits=5)
model_params = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "random_state": 42,
    "verbosity": -1,
}

preds = []
actuals = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    fold_model = LGBMRegressor(**model_params)
    fold_model.fit(X_train, y_train)
    y_pred = fold_model.predict(X_test)

    preds.extend(y_pred)
    actuals.extend(y_test.values)

# Final training on all data
model = LGBMRegressor(**model_params)
model.fit(X, y)

# -------- 4) Metrics --------
# squared=False was added in newer sklearn; compute RMSE manually for compatibility.
preds = np.asarray(preds)
actuals = np.asarray(actuals)
mse = mean_squared_error(actuals, preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(actuals, preds)
direction_accuracy = np.mean(np.sign(preds) == np.sign(actuals))

print("---- Model Performance ----")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Direction Accuracy: {direction_accuracy:.3%}")

# -------- 5) Predict tomorrow --------
last_row = X.iloc[[-1]]
pred_return = model.predict(last_row)[0]

last_price = df["Close"].iloc[-1].item()
predicted_price = float(last_price * np.exp(pred_return))

print("\n---- Prediction ----")
print(f"Last Close Price: {last_price:.2f}")
print(f"Predicted Return: {pred_return:.6f}")
print(f"Predicted Next Close: {predicted_price:.2f}")
