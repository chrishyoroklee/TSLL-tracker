"""
TSLA Next-Day Price Prediction MVP
----------------------------------
- Downloads TSLA historical daily prices
- Builds simple return, volatility, and microstructure features
- Trains a LightGBM regressor w/ walk-forward validation
- Predicts next-day return & price
"""

import re
from itertools import product

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
base_params = {
    "objective": "regression",
    "n_estimators": 600,
    "random_state": 42,
    "verbosity": -1,
}

grid_options = {
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [31, 63],
    "subsample": [0.75, 0.9],
    "colsample_bytree": [0.7, 0.9],
    "min_child_samples": [20],
    "reg_lambda": [0.0, 0.5],
}

grid_keys = list(grid_options.keys())
param_grid = []
max_combinations = 24
for idx, combo in enumerate(product(*(grid_options[key] for key in grid_keys))):
    candidate = dict(zip(grid_keys, combo))
    param_grid.append(candidate)
    if idx + 1 >= max_combinations:
        break

if not param_grid:
    raise ValueError("Parameter grid is empty; provide tuning options.")

best_result = None

for params in param_grid:
    candidate_params = {**base_params, **params}
    fold_preds = []
    fold_actuals = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        fold_model = LGBMRegressor(**candidate_params)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        fold_preds.extend(y_pred)
        fold_actuals.extend(y_test.values)

    fold_preds = np.asarray(fold_preds)
    fold_actuals = np.asarray(fold_actuals)
    rmse = np.sqrt(mean_squared_error(fold_actuals, fold_preds))
    mae = mean_absolute_error(fold_actuals, fold_preds)
    direction_accuracy = np.mean(np.sign(fold_preds) == np.sign(fold_actuals))

    result = {
        "params": candidate_params,
        "rmse": rmse,
        "mae": mae,
        "direction_accuracy": direction_accuracy,
    }

    if best_result is None or rmse < best_result["rmse"]:
        best_result = result

if best_result is None:
    raise RuntimeError("Unable to fit any LightGBM model during tuning.")

# Final training on all data with best params
model = LGBMRegressor(**best_result["params"])
model.fit(X, y)

# -------- 4) Metrics --------
rmse = best_result["rmse"]
mae = best_result["mae"]
direction_accuracy = best_result["direction_accuracy"]

print("---- Model Performance ----")
print(f"RMSE: {rmse:.6f}")
print(f"MAE:  {mae:.6f}")
print(f"Direction Accuracy: {direction_accuracy:.3%}")

print("\nBest Parameters:")
for key in sorted(best_result["params"]):
    print(f"{key}: {best_result['params'][key]}")

# -------- 5) Predict tomorrow --------
last_row = X.iloc[[-1]]
pred_return = model.predict(last_row)[0]

last_price = df["Close"].iloc[-1].item()
predicted_price = float(last_price * np.exp(pred_return))

print("\n---- Prediction ----")
print(f"Last Close Price: {last_price:.2f}")
print(f"Predicted Return: {pred_return:.6f}")
print(f"Predicted Next Close: {predicted_price:.2f}")
