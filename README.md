# TSLL-tracker

Daily TSLA forecasting utilities using LightGBM with engineered features and market context (SPY, VIX).

Quick start:

- Install deps (example):
  - `pip install yfinance lightgbm joblib scikit-learn pandas numpy`

- One‑off research script (original):
  - `python mvp_tsla_predictor.py` (kept for reference)

- Daily runner (saves/loads models, logs forecasts):
  - `python daily_runner.py --retrain never --models-dir models --log forecast_log.csv`
  - `python daily_runner.py --retrain weekly --window-years 3`
  - `python daily_runner.py --cv --retrain always --window-years 3`

Outputs:

- Models at `models/point_*.pkl` and `models/quant_*_*.pkl`
- Rolling log at `forecast_log.csv`
