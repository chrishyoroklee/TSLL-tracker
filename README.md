# TSLL-tracker — TSLA Multi‑Horizon Forecasts

LightGBM-based TSLA forecasting with engineered features, market context (SPY, VIX), quantile intervals, and a daily runner for automated next‑day and 5‑day predictions.

This repo contains both the original research MVP and a modular pipeline suitable for scheduled daily use with model persistence and logging.

## What’s Inside

- `mvp_tsla_predictor.py` — research script that downloads data, builds features, cross‑validates, trains t+1/t+5 models, and prints a forecast.
- `tsla_pipeline.py` — modular utilities: data fetch, feature engineering, cross‑validation, training, persistence, and inference.
- `daily_runner.py` — CLI script for daily/weekly operation: load or retrain models, predict, and append to a CSV log.
- Generated at runtime:
  - `models/point_*.pkl`, `models/quant_*_*.pkl` — saved LightGBM models
  - `forecast_log.csv` — rolling forecast log you can analyze or visualize

## Install

Use your Python environment (3.10+ recommended):

```
pip install -U pip
pip install yfinance lightgbm joblib scikit-learn pandas numpy
```

Optional (for your own scheduling or dashboards):

```
pip install schedule streamlit
```

If you’re using a project venv, activate it first (e.g., `.venv/bin/activate`).

## Quick Start

Run the original MVP (research):

```
python mvp_tsla_predictor.py
```

Run the daily runner (loads models if they exist; otherwise trains per policy):

```
python daily_runner.py --retrain never --models-dir models --log forecast_log.csv
python daily_runner.py --retrain weekly --window-years 3
python daily_runner.py --cv --retrain always --window-years 3
```

Common flags:

- `--start` History start date (default: 2015-01-01)
- `--end` Optional end date (default: tomorrow to capture the latest close)
- `--models-dir` Where to save/load models (default: `models`)
- `--log` Forecast log path (default: `forecast_log.csv`)
- `--retrain` Retraining policy: `never`, `weekly` (Mondays), or `always`
- `--window-years` Use recent N years when training (default: 3; 0 = full history)
- `--cv` Print cross‑validation metrics before training

## How It Works

Data sources (via `yfinance`):

- TSLA (auto_adjust=True)
- SPY (market proxy)
- ^VIX (volatility index)

The pipeline aligns SPY and VIX to TSLA’s trading-day index and forward/back fills gaps as needed. Earnings dates are retrieved and expanded to a ±3 trading‑day flag (wrapped in try/except so network hiccups don’t break the run).

Features (selected highlights):

- Price returns, lags, rolling means (3/5/10/21) and std (5/10/21)
- Microstructure proxies: volume ratio, 10‑day MA ratio, daily range
- Market context: SPY returns/rolls/vol; VIX level, change, and volatility
- Calendar signals: day‑of‑week, month, month‑end flag
- Earnings proximity flag

Targets and models:

- t+1 (`t1`) and t+5 (`t5`) log‑returns
- Point models: LightGBM regression
- Interval estimates: LightGBM quantile models at 0.1/0.5/0.9

Evaluation (in‑sample CV):

- TimeSeriesSplit with embargo (default splits=5, embargo=5)
- Metrics: RMSE, MAE, direction accuracy, q10–q90 coverage and width

Inference:

- Uses the latest feature row to predict next‑day and 5‑day returns
- Converts returns to prices via `price = last_close * exp(return)`
- Provides quantile return/price bands for uncertainty estimation

Logging and persistence:

- Models saved to `models/` using `joblib`
- Each run appends to `forecast_log.csv` with generated time, as‑of date, predicted‑for dates (business‑day offsets), predicted returns/prices, and quantiles

## Typical Workflows

Daily inference (no retrain):

```
python daily_runner.py --retrain never --models-dir models --log forecast_log.csv
```

Weekly retrain (Mondays) on a rolling 3‑year window:

```
python daily_runner.py --retrain weekly --window-years 3
```

Benchmark cross‑validation, then train and save:

```
python daily_runner.py --cv --retrain always --window-years 3
```

## Scheduling

Run once after market close using cron (example, local time 18:05, Mon–Fri):

```
5 18 * * 1-5 cd /path/to/TSLL-tracker && ./.venv/bin/python daily_runner.py --retrain weekly --window-years 3 >> logs/daily.log 2>&1
```

Or use a Python scheduler (if you add it to your own wrapper):

```python
import schedule, time
import subprocess

def run():
    subprocess.run(["python", "daily_runner.py", "--retrain", "weekly", "--window-years", "3"], check=True)

schedule.every().day.at("18:05").do(run)
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Caveats & Notes

- yfinance can rate‑limit or lag; the pipeline handles missing earnings data gracefully.
- Forecasts are for educational/research purposes only — not financial advice.
- Randomness is controlled via `random_state=42`, but LightGBM and rolling windows can still lead to minor variability.
- The log uses business‑day offsets for predicted‑for dates; holidays may shift when markets are closed.

## Acknowledgements

- [LightGBM](https://github.com/microsoft/LightGBM)
- [yfinance](https://github.com/ranaroussi/yfinance)

## Next Ideas

- Streamlit dashboard for recent forecasts and q10–q90 bands
- SQLite storage for logs and easier backfill of actuals
- Direction‑probability output and alerting when confidence is high
