"""
Daily forecast runner for TSLA multi-horizon models.

Usage examples:
  python daily_runner.py --retrain never --models-dir models --log forecast_log.csv
  python daily_runner.py --retrain weekly --window-years 3
  python daily_runner.py --cv --retrain always --window-years 3
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

from tsla_pipeline import (
    START_DATE,
    HORIZONS,
    QUANTILES,
    get_latest_data,
    build_features,
    evaluate_cv,
    train_models,
    save_models,
    load_models,
    latest_feature_row,
    restrict_training_window,
)


def append_log(
    log_path: str,
    as_of_date: pd.Timestamp,
    last_close: float,
    preds: Dict[str, dict],
) -> None:
    # Predicted-for dates using business days
    from pandas.tseries.offsets import BDay

    row = {
        "date_generated": pd.Timestamp.utcnow().tz_localize(None),
        "as_of_close_date": as_of_date,
        "last_close": last_close,
    }

    for k, horizon in HORIZONS.items():
        pred_for = as_of_date + BDay(horizon)
        row[f"pred_for_{k}"] = pred_for
        row[f"pred_ret_{k}"] = preds[k]["ret"]
        row[f"pred_price_{k}"] = preds[k]["price"]
        for q in QUANTILES:
            row[f"q_ret_{k}_{q}"] = preds[k]["q_ret"][q]
            row[f"q_price_{k}_{q}"] = preds[k]["q_price"][q]
        # Placeholders for future actuals
        row[f"actual_close_{k}"] = None
        row[f"actual_ret_{k}"] = None

    df_new = pd.DataFrame([row])

    if os.path.exists(log_path):
        df = pd.read_csv(log_path, parse_dates=["date_generated", "as_of_close_date"] + [c for c in df_new.columns if c.startswith("pred_for_")])
        df = pd.concat([df, df_new], ignore_index=True)
        # Drop exact duplicate as_of_close_date rows (keep last)
        df = df.drop_duplicates(subset=["as_of_close_date"], keep="last")
    else:
        df = df_new

    df.to_csv(log_path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Run daily TSLA forecasts (t+1, t+5)")
    ap.add_argument("--start", default=START_DATE, help="History start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="Optional end date (YYYY-MM-DD); defaults to tomorrow")
    ap.add_argument("--models-dir", default="models", help="Directory to save/load models")
    ap.add_argument("--log", default="forecast_log.csv", help="CSV path to append forecasts")
    ap.add_argument("--retrain", choices=["never", "weekly", "always"], default="never", help="Retraining policy")
    ap.add_argument("--window-years", type=int, default=3, help="Training window in years (0=all)")
    ap.add_argument("--cv", action="store_true", help="Print cross-validation metrics before training")
    args = ap.parse_args()

    # End date defaults to tomorrow to ensure current day completes
    end = None
    if args.end:
        end = pd.Timestamp(args.end)
    else:
        end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)

    # 1) Data
    base = get_latest_data(start=args.start, end=end)

    # 2) Features/targets
    fs = build_features(base)
    X, y = fs.X, fs.y

    # Optional: restrict training window
    Xtr, ytr = restrict_training_window(X, y, args.window_years)

    # 3) Model selection: load vs retrain
    need_train = False
    if args.retrain == "always":
        need_train = True
    elif args.retrain == "weekly":
        # Monday retrain policy
        need_train = fs.as_of_date.dayofweek == 0
    else:  # never
        need_train = False

    loaded = None if need_train else load_models(args.models_dir)

    if loaded is None:
        if args.cv:
            print("\n=== CROSS-VALIDATION ===")
            for k in HORIZONS.keys():
                m = evaluate_cv(Xtr, ytr[k])
                print(f"{k}: RMSE {m['rmse']:.6f}  MAE {m['mae']:.6f}  Dir {m['direction_acc']:.2%}  Cov {m['coverage']:.2%}  Width {m['int_width']:.6f}")

        pm, qm = train_models(Xtr, ytr)
        save_models(pm, qm, args.models_dir)
    else:
        pm, qm = loaded

    # 4) Inference
    latest_feat = latest_feature_row(X)
    preds = {
        **{
            k: {
                "ret": None,
                "price": None,
                "q_ret": {},
                "q_price": {},
            }
            for k in HORIZONS.keys()
        }
    }

    from tsla_pipeline import forecast_next
    preds = forecast_next(pm, qm, latest_feat, fs.last_close)

    # Console report
    print("\n=== FORECAST ===")
    print(f"As-of: {fs.as_of_date.date()}  Last Close: {fs.last_close:.2f}")
    for k in HORIZONS.keys():
        out = preds[k]
        print(f"\n{k}: return {out['ret']:.6f}  →  price {out['price']:.2f}")
        qret = {q: round(out['q_ret'][q], 6) for q in QUANTILES}
        qpx = {q: round(out['q_price'][q], 2) for q in QUANTILES}
        print("q returns:", qret)
        print("q prices:", qpx)

    # 5) Log to CSV
    append_log(args.log, fs.as_of_date, fs.last_close, preds)


if __name__ == "__main__":
    main()

