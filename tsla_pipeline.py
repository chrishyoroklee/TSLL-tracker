"""
TSLA forecasting pipeline utilities: data fetch, features, training,
model persistence, and inference.

Designed to be imported by a daily runner script.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
except ImportError as e:
    raise ImportError("Install LightGBM: pip install lightgbm") from e

try:
    import joblib
except ImportError as e:
    raise ImportError("Install joblib: pip install joblib") from e


# ---------------- Config ----------------
START_DATE = "2015-01-01"
HORIZONS: Dict[str, int] = {"t1": 1, "t5": 5}
QUANTILES = [0.1, 0.5, 0.9]
SPLITS = 5
EMBARGO = 5

POINT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "subsample": 0.75,
    "colsample_bytree": 0.9,
    "min_child_samples": 20,
    "reg_lambda": 0.5,
    "n_estimators": 600,
    "random_state": 42,
    "verbosity": -1,
}


def sanitize(c: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", c)


def _earnings_flag(idx: pd.DatetimeIndex) -> pd.Series:
    f = pd.Series(0, index=idx)
    try:
        er = yf.Ticker("TSLA").get_earnings_dates(limit=60)
        if er is not None and not er.empty:
            dates = er.index.tz_localize(None).normalize()
            for d in dates:
                for off in range(-3, 4):
                    t = d + pd.Timedelta(days=off)
                    if t in f.index:
                        f.loc[t] = 1
    except Exception:
        # Network/API issues should not break the pipeline.
        pass
    return f


@dataclass
class FeatureSet:
    X: pd.DataFrame
    y: Dict[str, pd.Series]
    last_close: float
    as_of_date: pd.Timestamp


def get_latest_data(
    start: str = START_DATE,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Download TSLA, SPY, VIX; align to TSLA index and return a base DataFrame.

    Columns: Open, High, Low, Close, Volume, SPY, VIX
    Index: trading days (TSLA calendar)
    """
    tsla = yf.download("TSLA", start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    vix = yf.download("^VIX", start=start, end=end, auto_adjust=True, progress=False, group_by="column")

    if tsla is None or tsla.empty:
        raise RuntimeError("TSLA download returned empty data.")

    tsla = tsla[["Open", "High", "Low", "Close", "Volume"]].copy()

    spy = spy[["Close"]].rename(columns={"Close": "SPY"}) if not spy.empty else pd.DataFrame(index=tsla.index)
    spy = spy.reindex(tsla.index).ffill().bfill()

    vix = vix[["Close"]].rename(columns={"Close": "VIX"}) if not vix.empty else pd.DataFrame(index=tsla.index)
    vix = vix.reindex(tsla.index).ffill().bfill()

    base = tsla.join(spy, how="left").join(vix, how="left")
    base = base.ffill().bfill()
    if base[["Close", "SPY", "VIX"]].isna().any().any():
        raise RuntimeError("Base NA after align.")
    return base


def build_features(base: pd.DataFrame) -> FeatureSet:
    """Build features and targets from aligned base frame.

    Returns FeatureSet with sanitized feature columns and dict of targets.
    """
    feat = pd.DataFrame(index=base.index)

    # Ensure Close is a 1D Series
    _close = base["Close"]
    if isinstance(_close, pd.DataFrame):
        _close = _close.iloc[:, 0]

    # Returns and volatility
    ret = np.log(_close / _close.shift(1))
    feat["ret_l1"] = ret.shift(1)
    feat["ret_l2"] = ret.shift(2)
    feat["r3"] = ret.rolling(3).mean()
    feat["r5"] = ret.rolling(5).mean()
    feat["r10"] = ret.rolling(10).mean()
    feat["r21"] = ret.rolling(21).mean()
    feat["vol5"] = ret.rolling(5).std()
    feat["vol10"] = ret.rolling(10).std()
    feat["vol21"] = ret.rolling(21).std()

    # Microstructure-like features
    feat["vol_ratio"] = base["Volume"] / base["Volume"].rolling(10).mean()
    feat["ma10_ratio"] = base["Close"] / base["Close"].rolling(10).mean()
    feat["range"] = (base["High"] - base["Low"]) / base["Close"]

    # Market context
    spy_ret = np.log(base["SPY"] / base["SPY"].shift(1))
    feat["spy_ret"] = spy_ret
    feat["spy_r5"] = spy_ret.rolling(5).mean()
    feat["spy_r21"] = spy_ret.rolling(21).mean()
    feat["spy_vol10"] = spy_ret.rolling(10).std()

    feat["vix"] = base["VIX"]
    feat["vix_chg"] = base["VIX"].pct_change()
    feat["vix_vol10"] = feat["vix_chg"].rolling(10).std()

    # Calendar and earnings
    idx = feat.index
    feat["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    feat["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    feat["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
    feat["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)
    feat["m_end"] = idx.is_month_end.astype(int)
    feat["er_flag"] = _earnings_flag(idx)

    # Targets: log-returns
    t1 = ret.shift(-1)
    t5 = np.log(_close.shift(-5) / _close)
    targets = pd.DataFrame({"t1": t1, "t5": t5}, index=feat.index)

    # Clean and assemble
    feature_cols = list(feat.columns)
    target_cols = list(targets.columns)

    full = feat.join(targets, how="inner").dropna(subset=feature_cols + target_cols)
    if full.empty:
        raise RuntimeError("No rows left after dropna — likely due to too-short start window.")

    X = full[feature_cols].copy()
    X.columns = [sanitize(c) for c in X.columns]
    y = {"t1": full["t1"], "t5": full["t5"]}

    as_of_date = X.index[-1]
    last_close = float(_close.loc[as_of_date])
    return FeatureSet(X=X, y=y, last_close=last_close, as_of_date=as_of_date)


def evaluate_cv(
    X: pd.DataFrame,
    y: pd.Series,
    quantiles=QUANTILES,
    splits: int = SPLITS,
    embargo: int = EMBARGO,
) -> Dict[str, float]:
    tscv = TimeSeriesSplit(n_splits=splits)
    preds, actual, qpreds = [], [], {q: [] for q in quantiles}

    for tr, te in tscv.split(X):
        tr = tr[:-embargo] if len(tr) > embargo else tr
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        m = LGBMRegressor(**POINT_PARAMS).fit(Xtr, ytr)
        preds.extend(m.predict(Xte))
        actual.extend(yte.values)

        for q in quantiles:
            qm = LGBMRegressor(**{**POINT_PARAMS, "objective": "quantile", "alpha": q})
            qm.fit(Xtr, ytr)
            qpreds[q].extend(qm.predict(Xte))

    preds, actual = np.array(preds), np.array(actual)
    rmse = float(np.sqrt(mean_squared_error(actual, preds)))
    mae = float(mean_absolute_error(actual, preds))
    da = float((np.sign(preds) == np.sign(actual)).mean())
    lo, hi = np.array(qpreds[0.1]), np.array(qpreds[0.9])
    cov = float(((actual >= lo) & (actual <= hi)).mean())
    iw = float((hi - lo).mean())
    return {"rmse": rmse, "mae": mae, "direction_acc": da, "coverage": cov, "int_width": iw}


def train_models(X: pd.DataFrame, y: Dict[str, pd.Series]):
    pm: Dict[str, LGBMRegressor] = {}
    qm: Dict[str, Dict[float, LGBMRegressor]] = {}
    for k in HORIZONS.keys():
        pm[k] = LGBMRegressor(**POINT_PARAMS).fit(X, y[k])
        qm[k] = {q: LGBMRegressor(**{**POINT_PARAMS, "objective": "quantile", "alpha": q}).fit(X, y[k]) for q in QUANTILES}
    return pm, qm


def save_models(pm, qm, models_dir: str = "models") -> None:
    os.makedirs(models_dir, exist_ok=True)
    for k in pm:
        joblib.dump(pm[k], os.path.join(models_dir, f"point_{k}.pkl"))
    for k in qm:
        for q in qm[k]:
            joblib.dump(qm[k][q], os.path.join(models_dir, f"quant_{k}_{q}.pkl"))


def load_models(models_dir: str = "models") -> Optional[Tuple[Dict[str, LGBMRegressor], Dict[str, Dict[float, LGBMRegressor]]]]:
    try:
        pm = {}
        qm: Dict[str, Dict[float, LGBMRegressor]] = {}
        for k in HORIZONS.keys():
            pm[k] = joblib.load(os.path.join(models_dir, f"point_{k}.pkl"))
            qm[k] = {}
            for q in QUANTILES:
                qm[k][q] = joblib.load(os.path.join(models_dir, f"quant_{k}_{q}.pkl"))
        return pm, qm
    except Exception:
        return None


def latest_feature_row(X: pd.DataFrame) -> pd.DataFrame:
    return X.iloc[[-1]].copy()


def forecast_next(
    pm: Dict[str, LGBMRegressor],
    qm: Dict[str, Dict[float, LGBMRegressor]],
    latest_feat: pd.DataFrame,
    last_close: float,
):
    preds = {}
    for k in HORIZONS.keys():
        r = float(pm[k].predict(latest_feat)[0])
        price = float(last_close * np.exp(r))
        qvals = {q: float(qm[k][q].predict(latest_feat)[0]) for q in QUANTILES}
        qpx = {q: float(last_close * np.exp(qvals[q])) for q in QUANTILES}
        preds[k] = {
            "ret": r,
            "price": price,
            "q_ret": qvals,
            "q_price": qpx,
        }
    return preds


def restrict_training_window(X: pd.DataFrame, y: Dict[str, pd.Series], years: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    if not years or years <= 0:
        return X, y
    cutoff = X.index[-1] - pd.Timedelta(days=int(365.25 * years))
    Xw = X.loc[X.index >= cutoff]
    yw = {k: y[k].loc[Xw.index] for k in y}
    return Xw, yw


