"""
TSLA Multi-Horizon Forecast MVP (Hardened)
------------------------------------------
- Uses explicit auto_adjust=True (yfinance)
- Aligns SPY/VIX to TSLA index
- Builds features into a separate 'feat' frame
- LightGBM point + quantile models for t+1 and t+5
"""

import re
import numpy as np
import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from lightgbm import LGBMRegressor
except ImportError:
    raise ImportError("Install with: pip install lightgbm")

# ---------------- Config ----------------
START_DATE = "2015-01-01"
HORIZONS = {"t1": 1, "t5": 5}
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

def sanitize(c): return re.sub(r"[^A-Za-z0-9_]", "_", c)

def earnings_flag(idx: pd.DatetimeIndex) -> pd.Series:
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
        pass
    return f

# ---------------- Download ----------------
# Ensure we include up to the previous business day to avoid same-day lag.
_today = pd.Timestamp.today().normalize()
_target = _today - BDay(1)
_end = pd.Timestamp(_target) + pd.Timedelta(days=1)

tsla = yf.download(
    "TSLA", start=START_DATE, end=_end, interval="1d", auto_adjust=True,
    actions=False, repair=True, progress=False, group_by="column"
)
spy = yf.download(
    "SPY", start=START_DATE, end=_end, interval="1d", auto_adjust=True,
    actions=False, repair=True, progress=False, group_by="column"
)
vix = yf.download(
    "^VIX", start=START_DATE, end=_end, interval="1d", auto_adjust=True,
    actions=False, repair=True, progress=False, group_by="column"
)

if tsla.empty:
    raise RuntimeError("TSLA download returned empty data.")

# Keep only needed cols; align indexes to TSLA
tsla = tsla[["Open","High","Low","Close","Volume"]].copy()

spy = spy[["Close"]].rename(columns={"Close":"SPY"})
spy = spy.reindex(tsla.index).ffill().bfill()

vix = vix[["Close"]].rename(columns={"Close":"VIX"})
vix = vix.reindex(tsla.index).ffill().bfill()

# Combine base frame
base = tsla.join(spy, how="left").join(vix, how="left")
base = base.ffill().bfill()
assert not base[["Close","SPY","VIX"]].isna().any().any(), "Base NA after align."

# ---------------- Features ----------------
feat = pd.DataFrame(index=base.index)

# price returns / trend / vol
# Ensure Close is a 1D Series (yfinance can sometimes yield (n,1) frames)
_close = base["Close"]
if isinstance(_close, pd.DataFrame):
    _close = _close.iloc[:, 0]
ret = np.log(_close / _close.shift(1))
feat["ret_l1"] = ret.shift(1)
feat["ret_l2"] = ret.shift(2)
feat["r3"]     = ret.rolling(3).mean()
feat["r5"]     = ret.rolling(5).mean()
feat["r10"]    = ret.rolling(10).mean()
feat["r21"]    = ret.rolling(21).mean()
feat["vol5"]   = ret.rolling(5).std()
feat["vol10"]  = ret.rolling(10).std()
feat["vol21"]  = ret.rolling(21).std()

# microstructure-ish
feat["vol_ratio"]  = base["Volume"] / base["Volume"].rolling(10).mean()
feat["ma10_ratio"] = base["Close"] / base["Close"].rolling(10).mean()
feat["range"]      = (base["High"] - base["Low"]) / base["Close"]

# market context
spy_ret = np.log(base["SPY"] / base["SPY"].shift(1))
feat["spy_ret"]   = spy_ret
feat["spy_r5"]    = spy_ret.rolling(5).mean()
feat["spy_r21"]   = spy_ret.rolling(21).mean()
feat["spy_vol10"] = spy_ret.rolling(10).std()

feat["vix"]       = base["VIX"]
feat["vix_chg"]   = base["VIX"].pct_change()
feat["vix_vol10"] = feat["vix_chg"].rolling(10).std()

# calendar
feat["dow_sin"]   = np.sin(2*np.pi*feat.index.dayofweek/7)
feat["dow_cos"]   = np.cos(2*np.pi*feat.index.dayofweek/7)
feat["month_sin"] = np.sin(2*np.pi*(feat.index.month-1)/12)
feat["month_cos"] = np.cos(2*np.pi*(feat.index.month-1)/12)
feat["m_end"]     = feat.index.is_month_end.astype(int)
feat["er_flag"]   = earnings_flag(feat.index)

# targets on base (use flattened close series)
t1 = ret.shift(-1)
t5 = np.log(_close.shift(-5) / _close)

targets = pd.DataFrame({"t1": t1, "t5": t5}, index=feat.index)

# ---------------- Clean / Assemble ----------------
feature_cols = list(feat.columns)
target_cols  = list(targets.columns)

# Sanity assert: all columns exist before filtering
missing_now = [c for c in feature_cols if c not in feat.columns] + \
              [c for c in target_cols if c not in targets.columns]
assert not missing_now, f"Missing columns before dropna: {missing_now}"

# Drop NA rows across features + targets
full = feat.join(targets, how="inner")
full = full.dropna(subset=feature_cols + target_cols)

if full.empty:
    # Print helpful debug
    print("DEBUG shapes:",
          "\nfeat:", feat.shape, "feat NA rows:", feat.isna().all(axis=1).sum(),
          "\ntargets:", targets.shape, "targets NA rows:", targets.isna().all(axis=1).sum())
    print("Head(feat):\n", feat.head())
    print("Head(targets):\n", targets.head())
    raise RuntimeError("No rows left after dropna — likely due to too-short start window.")

X = full[feature_cols].copy()
X.columns = [sanitize(c) for c in X.columns]
y = {"t1": full["t1"], "t5": full["t5"]}

# For inference, use the final feature row even if t5 target is NaN
feat_sanitized = feat.copy()
feat_sanitized.columns = [sanitize(c) for c in feat_sanitized.columns]
latest_feat = feat_sanitized.iloc[[-1]].copy()
last_close = float(_close.loc[feat_sanitized.index[-1]])

# ---------------- Eval ----------------
tscv = TimeSeriesSplit(n_splits=SPLITS)

def eval_target(yv):
    preds, actual, qpreds = [], [], {q: [] for q in QUANTILES}
    for tr, te in tscv.split(X):
        tr = tr[:-EMBARGO] if len(tr) > EMBARGO else tr
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = yv.iloc[tr], yv.iloc[te]

        m = LGBMRegressor(**POINT_PARAMS).fit(Xtr, ytr)
        preds.extend(m.predict(Xte)); actual.extend(yte.values)

        for q in QUANTILES:
            qm = LGBMRegressor(**{**POINT_PARAMS, "objective":"quantile", "alpha": q})
            qm.fit(Xtr, ytr)
            qpreds[q].extend(qm.predict(Xte))

    preds, actual = np.array(preds), np.array(actual)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    mae  = mean_absolute_error(actual, preds)
    da   = (np.sign(preds) == np.sign(actual)).mean()
    lo, hi = np.array(qpreds[0.1]), np.array(qpreds[0.9])
    cov = ((actual >= lo) & (actual <= hi)).mean()
    iw  = (hi - lo).mean()
    return rmse, mae, da, cov, iw

print("\n=== PERFORMANCE ===")
for k in ["t1","t5"]:
    rmse, mae, da, cov, iw = eval_target(y[k])
    print(f"\nHorizon {k}:")
    print(f"RMSE {rmse:.6f}  MAE {mae:.6f}  Dir {da:.2%}  Cov {cov:.2%}  Width {iw:.6f}")

# ---------------- Train final models ----------------
pm, qm = {}, {}
for k in ["t1","t5"]:
    pm[k] = LGBMRegressor(**POINT_PARAMS).fit(X, y[k])
    qm[k] = {q: LGBMRegressor(**{**POINT_PARAMS,"objective":"quantile","alpha":q}).fit(X, y[k])
             for q in QUANTILES}

# ---------------- Forecast ----------------
print("\n=== FORECAST ===")
print(f"Last Close: {last_close:.2f}")

for k in ["t1","t5"]:
    r = pm[k].predict(latest_feat)[0]
    price = last_close * np.exp(r)
    qvals = {q: qm[k][q].predict(latest_feat)[0] for q in QUANTILES}
    qpx = {q: last_close * np.exp(qvals[q]) for q in qvals}
    print(f"\n{k}: return {r:.6f}  →  price {price:.2f}")
    print("q returns:", {q: round(qvals[q], 6) for q in qvals})
    print("q prices:",  {q: round(qpx[q], 2)  for q in qpx})
