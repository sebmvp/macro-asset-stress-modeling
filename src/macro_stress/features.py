"""Log returns, lags, Gold volatility, and the 55-predictor matrix.

The supervised matrix matches the Milestone 2 / 3 notebooks:

  MACRO_LEVELS (8) + VOL_LEVELS (3) contemporaneous, each with lags 1/3/5
  gold_ret / spy_ret / btc_ret lags 1/3/5
  contemporaneous stress_hys_t and stress_any_t

That is 11 + 33 + 9 + 2 = 55 columns.

Leakage rules:
  * targets are shifted forward before any split
  * features at t use only information available at t
  * no random shuffling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .regimes import add_stress_flags

TRADING_DAYS = 252
GOLD_VOL_WINDOW = 30
LAGS = (1, 3, 5)

MACRO_LEVELS = (
    "Volatility_Index",
    "Yield_Curve_Spread",
    "High_Yield_Spread",
    "Financial_Stress_Index",
    "SPY_Drawdown",
    "Stock_Bond_Corr_90d",
    "SPY_RSI_14",
    "GLD_RSI_14",
)
VOL_LEVELS = (
    "SPY_Rolling_Vol_30d",
    "BTC_Rolling_Vol_30d",
    "Gold_Rolling_Vol_30d",
)
LEVEL_COLS = MACRO_LEVELS + VOL_LEVELS
RET_COLS = ("gold_ret", "spy_ret", "btc_ret")
STRESS_FEATURE_COLS = ("stress_hys_t", "stress_any_t")

FEATURE_COUNT = (
    len(LEVEL_COLS)  # contemporaneous
    + len(LEVEL_COLS) * len(LAGS)  # lags of macro + vol
    + len(RET_COLS) * len(LAGS)  # return lags
    + len(STRESS_FEATURE_COLS)
)


def log_returns(prices: pd.Series) -> pd.Series:
    """r_t = log(P_t) - log(P_{t-1}). First observation is NaN."""
    return np.log(prices) - np.log(prices.shift(1))


def annualized_rolling_vol(returns: pd.Series, window: int = GOLD_VOL_WINDOW) -> pd.Series:
    return returns.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def add_returns_and_gold_vol(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["gold_ret"] = log_returns(out["Gold"])
    out["spy_ret"] = log_returns(out["Equities_US"])
    out["btc_ret"] = log_returns(out["Crypto_Bitcoin"])
    out["Gold_Rolling_Vol_30d"] = annualized_rolling_vol(out["gold_ret"])
    return out


def feature_names() -> list[str]:
    """Column order matches the Milestone 2 / 3 notebooks."""
    names: list[str] = []
    for col in LEVEL_COLS:
        names.append(col)
        for lag in LAGS:
            names.append(f"{col}_lag{lag}")
    for col in RET_COLS:
        for lag in LAGS:
            names.append(f"{col}_lag{lag}")
    names.extend(STRESS_FEATURE_COLS)
    if len(names) != FEATURE_COUNT:
        raise RuntimeError(f"feature name count {len(names)} != {FEATURE_COUNT}")
    return names


def _lag_frame(out: pd.DataFrame) -> pd.DataFrame:
    for col in LEVEL_COLS:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    for col in RET_COLS:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    out["stress_hys_t"] = out["stress_hys"]
    out["stress_any_t"] = out["stress_any"]
    return out


def add_forward_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Shift outcomes forward so row t predicts t+1 / t+5."""
    out = df.copy()
    gold_next = out["gold_ret"].shift(-1)
    spy_next = out["spy_ret"].shift(-1)
    y = (gold_next > spy_next).astype(float)
    y[gold_next.isna() | spy_next.isna()] = np.nan
    out["y_gold_gt_spy"] = y
    out["y_gold_ret_1"] = gold_next
    out["y_spy_vol_1"] = out["SPY_Rolling_Vol_30d"].shift(-1)
    out["y_spy_vol_5"] = out["SPY_Rolling_Vol_30d"].shift(-5)
    out["y_btc_vol_1"] = out["BTC_Rolling_Vol_30d"].shift(-1)
    out["y_btc_vol_5"] = out["BTC_Rolling_Vol_30d"].shift(-5)
    out["y_vix_1"] = out["Volatility_Index"].shift(-1)
    out["y_vix_5"] = out["Volatility_Index"].shift(-5)
    if "stress_any" in out.columns:
        out["y_stress_any_5"] = out["stress_any"].shift(-5)
    if "stress_hys" in out.columns:
        out["y_stress_hys_5"] = out["stress_hys"].shift(-5)
    return out


def build_supervised_frame(
    raw: pd.DataFrame,
    stress_quantile: float = 0.66,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Returns, regimes, lags, and forward targets on a date-sorted panel."""
    out = add_returns_and_gold_vol(raw)
    out = add_stress_flags(out, quantile=stress_quantile, thresholds=thresholds)
    out = _lag_frame(out)
    out = add_forward_targets(out)
    return out


def drop_unusable(df: pd.DataFrame, extra: tuple[str, ...] = ()) -> pd.DataFrame:
    cols = feature_names() + list(extra)
    return df.dropna(subset=cols).reset_index(drop=True)


def X_from(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, feature_names()].astype(float)
