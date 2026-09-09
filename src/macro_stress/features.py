"""Log returns, lags, Gold volatility, and the 55-predictor matrix.

RECONSTRUCTED. The May 2026 report states that the final supervised matrix
has 55 predictors: contemporaneous macro-financial levels, 30-day rolling
volatilities, 1/3/5-day lags of those variables, Gold/SPY/BTC lagged
returns, and contemporaneous stress indicators. The exact column list was
not recovered. The list below is a documented reconstruction that:

  * produces 55 columns
  * includes every named feature in the report (spy_ret_lag1/3,
    gold_ret_lag1/3, High_Yield_Spread_lag5, Financial_Stress_Index_lag5,
    Volatility_Index, GLD_RSI_14, stress_any_t)
  * drops the first ~35 rows so the first usable date is 2014-11-21 on the
    canonical 4,150-row panel (30-day Gold vol warm-up + 5-day lag)

Leakage rules (report §3.3):
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

LEVEL_COLS = (
    "High_Yield_Spread",
    "Yield_Curve_Spread",
    "Financial_Stress_Index",
    "Volatility_Index",
    "Stock_Bond_Corr_90d",
    "SPY_Drawdown",
    "SPY_RSI_14",
    "GLD_RSI_14",
    "SPY_Rolling_Vol_30d",
    "BTC_Rolling_Vol_30d",
    "Gold_Rolling_Vol_30d",
)

RETURN_PREFIXES = ("gold", "spy", "btc")
STRESS_FEATURE_COLS = ("stress_any", "stress_all")

FEATURE_COUNT = (
    len(LEVEL_COLS)  # contemporaneous
    + len(LEVEL_COLS) * len(LAGS)  # lags
    + len(RETURN_PREFIXES) * len(LAGS)  # return lags
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
    names: list[str] = list(LEVEL_COLS)
    for col in LEVEL_COLS:
        for lag in LAGS:
            names.append(f"{col}_lag{lag}")
    for prefix in RETURN_PREFIXES:
        for lag in LAGS:
            names.append(f"{prefix}_ret_lag{lag}")
    names.append("stress_any_t")
    names.append("stress_all")
    if len(names) != FEATURE_COUNT:
        raise RuntimeError(f"feature name count {len(names)} != {FEATURE_COUNT}")
    return names


def _lag_frame(out: pd.DataFrame) -> pd.DataFrame:
    for col in LEVEL_COLS:
        for lag in LAGS:
            out[f"{col}_lag{lag}"] = out[col].shift(lag)
    for prefix in RETURN_PREFIXES:
        for lag in LAGS:
            out[f"{prefix}_ret_lag{lag}"] = out[f"{prefix}_ret"].shift(lag)
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
