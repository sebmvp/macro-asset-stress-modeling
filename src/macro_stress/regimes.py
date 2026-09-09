"""Stress-regime construction from the final report.

Baseline: 66th percentile of HYS, FSI, and VIX.
Published thresholds (full-sample 66th percentile on the 4,150-row panel):
    HYS >= 4.400
    FSI >= -0.172
    VIX >= 19.060

Descriptive analysis in the original project used full-sample quantiles.
That is a documented limitation for any live system.
"""

from __future__ import annotations

import pandas as pd

STRESS_QUANTILE = 0.66

# Anchors from the May 2026 report (Table 1). Used as checksums, not inputs,
# unless the caller passes use_historical_thresholds=True.
HISTORICAL_THRESHOLDS = {
    "High_Yield_Spread": 4.400,
    "Financial_Stress_Index": -0.172,
    "Volatility_Index": 19.060,
}

PROXY_COLUMNS = {
    "stress_hys": "High_Yield_Spread",
    "stress_fsi": "Financial_Stress_Index",
    "stress_vix": "Volatility_Index",
}


def compute_thresholds(
    df: pd.DataFrame,
    quantile: float = STRESS_QUANTILE,
) -> dict[str, float]:
    return {col: float(df[col].quantile(quantile)) for col in HISTORICAL_THRESHOLDS}


def add_stress_flags(
    df: pd.DataFrame,
    quantile: float = STRESS_QUANTILE,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add stress_hys / stress_fsi / stress_vix / stress_any / stress_all.

    Flags are contemporaneous (time t). They must not be shifted backward
    onto earlier rows — that would leak future stress into features.
    """
    out = df.copy()
    cuts = thresholds if thresholds is not None else compute_thresholds(out, quantile)
    for flag, col in PROXY_COLUMNS.items():
        out[flag] = (out[col] >= cuts[col]).astype(int)
    out["stress_any"] = (
        (out["stress_hys"] == 1) | (out["stress_fsi"] == 1) | (out["stress_vix"] == 1)
    ).astype(int)
    out["stress_all"] = (
        (out["stress_hys"] == 1) & (out["stress_fsi"] == 1) & (out["stress_vix"] == 1)
    ).astype(int)
    return out


def add_milestone1_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Milestone 1 used median HYS and a yield-curve sign split."""
    out = df.copy()
    out["high_stress_median"] = (
        out["High_Yield_Spread"] > out["High_Yield_Spread"].median()
    ).astype(int)
    out["inverted_yield"] = (out["Yield_Curve_Spread"] < 0).astype(int)
    return out
