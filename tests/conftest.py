"""Synthetic daily panel with the Kaggle schema. Not the real dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_stress.data import REQUIRED_COLUMNS


def make_panel(n: int = 120, start: str = "2018-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range(start, periods=n, freq="D")
    spy = 200 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n)))
    gold = 120 * np.exp(np.cumsum(rng.normal(0.0001, 0.007, n)))
    btc = 4000 * np.exp(np.cumsum(rng.normal(0.001, 0.04, n)))
    hys = 3.5 + np.abs(rng.normal(0, 0.8, n))
    fsi = rng.normal(-0.3, 0.4, n)
    vix = 15 + np.abs(rng.normal(0, 6, n))
    df = pd.DataFrame(
        {
            "Date": dates,
            "Equities_US": spy,
            "Equities_Tech": spy * 1.1,
            "Equities_Emerging": spy * 0.4,
            "Bonds_LongTerm": 90 + rng.normal(0, 2, n),
            "Gold": gold,
            "Oil": 50 + rng.normal(0, 3, n),
            "Volatility_Index": vix,
            "Crypto_Bitcoin": btc,
            "Yield_Curve_Spread": rng.normal(0.2, 0.4, n),
            "High_Yield_Spread": hys,
            "Financial_Stress_Index": fsi,
            "SPY_Drawdown": -np.abs(rng.normal(0.05, 0.04, n)),
            "SPY_Rolling_Vol_30d": np.abs(rng.normal(0.12, 0.04, n)),
            "BTC_Rolling_Vol_30d": np.abs(rng.normal(0.55, 0.08, n)),
            "Stock_Bond_Corr_90d": rng.normal(0.1, 0.2, n),
            "SPY_RSI_14": rng.uniform(30, 70, n),
            "GLD_RSI_14": rng.uniform(30, 70, n),
        }
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    assert not missing
    return df


@pytest.fixture
def panel() -> pd.DataFrame:
    return make_panel()
