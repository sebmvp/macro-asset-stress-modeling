"""Descriptive safe-haven tests from the final report.

Event-window calendar bounds were not recovered as exact timestamps.
The windows below are economically standard and marked RECONSTRUCTED;
cumulative returns will not be treated as a failed checksum if they differ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# RECONSTRUCTED bounds — original notebook date cuts were not recovered.
EVENT_WINDOWS = {
    "covid_2020": (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
    "inflation_2022": (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-14")),
    "banking_2023": (pd.Timestamp("2023-03-08"), pd.Timestamp("2023-03-24")),
}

M1_VOL_FEATURES = [
    "Volatility_Index",
    "Yield_Curve_Spread",
    "High_Yield_Spread",
    "Financial_Stress_Index",
    "Stock_Bond_Corr_90d",
    "SPY_Drawdown",
    "SPY_RSI_14",
    "SPY_Rolling_Vol_30d",
]


def mean_return_table(df: pd.DataFrame, flag: str) -> dict[str, float]:
    high = df[flag] == 1
    low = df[flag] == 0
    gold_high = float(df.loc[high, "gold_ret"].mean())
    spy_high = float(df.loc[high, "spy_ret"].mean())
    btc_high = float(df.loc[high, "btc_ret"].mean())
    spread_high = float((df.loc[high, "gold_ret"] - df.loc[high, "spy_ret"]).mean())
    spread_low = float((df.loc[low, "gold_ret"] - df.loc[low, "spy_ret"]).mean())
    delta = spread_high - spread_low
    ci_low, ci_high = bootstrap_delta(
        df.loc[high, "gold_ret"] - df.loc[high, "spy_ret"],
        df.loc[low, "gold_ret"] - df.loc[low, "spy_ret"],
    )
    return {
        "gold_high": gold_high,
        "spy_high": spy_high,
        "btc_high": btc_high,
        "delta_gold_spy": delta,
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "share_high": float(high.mean()),
    }


def bootstrap_delta(
    high: pd.Series,
    low: pd.Series,
    n_boot: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    high_v = high.dropna().to_numpy()
    low_v = low.dropna().to_numpy()
    if len(high_v) < 5 or len(low_v) < 5:
        return float("nan"), float("nan")
    draws = []
    for _ in range(n_boot):
        h = rng.choice(high_v, size=len(high_v), replace=True).mean()
        l = rng.choice(low_v, size=len(low_v), replace=True).mean()
        draws.append(h - l)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(lo), float(hi)


def vol_by_regime(df: pd.DataFrame, flag: str = "stress_any") -> dict[str, float]:
    high = df[flag] == 1
    out = {}
    for asset, col in (
        ("gold", "Gold_Rolling_Vol_30d"),
        ("spy", "SPY_Rolling_Vol_30d"),
        ("btc", "BTC_Rolling_Vol_30d"),
    ):
        out[f"{asset}_low"] = float(df.loc[~high, col].mean())
        out[f"{asset}_high"] = float(df.loc[high, col].mean())
    return out


def event_cumulative_returns(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result = {}
    for name, (start, end) in EVENT_WINDOWS.items():
        window = df.loc[(df["Date"] >= start) & (df["Date"] <= end)]
        result[name] = {
            "gold": float(window["gold_ret"].sum()),
            "spy": float(window["spy_ret"].sum()),
            "btc": float(window["btc_ret"].sum()),
            "n": int(len(window)),
            "start": str(start.date()),
            "end": str(end.date()),
        }
    return result


def rolling_corr_by_regime(
    df: pd.DataFrame,
    left: str,
    right: str,
    flag: str,
    window: int = 60,
) -> dict[str, float]:
    """Mean of a rolling correlation inside vs outside a stress flag."""
    corr = df[left].rolling(window, min_periods=window).corr(df[right])
    high = df[flag] == 1
    return {
        "low": float(corr.loc[~high].mean()),
        "high": float(corr.loc[high].mean()),
        "window": float(window),
    }


def max_drawdown(log_returns: pd.Series) -> float:
    """Peak-to-trough drawdown on a log-return path (wealth = exp(cumsum))."""
    wealth = pd.Series(np.exp(log_returns.fillna(0).to_numpy().cumsum()), dtype=float)
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def event_drawdowns(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, (start, end) in EVENT_WINDOWS.items():
        window = df.loc[(df["Date"] >= start) & (df["Date"] <= end)]
        result[name] = {
            "gold": max_drawdown(window["gold_ret"]),
            "spy": max_drawdown(window["spy_ret"]),
            "btc": max_drawdown(window["btc_ret"]),
        }
    return result


def left_tail_overlap(df: pd.DataFrame, asset: str, spy: str = "spy_ret", q: float = 0.05) -> float:
    """Share of SPY's worst-q days that are also worst-q days for `asset`."""
    spy_cut = df[spy].quantile(q)
    asset_cut = df[asset].quantile(q)
    spy_bad = df[spy] <= spy_cut
    asset_bad = df[asset] <= asset_cut
    n = int(spy_bad.sum())
    if n == 0:
        return float("nan")
    return float((spy_bad & asset_bad).sum() / n)


def threshold_robustness(df: pd.DataFrame, quantiles: tuple[float, ...] = (0.66, 0.75, 0.90)) -> dict:
    """Recompute Gold–SPY spread widening at several stress percentiles."""
    from .regimes import PROXY_COLUMNS, add_stress_flags

    usable = df.dropna(subset=["gold_ret", "spy_ret"]).copy()
    out: dict[str, dict[str, dict[str, float]]] = {}
    for q in quantiles:
        flagged = add_stress_flags(usable, quantile=q)
        key = f"p{int(round(q * 100))}"
        out[key] = {}
        for flag in list(PROXY_COLUMNS) + ["stress_any", "stress_all"]:
            stats = mean_return_table(flagged, flag)
            out[key][flag] = {
                "delta_gold_spy": stats["delta_gold_spy"],
                "ci_lower": stats["ci_lower"],
                "share_high": stats["share_high"],
            }
    return out


def spearman_stress_vs_spread(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Rank correlation between stress intensity and the Gold–SPY return spread."""
    usable = df.dropna(subset=["gold_ret", "spy_ret"]).copy()
    spread = usable["gold_ret"] - usable["spy_ret"]
    result: dict[str, dict[str, float]] = {}
    for name, col in (
        ("vix", "Volatility_Index"),
        ("hys", "High_Yield_Spread"),
        ("fsi", "Financial_Stress_Index"),
    ):
        pair = pd.DataFrame({"x": usable[col], "y": spread}).dropna()
        rx = pair["x"].rank().to_numpy(dtype=float)
        ry = pair["y"].rank().to_numpy(dtype=float)
        rho = float(np.corrcoef(rx, ry)[0, 1]) if len(pair) > 2 else float("nan")
        result[name] = {"rho": rho, "n": int(len(pair))}
    return result


def spy_beta(df: pd.DataFrame, flag: str = "stress_any") -> dict[str, float]:
    def _beta(frame: pd.DataFrame, ycol: str) -> float:
        x = frame["spy_ret"].to_numpy()
        y = frame[ycol].to_numpy()
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 10 or np.var(x) == 0:
            return float("nan")
        return float(np.cov(y, x, ddof=1)[0, 1] / np.var(x, ddof=1))

    high = df[flag] == 1
    return {
        "gold_calm": _beta(df.loc[~high], "gold_ret"),
        "gold_stress": _beta(df.loc[high], "gold_ret"),
        "btc_calm": _beta(df.loc[~high], "btc_ret"),
        "btc_stress": _beta(df.loc[high], "btc_ret"),
    }
