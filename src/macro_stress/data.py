"""Load and validate the Kaggle macro-stress panel.

Canonical source (not redistributed in this repo):
  Kanchana1990, Algorithmic Trading, Macro Stress, and Asset Regimes
  https://www.kaggle.com/datasets/kanchana1990/algorithmic-trading-macro-stress-and-asset-regimes
  File: Global_Market_Stress_and_Liquidity_Regimes.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CANONICAL_CSV_NAME = "Global_Market_Stress_and_Liquidity_Regimes.csv"
EXPECTED_N_ROWS = 4150
SAMPLE_START = pd.Timestamp("2014-10-17")
SAMPLE_END = pd.Timestamp("2026-02-25")

REQUIRED_COLUMNS = (
    "Date",
    "Equities_US",
    "Gold",
    "Crypto_Bitcoin",
    "Volatility_Index",
    "Yield_Curve_Spread",
    "High_Yield_Spread",
    "Financial_Stress_Index",
    "SPY_Drawdown",
    "SPY_Rolling_Vol_30d",
    "BTC_Rolling_Vol_30d",
    "Stock_Bond_Corr_90d",
    "SPY_RSI_14",
    "GLD_RSI_14",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def find_dataset(explicit: str | Path | None = None) -> Path | None:
    """Return the CSV path if present, else None."""
    if explicit is not None:
        path = Path(explicit)
        return path if path.is_file() else None
    candidates = [
        DATA_DIR / CANONICAL_CSV_NAME,
        DATA_DIR / "data.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    matches = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []
    if len(matches) == 1:
        return matches[0]
    return None


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("dataset is missing a Date column")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return human-readable issues. Empty list means the panel matches the report."""
    issues: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        return issues
    n = len(df)
    if n != EXPECTED_N_ROWS:
        issues.append(f"row count {n} != {EXPECTED_N_ROWS}")
    start, end = df["Date"].min(), df["Date"].max()
    if start != SAMPLE_START:
        issues.append(f"start {start.date()} != {SAMPLE_START.date()}")
    if end != SAMPLE_END:
        issues.append(f"end {end.date()} != {SAMPLE_END.date()}")
    return issues


def data_check(explicit: str | Path | None = None) -> dict:
    path = find_dataset(explicit)
    if path is None:
        return {"found": False, "path": None, "issues": ["csv not found"]}
    raw = load_raw(path)
    return {
        "found": True,
        "path": str(path),
        "n": int(len(raw)),
        "issues": validate_dataset(raw),
        "start": str(raw["Date"].min().date()),
        "end": str(raw["Date"].max().date()),
    }
