"""Chronological splits, walk-forward folds, and metrics.

No shuffling. Train dates are strictly before test dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

MODEL_E_TRAIN_END = pd.Timestamp("2023-11-25")
MODEL_E_TEST_START = pd.Timestamp("2023-11-26")
WALK_FORWARD_YEARS = range(2019, 2027)


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def chronological_split(
    df: pd.DataFrame,
    train_end: pd.Timestamp | None = None,
    test_start: pd.Timestamp | None = None,
    train_frac: float = 0.8,
) -> Split:
    if "Date" not in df.columns:
        raise ValueError("frame needs a Date column")
    ordered = df.sort_values("Date").reset_index(drop=True)
    if train_end is None or test_start is None:
        cut = int(len(ordered) * train_frac)
        train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    else:
        train = ordered.loc[ordered["Date"] <= train_end]
        test = ordered.loc[ordered["Date"] >= test_start]
    if train.empty or test.empty:
        raise ValueError("empty train or test split")
    if train["Date"].max() >= test["Date"].min():
        raise ValueError("train dates are not strictly before test dates")
    return Split(train=train.reset_index(drop=True), test=test.reset_index(drop=True))


def model_e_split(df: pd.DataFrame) -> Split:
    return chronological_split(
        df, train_end=MODEL_E_TRAIN_END, test_start=MODEL_E_TEST_START
    )


def walk_forward_folds(df: pd.DataFrame, years=WALK_FORWARD_YEARS) -> list[tuple[int, Split]]:
    """Expanding window: train on all dates before year Y, test on year Y."""
    ordered = df.sort_values("Date").reset_index(drop=True)
    folds: list[tuple[int, Split]] = []
    for year in years:
        start = pd.Timestamp(f"{year}-01-01")
        end = pd.Timestamp(f"{year}-12-31")
        train = ordered.loc[ordered["Date"] < start]
        test = ordered.loc[(ordered["Date"] >= start) & (ordered["Date"] <= end)]
        if train.empty or test.empty:
            continue
        if train["Date"].max() >= test["Date"].min():
            raise ValueError(f"walk-forward leak in {year}")
        folds.append((year, Split(train=train.reset_index(drop=True), test=test.reset_index(drop=True))))
    return folds


def classification_metrics(y_true, y_pred, y_proba=None) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        proba = np.clip(proba, 1e-7, 1 - 1e-7)
        out["roc_auc"] = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else 0.5
        out["pr_auc"] = float(average_precision_score(y_true, proba))
        out["brier"] = float(brier_score_loss(y_true, proba))
        out["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1]))
    return out


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def assert_chronological(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train["Date"].max() >= test["Date"].min():
        raise AssertionError("train is not strictly before test")
