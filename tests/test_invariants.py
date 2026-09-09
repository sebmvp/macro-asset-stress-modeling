"""Invariants for log returns, lags, targets, stress flags, and splits."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_panel
from macro_stress.evaluation import chronological_split, walk_forward_folds
from macro_stress.features import (
    FEATURE_COUNT,
    add_forward_targets,
    add_returns_and_gold_vol,
    build_supervised_frame,
    drop_unusable,
    feature_names,
    log_returns,
)
from macro_stress.regimes import add_stress_flags


def test_log_return_is_log_pt_minus_log_pt_minus_1(panel):
    r = log_returns(panel["Gold"])
    assert pd.isna(r.iloc[0])
    expected = np.log(panel["Gold"].iloc[5]) - np.log(panel["Gold"].iloc[4])
    assert r.iloc[5] == expected


def test_target_uses_next_day_returns_not_same_day(panel):
    frame = add_forward_targets(add_returns_and_gold_vol(panel))
    t = 10
    gold_next = frame["gold_ret"].iloc[t + 1]
    spy_next = frame["spy_ret"].iloc[t + 1]
    assert frame["y_gold_gt_spy"].iloc[t] == float(gold_next > spy_next)
    # Same-day comparison must not be the target.
    gold_now = frame["gold_ret"].iloc[t]
    spy_now = frame["spy_ret"].iloc[t]
    if (gold_now > spy_now) != (gold_next > spy_next):
        assert frame["y_gold_gt_spy"].iloc[t] != float(gold_now > spy_now)


def test_last_row_has_no_next_day_target(panel):
    frame = add_forward_targets(add_returns_and_gold_vol(panel))
    assert pd.isna(frame["y_gold_gt_spy"].iloc[-1])
    assert pd.isna(frame["y_spy_vol_5"].iloc[-1])
    assert pd.isna(frame["y_spy_vol_5"].iloc[-5])
    assert frame["y_spy_vol_5"].iloc[-6] == frame["SPY_Rolling_Vol_30d"].iloc[-1]


def test_chronological_split_is_ordered(panel):
    split = chronological_split(panel, train_frac=0.8)
    assert split.train["Date"].max() < split.test["Date"].min()
    assert split.train["Date"].is_monotonic_increasing
    assert split.test["Date"].is_monotonic_increasing


def test_lag_k_equals_value_k_days_earlier(panel):
    frame = build_supervised_frame(panel)
    t = 40
    assert frame["High_Yield_Spread_lag5"].iloc[t] == frame["High_Yield_Spread"].iloc[t - 5]
    assert frame["spy_ret_lag3"].iloc[t] == frame["spy_ret"].iloc[t - 3]
    assert frame["gold_ret_lag1"].iloc[t] == frame["gold_ret"].iloc[t - 1]


def test_features_do_not_contain_future_prices(panel):
    frame = build_supervised_frame(panel)
    t = 50
    # A lag feature at t must not equal the next day's raw series.
    assert frame["High_Yield_Spread_lag1"].iloc[t] != frame["High_Yield_Spread"].iloc[t + 1]
    assert frame["gold_ret_lag1"].iloc[t] != frame["gold_ret"].iloc[t + 1]
    # Contemporaneous Gold price is not a predictor; next-day Gold is not either.
    names = feature_names()
    assert "Gold" not in names
    assert "y_gold_gt_spy" not in names


def test_stress_any_is_or_and_stress_all_is_and(panel):
    flagged = add_stress_flags(panel)
    reconstructed_any = (
        (flagged["stress_hys"] == 1)
        | (flagged["stress_fsi"] == 1)
        | (flagged["stress_vix"] == 1)
    ).astype(int)
    reconstructed_all = (
        (flagged["stress_hys"] == 1)
        & (flagged["stress_fsi"] == 1)
        & (flagged["stress_vix"] == 1)
    ).astype(int)
    pd.testing.assert_series_equal(flagged["stress_any"], reconstructed_any, check_names=False)
    pd.testing.assert_series_equal(flagged["stress_all"], reconstructed_all, check_names=False)


def test_stress_threshold_is_sample_quantile(panel):
    flagged = add_stress_flags(panel, quantile=0.66)
    cut = panel["High_Yield_Spread"].quantile(0.66)
    expected = (panel["High_Yield_Spread"] >= cut).astype(int)
    pd.testing.assert_series_equal(flagged["stress_hys"], expected, check_names=False)


def test_feature_target_timestamps_align(panel):
    frame = drop_unusable(build_supervised_frame(panel), extra=("y_gold_gt_spy",))
    row = frame.iloc[20]
    # Feature row and target share the same Date; the outcome is the next day.
    assert row["Date"] == frame["Date"].iloc[20]
    raw = build_supervised_frame(panel)
    nxt = raw.loc[raw["Date"] > row["Date"]].iloc[0]
    assert row["y_gold_gt_spy"] == float(nxt["gold_ret"] > nxt["spy_ret"])


def test_walk_forward_is_expanding_and_ordered():
    long_panel = make_panel(n=500, start="2018-06-01")
    frame = drop_unusable(build_supervised_frame(long_panel), extra=("y_gold_gt_spy",))
    folds = walk_forward_folds(frame, years=range(2019, 2021))
    assert folds, "expected at least one fold"
    prev_train_n = 0
    for year, split in folds:
        assert split.train["Date"].max() < split.test["Date"].min()
        assert split.test["Date"].dt.year.eq(year).all()
        assert len(split.train) >= prev_train_n
        prev_train_n = len(split.train)


def test_reconstructed_matrix_has_55_predictors(panel):
    names = feature_names()
    assert len(names) == FEATURE_COUNT == 55
    assert len(set(names)) == 55
    frame = drop_unusable(build_supervised_frame(panel), extra=("y_gold_gt_spy",))
    assert not frame[names].isna().any().any()
    for required in (
        "spy_ret_lag1",
        "spy_ret_lag3",
        "gold_ret_lag1",
        "gold_ret_lag3",
        "High_Yield_Spread_lag5",
        "Financial_Stress_Index_lag5",
        "Volatility_Index",
        "GLD_RSI_14",
        "stress_any_t",
    ):
        assert required in names
