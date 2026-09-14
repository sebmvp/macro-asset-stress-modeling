"""The reconstruction must pin the historical project window."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from conftest import make_panel
from macro_stress.analysis import left_tail_overlap, max_drawdown
from macro_stress.data import SAMPLE_END, SAMPLE_START, load_raw, restrict_to_project_sample


def test_restrict_drops_rows_outside_the_published_window():
    extra = make_panel(n=5, start="2026-03-01")
    extra["Date"] = pd.to_datetime(extra["Date"])
    inside = make_panel(n=8, start="2018-01-01")
    inside["Date"] = pd.to_datetime(inside["Date"])
    mixed = pd.concat([inside, extra], ignore_index=True)
    out = restrict_to_project_sample(mixed)
    assert out["Date"].max() <= SAMPLE_END
    assert out["Date"].min() >= SAMPLE_START
    assert len(out) == len(inside)


def test_load_raw_restricts_updated_kaggle_extracts(tmp_path: Path):
    panel = make_panel(n=10, start="2018-06-01")
    future = make_panel(n=4, start="2027-01-01")
    combined = pd.concat([panel, future], ignore_index=True)
    path = tmp_path / "Global_Market_Stress_and_Liquidity_Regimes.csv"
    combined.to_csv(path, index=False)
    loaded = load_raw(path)
    assert loaded["Date"].max() <= SAMPLE_END
    assert (loaded["Date"] >= "2027-01-01").sum() == 0


def test_max_drawdown_is_non_positive():
    s = pd.Series([0.01, -0.02, -0.05, 0.03])
    assert max_drawdown(s) <= 0.0


def test_left_tail_overlap_bounds(panel):
    from macro_stress.features import add_returns_and_gold_vol

    frame = add_returns_and_gold_vol(panel).dropna(subset=["gold_ret", "spy_ret", "btc_ret"])
    gold = left_tail_overlap(frame, "gold_ret")
    btc = left_tail_overlap(frame, "btc_ret")
    assert 0.0 <= gold <= 1.0
    assert 0.0 <= btc <= 1.0
