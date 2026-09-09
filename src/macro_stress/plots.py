"""Figures for the reconstructed run. Original report figures stay in figures/."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .analysis import EVENT_WINDOWS


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2c2838",
            "axes.labelcolor": "#2c2838",
            "text.color": "#2c2838",
            "xtick.color": "#2c2838",
            "ytick.color": "#2c2838",
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.dpi": 140,
        }
    )


def plot_volatility_by_regime(df: pd.DataFrame, path: Path, flag: str = "stress_any") -> None:
    _style()
    high = df[flag] == 1
    labels = ["Gold", "SPY", "BTC"]
    cols = ["Gold_Rolling_Vol_30d", "SPY_Rolling_Vol_30d", "BTC_Rolling_Vol_30d"]
    low_means = [df.loc[~high, c].mean() for c in cols]
    high_means = [df.loc[high, c].mean() for c in cols]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([i - 0.18 for i in x], low_means, width=0.36, label="Low stress", color="#c4b5fd")
    ax.bar([i + 0.18 for i in x], high_means, width=0.36, label="High stress", color="#7c3aed")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("Annualized 30-day volatility")
    ax.set_title(f"Volatility by {flag} regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_event_windows(df: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), sharey=True)
    titles = {
        "covid_2020": "COVID crash 2020",
        "inflation_2022": "Inflation/rates 2022",
        "banking_2023": "Banking stress 2023",
    }
    colors = {"gold_ret": "#8b5cf6", "spy_ret": "#6b6578", "btc_ret": "#a1a1aa"}
    for ax, (name, (start, end)) in zip(axes, EVENT_WINDOWS.items()):
        window = df.loc[(df["Date"] >= start) & (df["Date"] <= end)].copy()
        for col, color in colors.items():
            ax.plot(window["Date"], window[col].cumsum(), color=color, lw=1.6)
        ax.set_title(titles[name], fontsize=10)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        ax.axhline(0, color="#d4d4d8", lw=0.8)
    axes[0].set_ylabel("Cumulative log return")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_model_e_auc(rows: list[dict], path: Path) -> None:
    _style()
    names = [r["model"] for r in rows]
    aucs = [r.get("roc_auc", float("nan")) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(names, aucs, color="#7c3aed")
    ax.set_ylim(0.45, 0.75)
    ax.set_ylabel("ROC–AUC")
    ax.set_title("Model E ranking (reconstructed run)")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
