# Methodology

Reconstructed from the May 2026 report. Exact original hyperparameters
and some implementation constants were not recovered. See
[RECOVERY_NOTES.md](RECOVERY_NOTES.md).

## Pipeline

```
Raw daily panel (Kaggle)
        ↓
Log returns + rolling features
        ↓
Stress regimes (66th percentile of HYS / FSI / VIX)
        ↓
Descriptive safe-haven tests
  (means, bootstrap CIs, volatility, rolling correlation, events)
        ↓
Predictive models  (E: Gold>SPY tomorrow; F: volatility; G: future stress)
        ↓
Robustness  (thresholds, SHAP, walk-forward)
        ↓
Interpretation
```

No random shuffling anywhere. Targets are shifted forward *before* the
train/test cut.

## Returns

Daily log return:

\[
r_{i,t} = \log P_{i,t} - \log P_{i,t-1}
\]

Price levels were non-stationary; returns were the modeling object.
Gold volatility is reconstructed as a 30-day rolling standard deviation
of Gold log returns, annualized by \(\sqrt{252}\). SPY and Bitcoin
30-day volatilities are taken from the panel.

## Stress regimes

Baseline cut: **66th percentile** of each proxy on the full 4,150-day
sample. That isolates elevated stress while leaving enough observations
for tests and models. It is a retrospective, full-sample quantile — a
documented limitation for any live system.

| Flag | Rule (published) | Share of days |
|---|---|---|
| `stress_hys` | High_Yield_Spread ≥ 4.400 | 34.2% |
| `stress_fsi` | Financial_Stress_Index ≥ −0.172 | 34.1% |
| `stress_vix` | Volatility_Index ≥ 19.060 | 34.0% |
| `stress_any` | at least one flag | 56.3% |
| `stress_all` | all three flags | 14.3% |

Robustness repeats the safe-haven tests at the 75th and 90th percentiles.

## Feature matrix (Model E / F / G)

The report states **55 predictors**. The exact column list was not
recovered. The reconstruction in `src/macro_stress/features.py`:

- contemporaneous macro-financial levels and 30-day vols (11)
- 1 / 3 / 5-day lags of those levels (33)
- Gold / SPY / BTC return lags at 1 / 3 / 5 days (9)
- contemporaneous `stress_any` and `stress_all` (2)

That is 55 columns, and it includes every feature named in the report
(`spy_ret_lag1/3`, `gold_ret_lag1/3`, `High_Yield_Spread_lag5`,
`Financial_Stress_Index_lag5`, `Volatility_Index`, `GLD_RSI_14`,
`stress_any_t`).

Leakage rules:

- features at \(t\) use only information available at \(t\)
- \(y_{t}\) is the *next* day's outcome (`shift(-1)` / `shift(-5)`)
- the last rows without a future target are dropped
- chronological split, never shuffled

## Splits

**Model E headline split** (published):

- train: 2014-11-21 → 2023-11-25
- test: 2023-11-26 → 2026-02-24
- 822 test rows; Gold beats SPY on 38.3% of them

**Walk-forward:** expanding window, test years 2019–2026. Train on all
dates strictly before 1 January of the test year.

**Model F / G:** chronological 80/20 on the usable supervised frame.

## Models

| Task | Target | Models |
|---|---|---|
| **E** | \(1\{r^{\text{Gold}}_{t+1} > r^{\text{SPY}}_{t+1}\}\) | Majority, VIX rule, logistic (balanced), XGBoost, LightGBM, CatBoost, Random Forest, MLP, stacking |
| **F** | SPY 30d vol, BTC 30d vol, VIX at \(h=1,5\) | Persistence, Ridge, XGBoost |
| **G** | `stress_any` at \(h=5\) (robustness target) | Majority, persistence, logistic, XGBoost ± class weight / SMOTE / threshold |

The original high-yield-stress-at-h=5 target had a ~0.7% positive rate
on the test window, which made accuracy useless. The report switched
the headline G metric to PR–AUC on `stress_any`.

Hyperparameters other than "class-balanced logistic" were not recovered.
The reconstruction uses library defaults plus a small documented XGBoost
setting (`n_estimators=200`, `max_depth=4`, `learning_rate=0.05`,
`random_state=42`). Those choices are **not** tuned to published scores.

## Event windows (reconstructed bounds)

Exact notebook timestamps were not recovered. Economically standard
windows used in the reconstruction:

| Episode | Reconstruction bounds |
|---|---|
| COVID crash | 2020-02-19 → 2020-03-23 |
| Inflation / rate shock | 2022-01-03 → 2022-10-14 |
| Banking stress | 2023-03-08 → 2023-03-24 |

Published cumulative returns and max drawdowns in the report remain the
source of truth if these bounds produce different path sums.
