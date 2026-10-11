# Methodology

Pipeline used in the Jupyter notebooks (`notebooks/01`–`04`) and in `src/macro_stress`.

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
Gold volatility is a 30-day rolling standard deviation of Gold log
returns, annualized by \(\sqrt{252}\). SPY and Bitcoin 30-day
volatilities are taken from the panel.

## Stress regimes

Baseline cut: **66th percentile** of each proxy on the full 4,150-day
sample. That isolates elevated stress while leaving enough observations
for tests and models. It is a retrospective, full-sample quantile — a
documented limitation for any live system.

| Flag | Rule | Share of days |
|---|---|---|
| `stress_hys` | High_Yield_Spread ≥ 4.400 | 34.2% |
| `stress_fsi` | Financial_Stress_Index ≥ −0.172 | 34.1% |
| `stress_vix` | Volatility_Index ≥ 19.060 | 34.0% |
| `stress_any` | at least one flag | 56.3% |
| `stress_all` | all three flags | 14.3% |

Robustness repeats the safe-haven tests at the 75th and 90th percentiles.

## Feature matrix (Model E / F / G)

**55 predictors**, matching the course notebooks:

- contemporaneous macro-financial levels and 30-day vols (11)
- 1 / 3 / 5-day lags of those levels (33)
- Gold / SPY / BTC return lags at 1 / 3 / 5 days (9)
- contemporaneous `stress_hys_t` and `stress_any_t` (2)

Named features include `spy_ret_lag1/3`, `gold_ret_lag1/3`,
`High_Yield_Spread_lag5`, `Financial_Stress_Index_lag5`,
`Volatility_Index`, `GLD_RSI_14`, `stress_any_t`.

Leakage rules:

- features at \(t\) use only information available at \(t\)
- \(y_{t}\) is the *next* day's outcome (`shift(-1)` / `shift(-5)`)
- the last rows without a future target are dropped
- chronological split, never shuffled

## Splits

**Model E headline split:**

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
| **G** | `stress_any` at \(h=5\) | Majority, persistence, logistic, XGBoost ± class weight / SMOTE / threshold |

The high-yield-stress-at-h=5 target had a ~0.7% positive rate on the
test window, which made accuracy useless. Headline G metric is PR–AUC
on `stress_any`.

XGBoost (Model E): `n_estimators=400`, `max_depth=4`,
`learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9`,
`random_state=42`. Logistic uses `class_weight="balanced"`.

## Event windows

| Episode | Bounds |
|---|---|
| COVID crash | 2020-02-15 → 2020-04-30 |
| Inflation / rate shock | 2022-06-01 → 2022-10-31 |
| Banking stress | 2023-03-01 → 2023-04-30 |
