# Results

Numbers in this file are **published project results** from the May 2026
report unless a row is marked reconstructed. Do not replace them with
whatever `make reproduce` happens to print.

## Safe-haven returns

Under `stress_vix` (strongest evidence):

| Asset | Mean daily log return |
|---|---|
| Gold | +0.00068 |
| SPY | −0.00092 |
| Bitcoin | −0.00008 |

Gold–SPY spread widening: **+0.00247**, bootstrap CI lower bound **+0.00157**.

| Regime | Gold–SPY widening | CI lower bound |
|---|---|---|
| `stress_vix` | +0.00247 | +0.00157 |
| `stress_any` | +0.00096 | +0.00029 |
| `stress_all` | +0.00191 | +0.00040 |
| `stress_fsi` | −0.00012 | (no Gold advantage) |

Gold is a **conditional** safe haven, not a universal one.

## Threshold robustness and deciles

VIX-based widening survives 66th, 75th, and 90th percentile cuts.
`stress_any` survives 66% and 75%, not 90%. HYS and FSI do not produce
robust widening.

Spearman correlation between stress intensity and the Gold–SPY spread:

| Stress | ρ | p |
|---|---|---|
| VIX decile | +0.0859 | < 0.001 |
| High-yield spread | −0.0257 | 0.0979 |
| Financial stress index | −0.0305 | 0.0493 |

Defensive Gold behavior increases most consistently with
volatility-style stress.

## Volatility (`stress_any`)

| Asset | Calm | Stress | Change | p |
|---|---|---|---|---|
| Gold 30d vol | 0.1040 | 0.1298 | +24.8% | < 0.001 |
| SPY 30d vol | 0.0853 | 0.1521 | +78.3% | < 0.001 |
| Bitcoin 30d vol | 0.6079 | 0.6095 | ~0 | 0.8592 |

Bitcoin being unchanged is **not** evidence that it is defensive. It is
evidence that it is extremely volatile in both regimes.

## Correlation (60-day rolling)

Gold–SPY:

| Regime | Calm → stress |
|---|---|
| `stress_hys` | 0.016 → −0.003 |
| `stress_fsi` | 0.045 → −0.059 |
| `stress_any` | 0.034 → −0.010 |
| `stress_vix` | −0.032 → +0.089 |

Gold sometimes decouples from equities; the correlation evidence depends
on the regime definition.

Gold–Bitcoin generally **rises** in stress (`stress_any` 0.055 → 0.110;
`stress_all` 0.061 → 0.236).

## Event studies

| Window | Gold | SPY | BTC | Gold max DD | SPY max DD | BTC max DD |
|---|---|---|---|---|---|---|
| COVID 2020 | +0.0637 | −0.1445 | −0.1748 | −0.1338 | −0.4112 | −0.7131 |
| Inflation / rates 2022 | −0.1192 | −0.0585 | −0.4390 | — | — | — |
| Banking 2023 | +0.0848 | +0.0523 | +0.2346 | — | — | — |

COVID is the strongest real-world safe-haven example. 2022 is the
counterexample that makes the conclusion conditional. 2023 Gold beats
SPY, but Bitcoin rallies more.

## Bitcoin vs Gold

| | Calm | Stress |
|---|---|---|
| Gold β to SPY | +0.1569 | +0.0199 |
| Bitcoin β to SPY | +0.6657 | +0.8617 |

Gold becomes *less* equity-sensitive in stress. Bitcoin becomes *more*.

Worst 5% day overlap with SPY: Gold 13.5%, Bitcoin 16.3%.

Bitcoin does not behave like "digital Gold" under these tests.

## Model E (will Gold beat SPY tomorrow?)

822 test rows; Gold > SPY on 38.3% of them.

| Model | Accuracy | Bal. acc | F1 | ROC–AUC | Brier | Log loss |
|---|---|---|---|---|---|---|
| Majority | — | — | — | 0.5000 | — | — |
| VIX rule | — | — | — | 0.5040 | — | — |
| Logistic | — | — | — | 0.6091 | — | — |
| MLP | — | — | — | 0.5673 | — | — |
| Random Forest | — | **0.6467** | **0.5833** | 0.6622 | — | — |
| LightGBM | — | — | — | 0.6687 | — | — |
| CatBoost | — | 0.6179 | — | 0.6722 | — | — |
| Stacking | — | — | — | 0.6800 | — | — |
| **XGBoost** | **0.6460** | 0.5922 | 0.4393 | **0.6835** | 0.2210 | 0.6317 |

XGBoost is best by ranking (ROC–AUC). Random Forest is best on balanced
classification metrics. That is more nuanced than "one model won."

Themed SHAP (Model E): cross-asset momentum 39.5%, macro stress 20.0%,
RSI 17.3%, volatility 14.3%, other 9.0%. Model E is a
**momentum + stress** classifier, not a pure macro-stress model.
Gain importance did not agree strongly with SHAP / permutation.

Walk-forward mean ROC–AUC **0.6571** (σ = 0.0289); all eight folds > 0.55.

## Model F (volatility)

| Target | Persistence RMSE (R²) | Ridge RMSE (R²) | XGBoost RMSE (R²) |
|---|---|---|---|
| SPY 30d vol h=1 | 0.0099 (0.9786) | **0.0097 (0.9796)** | 0.0164 (0.9421) |
| SPY 30d vol h=5 | 0.0248 (0.8670) | **0.0210 (0.9046)** | 0.0246 (0.8699) |
| BTC 30d vol h=5 | **0.0608** | 0.0612 | 0.0755 |
| VIX h=5 | **3.2783 (0.4950)** | 3.2904 (0.4912) | 5.5236 (−0.4337) |

More complexity is not automatically better. Persistent volatility
rewards persistence and Ridge.

## Model G (future stress)

Headline: `stress_any` at h=5, **PR–AUC**.

| Model | PR–AUC |
|---|---|
| Majority | 0.2433 |
| Persistence | 0.5240 |
| **Logistic (balanced)** | **0.7095** |
| XGBoost default | 0.6429 |
| XGBoost weighted | 0.6518 |
| XGBoost + SMOTE | 0.6518 |
| XGBoost + tuned threshold | 0.6518 |

Logistic wins on PR–AUC. Threshold tuning helped validation and
generalized poorly (regime drift).

## What failed

- **MLP:** ~100% train accuracy, ~53% test accuracy (~47 pp gap). Seven
  of its top-ten important features had significant train/test shift
  (\(p < 0.01\)). Regularization improved Brier (~0.44 → ~0.24) but
  ROC–AUC stayed ~0.51. Covariate shift, not only overfitting.
- **XGBoost on VIX h=5:** negative out-of-sample R².
- **Next-day Gold returns (Milestone 1):** Ridge RMSE 0.010764 did not
  beat a zero baseline (0.010732). That is why the project moved off
  "predict Gold tomorrow" and onto regimes and relative performance.

## Three conclusions

1. Stress measures align with volatility and carry usable market-state
   information.
2. Gold is a **conditional** safe haven — strongest under VIX-style
   stress and COVID-like equity/liquidity shocks, not universally.
3. Bitcoin does **not** behave like digital Gold in this sample.
