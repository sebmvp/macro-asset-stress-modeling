<p align="center">
  <img src="assets/macro-header.svg" alt="Macro Stress &amp; Asset Behavior — does Gold actually protect you when markets get stressed, and does Bitcoin behave anything like it?" width="100%"/>
</p>

# Macro Stress & Asset Behavior

**When markets get stressed, does Gold actually protect you — and does Bitcoin behave anything like it?**

We asked that as a CompSci 390B group research project (Spring 2026). Using 4,150 daily observations, we tested common stress definitions, event windows, volatility behavior, and predictive models.

The short answer: **Gold is a conditional safe haven**, not a universal one. Evidence is strongest under volatility-driven stress, composite stress, and the COVID-style liquidity/equity shock. Gold is weaker under the 2022 inflation/rate shock. Bitcoin fails the same tests — it does not earn the "digital gold" label in this sample.

Nonlinear models improved one directional classification task (will Gold beat SPY tomorrow?). Simple baselines stayed stronger for persistent volatility. That mismatch is one of the project's better modeling lessons: complexity has to match the target.

This repository is the **September 2026 recovery**: original notebooks were not available to package, so the runnable code here reconstructs the published methodology. The May 2026 report remains the scholarly record. See [docs/PROVENANCE.md](docs/PROVENANCE.md).

---

## What we wanted to know

1. **Stress and volatility.** Are volatility and market-risk measures systematically higher during macro-financial stress?
2. **Gold as a safe haven.** Does Gold behave defensively relative to SPY during stress — and is that robust to how you define "stress"?
3. **Bitcoin comparison.** Does Bitcoin behave like Gold under the same definitions, or like a high-beta risk asset?
4. **Predictive modeling.** Can lagged macro-financial indicators and market-state variables improve forecasting beyond simple baselines?

A safe haven isn't an asset with a good average return. It's an asset that defends *exactly when* risky assets are under pressure.

## Data

Kaggle dataset *Algorithmic Trading, Macro Stress, and Asset Regimes* — **4,150 daily observations, 2014-10-17 through 2026-02-25**. The CSV is **not** in this repo (CC BY-NC-SA 4.0). Acquisition: [data/README.md](data/README.md).

- **Assets:** Gold, U.S. equities (`Equities_US` as SPY), Bitcoin — modeled as daily log returns \(r_t = \log P_t - \log P_{t-1}\).
- **Macro-financial variables:** VIX, high-yield spread, financial stress index, yield-curve spread, 90-day stock–bond correlation, SPY drawdown, RSI, 30-day rolling volatility.

## Stress regimes

Baseline: 66th percentile of three proxies.

| Regime | Rule | Share of days |
|---|---|---|
| `stress_hys` | High-yield spread ≥ 4.400 | 34.2% |
| `stress_fsi` | Financial stress index ≥ −0.172 | 34.1% |
| `stress_vix` | VIX ≥ 19.060 | 34.0% |
| `stress_any` | At least one flag high | 56.3% |
| `stress_all` | All three flags high | 14.3% |

Threshold robustness was re-checked at the 75th and 90th percentiles. The VIX-based Gold–SPY result survives all three; `stress_any` does not survive the 90th.

## Modeling

Three supervised tasks, chronological splits, **no shuffling**. Targets are shifted forward before splitting.

- **Model E** — does Gold beat SPY tomorrow? Majority, VIX rule, class-balanced logistic, XGBoost, LightGBM, CatBoost, Random Forest, MLP, stacking. 55 predictors.
- **Model F** — 30-day rolling volatility for SPY, Bitcoin, and VIX at *h* = 1 and *h* = 5: persistence vs Ridge vs XGBoost.
- **Model G** — stress five trading days ahead. Headline metric: PR–AUC. Class weighting, SMOTE, and threshold tuning were tried; threshold tuning did not generalize.

## What surprised us

**Gold's safe-haven status is real but conditional.** Under VIX stress, Gold's mean daily log return is +0.00068 while SPY's is −0.00092 — the spread widens by 0.00247 with a bootstrap CI excluding zero. The same test under financial-stress-index regimes shows *no* Gold advantage. In the 2022 inflation/rate-hike window Gold lost more than SPY.

**Bitcoin is not digital gold in this sample.** Gold's beta to SPY *falls* from +0.157 in calm periods to +0.020 during stress. Bitcoin's moves the opposite way: +0.666 calm → +0.862 stressed. Bitcoin's 30-day volatility sits near 0.61 in *both* regimes (*p* = 0.859). Its worst days overlap SPY's worst days more than Gold's do (16.3% vs 13.5%).

**Complex models were not automatically better.** XGBoost won the directional ranking task (ROC–AUC **0.6835** vs 0.6091 logistic and 0.5040 for a VIX rule). For slow-moving volatility, persistence and Ridge beat XGBoost on nearly every target-horizon pair. XGBoost collapsed to *negative* out-of-sample R² on 5-day VIX.

**The MLP failed instructively.** ~100% train accuracy, ~53% test accuracy. Seven of its top-ten SHAP-ranked features fail a KS train/test test (*p* < 0.01). Heavier regularization fixed calibration (Brier 0.44 → 0.24) but not ranking (ROC–AUC stayed ≈ 0.51). That is covariate shift, not just overfitting.

**The classifier isn't a stress indicator in disguise.** Themed SHAP: cross-asset momentum 39.5%, macro-stress 20.0%, RSI 17.3%, volatility 14.3%, other 9.0%.

## Validation

- **Chronological 80/20** for headline results — train 2014-11-21 → 2023-11-25, test 2023-11-26 → 2026-02-24 (822 test rows; Gold beats SPY on 38.3% of them).
- **Expanding-window walk-forward** for Model E, test years 2019–2026. Mean ROC–AUC **0.6571** (σ = 0.0289); all eight folds > 0.55.

Walk-forward was implemented in the original pipeline by Dhruv. The code in this repo reimplements the published design.

## Reproduce

```bash
make setup          # python 3.13 venv + package (macOS: also installs libomp for XGBoost)
make test           # leakage / transform invariants (no Kaggle CSV required)
# follow data/README.md to download the CSV into data/
make data-check     # 4,150 rows, dates, required columns
make reproduce      # features, Model E/F/G, walk-forward, figures
```

`make reproduce` writes `results/latest/run.json` and `results/latest/DISCREPANCY.md`. Published scores (XGBoost 0.6835, etc.) are **checksums**. The reconstruction uses library defaults where original hyperparameters were not recovered. It is not tuned to hit those numbers.

Milestone 1 Gold-return Ridge (RMSE 0.010764, no better than a zero baseline) and the Model F negative result (XGBoost losing to persistence on sticky volatility) are part of the story. Don't hide them.

## Figures (from the original report)

<table>
<tr>
<td width="50%"><img src="figures/fig2-stress-window-returns.png" alt="Cumulative log returns for Gold, SPY, and Bitcoin across the 2020 COVID crash, 2022 inflation/rates, and 2023 banking stress windows"/></td>
<td width="50%"><img src="figures/fig1-volatility-by-regime.png" alt="Annualized 30-day volatility by asset under low vs high stress regimes"/></td>
</tr>
<tr>
<td><b>Stress windows.</b> Gold protects capital most clearly in the COVID crash, is weak in the 2022 inflation/rate shock, and is positive but not dominant in 2023 banking stress.</td>
<td><b>Volatility by regime.</b> Gold's volatility rises in stress but stays far below Bitcoin's and expands much less than SPY's. Bitcoin is simply always volatile.</td>
</tr>
<tr>
<td><img src="figures/fig3-model-roc-auc.png" alt="ROC-AUC comparison across Majority, VIX rule, Logistic, XGBoost, CatBoost, and Random Forest models"/></td>
<td><img src="figures/fig4-themed-shap.png" alt="Themed SHAP attribution for Model E: momentum 39.5%, macro stress 20%, RSI 17.3%, volatility 14.3%, other 9%"/></td>
</tr>
<tr>
<td><b>Model E ranking.</b> Nonlinear models improve the Gold-vs-SPY ranking task; XGBoost leads by ROC–AUC.</td>
<td><b>What drives it.</b> Most of XGBoost's signal comes from lagged cross-asset returns, not the stress indicators themselves.</td>
</tr>
</table>

## Full report

[Algorithmic Prediction of Trade and Value of Monetary and Currency-Based Assets](report/Macro_Asset_Volatility_Modeling.pdf) (May 2026) — methodology, tables, references.

## Who did what

Group project. The boundaries matter.

- **Dhruv Kartik** implemented the empirical pipeline: regime construction, conditional return/volatility analysis, event studies, nonlinear modeling, SHAP, threshold robustness, and walk-forward validation.
- **Sebastian Vaskes Pimentel** contributed to model design, training, evaluation, and the comparison across Logistic Regression, Ridge, XGBoost, LightGBM, CatBoost, Random Forest, MLP, and stacking.
- We collaborated on the research questions, feature choices, safe-haven interpretation, limitations, and conclusions.

**Repository recovery / reproducibility packaging:** Sebastian Vaskes Pimentel, September 2026.

## Limitations

These are part of the result, not a footnote.

- **Regime sensitivity.** VIX-based stress produces the robust safe-haven result; HYS and FSI stress do not.
- **Full-sample threshold bias.** Descriptive quantiles use the whole sample — fine retrospectively, leaky in a live system.
- **Covariate shift.** The MLP failure is direct evidence. Walk-forward helps; it does not eliminate regime drift.
- **Event-window selection.** Economically meaningful windows; different dates change cumulative returns.
- **U.S.-centric sample starting 2014.** Enough for modern Bitcoin, not enough for Gold across many historical cycles. No costs, spreads, taxes, or portfolio constraints.
- **Multiple testing.** Many regimes, thresholds, horizons, and models.

No license file is included. The course report is the scholarly record. The Kaggle dataset remains under CC BY-NC-SA 4.0 on Kaggle.
