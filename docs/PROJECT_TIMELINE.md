# Project timeline

CompSci 390B, Spring 2026. The academic title was *Algorithmic Prediction
of Trade and Value of Monetary and Currency-Based Assets*. The GitHub
title is the shorter question the work actually answered.

## Proposal

Broad question: can macro-financial stress indicators help explain or
predict behavior across Gold, U.S. equities, and Bitcoin?

## Phase 1 — problem + data

Kaggle panel, 4,150 daily observations, 2014-10-17 through 2026-02-25.
Log returns because prices were non-stationary.

## Phase 2 — Milestone 1 / early EDA

Stress variables clustered with volatility. Short-horizon Gold returns
had weak contemporaneous correlations with macro variables. Medium-lag
Granger tests suggested equity → Gold information. Next-day Gold return
prediction was hard (Ridge RMSE 0.010764, no better than a zero
baseline). Volatility forecasting was much easier because volatility is
persistent (SPY 30d vol, h=5: persistence RMSE ~0.0248, Ridge ~0.0210,
Ridge R² ~0.9046 in the later pipeline).

That shifted the question from "can we predict daily Gold returns?" to
"when does Gold actually behave defensively, and under what kind of
stress?"

## Phase 3 — Milestone 2 / regimes

Explicit 66th-percentile stress flags (`stress_hys`, `stress_fsi`,
`stress_vix`, `stress_any`, `stress_all`). Safe-haven tests, event
studies, first nonlinear classifiers.

## Phase 4 — nonlinear models, SHAP, robustness

Model E / F / G, themed SHAP, threshold robustness at 75th and 90th
percentiles, expanding-window walk-forward 2019–2026.

## Phase 5 — final report (May 2026)

Three conclusions: stress tracks volatility; Gold is a conditional safe
haven; Bitcoin is not digital Gold in this sample. Modeling lesson:
nonlinear models help the directional task; persistence and Ridge remain
hard to beat on sticky volatility.

## Phase 6 — repository recovery (September 2026)

Original notebooks were not recovered. This repository reconstructs the
published methodology, packages the report and course slides, and
separates published checksums from reconstructed runs.
