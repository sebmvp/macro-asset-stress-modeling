# Figures

`figures/` holds the **published report figures** exported from the May 2026
final report. They are the visual record of the original analysis.

| File | What it shows |
|---|---|
| `fig1-volatility-by-regime.png` | Annualized 30-day volatility, Gold / SPY / Bitcoin, low vs high stress |
| `fig2-stress-window-returns.png` | Cumulative log returns in the 2020 / 2022 / 2023 event windows |
| `fig3-model-roc-auc.png` | Model E ROC–AUC for Majority, VIX rule, logistic, XGBoost, CatBoost, RF |
| `fig4-themed-shap.png` | Themed SHAP shares for Model E (published checksums) |

`make reproduce` writes a **reconstructed** copy of some of these under
`figures/reproduced/` (gitignored). Those plots use the reconstructed
pipeline and library-default hyperparameters. They are not a replacement
for the report figures, and they are not retuned to match published scores.

If a reconstructed plot and a report figure disagree, the report figure
and the published numbers in `results/historical/anchors.json` win.
