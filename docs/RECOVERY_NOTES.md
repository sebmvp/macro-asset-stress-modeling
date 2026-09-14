# Recovery notes

The original CompSci 390B course workspace that held the analysis notebooks
was lost after the project was submitted. What survived is enough to
reconstruct the methodology and to treat the published numbers as a
checksum, not enough to claim a byte-for-byte restoration of the Spring
2026 code.

This file is the public recovery record. It does not list private
filesystem paths, chat exports, or LMS dumps.

## What survived

| Artifact | Status |
|---|---|
| Final report PDF (`report/Macro_Asset_Volatility_Modeling.pdf`, May 2026) | Recovered and committed |
| Report figures in `figures/` | Recovered (exported from the report) |
| Course final slides (`presentation/course-final-presentation.pptx`) | Recovered and committed as the original 12-slide deck |
| Published metrics, regime thresholds, event-study returns, SHAP theme shares, walk-forward means | Recovered from the report into `results/historical/anchors.json` |
| Public Kaggle dataset identity, license, and schema | Recovered; CSV is downloaded, not committed |
| Header graphic `assets/macro-header.svg` | Original to this repository (September 2026) |

## What did not survive

No original analysis notebooks, `.py` scripts, HTML notebook exports, or
Jupyter checkpoints were present at packaging time. A historical HTML
export named approximately `milestone1_gold_comprehensive` is known to
have existed on a collaborator's machine; it was not found in Sebastian's
local copies.

**Do not call the Python in `src/` "recovered Spring 2026 code."**
It is a September 2026 reconstruction of the published methodology.

## How the reconstruction is labeled

| Component | Label | Basis |
|---|---|---|
| Log-return transform | RECONSTRUCTED | Report equation \(r_t = \log P_t - \log P_{t-1}\) |
| Stress flags at the 66th percentile | RECONSTRUCTED | Report Table 1; published thresholds used as checksums |
| 55-predictor matrix | RECONSTRUCTED | Report §3.3 describes feature *families*, not the exact column list. `features.py` includes every named feature in the report and yields 55 columns. |
| Target shift + chronological split | RECONSTRUCTED | Report §3.3 / §4.3 |
| Model E / F / G families | RECONSTRUCTED | Report §6. Hyperparameters other than "class-balanced logistic" and "chronological evaluation" were not recovered. Defaults are used. **Not tuned to published scores.** |
| Walk-forward expanding yearly folds | RECONSTRUCTED | Report Table 7. Original implementation was Dhruv Kartik's pipeline; this is a methodology reimplementation. |
| Event-window calendar bounds | RECONSTRUCTED | Episodes named in the report; exact start/end timestamps were not recovered. |
| SHAP theme shares 39.5 / 20.0 / 17.3 / 14.3 / 9.0 | HISTORICAL CHECKSUM | Quoted from the report |
| Model E / F / G published metrics | HISTORICAL CHECKSUM | Quoted from the report |

Where a reconstructed run and a published number differ, the report wins.
`results/latest/DISCREPANCY.md` (produced by `make reproduce`, not
committed) is the place to record that difference. Do not retune
hyperparameters to chase checksums.

## Authorship

From the May 2026 report, which remains the scholarly record:

- **Dhruv Kartik** implemented the empirical pipeline: regime construction,
  conditional return/volatility analysis, event studies, nonlinear
  modeling, SHAP, threshold robustness, and walk-forward validation.
- **Sebastian Vaskes Pimentel** contributed to model design, training,
  evaluation, and comparison across Logistic Regression, Ridge, XGBoost,
  LightGBM, CatBoost, Random Forest, MLP, and stacking.
- **Nathan Dennis** led much of the written report structure and
  integration.
- The group collaborated on research questions, feature choices,
  safe-haven interpretation, limitations, and conclusions.

**Repository recovery / reproducibility packaging:**
Sebastian Vaskes Pimentel, September 2026.
