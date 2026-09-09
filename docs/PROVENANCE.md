# Provenance

This file is the public recovery record. It does not list private filesystem
paths, chat exports, or course-LMS dumps.

## What was recovered

| Artifact | Stage | Status |
|---|---|---|
| Final report PDF (`report/Macro_Asset_Volatility_Modeling.pdf`) | Final (May 2026) | Recovered |
| Report figures in `figures/` | Final | Recovered (exported from the report) |
| Header graphic `assets/macro-header.svg` | Public writeup, Sep 2026 | Original to this repo |
| Milestone 1 / 2 / 3 PDFs, proposal, final slides | Course submissions | Recovered locally; **not** committed (course artifacts) |
| Obsidian notes for Milestone 1–2 | Working notes | Recovered locally; **not** committed |

## What was not recovered

No original analysis notebooks, `.py` scripts, HTML notebook exports
(`milestone1_gold_comprehensive.html` or similar), Jupyter checkpoints, or
the Kaggle CSV were present in the public repo or in recoverable project
directories at packaging time.

A historical HTML export named approximately `milestone1_gold_comprehensive`
is known to have existed on a collaborator's machine. It was not found in
Sebastian's local copies.

**Do not call the Python in `src/` "recovered Spring 2026 code."**
It is a September 2026 reconstruction of the published methodology.

## Classification of this repository's code

| Component | Label | Basis |
|---|---|---|
| Log-return transform | RECONSTRUCTED | Report equation \(r_t = \log P_t - \log P_{t-1}\) |
| Final stress flags at the 66th percentile | RECONSTRUCTED | Report Table 1; published thresholds used as checksums |
| 55-predictor matrix | RECONSTRUCTED | Report §3.3 describes the *families* of features, not the exact column list. The list in `features.py` is a documented reconstruction that includes every named feature in the report and yields 55 columns. |
| Target shift + chronological split | RECONSTRUCTED | Report §3.3 / §4.3 |
| Model E/F/G families | RECONSTRUCTED | Report §6. Hyperparameters other than "class-balanced logistic" and "chronological evaluation" were not recovered. Defaults are used. **Not tuned to published scores.** |
| Walk-forward expanding yearly folds | RECONSTRUCTED | Report Table 7. Original implementation was Dhruv's pipeline; this is a methodology reimplementation. |
| Event-window calendar bounds | RECONSTRUCTED | Episodes named in the report; exact start/end timestamps were not recovered. |
| SHAP themed shares (39.5 / 20.0 / 17.3 / 14.3 / 9.0) | HISTORICAL CHECKSUM | Quoted from the report; not recomputed unless `shap` is installed on a full run |
| Milestone 1 Gold Ridge table | HISTORICAL CHECKSUM | Feature list only partially specified |
| Milestone 1 5-day SPY vol features | RECONSTRUCTED, better specified | Milestone 1 lists the eight inputs explicitly |

## Authorship (do not rewrite)

From the May 2026 report, which remains the scholarly record:

- **Dhruv Kartik** implemented the empirical pipeline: regime construction,
  conditional return/volatility analysis, event studies, nonlinear modeling,
  SHAP, threshold robustness, and walk-forward validation.
- **Sebastian Vaskes Pimentel** contributed to model design, training,
  evaluation, and comparison across Logistic Regression, Ridge, XGBoost,
  LightGBM, CatBoost, Random Forest, MLP, and stacking.
- The two of us collaborated on research questions, feature choices,
  safe-haven interpretation, limitations, and conclusions.

**Repository recovery / reproducibility packaging:**
Sebastian Vaskes Pimentel, September 2026.

The original course PDF lists a third enrolled name. That student did not
implement the analysis, did not present, and is not credited in this
repository's documentation.

## Dataset

Kaggle, CC BY-NC-SA 4.0. Not committed. See `data/README.md`.
