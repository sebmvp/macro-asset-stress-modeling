# Reproducibility

The May 2026 report is the scholarly record. The code in `src/` is a
September 2026 reconstruction of that methodology. Published scores are
checksums, not training targets.

## Commands

```bash
make setup          # Python 3.13 venv + package
                    # macOS: also installs libomp for XGBoost via Homebrew
make test           # leakage / transform invariants (no Kaggle CSV)
make download       # Kaggle API → data/Global_Market_Stress_and_Liquidity_Regimes.csv
make data-check     # 4,150 rows, dates, required columns (after date restriction)
make reproduce      # features, descriptive tests, Model E/F/G, walk-forward, figures
```

Without `make`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/download_data.py
python -m macro_stress.cli data-check
python -m macro_stress.cli reproduce
```

Optional extras for LightGBM / CatBoost / SHAP / SMOTE:

```bash
pip install -e ".[full]"
```

## What you should expect

| Check | Pass condition |
|---|---|
| `make test` | leakage invariants on a synthetic panel |
| `make data-check` | restricted panel has 4,150 rows, 2014-10-17 → 2026-02-25, required columns |
| `make reproduce` | writes `results/latest/run.json` and `results/latest/DISCREPANCY.md` |

`make reproduce` will **not** necessarily reprint XGBoost ROC–AUC 0.6835
or Model G PR–AUC 0.7095. Original hyperparameters were not recovered.
A miss is not a bug to be optimized away.

## Date restriction

The public Kaggle file can be updated. `load_raw()` always keeps
2014-10-17 through 2026-02-25, which is the historical project sample.

## Seeds

Where a library accepts a seed, the reconstruction uses `random_state=42`
and XGBoost `n_jobs=1`. Bootstrap CIs in `analysis.py` use
`numpy.random.default_rng(42)`.

## Paths

All paths are relative to the repository root. There are no
machine-specific absolute paths in the package.

## macOS / XGBoost

XGBoost on macOS needs OpenMP. `make setup` runs `brew install libomp`
when Homebrew is present.

## Data license

The CSV is CC BY-NC-SA 4.0 on Kaggle and is gitignored. See
[DATA_LICENSE_AND_PROVENANCE.md](DATA_LICENSE_AND_PROVENANCE.md).
