# Reproducibility

Primary path is the Jupyter notebooks. `src/macro_stress` is the shared
library those notebooks and `make reproduce` both use.

## Commands

```bash
make setup          # Python venv + package
                    # macOS: also installs libomp for XGBoost via Homebrew
make test           # leakage / transform invariants (no Kaggle CSV)
make download       # Kaggle API → data/Global_Market_Stress_and_Liquidity_Regimes.csv
make data-check     # 4,150 rows, dates, required columns
make notebooks      # execute notebooks/01–04
make reproduce      # batch CLI run of features, Model E/F/G, walk-forward
```

Without `make`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,notebook,full]"
pytest
python scripts/download_data.py
jupyter notebook notebooks/00_index.ipynb
```

## What you should expect

| Check | Pass condition |
|---|---|
| `make test` | leakage invariants on a synthetic panel |
| `make data-check` | restricted panel has 4,150 rows, 2014-10-17 → 2026-02-25, required columns |
| notebooks 01–04 | execute top to bottom against the CSV and show tables/plots |

## Date restriction

The public Kaggle file can be updated. `load_raw()` always keeps
2014-10-17 through 2026-02-25.

## Seeds

`random_state=42` where a library accepts a seed. Bootstrap CIs in
`analysis.py` use `numpy.random.default_rng(42)`.

## Paths

All paths are relative to the repository root.

## macOS / XGBoost

XGBoost on macOS needs OpenMP. `make setup` runs `brew install libomp`
when Homebrew is present.

## Data license

The CSV is CC BY-NC-SA 4.0 on Kaggle and is gitignored. See
[DATA_LICENSE_AND_PROVENANCE.md](DATA_LICENSE_AND_PROVENANCE.md).
