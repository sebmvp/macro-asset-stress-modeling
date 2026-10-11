# Notebooks

Run in order. Kernel cwd can be the repo root or `notebooks/` — both resolve the CSV.

| Notebook | What it does |
|---|---|
| `00_index.ipynb` | Data check and reading order |
| `01_milestone1_baselines.ipynb` | Stationarity, Granger, Gold-return baselines, 5-day SPY vol |
| `02_milestone2_safehaven_and_modeling.ipynb` | Stress regimes, safe-haven tests, Models E / F / G |
| `03_milestone3_interpretability_and_robustness.ipynb` | Thresholds, extra models, SHAP, walk-forward, Model G PR-AUC |
| `04_milestone3_direct_safehaven_findings.ipynb` | Deciles, Bitcoin vs Gold, themed SHAP, MLP diagnostic |

```bash
make setup
# CSV at data/Global_Market_Stress_and_Liquidity_Regimes.csv
jupyter notebook notebooks/00_index.ipynb
```

Or: `make notebooks` to execute 01–04 with the project venv.
