# Dataset

The original analysis used the Kaggle dataset:

**Algorithmic Trading: Macro Stress & Asset Regimes**
Author: [Kanchana1990](https://www.kaggle.com/kanchana1990)
https://www.kaggle.com/datasets/kanchana1990/algorithmic-trading-macro-stress-and-asset-regimes

License on Kaggle: **CC BY-NC-SA 4.0**. This repository does **not** redistribute the CSV.

Kaggle reports underlying sources as Yahoo Finance / `yfinance` (asset prices) and FRED / `pandas-datareader` (macro indicators).

Column inventory: [data_schema.csv](data_schema.csv).

## Expected file

```
data/Global_Market_Stress_and_Liquidity_Regimes.csv
```

| Check | Expected |
|---|---|
| Rows (project sample) | 4,150 |
| Start | 2014-10-17 |
| End | 2026-02-25 |
| Required columns | `Date`, `Equities_US`, `Gold`, `Crypto_Bitcoin`, `Volatility_Index`, `Yield_Curve_Spread`, `High_Yield_Spread`, `Financial_Stress_Index`, `SPY_Drawdown`, `SPY_Rolling_Vol_30d`, `BTC_Rolling_Vol_30d`, `Stock_Bond_Corr_90d`, `SPY_RSI_14`, `GLD_RSI_14` |

The public Kaggle extract can continue to grow. `load_raw()` always restricts to the historical project window above. `make data-check` verifies the restricted panel.

## How to acquire

### Option A — Kaggle API

1. Create a Kaggle account and accept the dataset terms.
2. Put an API token in `~/.kaggle/kaggle.json` (`chmod 600`).
3. From the repo root:

```bash
make setup
make download
make data-check
```

Equivalent:

```bash
python scripts/download_data.py
# or
kaggle datasets download -d kanchana1990/algorithmic-trading-macro-stress-and-asset-regimes -p data --unzip
```

### Option B — browser

Download the CSV from the Kaggle page and save it as
`data/Global_Market_Stress_and_Liquidity_Regimes.csv`.

If the unzipped filename differs, rename it (or pass `--data path/to/file.csv` to the CLI).

Tests do **not** need the full dataset. They use a tiny synthetic fixture.
