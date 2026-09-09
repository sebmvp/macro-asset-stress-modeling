# Dataset

The original analysis used the Kaggle dataset:

**Algorithmic Trading, Macro Stress, and Asset Regimes**  
Author: [Kanchana1990](https://www.kaggle.com/kanchana1990)  
https://www.kaggle.com/datasets/kanchana1990/algorithmic-trading-macro-stress-and-asset-regimes

License on Kaggle: **CC BY-NC-SA 4.0**. This repository does **not** redistributed the CSV.

## Expected file

```
data/Global_Market_Stress_and_Liquidity_Regimes.csv
```

| Check | Expected |
|---|---|
| Rows | 4,150 |
| Start | 2014-10-17 |
| End | 2026-02-25 |
| Required columns | `Date`, `Equities_US`, `Gold`, `Crypto_Bitcoin`, `Volatility_Index`, `Yield_Curve_Spread`, `High_Yield_Spread`, `Financial_Stress_Index`, `SPY_Drawdown`, `SPY_Rolling_Vol_30d`, `BTC_Rolling_Vol_30d`, `Stock_Bond_Corr_90d`, `SPY_RSI_14`, `GLD_RSI_14` |

`make data-check` verifies those properties.

## How to acquire

1. Create a Kaggle account and accept the dataset terms.
2. Install the Kaggle CLI and put an API token in `~/.kaggle/kaggle.json`.
3. From the repo root:

```bash
kaggle datasets download -d kanchana1990/algorithmic-trading-macro-stress-and-asset-regimes -p data --unzip
```

The download should produce `Global_Market_Stress_and_Liquidity_Regimes.csv`. If the unzipped name differs, rename it to that filename (or pass `--data path/to/file.csv` to the CLI).

Tests do **not** need the full dataset. They use a tiny synthetic fixture.
