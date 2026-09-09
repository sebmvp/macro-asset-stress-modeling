"""Run the reconstructed analysis and compare against published anchors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import (
    M1_VOL_FEATURES,
    event_cumulative_returns,
    mean_return_table,
    spy_beta,
    vol_by_regime,
)
from .data import (
    CANONICAL_CSV_NAME,
    DATA_DIR,
    REPO_ROOT,
    find_dataset,
    load_raw,
    validate_dataset,
)
from .evaluation import (
    chronological_split,
    classification_metrics,
    model_e_split,
    regression_metrics,
    walk_forward_folds,
)
from .features import (
    FEATURE_COUNT,
    build_supervised_frame,
    drop_unusable,
    feature_names,
)
from .models import (
    fit_logistic,
    fit_ridge_regressor,
    fit_xgboost_classifier,
    fit_xgboost_regressor,
    majority_predict,
    persistence_forecast,
    vix_rule_predict,
)
from .plots import plot_event_windows, plot_model_e_auc, plot_volatility_by_regime
from .regimes import HISTORICAL_THRESHOLDS, compute_thresholds

RESULTS_DIR = REPO_ROOT / "results" / "latest"
FIGURE_DIR = REPO_ROOT / "figures" / "reproduced"
ANCHORS_PATH = REPO_ROOT / "results" / "historical" / "anchors.json"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def run_model_e(frame: pd.DataFrame) -> list[dict]:
    split = model_e_split(drop_unusable(frame, extra=("y_gold_gt_spy",)))
    y_tr = split.train["y_gold_gt_spy"].astype(int)
    y_te = split.test["y_gold_gt_spy"].astype(int)
    maj_pred, maj_proba = majority_predict(y_tr, len(split.test))
    vix_pred, vix_score = vix_rule_predict(split.train, split.test)
    rows = [
        {
            "model": "Majority",
            **classification_metrics(y_te, maj_pred, maj_proba),
            "n_test": int(len(split.test)),
            "positive_rate": float(y_te.mean()),
            "train_start": str(split.train["Date"].min().date()),
            "train_end": str(split.train["Date"].max().date()),
            "test_start": str(split.test["Date"].min().date()),
            "test_end": str(split.test["Date"].max().date()),
        },
        {"model": "VIX rule", **classification_metrics(y_te, vix_pred, vix_score)},
        {"model": "Logistic", **fit_logistic(split.train, split.test, "y_gold_gt_spy")},
        {
            "model": "XGBoost",
            **fit_xgboost_classifier(split.train, split.test, "y_gold_gt_spy"),
        },
    ]
    return rows


def run_model_f(frame: pd.DataFrame) -> list[dict]:
    rows = []
    tasks = [
        ("SPY 30d vol h=1", "y_spy_vol_1", "SPY_Rolling_Vol_30d"),
        ("SPY 30d vol h=5", "y_spy_vol_5", "SPY_Rolling_Vol_30d"),
        ("BTC 30d vol h=5", "y_btc_vol_5", "BTC_Rolling_Vol_30d"),
        ("VIX h=5", "y_vix_5", "Volatility_Index"),
    ]
    feats = feature_names()
    for label, target, current in tasks:
        usable = drop_unusable(frame, extra=(target,))
        split = chronological_split(usable)
        rows.append(
            {
                "task": label,
                "model": "Persistence",
                **persistence_forecast(split.test[current], split.test[target]),
            }
        )
        rows.append(
            {
                "task": label,
                "model": "Ridge",
                **fit_ridge_regressor(split.train, split.test, target, feats),
            }
        )
        rows.append(
            {
                "task": label,
                "model": "XGBoost",
                **fit_xgboost_regressor(split.train, split.test, target, feats),
            }
        )
    return rows


def run_model_g(frame: pd.DataFrame) -> list[dict]:
    usable = drop_unusable(frame, extra=("y_stress_any_5",))
    split = chronological_split(usable)
    y_tr = split.train["y_stress_any_5"].astype(int)
    y_te = split.test["y_stress_any_5"].astype(int)
    maj_pred, maj_proba = majority_predict(y_tr, len(split.test))
    persist_pred = split.test["stress_any"].astype(int).to_numpy()
    persist_score = persist_pred.astype(float)
    return [
        {"model": "Majority", **classification_metrics(y_te, maj_pred, maj_proba)},
        {
            "model": "Regime persistence",
            **classification_metrics(y_te, persist_pred, persist_score),
        },
        {"model": "Logistic", **fit_logistic(split.train, split.test, "y_stress_any_5")},
        {
            "model": "XGBoost",
            **fit_xgboost_classifier(split.train, split.test, "y_stress_any_5"),
        },
    ]


def run_milestone1_vol(frame: pd.DataFrame) -> list[dict]:
    """M1 5-day SPY vol — feature list is specified in the milestone report."""
    from sklearn.ensemble import RandomForestRegressor

    usable = frame.dropna(subset=M1_VOL_FEATURES + ["y_spy_vol_5"]).reset_index(drop=True)
    split = chronological_split(usable)
    rows = [
        {
            "model": "Persistence",
            **persistence_forecast(split.test["SPY_Rolling_Vol_30d"], split.test["y_spy_vol_5"]),
        }
    ]
    ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    ridge.fit(split.train[M1_VOL_FEATURES], split.train["y_spy_vol_5"])
    rows.append(
        {
            "model": "Ridge",
            **regression_metrics(
                split.test["y_spy_vol_5"], ridge.predict(split.test[M1_VOL_FEATURES])
            ),
        }
    )
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1)
    rf.fit(split.train[M1_VOL_FEATURES], split.train["y_spy_vol_5"])
    rows.append(
        {
            "model": "RandomForest",
            **regression_metrics(
                split.test["y_spy_vol_5"], rf.predict(split.test[M1_VOL_FEATURES])
            ),
        }
    )
    return rows


def run_walk_forward(frame: pd.DataFrame) -> list[dict]:
    usable = drop_unusable(frame, extra=("y_gold_gt_spy",))
    rows = []
    for year, split in walk_forward_folds(usable):
        metrics = fit_xgboost_classifier(split.train, split.test, "y_gold_gt_spy")
        rows.append(
            {
                "year": year,
                "train_n": int(len(split.train)),
                "test_n": int(len(split.test)),
                "positive_rate": float(split.test["y_gold_gt_spy"].mean()),
                **metrics,
            }
        )
    return rows


def compare_to_anchors(payload: dict) -> list[dict]:
    if not ANCHORS_PATH.is_file():
        return []
    raw_anchors = json.loads(ANCHORS_PATH.read_text())
    anchors = raw_anchors["items"] if isinstance(raw_anchors, dict) else raw_anchors
    notes = []
    for item in anchors:
        notes.append(
            {
                "id": item["id"],
                "stage": item["stage"],
                "historical": item["value"],
                "note": item["note"],
            }
        )
    payload["anchor_notes"] = notes
    return notes


def reproduce(data_path: str | None = None) -> dict:
    path = find_dataset(data_path)
    if path is None:
        raise FileNotFoundError(
            "Dataset CSV not found. Place "
            f"{CANONICAL_CSV_NAME} in {DATA_DIR} (see data/README.md)."
        )
    raw = load_raw(path)
    issues = validate_dataset(raw)
    frame = build_supervised_frame(raw)
    usable_e = drop_unusable(frame, extra=("y_gold_gt_spy",))

    descriptive = {
        "n_raw": int(len(raw)),
        "n_supervised_e": int(len(usable_e)),
        "feature_count": FEATURE_COUNT,
        "computed_thresholds": compute_thresholds(raw),
        "historical_thresholds": HISTORICAL_THRESHOLDS,
        "validation_issues": issues,
        "first_usable_date": str(usable_e["Date"].min().date()),
        "last_usable_date": str(usable_e["Date"].max().date()),
        "regimes": {
            flag: mean_return_table(frame.dropna(subset=["gold_ret", "spy_ret"]), flag)
            for flag in ("stress_vix", "stress_any", "stress_all", "stress_fsi", "stress_hys")
        },
        "vol_stress_any": vol_by_regime(frame.dropna(subset=["Gold_Rolling_Vol_30d"])),
        "event_windows_reconstructed": event_cumulative_returns(frame),
        "betas_stress_any": spy_beta(frame.dropna(subset=["gold_ret", "spy_ret", "btc_ret"])),
    }
    model_e = run_model_e(frame)
    model_f = run_model_f(frame)
    model_g = run_model_g(frame)
    m1_vol = run_milestone1_vol(frame)
    walk = run_walk_forward(frame)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_volatility_by_regime(frame.dropna(subset=["Gold_Rolling_Vol_30d"]), FIGURE_DIR / "fig1-volatility-by-regime.png")
    plot_event_windows(frame, FIGURE_DIR / "fig2-stress-window-returns.png")
    plot_model_e_auc(model_e, FIGURE_DIR / "fig3-model-roc-auc.png")

    payload = {
        "status": "reconstructed_run",
        "dataset": str(path.name),
        "dataset_issues": issues,
        "descriptive": descriptive,
        "model_e": model_e,
        "model_f": model_f,
        "model_g": model_g,
        "milestone1_vol": m1_vol,
        "walk_forward": walk,
        "walk_forward_mean_roc_auc": (
            float(pd.Series([r["roc_auc"] for r in walk]).mean()) if walk else None
        ),
    }
    compare_to_anchors(payload)
    _write_json(RESULTS_DIR / "run.json", payload)
    (RESULTS_DIR / "DISCREPANCY.md").write_text(_discrepancy_markdown(payload))
    return payload


def _discrepancy_markdown(payload: dict) -> str:
    lines = [
        "# Reconstructed run vs published anchors",
        "",
        "This run is a **September 2026 reconstruction**. It is not the original",
        "notebook. Published scores are checksums. Differences are expected when",
        "hyperparameters, the exact 55-column list, or event-window bounds differ.",
        "",
        f"- Dataset issues: `{payload.get('dataset_issues')}`",
        f"- Feature count: {payload['descriptive']['feature_count']}",
        f"- First usable date: {payload['descriptive']['first_usable_date']}",
        f"- Model E test n / positive rate: see model_e Majority row",
        "",
        "## Model E (reconstructed)",
        "",
        "| Model | ROC–AUC | Accuracy | Bal. acc | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in payload["model_e"]:
        lines.append(
            f"| {row['model']} | {row.get('roc_auc', float('nan')):.4f} | "
            f"{row.get('accuracy', float('nan')):.4f} | "
            f"{row.get('balanced_acc', float('nan')):.4f} | "
            f"{row.get('f1', float('nan')):.4f} |"
        )
    lines += [
        "",
        "Published XGBoost ROC–AUC checksum: **0.6835**. Do not treat a miss as a bug",
        "to be optimized away.",
        "",
        "## Walk-forward mean ROC–AUC",
        "",
        f"- Reconstructed mean: `{payload.get('walk_forward_mean_roc_auc')}`",
        "- Published mean: `0.6571` (σ = 0.0289)",
        "",
    ]
    return "\n".join(lines) + "\n"

