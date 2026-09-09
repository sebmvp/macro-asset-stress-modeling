"""Model families from the final report.

Hyperparameters are library defaults plus the choices the report states
explicitly (class-balanced logistic, chronological evaluation). Original
XGBoost / RF / MLP settings were not recovered and are NOT tuned here
to chase published scores. Those scores are checksums, not targets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import classification_metrics, regression_metrics
from .features import X_from


def majority_predict(y_train: pd.Series, n: int) -> tuple[np.ndarray, np.ndarray]:
    majority = int(y_train.mode().iloc[0])
    pred = np.full(n, majority, dtype=int)
    # Constant score → ROC-AUC 0.5 by construction.
    proba = np.full(n, 0.5)
    return pred, proba


def vix_rule_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Predict Gold > SPY when VIX is above the training median."""
    threshold = float(train["Volatility_Index"].median())
    score = test["Volatility_Index"].to_numpy(dtype=float)
    pred = (score > threshold).astype(int)
    # Rank by VIX level so ROC-AUC is defined.
    return pred, score


def fit_logistic(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train[target].astype(int)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(test[target], pred, proba)


def fit_xgboost_classifier(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    from xgboost import XGBClassifier

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train[target].astype(int)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(test[target], pred, proba)


def fit_random_forest(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=1,
        class_weight="balanced",
    )
    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train[target].astype(int)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(test[target], pred, proba)


def fit_mlp(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    """Regularized MLP — not the overfit Milestone-2 net."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    alpha=1e-2,
                    early_stopping=True,
                    max_iter=400,
                    random_state=42,
                ),
            ),
        ]
    )
    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train[target].astype(int)
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(test[target], pred, proba)


def persistence_forecast(current: pd.Series, y_true: pd.Series) -> dict:
    return regression_metrics(y_true, current)


def fit_ridge_regressor(
    train: pd.DataFrame, test: pd.DataFrame, target: str, feature_cols: list[str]
) -> dict:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    pipe.fit(train[feature_cols], train[target])
    pred = pipe.predict(test[feature_cols])
    return regression_metrics(test[target], pred)


def fit_xgboost_regressor(
    train: pd.DataFrame, test: pd.DataFrame, target: str, feature_cols: list[str]
) -> dict:
    from xgboost import XGBRegressor

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    model.fit(train[feature_cols], train[target])
    pred = model.predict(test[feature_cols])
    return regression_metrics(test[target], pred)
