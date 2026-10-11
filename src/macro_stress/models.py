"""Model families used in the Milestone 2 / 3 notebooks."""

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

XGB_CLASSIFIER_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

RF_CLASSIFIER_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "min_samples_leaf": 10,
    "class_weight": "balanced",
    "n_jobs": -1,
    "random_state": 42,
}

LGBM_CLASSIFIER_PARAMS = {
    "n_estimators": 500,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
    "verbose": -1,
}

CATBOOST_CLASSIFIER_PARAMS = {
    "iterations": 500,
    "depth": 5,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "verbose": False,
}


def make_xgboost_classifier(**overrides):
    from xgboost import XGBClassifier

    params = {**XGB_CLASSIFIER_PARAMS, **overrides}
    return XGBClassifier(**params)


def make_random_forest_classifier(**overrides):
    params = {**RF_CLASSIFIER_PARAMS, **overrides}
    return RandomForestClassifier(**params)


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


def _fit_sklearnish_classifier(clf, train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    X_tr, X_te = X_from(train), X_from(test)
    y_tr = train[target].astype(int)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return classification_metrics(test[target], pred, proba)


def fit_xgboost_classifier(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    return _fit_sklearnish_classifier(make_xgboost_classifier(), train, test, target)


def fit_random_forest(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    return _fit_sklearnish_classifier(make_random_forest_classifier(), train, test, target)


def fit_optional_boosting(train: pd.DataFrame, test: pd.DataFrame, target: str) -> list[dict]:
    """LightGBM / CatBoost if installed. Missing extras are skipped, not faked."""
    rows: list[dict] = []
    try:
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(**LGBM_CLASSIFIER_PARAMS)
        rows.append({"model": "LightGBM", **_fit_sklearnish_classifier(clf, train, test, target)})
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier

        clf = CatBoostClassifier(**CATBOOST_CLASSIFIER_PARAMS)
        rows.append({"model": "CatBoost", **_fit_sklearnish_classifier(clf, train, test, target)})
    except ImportError:
        pass
    return rows


def fit_mlp(train: pd.DataFrame, test: pd.DataFrame, target: str) -> dict:
    """MLP with the Milestone 3 bake-off settings (64, 64), alpha=1e-3."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 64),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
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
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )
    model.fit(train[feature_cols], train[target])
    pred = model.predict(test[feature_cols])
    return regression_metrics(test[target], pred)
