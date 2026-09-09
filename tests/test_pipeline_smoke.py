"""Smoke the supervised pipeline on the synthetic panel (not the Kaggle CSV)."""

from macro_stress.evaluation import chronological_split, classification_metrics
from macro_stress.features import build_supervised_frame, drop_unusable
from macro_stress.models import fit_logistic, majority_predict


def test_model_e_pipeline_runs_on_fixture(panel):
    frame = drop_unusable(build_supervised_frame(panel), extra=("y_gold_gt_spy",))
    split = chronological_split(frame)
    y_te = split.test["y_gold_gt_spy"].astype(int)
    pred, proba = majority_predict(split.train["y_gold_gt_spy"].astype(int), len(split.test))
    maj = classification_metrics(y_te, pred, proba)
    assert maj["roc_auc"] == 0.5
    logit = fit_logistic(split.train, split.test, "y_gold_gt_spy")
    assert 0.0 <= logit["roc_auc"] <= 1.0
    assert "accuracy" in logit
