from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import cross_val_predict


def logloss_and_pr_auc(y_true, probas) -> dict:
    return {
        "logloss": float(log_loss(y_true, probas)),
        "pr_auc": float(average_precision_score(y_true, probas)),
    }


def expected_calibration_error(y_true, probas, n_bins: int = 10) -> float:
    """Mean |accuracy - confidence| weighted by bin mass (uniform bins)."""
    y_true = np.asarray(y_true)
    probas = np.asarray(probas)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probas, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - probas[m].mean())
    return float(ece)


def calibration_scores(y_true, probas, n_bins: int = 10) -> dict:
    return {
        **logloss_and_pr_auc(y_true, probas),
        "ece": expected_calibration_error(y_true, probas, n_bins),
    }



def calculate_oof_metrics(best_pipeline, X, y, kfolds) -> dict:
    """ **at each cross validation round**, we calculate the probas of the model on the **out-of-fold** samples, 
    then concatenate the results to one vector of same shape as `y`
    """ 
    oof_concat_probas = cross_val_predict(best_pipeline, X, y, cv=kfolds, method="predict_proba")[:, 1]
    return logloss_and_pr_auc(y, oof_concat_probas)
