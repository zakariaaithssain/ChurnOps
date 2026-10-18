import numpy as np

from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import cross_val_predict


def logloss_and_pr_auc(y_true, probas) -> dict:
    return {
        "pr_auc": round(average_precision_score(y_true, probas), 4),
        "logloss": round(log_loss(y_true, probas), 4)
    }


def brier_skill_score(bs: float, bs_ref: float) -> float:
    """`1 - BS_model / BS_ref`, with the non calibrated logistic regression as ref.

    Takes scalar Brier scores (see brier_score). Positive means better
    than the reference, 0 equals it. NaN when the reference is perfect.
    """
    if bs_ref == 0:
        return None
    return round(1.0 - bs / bs_ref, 4)


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
    return round(float(ece), 4)


def calibration_scores(y_true, probas, n_bins: int = 10, ref_probas=None) -> dict:
    scores = {
        "expected_calibration_error": expected_calibration_error(y_true, probas, n_bins),
        **logloss_and_pr_auc(y_true, probas),
        "brier_loss" : round(brier_score_loss(y_true, probas), 4)
    }
    if ref_probas is not None:
        scores["brier_skill_score_logireg_as_ref"] = brier_skill_score(scores["brier_loss"], brier_score_loss(y_true, ref_probas))
    return scores



def calculate_oof_metrics(best_pipeline, X, y, kfolds) -> dict:
    """ **at each cross validation round**, we calculate the probas of the model on the **out-of-fold** samples, 
    then concatenate the results to one vector of same shape as `y`
    """ 
    oof_concat_probas = cross_val_predict(best_pipeline, X, y, cv=kfolds, method="predict_proba")[:, 1]
    return logloss_and_pr_auc(y, oof_concat_probas)
