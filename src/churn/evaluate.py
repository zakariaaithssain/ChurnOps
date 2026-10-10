from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import cross_val_predict


def oof_scores(y_true, probas) -> dict:
    return {
        "logloss": float(log_loss(y_true, probas)),
        "pr_auc": float(average_precision_score(y_true, probas)),
    }



def refit_save_best_pipeline(best_pipeline, X, y, save_path: Path, prefix: str):
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, save_path / f"{prefix}_pipeline.joblib")



def calculate_save_oof_metrics(builder, best_pipeline, X, y, kfolds, save_path: Path, prefix: str) -> dict : 
    # at each cv iteration, we calculate the probas of the model on the out-of-fold samples
    # then concatenate the results to one vector of same shape as y
    oof_concat_probas = cross_val_predict(best_pipeline, X, y, cv=kfolds, method="predict_proba")[:, 1]
    scores = oof_scores(y, oof_concat_probas)
    oof = pd.DataFrame({
        "msno": builder.df["msno"].values,
        "label": y.values,
        "out-of-fold proba": oof_concat_probas,
    })
    oof.to_csv(save_path / f"{prefix}_oof.csv", index=False)
    return scores
