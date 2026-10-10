import optuna
from optuna import Trial
from sklearn.base import clone
from sklearn.metrics import log_loss

from churn.config import LR_OPTUNA_TRIALS, LGBM_OPTUNA_TRIALS, OPTUNA_DSN

PRUNER = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)


def _pruned_cv_neg_log_loss(trial: Trial, estimator, X, y, kfolds) -> float:
    """Mean neg_log_loss over folds, reporting the running mean after each
    fold so the pruner can stop bad trials early instead of scoring all folds."""
    fold_scores: list[float] = []
    for step, (tr, te) in enumerate(kfolds.split(X, y)):
        est = clone(estimator)
        X_tr = X.iloc[tr] if hasattr(X, "iloc") else X[tr]
        X_te = X.iloc[te] if hasattr(X, "iloc") else X[te]
        y_tr = y.iloc[tr] if hasattr(y, "iloc") else y[tr]
        y_te = y.iloc[te] if hasattr(y, "iloc") else y[te]
        est.fit(X_tr, y_tr)
        p = est.predict_proba(X_te)[:, 1]
        fold_scores.append(-float(log_loss(y_te, p, labels=[0, 1])))
        trial.report(sum(fold_scores) / len(fold_scores), step)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return sum(fold_scores) / len(fold_scores)


def run_logireg_study(builder, X, y, kfolds, n_trials=LR_OPTUNA_TRIALS):
    def _lireg_objective(trial: Trial):
        C = trial.suggest_float('C', 1e-3, 1, log=True)
        spline_n_knots = trial.suggest_int('n_knots', 4, 8)

        estimator = builder.create_pipeline(C=C, n_knots=spline_n_knots)
        return _pruned_cv_neg_log_loss(trial, estimator, X, y, kfolds)

    study = optuna.create_study(
        direction="maximize",
        pruner=PRUNER,
        study_name="logireg_study",
        storage=OPTUNA_DSN,
        load_if_exists=True)

    study.optimize(_lireg_objective, n_trials=n_trials, n_jobs=8) #parallelize only for logireg, because it has none built in. 
    return study


def run_lightgbm_study(builder, X, y, kfolds, n_trials=LGBM_OPTUNA_TRIALS):
    def _lightgbm_objective(trial: Trial):
        num_leaves = trial.suggest_int("num_leaves", 15, 127)
        min_child_samples = trial.suggest_int("min_child_samples", 50, 500)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 50.0, log=True)

        estimator = builder.create_pipeline(
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda)
        return _pruned_cv_neg_log_loss(trial, estimator, X, y, kfolds)

    study = optuna.create_study(
        direction="maximize",
        pruner=PRUNER,
        study_name="lgbm_study",
        storage=OPTUNA_DSN,
        load_if_exists=True)

    study.optimize(_lightgbm_objective, n_trials=n_trials) # don't use parallelism, already used in model level. 
    return study
