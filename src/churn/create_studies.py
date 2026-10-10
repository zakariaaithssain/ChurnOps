import optuna
from optuna import Trial
from sklearn.model_selection import cross_val_score

from churn.config import N_OPTUNA_TRIALS


def run_logireg_study(builder, X, y, kfolds, n_trials=N_OPTUNA_TRIALS):
    def _lireg_objective(trial: Trial):
        C = trial.suggest_float('C', 1e-3, 1, log=True)
        spline_n_knots = trial.suggest_int('n_knots', 4, 8)

        estimator = builder.create_pipeline(C=C, n_knots=spline_n_knots)
        return cross_val_score(estimator, X, y, cv=kfolds, scoring="neg_log_loss").mean()

    study = optuna.create_study(direction="maximize", study_name="logireg_study")
    study.optimize(_lireg_objective, n_trials=n_trials, n_jobs=-1)
    return study


def run_lightgbm_study(builder, X, y, kfolds, n_trials=N_OPTUNA_TRIALS):
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
        return cross_val_score(estimator, X, y, cv=kfolds, scoring="neg_log_loss").mean()

    study = optuna.create_study(direction="maximize", study_name="lgbm_study")
    study.optimize(_lightgbm_objective, n_trials=n_trials, n_jobs=-1)
    return study


