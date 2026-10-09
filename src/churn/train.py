import optuna
from optuna import Trial

from sklearn.model_selection import cross_val_score, StratifiedKFold

try:
    from churn.create_models import MyLogisticRegression, MyLightGBM
    from churn.config import FEATURES_FILE, R1_WIN_LEN, SEED
except ImportError:  # running as script inside src/churn/
    from create_models import MyLogisticRegression, MyLightGBM
    from config import FEATURES_FILE, R1_WIN_LEN, SEED
N_TRIALS = 100
FOLDS = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)



def run_lireg_study(kfolds): 
    lireg_builder = MyLogisticRegression(filepath=FEATURES_FILE, window_len=R1_WIN_LEN)
    lireg_X, lireg_y = lireg_builder.create_X_y()

    def _lireg_objective(trial : Trial): 
        C = trial.suggest_float('C', 1e-3, 1, log=True)
        spline_n_knots = trial.suggest_int('n_knots', 4, 8)
        
        estimator = lireg_builder.create_pipeline(C=C, n_knots=spline_n_knots)
        return cross_val_score(estimator, lireg_X, lireg_y, cv=kfolds, scoring="neg_log_loss").mean()
    
    study = optuna.create_study(direction="maximize")
    study.optimize(_lireg_objective, n_trials=N_TRIALS, n_jobs=-1)
    return study




def run_lightgbm_study(kfolds):
    lightgbm_builder = MyLightGBM(filepath=FEATURES_FILE)
    lightgbm_X, lightgbm_y = lightgbm_builder.create_X_y()

    def _lightgbm_objective(trial: Trial):
        num_leaves = trial.suggest_int("num_leaves", 15, 127)
        min_child_samples = trial.suggest_int("min_child_samples", 50, 500)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 50.0, log=True)

        estimator = lightgbm_builder.create_pipeline(
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
        )
        return cross_val_score(estimator, lightgbm_X, lightgbm_y, cv=kfolds, scoring="neg_log_loss").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(_lightgbm_objective, n_trials=N_TRIALS, n_jobs=-1)
    return study


# def lightgbm_objective(trial: Trial): 

#     num_leaves = trial.suggest_int('num_leaves', 15, 127)
#     min_child_samples = trial.suggest_int('min_child_samples', 50, 500)
#     colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1)
#     reg_alpha = trial.suggest_float('reg_alpha', 1e-3, 5)
#     reg_lambda = trial.suggest_float('reg_lambda', 0.1, 50)

