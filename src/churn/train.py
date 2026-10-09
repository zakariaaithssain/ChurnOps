import optuna
from optuna import Trial
from create_models import create_logistic_regression, create_gradient_boosting




def logistic_regression_objective(trial : Trial): 
    C = trial.suggest_float('C', 1e-3, 1)
    spline_n_knots = trial.suggest_int('n_knots', 4, 8)
