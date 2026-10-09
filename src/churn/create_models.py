import pandas as pd 

from pathlib import Path
from typing import Literal

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin, BaseEstimator

from lightgbm import LGBMClassifier

from config import * 
# we avoid using class_weight (logiReg) or is_unbalance (lightgbm) as they will distort probabilities


# logistic regression (baseline)
def create_logistic_regression(filepath : str, window_len: Literal[28, 30], C:float, n_knots: int):
    # from EDA conclusions
    ENGINEERED_FOR_LR = ["age_is_missing", "is_plan_30", "is_city_1", "dte_is_last_day",
                "has_cancel", "has_no_autorenew", "has_discount", "noauto_inactive"]
    # too correlated with others
    CORR_DROP  = ["plays_30d","plays_90d","secs_7d","secs_90d","secs_prev30d",
                "total_paid","last_list_price","secs_trend_30d_vs_90d"]
    # U-shaped corr with label (both decline AND growth predict churn)
    # linear term cannot represent a U, so each gets a spline expansion BESIDE its raw column.
    U_SHAPED = ["secs_trend_30d_vs_prev30d", "active_days_delta"]


    df =  pd.read_parquet(Path(filepath)).copy()
    y = df['label']
    X = df.drop(columns=['label', 'msno'])

    # add engineered features: 
    X['age_is_missing'] = X['age'].isna().astype(float)
    X['is_plan_30'] = (X['last_plan_days'] == 30).astype(float)
    X['is_city_1'] = (pd.to_numeric(X['city'], errors='coerce') == 1).astype(float)
    X['dte_is_last_day'] = (X['days_to_expiry'] == window_len).astype(float)
    X['has_cancel'] = (X['n_cancel'] > 0).astype(float)
    X['has_no_autorenew'] = (X['n_no_autorenew'] > 0).astype(float)
    X['has_discount'] = (X['n_discounted'] > 0).astype(float)
    X['noauto_inactive'] = ((X["last_is_autorenew"] == 0) & (X["active_days_30d"] == 0)).astype(float)

    # encode missingness
    for cat in CATEGORICAL: 
        X[cat] = X[cat].astype('string').fillna("MISSING")

    base = df.columns.difference(CATEGORICAL)
    X[base] = X[base].astype("float64") 

    flags = ENGINEERED_FOR_LR + ["has_member_row","has_logs","last_is_autorenew","last_is_cancel"]
    NUMERICAL = [c for c in df.columns
                if c not in CATEGORICAL + flags + CORR_DROP]

    # transformations derived from EDA
    for_numerical = Pipeline([
                        ('median_impute', SimpleImputer(strategy='median', add_indicator=True)),
                        ('quantile_transfrom', QuantileTransformer(n_quantiles=200, output_distribution='normal', random_state=SEED))
                        
    ])
    for_ushaped = Pipeline([
                        ('median_impute', SimpleImputer(strategy='median')), 
                        ('quantile_transform', QuantileTransformer(output_distribution='uniform', n_quantiles=200, random_state=SEED)), 
                        ('spline_transform', SplineTransformer(n_knots=n_knots, degree=3, extrapolation='constant'))
    ])

    onehotencoder = lambda f : OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=f)

    preprocess = ColumnTransformer([
        ("numerical", for_numerical, NUMERICAL),
        ("u_shaped", for_ushaped, U_SHAPED), 
        ("flags", "passthrough", flags), 
        ("categorical_1", onehotencoder(0.01), ["city", "registeredvia", "gender"]), 
        ("categorical_2", onehotencoder(1000), ["last_payment_method"])
    ])

    return Pipeline([
        ("preprocess", preprocess), 
        ("lr_estimator", LogisticRegression(C=C, max_iter=2000, random_state=SEED))
    ]), X, y



# gradient boosting, handles missing data well (especially mnar)
# no need for engineered features, correlated features dropping, 
# multivariate interactions modeling, nor any transformation. 
def create_gradient_boosting(
                    filepath: str, 
                    min_lumping_count : int = 1000,
                    num_leaves : int = 31,
                    min_child_samples: int = 20,
                    colsample_bytree: float = 1,
                    reg_alpha: float = 0,
                    reg_lambda: float = 0
                    ): 
    df =  pd.read_parquet(Path(filepath)).copy()
    y = df['label']
    X = df.drop(columns=['label', 'msno'])
    for feature in CATEGORICAL: 
        # for lightgbm native handling
        X[feature] = X[feature].astype("category")
    
    return Pipeline([
        ("lumping", LumpingTransformer(min_count=min_lumping_count)),
        ("lightgbm_estimator", LGBMClassifier(min_child_samples=min_child_samples, num_leaves=num_leaves,
                                            colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                                            reg_lambda=reg_lambda, learning_rate=0.1, random_state=SEED))
    ]), X, y 






# custom Transformer for Lumping rare categories to "Other"
# for each categ feature, group classes that occur less than min_count into "Other" 
# "missing" are always kept even if less than min_count. 
# this prevents lightgbm from overfitting rare classes while keeping MNAR signal ( "missing")

class LumpingTransformer(TransformerMixin, BaseEstimator): 
    def __init__(self, min_count: int = 1000):
        self.min_count = min_count
        self.categ_dense_classes_ = None

    def fit(self, X, y = None): 
        self.categ_dense_classes_ = {}
        for c in CATEGORICAL:
            vals = X[c].astype("string").fillna("MISSING")
            counts = vals.groupby(vals).transform("count")
            keep = sorted(set(vals[counts >= self.min_count]) | {"MISSING"})
            self.categ_dense_classes_[c] = keep

        return self

    def transform(self, X): 
        X = X.copy()
        for categ, keep in self.categ_dense_classes_.items():
            vals = X[categ].astype("string").fillna("MISSING")
            vals = vals.where(vals.isin(keep), "OTHER")
            X[categ] = pd.Categorical(vals, categories=keep + ["OTHER"])
        return X


