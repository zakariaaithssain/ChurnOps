import joblib
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

from churn.config import CATEGORICAL, SEED, N_JOBS


# we avoid using class_weight (logiReg) or is_unbalance (lightgbm) as they will distort probabilities



class LogisticRegressionPipeline: 
    def __init__(self, filepath : str, window_len : Literal[28, 30]): 
        self.filepath = filepath
        self.window_len = window_len
        self.df =  (pd.read_parquet(Path(self.filepath))
                 .copy())

        # from EDA conclusions
        self.ENGINEERED_FOR_LR = ["age_is_missing", "is_plan_30", "is_city_1", "dte_is_last_day",
                    "has_cancel", "has_no_autorenew", "has_discount", "noauto_inactive"]
        self.flags = self.ENGINEERED_FOR_LR + ["has_member_row","has_logs","last_is_autorenew","last_is_cancel"]
        # too correlated with others
        self.CORR_DROP  = ["plays_30d","plays_90d","secs_7d","secs_90d","secs_prev30d",
                    "total_paid","last_list_price","secs_trend_30d_vs_90d"]
        
        self.NUMERICAL = [c for c in self.df.columns if c not in CATEGORICAL + self.flags + self.CORR_DROP + ["label", "msno"]]
        # U-shaped corr with label (both decline AND growth predict churn)
        # linear term cannot represent a U, so each gets a spline expansion BESIDE its raw column.
        self.U_SHAPED = ["secs_trend_30d_vs_prev30d", "active_days_delta"]

        





    def create_pipeline(
        self,
        C : float,
        n_knots : int
    ): 
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
            ("numerical", for_numerical, self.NUMERICAL),
            ("u_shaped", for_ushaped, self.U_SHAPED), 
            ("flags", "passthrough", self.flags), 
            ("categorical_1", onehotencoder(0.01), ["city", "registeredvia", "gender"]), 
            ("categorical_2", onehotencoder(1000), ["last_payment_method"])
        ])

        return Pipeline([
            ("preprocess", preprocess), 
            ("lr_estimator", LogisticRegression(C=C, max_iter=2000, random_state=SEED))
        ])



    def create_X_y(self): 
        y = self.df['label']
        X = engineer_lr_features(self.df.drop(columns=['label', 'msno']), self.window_len)
        return X, y
        


class LightGBMPipeline:
    # gradient boosting handles missing data well (especially MNAR):
    # no engineered features, no correlated-feature dropping,
    # no multivariate interaction modeling, nor any transformation.
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = pd.read_parquet(Path(self.filepath)).copy()

    def create_X_y(self):
        y = self.df["label"]
        X = coerce_lgbm_dtypes(self.df.drop(columns=["label", "msno"]))
        return X, y

    def create_pipeline(
        self,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        min_lumping_count: int = 1000,
    ):
        return Pipeline([
            ("lumping", LumpingTransformer(min_count=min_lumping_count)),
            ("lightgbm_estimator", LGBMClassifier(
                num_leaves=num_leaves,
                min_child_samples=min_child_samples,
                colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                learning_rate=0.1, #no need to tune
                random_state=SEED,
                verbose=-1,
                n_jobs = N_JOBS, # for lightgbm, set parallelism in model level, not cv nor trials level. 
            )),
        ])
        
# helpers 

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




def engineer_lr_features(X: pd.DataFrame, window_len: int) -> pd.DataFrame:
    """Add the EDA-derived LR flags + missingness encoding.
    run on train, calibration, and test data. 
    X must be the raw feature frame (df.drop(columns = ["label", "msno"])).
    """
    X = X.copy()
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

    base = X.columns.difference(CATEGORICAL)
    X[base] = X[base].astype("float64")
    return X


def coerce_lgbm_dtypes(X: pd.DataFrame) -> pd.DataFrame:
    """Same numeric coercion as LightGBMPipeline.create_X_y (guards against
    Decimal columns from Postgres); categoricals are handled by LumpingTransformer."""
    X = X.copy()
    base = X.columns.difference(CATEGORICAL)
    X[base] = X[base].astype("float64")  # guards against Decimal columns from Postgres
    return X


def refit_save_best_pipeline(best_pipeline, X, y, save_path: Path, prefix: str):
    best_pipeline.fit(X, y)
    joblib.dump(best_pipeline, save_path / f"{prefix}_pipeline.joblib")










