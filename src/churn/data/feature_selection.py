"""Feature engineering and model-specific feature lists.

Trimmed plan (evidence: eda.ipynb, cell c12-decisions).

Deliberately DROPPED from the plan -- do not re-add without new evidence:
- Day-28 investigation + stratified drop-vs-keep ablation -> replaced by the
  cheap round-invariant flag `dte_is_last_day` (last day of the label month:
  day 28 in r1, day 31 in r2).
- Trend-NULL rebuild and state encoding -> LR gets missing indicators,
  LightGBM handles NaN natively.
- avg_paid-within-plan, age-within-subset, and days_since_last_log inside the
  autorenew-off group -> skipped; the write-up must NOT claim price sensitivity.
- Dev-only EDA rerun -> noted as a minor limitation. The choices below are
  coarse enough to be robust to it. Keep the dev/cal split for everything
  from here on.
- Target encoding, payment risk tiers, `is_single_day`, 1st-99th percentile
  winsorizing, and the correlation-pruning ablation -> all dropped.
"""
import pandas as pd
from typing import Literal

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, SplineTransformer
from sklearn.linear_model import LogisticRegression



# Shared categoricals. Rare-level handling differs per model (see below).
CATS = ["city", "registeredvia", "gender", "last_payment_method"]
# see eda.ipynb
# Hand-built flags. LR takes them as linear terms (it cannot find interactions
# on its own); LightGBM EXCLUDES them all -- trees recover them and they would
# only dilute SHAP importance.
ENGINEERED = ["age_is_missing", "is_plan_30", "is_city_1", "dte_is_last_day",
              "has_cancel", "has_no_autorenew", "has_discount", "noauto_inactive"]

def add_features(df : pd.DataFrame, win_len : Literal[28, 31]):          # win_len: 28 for r1, 31 for r2
    df = df.copy()
    # city 1.0 churns ~3.7% vs 7-10% elsewhere; compute BEFORE the string cast below.
    df["is_city_1"] = (pd.to_numeric(df["city"], errors="coerce") == 1).astype(float)
    for c in CATS:
        df[c] = df[c].astype("string").fillna("MISSING")
    base = df.columns.difference(CATS + ["msno", "label"])
    df[base] = df[base].astype("float64")   # guards against Decimal columns from Postgres
    df["age_is_missing"]  = df["age"].isna().astype(float)
    # 30-day plan churns ~4.5% vs 35-100% for every other length.
    df["is_plan_30"]      = (df["last_plan_days"] == 30).astype(float)
    # Round-invariant stand-in for "day 28": last day of the label month.
    df["dte_is_last_day"] = (df["days_to_expiry"] == win_len).astype(float)
    df["has_cancel"]      = (df["n_cancel"] > 0).astype(float)
    df["has_no_autorenew"] = (df["n_no_autorenew"] > 0).astype(float)
    df["has_discount"]    = (df["n_discounted"] > 0).astype(float)
    # Key interaction LR cannot learn alone: silent + no auto-renew churns ~90%.
    df["noauto_inactive"] = ((df["last_is_autorenew"] == 0) & (df["active_days_30d"] == 0)).astype(float)
    return df



# --- Logistic regression ---
# Manual drop list: one of each near-duplicate pair (Spearman |r| > 0.9), plus
# the weakest trend (30d_vs_90d, r = 0.84 with the cleaner 30d_vs_prev30d).
LR_DROP  = ["plays_30d","plays_90d","secs_7d","secs_90d","secs_prev30d",
            "total_paid","last_list_price","secs_trend_30d_vs_90d"]
# U-shaped vs churn (both decline AND growth predict churn): linear term cannot
# represent a U, so each gets a spline expansion BESIDE its raw column.
U_SHAPED = ["secs_trend_30d_vs_prev30d", "active_days_delta"]

def make_logistic_reg(df, C=0.1):
    # Tune ONLY C over {0.01, 0.1, 1} by mean log loss in 5-fold CV.
    # No class_weight.
    flags = ENGINEERED + ["has_member_row","has_logs","last_is_autorenew","last_is_cancel"]
    nums = [c for c in df.columns
            if c not in CATS + flags + LR_DROP + ["msno","label"]]
    # Median-impute with missing indicators, then gaussianize. This absorbs the
    # skew, the outliers and the 761,224 trend explosion, so no clipping or
    # winsorizing is needed. Inside the pipeline => refit per fold.
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                    ("q", QuantileTransformer(output_distribution="normal", n_quantiles=200))])
    spl = Pipeline([("imp", SimpleImputer(strategy="median")),
                    ("q", QuantileTransformer(output_distribution="uniform", n_quantiles=200)),
                    ("sp", SplineTransformer(n_knots=6, degree=3, extrapolation="constant"))])
    # Rare lumping: 1% frequency for city/registeredvia/gender, 1,000 users for
    # payment_method. Unseen r2 levels fall into the rare bucket.
    oh = lambda f: OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=f)
    pre = ColumnTransformer([
        ("num", num, nums), ("spl", spl, U_SHAPED), ("flags", "passthrough", flags),
        ("ca", oh(0.01), ["city", "registeredvia", "gender"]),
        ("cb", oh(1000), ["last_payment_method"])])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C, max_iter=2000))])


# --- LightGBM ---
# Keeps ALL raw features, correlated windows included. Drops only the id, the
# target, and the ENGINEERED flags (trees recover them; see module docstring).
# No transforms, winsorizing, binning or interactions. NaN left untouched --
# no imputation at all.

def gbm_features(df):
    return [c for c in df.columns if c not in ENGINEERED + ["msno", "label"]]

# Frequency-based lumping (NOT label-based, so fitting on dev is safe):
# levels below min_n users become "OTHER". "MISSING" is always kept.
# Call fit_levels on dev AFTER add_features, apply the same dict to r2 so both
# rounds see identical category lists.
def fit_levels(train, min_n=1000):
    return {c: sorted(set(train[c].value_counts().loc[lambda s: s >= min_n].index) | {"MISSING"})
            for c in CATS}

def apply_levels(df, levels):
    df = df.copy()
    for c, keep in levels.items():
        vals = df[c].where(df[c].isin(keep), "OTHER").astype(str)
        df[c] = pd.Categorical(vals, categories=keep + ["OTHER"])
    return df

# Categorical regularization stops tiny payment methods from overfitting.
# Fit with early stopping (100 rounds) on log loss, learning_rate=0.05.
# n_estimators is only a cap. No scale_pos_weight, no sampling.
GBM_PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=0.05,
                  num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, min_data_per_group=500,
                  cat_smooth=50, n_estimators=2000, random_state=42, n_jobs=-1, verbose=-1)
# Light tuning only (features matter more here): 15-trial random search over
# this grid, selected by log loss.
GBM_SEARCH_SPACE = dict(num_leaves=[15, 31, 63],
                        min_child_samples=[50, 100, 300],
                        colsample_bytree=[0.6, 0.8],
                        reg_lambda=[1, 10])
