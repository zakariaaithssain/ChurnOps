from pathlib import Path
import os


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / str("data")
RAW_DATA = DATA_DIR / str("raw")
ARTIFACTS_DIR = ROOT / str("artifacts")
CALIB_DATA = DATA_DIR / str("r2_calib_data.joblib")
TEST_DATA = DATA_DIR / str("r2_test_data.joblib")

R2_CALIB_FRAC = 0.2
CALIB_METHODS = ("sigmoid", "isotonic")

SCHEMAS = ["r1", "r2"]
RAW_SCHEMA = "raw"
DATA_DSN = os.getenv(
    "CHURN_DATA_DSN", "postgresql://kkbox:postgrespass@localhost:5432/kkbox_data"
)

FEATURES_COLS = [
    "msno",
    "label",
    "city",
    "registeredvia",
    "gender",
    "age",
    "days_since_registration",
    "has_member_row",
    "n_trans",
    "n_trans_90d",
    "n_cancel",
    "n_no_autorenew",
    "n_payment_methods",
    "n_distinct_plans",
    "n_discounted",
    "total_paid",
    "avg_paid",
    "days_since_first_trans",
    "last_payment_method",
    "last_plan_days",
    "last_list_price",
    "last_amount_paid",
    "last_is_autorenew",
    "last_is_cancel",
    "days_to_expiry",
    "has_logs",
    "active_days_7d",
    "active_days_30d",
    "active_days_90d",
    "active_days_prev30d",
    "secs_7d",
    "secs_30d",
    "secs_90d",
    "secs_prev30d",
    "plays_30d",
    "plays_90d",
    "skip_rate_90d",
    "completion_rate_90d",
    "repeat_ratio_90d",
    "secs_per_play_90d",
    "max_secs_day_90d",
    "std_secs_90d",
    "days_since_last_log",
    "secs_trend_7d_vs_30d",
    "secs_trend_30d_vs_90d",
    "secs_trend_30d_vs_prev30d",
    "active_days_delta",
]

CATEGORICAL_COLS = ["city", "registeredvia", "gender", "last_payment_method"]

BOOL_COLS = ["has_member_row", "last_is_autorenew", "last_is_cancel", "has_logs"]

SEED = 37
LR_OPTUNA_TRIALS = 20
LGBM_OPTUNA_TRIALS = 100
OPTUNA_DSN = os.getenv(
    "OPTUNA_DSN", "postgresql://kkbox:postgrespass@localhost:5432/optuna_studies"
)

N_JOBS = 8  # I have 8 physical cores

MLFLOW_EXPERIMENT = "KKBox_Churn_Prediction"
MLFLOW_PARENT_RUN = "Full_Training_Pipeline"


# sql files names
CREATE_SCHEMAS = "create_schemas.sql"
LOAD_ROUND = "load_round.sql"
RESET = "reset.sql"
BUILD_FEATURES = "build_features.sql"

CHUNK = 16 * 1024 * 1024
WORK_MEM = "512MB"
SYNCHRONOUS_COMMIT = "off"

# lookback_days must be >= the longest window used in build_features.sql (90 days).
# labels are relative to RAW_DATA. features are relative to DATA_DIR.
# win_len is the label-window length in days, kept per round next to win_end.
# r1 is train, r2 is test. user_logs_v2.csv is intentionally unused:
# both rounds reuse the shared user_logs.csv filtered by cutoff.
ROUNDS = {
    "r1": {
        "cutoff": "2017-01-31",
        "lookback_days": 90,
        "labels": "train.csv",
        "win_end": "2017-02-28",
        "win_len": 28,
        "features": "r1_features.parquet",
    },
    "r2": {
        "cutoff": "2017-02-28",
        "lookback_days": 90,
        "labels": "test/churn_comp_refresh/train_v2.csv",
        "win_end": "2017-03-31",
        "win_len": 30,
        "features": "r2_features.parquet",
    },
}

FEATURES_FILES = {
    round_name: DATA_DIR / str(cfg["features"]) for round_name, cfg in ROUNDS.items()
}

# COPY maps columns by position: these lists must match the CSV column order.
RAW_COLS = {
    "labels": ["msno", "ischurn"],
    "members": [
        "msno",
        "city",
        "bd",
        "gender",
        "registeredvia",
        "registrationinitdate",
    ],
    "transactions": [
        "msno",
        "paymentmethodid",
        "paymentplandays",
        "planlistprice",
        "actualamountpaid",
        "isautorenew",
        "transactiondate",
        "membershipexpiredate",
        "iscancel",
    ],
    "userlogs": [
        "msno",
        "logdate",
        "num25",
        "num50",
        "num75",
        "num985",
        "num100",
        "numunq",
        "totalsecs",
    ],
}


# userlogs last (huge, shared by both rounds via cutoff filtering)
RAW_LOADS = [
    (f"{RAW_SCHEMA}.members", "members", "members_v3.csv"),
    (f"{RAW_SCHEMA}.transactions_v1", "transactions", "transactions.csv"),
    (
        f"{RAW_SCHEMA}.transactions_v2",
        "transactions",
        "test/churn_comp_refresh/transactions_v2.csv",
    ),
    (f"{RAW_SCHEMA}.userlogs", "userlogs", "user_logs.csv"),
]

# NOTE: these run with search_path set to (round, raw), so round tables are
# unqualified and the shared members table is qualified as raw.members.
RANGES = """
SELECT 'labels' AS tbl, COUNT(*) AS n, NULL::date AS min_d, NULL::date AS max_d FROM labels
UNION ALL SELECT 'transactions', COUNT(*), MIN(transactiondate), MAX(transactiondate) FROM transactions
UNION ALL SELECT 'userlogs',     COUNT(*), MIN(logdate),         MAX(logdate)         FROM userlogs
"""
COVERAGE = f"""
SELECT COUNT(*)           AS labeled_users,
       COUNT(t.msno)      AS with_trans,
       COUNT(u.msno)      AS with_logs,
       COUNT(m.msno)      AS with_member,
       AVG(l.ischurn::int) AS churn_rate
FROM labels l
LEFT JOIN (SELECT DISTINCT msno FROM transactions) t USING (msno)
LEFT JOIN (SELECT DISTINCT msno FROM userlogs)     u USING (msno)
LEFT JOIN {RAW_SCHEMA}.members m USING (msno)
"""
