from pathlib import Path
import os

FEATURES_FILE = Path("data/r1_features.parquet")
CATEGORICAL = ["city", "registeredvia", "gender", "last_payment_method"]
SEED = 37
R1_WIN_LEN = 28
R2_WIN_LEN = 30
LR_OPTUNA_TRIALS = 20
LGBM_OPTUNA_TRIALS = 100
DATA_DSN = os.getenv("CHURN_DATA_DSN", "postgresql://kkbox:postgrespass@localhost:5432/kkbox_data")
OPTUNA_DSN = os.getenv("OPTUNA_DSN", "postgresql://kkbox:postgrespass@localhost:5432/optuna_studies")
SCHEMAS = ['r1', 'r2']
