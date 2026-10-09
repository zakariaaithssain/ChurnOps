from pathlib import Path


FEATURES_FILE = Path("data/r1_features.parquet")
CATEGORICAL = ["city", "registeredvia", "gender", "last_payment_method"]
SEED = 37
R1_WIN_LEN = 28
R2_WIN_LEN = 30