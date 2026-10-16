import joblib

from typing import Literal

from churn.config import ARTIFACTS_DIR, ROUNDS


ModelName = Literal["logireg", "lgbm"]
CalibMethod = Literal["sigmoid", "isotonic"]

R2_WIN_LEN = ROUNDS["r2"]["win_len"]



