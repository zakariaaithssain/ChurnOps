from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from churn.config import (
    ARTIFACTS_DIR, CALIB_DATA, CALIB_METHODS, FEATURES, R2_CALIB_FRAC,
    ROUNDS, SEED, TEST_DATA,
)
from churn.create_pipelines import coerce_lgbm_dtypes, engineer_lr_features




ModelName = Literal["logireg", "lgbm"]
CalibMethod = Literal["sigmoid", "isotonic"]

R2_WIN_LEN = ROUNDS["r2"]["win_len"]


def split_r2_calib_test(r2_features: Path = FEATURES["r2"],
                        calib_frac: float = R2_CALIB_FRAC,
                        seed: int = SEED,
                        overwrite: bool = False) -> dict:
    """Stratified r2 split into calib/test, saved as (X, y) joblibs.

    X is the raw feature frame (no label/msno); per-model transformations
    are applied later via churn.create_pipelines helpers so calib/test go
    through exactly the same path as training data. Fail-fast if the split
    exists unless overwrite=True.
    """
    if not overwrite and (CALIB_DATA.is_file() or TEST_DATA.is_file()):
        raise FileExistsError(f"{CALIB_DATA} or {TEST_DATA} already exists.")

    df = pd.read_parquet(Path(r2_features)).copy()
    X = df.drop(columns=["label", "msno"])
    y = df["label"]

    X_calib, X_test, y_calib, y_test = train_test_split(
        X, y, train_size=calib_frac, stratify=y, random_state=seed)
    
    CALIB_DATA.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_calib, y_calib), CALIB_DATA)
    joblib.dump((X_test, y_test), TEST_DATA)
    stats = {
        "calib_rows": int(len(y_calib)), "test_rows": int(len(y_test)),
        "calib_rate": float(y_calib.mean()), "test_rate": float(y_test.mean()),
        "calib_path": str(CALIB_DATA), "test_path": str(TEST_DATA),
    }
    print(f"calib: {stats['calib_rows']:,} rows (rate={stats['calib_rate']:.4f}) -> {CALIB_DATA}")
    print(f"test:  {stats['test_rows']:,} rows (rate={stats['test_rate']:.4f}) -> {TEST_DATA}")
    return stats



def load_pipeline(model: ModelName):
    pipeline_file = ARTIFACTS_DIR / f"{model}_pipeline.joblib"
    if not pipeline_file.is_file():
        raise FileNotFoundError(f"{pipeline_file} does not exist, did you save {model} best pipeline?")
    return joblib.load(pipeline_file)



def prepare_X(model: ModelName, X: pd.DataFrame) -> pd.DataFrame:
    """apply the model's training-time transformations to raw X.  
    X must be the raw feature frame (df.drop(columns = ["label", "msno"]))"""
    if model == "logireg":
        return engineer_lr_features(X, R2_WIN_LEN)
    return coerce_lgbm_dtypes(X)



def calibrate_model(model: ModelName, methods: tuple = CALIB_METHODS) -> dict:
    """calibrate and save model per calibration method"""

    def _fit_calibrator(model: ModelName, method: CalibMethod):
        """fit a prefit calibrator on the calib split (base estimator untouched).
        """
        if not CALIB_DATA.is_file():
            raise FileNotFoundError(f"{CALIB_DATA} does not exist, run `split` first to create it.")
        X_calib, y_calib = joblib.load(CALIB_DATA)

        pipeline_file = ARTIFACTS_DIR / f"{model}_pipeline.joblib"
        if not pipeline_file.is_file():
            raise FileNotFoundError(f"{pipeline_file} does not exist, did you save {model} best pipeline?")
        pipeline = joblib.load(pipeline_file)

        calibrator = CalibratedClassifierCV(FrozenEstimator(pipeline, method=method))
        calibrator.fit(prepare_X(model, X_calib), y_calib)
        return calibrator

    fitted = {m: _fit_calibrator(model, m) for m in methods}
    out = ARTIFACTS_DIR / f"calib_{model}_pipeline.joblib"
    joblib.dump(fitted, out)


