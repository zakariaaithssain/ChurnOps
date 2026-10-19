import joblib
import mlflow
import pandas as pd

from pathlib import Path
from typing import Literal, cast

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import train_test_split

from churn.create_pipelines import coerce_lgbm_dtypes, engineer_lr_features
from churn.config import (
    ARTIFACTS_DIR,
    CALIB_DATA,
    CALIB_METHODS,
    FEATURES_FILES,
    R2_CALIB_FRAC,
    ROUNDS,
    SEED,
    TEST_DATA,
)


ModelName = Literal["logireg", "lgbm"]
CalibMethod = Literal["sigmoid", "isotonic"]
R2_WIN_LEN = ROUNDS["r2"]["win_len"]


def split_r2_calib_test(
    r2_features: Path = FEATURES_FILES["r2"],
    calib_frac: float = R2_CALIB_FRAC,
    seed: int = SEED,
    overwrite: bool = False,
) -> dict:
    """Stratified r2 split into calib/test, saved as (X, y) joblibs."""
    if not overwrite and (CALIB_DATA.is_file() or TEST_DATA.is_file()):
        raise FileExistsError(f"{CALIB_DATA} or {TEST_DATA} already exists.")

    df = pd.read_parquet(Path(r2_features)).copy()
    X = df.drop(columns=["label", "msno"])
    y = df["label"]

    X_calib, X_test, y_calib, y_test = train_test_split(
        X, y, train_size=calib_frac, stratify=y, random_state=seed
    )

    CALIB_DATA.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((X_calib, y_calib), CALIB_DATA)
    joblib.dump((X_test, y_test), TEST_DATA)

    stats = {
        "calib_rows": len(y_calib),
        "test_rows": len(y_test),
        "calib_rate": float(y_calib.mean()),
        "test_rate": float(y_test.mean()),
    }

    # log split params to parent mlflow run
    mlflow.log_params(
        {
            "calib_frac": calib_frac,
            "split_seed": seed,
            "calib_source_data": str(r2_features),
        }
    )
    mlflow.log_metrics(
        {
            "calib_rows": float(stats["calib_rows"]),
            "test_rows": float(stats["test_rows"]),
            "calib_target_rate": float(stats["calib_rate"]),
            "test_target_rate": float(stats["test_rate"]),
        }
    )

    return stats


def calibrate_model(model: ModelName, methods: tuple = CALIB_METHODS) -> None:
    """calibrate and save model per calibration method"""

    def _prepare_X(model: ModelName, X: pd.DataFrame) -> pd.DataFrame:
        if model == "logireg":
            return engineer_lr_features(X, cast(int, R2_WIN_LEN))
        return coerce_lgbm_dtypes(X)

    def _fit_calibrator(model: ModelName, method: CalibMethod):
        if not CALIB_DATA.is_file():
            raise FileNotFoundError(
                f"{CALIB_DATA} does not exist, run `split` first to create it."
            )
        X_calib, y_calib = joblib.load(CALIB_DATA)

        pipeline_file = ARTIFACTS_DIR / f"{model}_pipeline.joblib"
        if not pipeline_file.is_file():
            raise FileNotFoundError(
                f"{pipeline_file} does not exist, did you save {model} best pipeline?"
            )
        pipeline = joblib.load(pipeline_file)

        calibrator = CalibratedClassifierCV(FrozenEstimator(pipeline), method=method)
        calibrator.fit(_prepare_X(model, X_calib), y_calib)
        return calibrator

    # log to isolated mlflow child run per model family calibration
    with mlflow.start_run(run_name=f"Calibration_{model}", nested=True):
        mlflow.log_param("model_family", model)
        mlflow.log_param("calibration_methods", list(methods))

        fitted = {m: _fit_calibrator(model, m) for m in methods}
        out = ARTIFACTS_DIR / f"calib_{model}_pipeline.joblib"
        joblib.dump(fitted, out)

        # log the dict of model per calib method
        mlflow.log_artifact(str(out), artifact_path="calibrated_models")
        print(f"Logged calibrated models for {model} to MLflow artifacts.")


def run_calibration(overwrite_split: bool = False):
    """Orchestrator called inside main.py active MLflow Parent context"""
    split_r2_calib_test(seed=SEED, overwrite=overwrite_split)
    for _model in ["logireg", "lgbm"]:
        calibrate_model(cast(ModelName, _model))
