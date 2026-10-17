import argparse
import joblib

from pathlib import Path
from typing import Literal

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




def calibrate_model(model: ModelName, methods: tuple = CALIB_METHODS) -> dict:
    """calibrate and save model per calibration method"""


    def _prepare_X(model: ModelName, X: pd.DataFrame) -> pd.DataFrame:
        """apply the model's training-time transformations to raw X.  
        X must be the raw feature frame (df.drop(columns = ["label", "msno"]))"""
        if model == "logireg":
            return engineer_lr_features(X, R2_WIN_LEN)
        return coerce_lgbm_dtypes(X)


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
        calibrator.fit(_prepare_X(model, X_calib), y_calib)
        return calibrator


    fitted = {m: _fit_calibrator(model, m) for m in methods}
    out = ARTIFACTS_DIR / f"calib_{model}_pipeline.joblib"
    joblib.dump(fitted, out)



#orchestrator for main.py CLI 
def run_calibration(overwrite_split: bool = False):
    split_r2_calib_test(seed=SEED, overwrite=overwrite_split)
    for _model in ["logireg", "lgbm"]:
        calibrate_model(_model)



if __name__ == "__main__": 

    ap = argparse.ArgumentParser(description="split r2 calib/test + fit prefit calibrators (thin wrapper over churn.calibrate)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_split = sub.add_parser("split", help="stratified r2 split via split_r2_calib_test")
    p_split.add_argument("--r2-features", default=str(FEATURES["r2"]))
    p_split.add_argument("--calib-frac", type=float, default=R2_CALIB_FRAC)
    p_split.add_argument("--seed", type=int, default=SEED)
    p_split.add_argument("--overwrite", action="store_true")

    p_cal = sub.add_parser("calibrate", help="fit prefit calibrators via calibrate_model")
    p_cal.add_argument("--model", choices=["logireg", "lgbm"], required=True)
    p_cal.add_argument("--methods", nargs="+", choices=list(CALIB_METHODS), default=list(CALIB_METHODS))

    args = ap.parse_args()
    if args.cmd == "split":
        split_r2_calib_test(Path(args.r2_features), args.calib_frac, args.seed, args.overwrite)
    if args.cmd == "calibrate":
        calibrate_model(args.model, tuple(args.methods))
