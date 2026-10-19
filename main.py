import argparse
import json
import mlflow
import mlflow.sklearn

from pathlib import Path
from sklearn.model_selection import StratifiedKFold

from churn.config import ARTIFACTS_DIR, FEATURES, ROUNDS, SEED, LR_OPTUNA_TRIALS, LGBM_OPTUNA_TRIALS, MLFLOW_EXPERIMENT, MLFLOW_PARENT_RUN
from churn.create_pipelines import LightGBMPipeline, LogisticRegressionPipeline
from churn.metrics import brier_skill_score, calculate_oof_metrics
from churn.create_studies import run_lightgbm_study, run_logireg_study
from churn.create_pipelines import refit_save_best_pipeline
from churn.calibrate import run_calibration


def main(argv=None):
    ap = argparse.ArgumentParser(description="hyperparam tune + refit Logistic Regression and LightGBM on KKBOX round data (r1=train, r2=test)")
    ap.add_argument("--stage", choices=["train", "calib", "all"], default="all", help="train (tune+refit), calib (split r2 + fit calibrators), or all")
    ap.add_argument("--overwrite-train", action="store_true")
    ap.add_argument("--schema", choices=list(ROUNDS), default="r1", help="round to train on (features/ win_len come from config.ROUNDS)")
    ap.add_argument("--data-file", default=None, help="override features parquet filepath (default in config.FEATURES[schema])")
    ap.add_argument("--save-path", default=str(ARTIFACTS_DIR))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--overwrite-split", action="store_true", help="overwrite existing r2 calib/test split")
    args = ap.parse_args(argv)

    if args.stage == "calib":
        run_calibration(args.overwrite_split)
        return

    out = Path(args.save_path)
    if out.exists() and not args.overwrite_train: 
        raise RuntimeError(f"{out} already exists which might indicate that models are already trained. use --overwrite-train to retrain.")
    
    out.mkdir(parents=True, exist_ok=True)
    kfolds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    data_path = Path(args.data_file) if args.data_file else FEATURES[args.schema]
    window_len = ROUNDS[args.schema]["win_len"]
    
    #log global pipeline configurations to the Parent Run
    mlflow.log_params({
        "stage": args.stage,
        "schema": args.schema,
        "folds": args.folds,
        "seed": SEED
    })

    summary: dict = {
        "schema": args.schema,
        "data_file": str(data_path),
        "folds": args.folds,
        "n_trials": {},
        "seed": SEED
        }

    # ==========================================
    # LOGISTIC REGRESSION (CHILD RUN 1)
    # ==========================================
    with mlflow.start_run(run_name="Logistic_Regression", nested=True):
        logireg_builder = LogisticRegressionPipeline(filepath=data_path, window_len=window_len)
        X, y = logireg_builder.create_X_y()

        logireg_study = run_logireg_study(logireg_builder, X, y, kfolds, LR_OPTUNA_TRIALS)
        logireg_best_hyperp = dict(logireg_study.best_params)

        mlflow.log_params(logireg_best_hyperp)
        mlflow.log_metric("best_neg_log_loss", logireg_study.best_value)
        mlflow.log_metric("optuna_trials", len(logireg_study.trials))

        logireg_best_pipe = logireg_builder.create_pipeline(**logireg_best_hyperp)
        refit_save_best_pipeline(logireg_best_pipe, X, y, out, "logireg")
        logireg_oof_scores = calculate_oof_metrics(logireg_best_pipe, X, y, kfolds)

        mlflow.log_metrics(logireg_oof_scores)
        mlflow.sklearn.log_model(logireg_best_pipe, artifact_path="model")

        summary["logireg"] = {"params": logireg_best_hyperp, "best_neg_log_loss": logireg_study.best_value, **logireg_oof_scores}
        summary["n_trials"]["logireg"] = len(logireg_study.trials)

    # ==========================================
    # LIGHTGBM (CHILD RUN 2)
    # ==========================================
    with mlflow.start_run(run_name="LightGBM", nested=True):
        lgbm_builder = LightGBMPipeline(filepath=data_path)
        X, y = lgbm_builder.create_X_y()

        lgbm_study = run_lightgbm_study(lgbm_builder, X, y, kfolds, LGBM_OPTUNA_TRIALS)
        lgbm_best_hyperp = dict(lgbm_study.best_params)
        print(f"LightGBM best {lgbm_best_hyperp} neg_log_loss={lgbm_study.best_value:.4f}")

        mlflow.log_params(lgbm_best_hyperp)
        mlflow.log_metric("best_neg_log_loss", lgbm_study.best_value)
        mlflow.log_metric("optuna_trials", len(lgbm_study.trials))

        lgbm_best_pipe = lgbm_builder.create_pipeline(**lgbm_best_hyperp)
        refit_save_best_pipeline(lgbm_best_pipe, X, y, out, "lgbm")
        lgbm_oof_scores = calculate_oof_metrics(lgbm_best_pipe, X, y, kfolds)

        mlflow.log_metrics(lgbm_oof_scores)
        mlflow.sklearn.log_model(lgbm_best_pipe, artifact_path="model")

        summary["lgbm"] = {"params": lgbm_best_hyperp, "best_neg_log_loss": lgbm_study.best_value, **lgbm_oof_scores}
        summary["n_trials"]["lgbm"] = len(lgbm_study.trials)

    # ==========================================
    # CROSS-MODEL SUMMARY
    # ==========================================
    summary["logireg"]["brier_skill_score_logireg_as_ref"] = 0.0
    summary["lgbm"]["brier_skill_score_logireg_as_ref"] = brier_skill_score(summary["lgbm"]["brier"], summary["logireg"]["brier"])
    summary_file = out / "train_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    mlflow.log_artifact(str(summary_file))
    print("wrote summary", out)
    
    # calibration after train
    if args.stage == "all":
        run_calibration(args.overwrite_split)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=MLFLOW_PARENT_RUN):
        main()
