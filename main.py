import argparse
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from churn.config import ARTIFACTS_DIR, FEATURES, ROUNDS, SEED, LR_OPTUNA_TRIALS, LGBM_OPTUNA_TRIALS
from churn.calibrate import run_calibration
from churn.create_pipelines import LightGBMPipeline, LogisticRegressionPipeline
from churn.metrics import calculate_oof_metrics
from churn.create_studies import run_lightgbm_study, run_logireg_study
from churn.create_pipelines import refit_save_best_pipeline


def main(argv=None):
    ap = argparse.ArgumentParser(description="hyperparam tune + refit Logistic Regression and LightGBM on KKBOX round data (r1=train, r2=test)")
    ap.add_argument("--schema", choices=list(ROUNDS), default="r1", help="round to train on (features/ win_len come from config.ROUNDS)")
    ap.add_argument("--data-file", default=None, help="override parquet path (default: config.FEATURES[schema])")
    ap.add_argument("--save-path", default=str(ARTIFACTS_DIR))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-logreg", action="store_true")
    ap.add_argument("--skip-lgbm", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="after training, fit prefit calibrators on the r2-calib split and evaluate on r2-test")
    args = ap.parse_args(argv)

    out = Path(args.save_path)
    out.mkdir(parents=True, exist_ok=True)

    kfolds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    data_path = Path(args.data_file) if args.data_file else FEATURES[args.schema]
    window_len = ROUNDS[args.schema]["win_len"]
    # NOTE: studies use load_if_exists=True, so optimize(n_trials=N) appends N
    # trials to whatever is already stored. Report actual cumulative totals,
    # not the per-run request, otherwise the summary understates the budget
    # after more than one invocation.
    summary: dict = {"schema": args.schema, "data_file": str(data_path), "folds": args.folds, "n_trials": {}, "seed": args.seed}

    write_summary = False
    trained: list = []

    if not args.skip_logreg:
        builder = LogisticRegressionPipeline(filepath=data_path, window_len=window_len)
        X, y = builder.create_X_y()

        logireg_study = run_logireg_study(builder, X, y, kfolds, LR_OPTUNA_TRIALS)
        logireg_best_hyperp = dict(logireg_study.best_params)
        print(f"Logistic Regression best {logireg_best_hyperp} neg_log_loss={logireg_study.best_value:.4f}")

        logireg_best_pipe = builder.create_pipeline(**logireg_best_hyperp)
        refit_save_best_pipeline(logireg_best_pipe, X, y, out, "logireg")

        logireg_scores = calculate_oof_metrics(logireg_best_pipe, X, y, kfolds)
        print(f"Logistic Regression out-of-fold probas: {logireg_scores}")

        summary["logireg"] = {"params": logireg_best_hyperp, "best_neg_log_loss": logireg_study.best_value, **logireg_scores}
        summary["n_trials"]["logireg"] = len(logireg_study.trials)
        write_summary = True
        trained.append("logireg")

    if not args.skip_lgbm:
        builder = LightGBMPipeline(filepath=data_path)
        X, y = builder.create_X_y()

        lgbm_study = run_lightgbm_study(builder, X, y, kfolds, LGBM_OPTUNA_TRIALS)
        lgbm_best_hyperp = dict(lgbm_study.best_params)
        print(f"LightGBM best {lgbm_best_hyperp} neg_log_loss={lgbm_study.best_value:.4f}")

        lgbm_best_pipe = builder.create_pipeline(**lgbm_best_hyperp)
        refit_save_best_pipeline(lgbm_best_pipe, X, y, out, "lgbm")

        lgbm_scores = calculate_oof_metrics(lgbm_best_pipe, X, y, kfolds)
        print(f"LightGBM out-of-fold probas: {lgbm_scores}")

        summary["lgbm"] = {"params": lgbm_best_hyperp, "best_neg_log_loss": lgbm_study.best_value, **lgbm_scores}
        summary["n_trials"]["lgbm"] = len(lgbm_study.trials)
        write_summary = True
        trained.append("lgbm")

    if args.calibrate:
        if not trained:
            print("nothing trained, skipping calibration")
        else:
            if args.schema == "r2":
                print("warning: base pipelines trained on r2, calibrating on an r2 subset is optimistic")
            summary["calibration"] = run_calibration(models=tuple(trained))
            write_summary = True

    if write_summary:
        (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
        print("wrote summary", out)


if __name__ == "__main__":
    main()
