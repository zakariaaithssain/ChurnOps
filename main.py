import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from sklearn.model_selection import StratifiedKFold

from churn.config import FEATURES_FILE, R1_WIN_LEN, SEED, N_OPTUNA_TRIALS
from churn.create_pipelines import LightGBMPipeline, LogisticRegressionPipeline
from churn.evaluate import calculate_save_oof_metrics, refit_save_best_pipeline
from churn.create_studies import run_lightgbm_study, run_logireg_study


def main(argv=None):
    ap = argparse.ArgumentParser(description="hyperparam tune + refit Logistic Regression and LightGBM on KKBOX r1 data")
    ap.add_argument("--data-file", default=str(FEATURES_FILE))
    ap.add_argument("--save-path", default=str(ROOT / "artifacts"))
    ap.add_argument("--n-trials", type=int, default=N_OPTUNA_TRIALS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-logreg", action="store_true")
    ap.add_argument("--skip-lgbm", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.save_path)
    out.mkdir(parents=True, exist_ok=True)

    kfolds = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    data_path = Path(args.data_file)
    summary: dict = {"folds": args.folds, "n_trials": args.n_trials, "seed": args.seed}

    write_summary = False
    if not args.skip_logreg:
        builder = LogisticRegressionPipeline(filepath=data_path, window_len=R1_WIN_LEN)
        X, y = builder.create_X_y()

        logireg_study = run_logireg_study(builder, X, y, kfolds, args.n_trials)
        logireg_best_hyperp = dict(logireg_study.best_params)
        print(f"Logistic Regression best {logireg_best_hyperp} neg_log_loss={logireg_study.best_value:.4f}")

        logireg_best_pipe = builder.create_pipeline(**logireg_best_hyperp)
        refit_save_best_pipeline(logireg_best_pipe, X, y, out, "logireg")
        
        logireg_scores = calculate_save_oof_metrics(builder, logireg_best_pipe, X, y, kfolds, out, "logireg")
        print(f"Logistic Regression out-of-fold probas: {logireg_scores}")

        (out / "logireg_best.json").write_text(json.dumps(
            {"params": logireg_best_hyperp, "best_neg_log_loss": logireg_study.best_value, "out_of_fold_probas": logireg_scores}, indent=2))
        summary["logireg"] = {"params": logireg_best_hyperp, **logireg_scores}
        write_summary = True

    if not args.skip_lgbm:
        builder = LightGBMPipeline(filepath=data_path)
        X, y = builder.create_X_y()

        lgbm_study = run_lightgbm_study(builder, X, y, kfolds, args.n_trials)
        lgbm_best_hyperp = dict(lgbm_study.best_params)
        print(f"LightGBM best {lgbm_best_hyperp} neg_log_loss={lgbm_study.best_value:.4f}")

        lgbm_best_pipe = builder.create_pipeline(**lgbm_best_hyperp)
        refit_save_best_pipeline(lgbm_best_pipe, X, y, out, "lgbm")

        lgbm_scores = calculate_save_oof_metrics(builder, lgbm_best_pipe, X, y, kfolds, out, "lgbm")
        print(f"LightGBM out-of-fold probas: {lgbm_scores}")

        lumping_levels = lgbm_best_pipe.named_steps["lumping"].categ_dense_classes_
        (out / "lgbm_lumping_levels.json").write_text(json.dumps(lumping_levels, indent=2))
        (out / "lgbm_best.json").write_text(json.dumps(
            {"params": lgbm_best_hyperp, "best_neg_log_loss": lgbm_study.best_value, "out_of_fold_probas": lgbm_scores}, indent=2))
        summary["lgbm"] = {"params": lgbm_best_hyperp, **lgbm_scores}
        write_summary = True


    if write_summary: 
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print("wrote summary", out)


if __name__ == "__main__":
    main()
