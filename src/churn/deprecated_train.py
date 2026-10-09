"""Train both churn models with cross-validation on the whole of r1.

Setup: r1 is entirely training data. Calibration happens later on r2
(r2 split into cal/test) -- see eda.ipynb, cell c12-decisions.

- Logistic regression: grid over C in {0.01, 0.1, 1}, selected by mean
  log loss in 5-fold CV (no class_weight). Refit the winner on full r1.
- LightGBM: 15-trial random search over GBM_SEARCH_SPACE, selected by mean
  log loss in 5-fold CV with early stopping (100 rounds) on log loss.
  No scale_pos_weight, no sampling. Refit the winner on full r1 with
  n_estimators = mean best-iteration across folds.

Folds are stratified by label x dte_is_last_day (day-28 cohort artifact).

Usage:
    .venv/bin/python src/churn/train.py
    .venv/bin/python src/churn/train.py --n-trials 2 --folds 2  # smoke test

Outputs (artifacts/):
    levels.json         category levels fit on full r1 (apply to r2 later)
    lr_cv.json          C-grid CV table (mean/std log loss + PR-AUC)
    lr_pipeline.joblib  LR pipeline refit on full r1 with best C
    lr_oof.parquet      LR out-of-fold P(churn) with msno + label
    gbm_trials.json     15-trial random-search CV table
    gbm_model.joblib    LGBM refit on full r1
    gbm_oof.parquet     GBM out-of-fold P(churn) with msno + label
    summary.json        best params + CV scores for both models
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import ParameterSampler, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]  # repo root (contains data/)
sys.path.insert(0, str(ROOT / "src"))

from churn.data.feature_selection import (  # noqa: E402
    CATS,
    GBM_PARAMS,
    GBM_SEARCH_SPACE,
    add_features,
    apply_lumping,
    fit_lumping,
    gbm_features,
    make_logistic_reg,
)

LR_C_GRID = [0.01, 0.1, 1.0]


def cv_folds(y, dte_flag, n_splits, seed):
    stratum = y.astype(str) + "_" + dte_flag.astype(str)
    return list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(y, stratum)
    )


def fold_scores(y_true, p):
    return {"logloss": float(log_loss(y_true, p)), "pr_auc": float(average_precision_score(y_true, p))}


def run_lr_cv(X, y, folds, out):
    rows, oof = [], []
    for C in LR_C_GRID:
        fold_ll, fold_pr = [], []
        for i, (tr, va) in enumerate(folds):
            t0 = time.time()
            pipe = make_logistic_reg(X, C=C)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            p = pipe.predict_proba(X.iloc[va])[:, 1]
            s = fold_scores(y.iloc[va], p)
            fold_ll.append(s["logloss"])
            fold_pr.append(s["pr_auc"])
            oof.append(pd.DataFrame({"fold": i, "C": C, "p": p}, index=y.iloc[va].index))
            print(f"  LR C={C} fold {i}: logloss={s['logloss']:.4f} "
                  f"pr_auc={s['pr_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        rows.append({"C": C,
                     "mean_logloss": float(pd.Series(fold_ll).mean()),
                     "std_logloss": float(pd.Series(fold_ll).std()),
                     "mean_pr_auc": float(pd.Series(fold_pr).mean())})
    cv = pd.DataFrame(rows).sort_values("mean_logloss")
    cv.to_json(out / "lr_cv.json", orient="records", indent=2)
    best_c = float(cv.iloc[0]["C"])
    print(f"LR best C={best_c} (mean logloss={cv.iloc[0]['mean_logloss']:.4f})")
    return best_c, pd.concat(oof)


def run_gbm_cv(G, y, folds, n_trials, seed, out):
    trials = list(ParameterSampler(GBM_SEARCH_SPACE, n_iter=n_trials, random_state=seed))
    rows, oof = [], []
    for t, params in enumerate(trials):
        fold_ll, fold_pr, iters = [], [], []
        for i, (tr, va) in enumerate(folds):
            t0 = time.time()
            clf = lgb.LGBMClassifier(**{**GBM_PARAMS, **params})
            clf.fit(G.iloc[tr], y.iloc[tr],
                    eval_X=G.iloc[va], eval_y=y.iloc[va],
                    categorical_feature=CATS,
                    callbacks=[lgb.early_stopping(100, verbose=False)])
            p = clf.predict_proba(G.iloc[va])[:, 1]
            s = fold_scores(y.iloc[va], p)
            fold_ll.append(s["logloss"])
            fold_pr.append(s["pr_auc"])
            iters.append(int(clf.best_iteration_))
            oof.append(pd.DataFrame({"fold": i, "trial": t, "p": p}, index=y.iloc[va].index))
            print(f"  GBM trial {t} {params} fold {i}: logloss={s['logloss']:.4f} "
                  f"pr_auc={s['pr_auc']:.4f} best_iter={clf.best_iteration_} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        rows.append({"trial": t, **{k: (v if not isinstance(v, bool) else int(v))
                                    for k, v in params.items()},
                     "mean_logloss": float(pd.Series(fold_ll).mean()),
                     "std_logloss": float(pd.Series(fold_ll).std()),
                     "mean_pr_auc": float(pd.Series(fold_pr).mean()),
                     "mean_best_iter": float(pd.Series(iters).mean())})
    cv = pd.DataFrame(rows).sort_values("mean_logloss")
    cv.to_json(out / "gbm_trials.json", orient="records", indent=2)
    best = cv.iloc[0].to_dict()
    print(f"GBM best trial={int(best['trial'])} "
          f"(mean logloss={best['mean_logloss']:.4f}, mean_best_iter={best['mean_best_iter']:.0f})")
    return best, pd.concat(oof)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "r1_features.parquet"))
    ap.add_argument("--out", default=str(ROOT / "artifacts"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--skip-lr", action="store_true")
    ap.add_argument("--skip-gbm", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {a.data} ...", flush=True)
    df = pd.read_parquet(a.data)
    print(f"r1 shape: {df.shape}, churn rate: {df['label'].mean():.4f}", flush=True)

    F = add_features(df, win_len=28)  # r1 label month has 28 days
    y = F["label"].astype(int)
    folds = cv_folds(y, F["dte_is_last_day"], a.folds, a.seed)
    print(f"{a.folds}-fold CV stratified by label x dte_is_last_day", flush=True)
    summary = {"n": len(F), "churn_rate": float(y.mean()), "folds": a.folds, "seed": a.seed}

    if not a.skip_lr:
        X = F.drop(columns=["label"])
        best_c, lr_oof = run_lr_cv(X, y, folds, out)
        lr_oof = lr_oof[lr_oof["C"] == best_c].drop(columns="C")
        lr_oof = lr_oof.assign(msno=F["msno"], label=y.values).reset_index(drop=True)
        lr_oof.to_parquet(out / "lr_oof.parquet", index=False)
        print("refitting LR on full r1 ...", flush=True)
        final_lr = make_logistic_reg(X, C=best_c).fit(X, y)
        joblib.dump(final_lr, out / "lr_pipeline.joblib")
        summary["lr"] = {"best_C": best_c}

    if not a.skip_gbm:
        # Frequency-based lumping (no label use) -> fit once on full r1.
        levels = fit_lumping(F, min_n=1000)
        with open(out / "levels.json", "w") as f:
            json.dump(levels, f)
        G = apply_lumping(F, levels)[gbm_features(F)]
        best, gbm_oof = run_gbm_cv(G, y, folds, a.n_trials, a.seed, out)
        trial = int(best["trial"])
        gbm_oof = gbm_oof[gbm_oof["trial"] == trial].drop(columns="trial")
        gbm_oof = gbm_oof.assign(msno=F["msno"], label=y.values).reset_index(drop=True)
        gbm_oof.to_parquet(out / "gbm_oof.parquet", index=False)
        n_est = max(1, int(round(best["mean_best_iter"])))
        final_params = {k: v for k, v in GBM_PARAMS.items() if k != "n_estimators"}
        final_params.update({k: best[k] for k in GBM_SEARCH_SPACE})
        print(f"refitting GBM on full r1 (n_estimators={n_est}) ...", flush=True)
        final_gbm = lgb.LGBMClassifier(**final_params, n_estimators=n_est)
        final_gbm.fit(G, y, categorical_feature=CATS)
        joblib.dump(final_gbm, out / "gbm_model.joblib")
        summary["gbm"] = {"best_trial": trial,
                          "params": {k: best[k] for k in GBM_SEARCH_SPACE},
                          "n_estimators": n_est}

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
