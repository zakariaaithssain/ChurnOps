# end-to-end MLOps Project for KKBox Churn Prediction.

## Overview

this project contains the following stages:

1. Postgres (`src/churn/data` and `src/churn/sql`)
   1. load raw data to Postgres.
   2. construct the aggregated features table.
   3. load the features table(s) from the database.
2. *(NoteBook)* Exploratory Data Analysis (`src/churn/data/eda.ipynb`)
3. Feature Engineering and Pipelines Construction (`src/churn/create_pipelines.py`)
4. Models Training and Tuning (`src/churn/create_studies.py`)
5. Models Calibration (`src/churn/calibrate.py`)
6. Data Drift and Classification Quality Monitoring (`src/churn/monitor.py`)
7. Final results are in `results.ipynb`.

## How to Run

we use **`uv`** for dependency and environment management.

```bash
# sync dependencies and install the local 'churn' package
uv sync
```

we use **MLFlow** to log metrics and artifacts, an MLFlow server is included in `docker-compose` along with **Postgres** and **PgAdmin** services.

```bash
docker compose up -d
```

you can configure the env variables, see `docker-compose.yaml`.

### 1. Database Initialization & ETL

before training models, you must initialize the PostgreSQL database, populate it with raw data, build target rounds, and export the feature matrices to local Parquet files.

#### A. Initialize Schemas

create the necessary database schemas and structure as defined in `src/churn/config.py`:

```bash
uv run python -m churn.execute_queries init
```

#### B. Load Raw Data

truncates the raw staging tables and streams raw CSV data from configured paths directly into the PostgreSQL database:

```bash
uv run python -m churn.execute_queries load-raw
```

#### C. Build a Specific Evaluation Round

extracts and windows the transactional and user logs for a specific round configuration (`r1` or `r2`) up to its defined cutoff date:

```bash
uv run python -m churn.execute_queries round r1
```

**Note:** This command uses strict constraints; rerun attempts will deliberately fail if tables already exist to protect raw partition structures from accidental data loss.

#### D. Engineer Features for a Round

executes complex SQL window functions and aggregations to construct the engineering features table:

```bash
uv run python -m churn.execute_queries build-features r1
```

#### E. Reset the Database Environment

if you need to completely wipe the slate clean, you can drop all project schemas and data. This command requires manual console confirmation:

```bash
uv run python -m churn.execute_queries reset
```

---

#### Export Features to Local Parquet

once your PostgreSQL features table (`train_features`) is populated, extract it into highly efficient local Parquet binaries to serve as inputs for the model training CLI.

```bash
# import schema r1 features
uv run python -m churn.import_from_postgres r1

# import schema r2 features
uv run python -m churn.import_from_postgres r2
```

The data will automatically save to filepath mapped in `config.FEATURES_FILES` ready for the remaining stages.

---

#### Data Workflow CLI Commands Quick Reference

| Script / Command | Target / Arguments | Description |
| :--- | :--- | :--- |
| `churn.execute_queries init` | *None* | Sets up base database schemas. |
| `churn.execute_queries load-raw` | *None* | Ingests CSVs into raw database partitions. |
| `churn.execute_queries round` | `[r1 \| r2]` | Filters and aligns source timelines to a specific round cutoff. |
| `churn.execute_queries build-features` | `[r1 \| r2]` | Compiles target feature vectors inside the database. |
| `churn.execute_queries reset` | *None* | Destroys all generated tables/schemas (requires confirmation). |
| `churn.import_from_postgres` | `[r1 \| r2]` | Streams data from Postgres into localized `.parquet` matrices. |

---
### 2. Model Training, Calibration and Monitoring

stages **3 to 6** are orchestrated using `main.py` that defines a **CLI** supporting four execution stages via the `--stage` argument.

#### A. Full Pipeline: Tune, Refit & Calibrate (Default)

tunes hyperparameters using Optuna for both Logistic Regression and LightGBM, refits the best pipelines, saves the models, and runs probability calibration on the validation set.

```bash
uv run main.py --stage train-calib
```

#### B. Hyperparameter Tuning & Refit Only

runs hyperparameter optimization and saves the final models to the artifacts directory, bypassing calibration.

```bash
uv run main.py --stage train
```

**Note:** if the save path already contains artifacts, the script will error out. Use the `--overwrite-train` flag to force retraining.

#### C. Post-Train Probability Calibration

if you already have models trained, you can run the calibration phase independently:

```bash
uv run main.py --stage calib
```

#### D. Drift & Performance Monitoring

```bash
#run Evidently data drift detection and classification quality tests on a production/inference dataset:
uv run main.py --stage monitor --inference-data path/to/inference.parquet

# run evidently ui
uv run evidently ui --workspace evidently_workspace
```

#### run using MLproject  
we can also use MLFlow to run the exact commands available in the CLI: 
```bash 
#general command
mlflow run . [-P arg=value] --env-manager=uv
```
**examples:**  
```bash
mlflow run . -P stage=monitor -P data-file=path/to/inference.parquet --env-manager=uv
```
```bash
mlflow run . -P overwrite-train=True -P overwrite-split=True --env-manager=uv
```

#### main CLI Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--stage` | `train-calib` | Pipeline stage: `train`, `calib`, `train-calib`, or `monitor`. |
| `--overwrite-train` | *None* | Flag to overwrite existing models in the artifact folder. |
| `--schema` | `r1` | Round target configuration from `config.ROUNDS`. |
| `--data-file` | *None* | Override path for the features parquet file. |
| `--save-path` | Config Path | Directory to store local pipeline output artifacts. |
| `--folds` | `5` | Number of splits for `StratifiedKFold` cross-validation. |
| `--overwrite-split` | *None* | Flag to overwrite existing calibration validation splits. |
| `--inference-data` | *None* | Path to data for drift monitoring (Required if stage is `monitor`). |

--- 

## Features Table Construction Details

we construct the features table by aggregating different user data from different tables (see `src/churn/sql/build_features.sql`)

Every feature below is computed from data **on or before the cutoff** (`round1: 2017-01-31, round2: 2017-02-28`), and there is one row per user. All log windows count back from the cutoff, so "30d" means the 30 days ending on the cutoff.

**Identity and target**

| Column | Meaning |
|---|---|
| `msno` | Anonymized user ID |
| `label` | (target variable) 1 if the user churned, i.e. **made no new subscription within 30 days after their membership expired.** |

**Member profile** (doesn't cover every user)

| Feature | Meaning | Notes |
|---|---|---|
| `city` | Anonymous city code. | Categorical. |
| `registeredvia` | Registration channel code. | Categorical. |
| `gender` | Male, female or NULL. | Heavily missing, we treat the missingness itself as a signal (MNAR). |
| `age` | Age from `bd`, kept only if it is between 10 and 90. | date. | NULL if registered after the cutoff or no member row. |
| `has_member_row` | Whether a member row exists. | Lets the model use missing demographics as a signal. |

**Transaction history** (all transactions up to the cutoff)

| Feature | Meaning |
|---|---|
| `n_trans` | Lifetime number of transactions. This is a proxy for how many billing cycles the user has been through. |
| `n_trans_90d` | Transactions in the last 90 days. Most users renew monthly, so this is usually about 3. |
| `n_cancel` | Number of cancel transactions. `is_cancel` marks an active cancellation, which doesn't necessarily mean the user churned. |
| `n_no_autorenew` | Number of transactions without auto-renew. Manual renewers are a major churn-risk group. |
| `n_payment_methods` | Distinct payment methods used. |
| `n_distinct_plans` | Distinct plan lengths used. Switching plans suggests an engaged user. |
| `n_discounted` | Transactions where the amount paid was below the list price (promotions). |
| `total_paid` | Lifetime amount paid. Amounts are in New Taiwan Dollars. |
| `avg_paid` | Average amount paid per transaction. |
| `days_since_first_trans` | Customer tenure since the first transaction. It is left-censored, because the history starts around 2015-01-01. |

**Last transaction** (the latest by date)

| Feature | Meaning |
|---|---|
| `last_payment_method` | Payment method code, categorical. |
| `last_plan_days` | Plan length in days (7, 30, 90, 365 and so on). |
| `last_list_price` / `last_amount_paid` | Plan list price and amount actually paid.  is one of the strongest churn signals. |
| `last_is_cancel` | Whether the last transaction was a cancellation. Also very strong. |
| `days_to_expiry` | Expiry date minus cutoff. the position inside the label month (1-28 for r1, 1-31 for r2). |

**Listening activity** (logs in the 90 days before the cutoff)

Windows: 7d, 30d and 90d are nested, since 90d contains 30d, which contains 7d. `prev30d` is the separate stretch from 60 to 31 days before the cutoff, which gives a clean "before" period to compare against.

| Feature | Meaning |
|---|---|
| `has_logs` | Whether the user has any log row in the 90-day window. |
| `active_days_7d / 30d / 90d / prev30d` | Number of days with listening activity in each window. 0 when there are no logs. |
| `secs_7d / 30d / 90d / prev30d` | Total listening seconds per window. Daily values are clipped to 0-86,400 to remove corrupt rows. |
| `plays_30d`, `plays_90d` | Total songs played, summed over all five completion buckets. |
| `skip_rate_90d` | Share of plays abandoned before 50%, i.e. the 25% and 50% buckets divided by total plays. `num_25` counts songs played less than 25% of their length and `num_50` counts those played 25-50%. |
| `completion_rate_90d` | Share of plays that reached at least 75%. That is the `num_985` and `num_100` buckets, songs played over 75% of their length. |
| `repeat_ratio_90d` | Daily unique songs summed, divided by total plays. Lower means more repeat listening. It is a ratio of summed daily counts, not true global uniqueness. |
| `secs_per_play_90d` | Average seconds per play, which is roughly average session depth. |
| `max_secs_day_90d` | Seconds on the user's heaviest listening day. |
| `std_secs_90d` | Standard deviation of daily seconds across active days. It is NULL for users with only one active day. |
| `days_since_last_log` | Cutoff minus the last listening date. It is NULL if there are no logs and can only range from 0 to 89. |

**Trend features**

| Feature | Meaning |
|---|---|
| `secs_trend_7d_vs_30d` | Average daily seconds in the last 7 days divided by the last 30 days. Below 1 means recent decline. |
| `secs_trend_30d_vs_90d` | Last 30 days of seconds divided by the previous 30. This is the cleanest decline signal. |
| `active_days_delta` | Active days in the last 30 minus the previous 30. |

## decisions:

---

- **models:**

  we use two models here: `LightGBM` from our inductive bias about this task, as it's known to perform well on large tabular datasets, handles categorical variables well with no encoding, doesn't require feature normalization, highly optimized, and natively learns the multivariate relationships with the target and also the underlying information behind MNAR missing values, which would be lost with imputation.

  `Logistic Regression` is used as a baseline to justify the usage of `LightGBM`. we use EDA to construct a features engineering pipeline that would maximize the performance of this model (manual flags, correlation-drop, median+indicator+ quantile transformation, splines for U-shaped trends, and lumped one-hot encoding), conversely to LightGBM minimal preprocessing (rare categories lumping only, to avoid overfitting)

---

---

- **primary metrics:**

  we use `Log Loss` (cross-entropy) as the tuning metric, as it penalizes distorted probabilities heavily, so it optimizes for calibrated models.

  we use `ECE` (expected calibration error) as the main comparison metric, as it reflects direcly the models calibration.

  we also use `PR-AUC` as the diagnostic for the models performance on this heavily imbalanced data.

---

---

- **class imbalance handling:**

  we avoided using models hyperparams that would handle this (`class_weight` for LogisticRegression and `is_unbalance` for LightGBM) as they multiply the losses by class weight, which pushes the models to act as if the prior is 50/50, thus distorting the probabilities and breaking calibration.

  this is a safe choice because **we have enough minority samples to learn the patterns** (~50k minority samples), combined with the usage of stratisfied cross validation folds.

---

---

- **data split:**

  we train the models on the features constructed from the Round 1 data.

  Round 2 data is split into calibration and test sets.

---

## results:

the winner model is `LightGBM calibrated using Isotonic method`, with the following main metrics:

- `Expected Calibration Error = 0.0009`.
- `Precision-Recall AUC = 0.53` (0.07 base rate).
- `Lift = Precision / Base Rate = 0.50 / 0.07 = ~7x`

this means that a sample flagged by our model is ~7 times more likely to be a true positive than an sample chosen at random from the dataset. this is an exceptionally strong result for an imbalanced dataset.

**NOTE:** for more results, comparisons, plots and SHAP values, see `results.ipynb`.

---
