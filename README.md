Every feature below is computed from data on or before the cutoff (r1: 2017-01-31, r2: 2017-02-28), and there is one row per user. All log windows count back from the cutoff, so "30d" means the 30 days ending on the cutoff.

**Identity and target**

| Column | Meaning |
|---|---|
| `msno` | Anonymized user ID. Never a model input. |
| `label` | 1 if the user churned, i.e. made no new subscription within 30 days after their membership expired. This is the target. |

**Member profile** (from `members_v3`, which doesn't cover every user)

| Feature | Meaning | Notes |
|---|---|---|
| `city` | Anonymous city code. | Categorical. NULL when there is no member row. |
| `registeredvia` | Registration channel code. | Categorical. |
| `gender` | Male, female or NULL. | Heavily missing, so the missingness is itself a signal. |
| `age` | Age from `bd`, kept only if it is between 10 and 90. |  date. | NULL if registered after the cutoff or no member row. |
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

**Last transaction** (the latest by date, ties broken by the later expiry)

| Feature | Meaning |
|---|---|
| `last_payment_method` | Payment method code, categorical. |
| `last_plan_days` | Plan length in days (7, 30, 90, 365 and so on). |
| `last_list_price` / `last_amount_paid` | Plan list price and amount actually paid.  is one of the strongest churn signals. |
| `last_is_cancel` | Whether the last transaction was a cancellation. Also very strong. |
| `days_to_expiry` | Expiry date minus cutoff. After your filter it is the position inside the label month (1-28 for r1, 1-31 for r2). |

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
| `secs_trend_30d_vs_90d` |  Last 30 days of seconds divided by the previous 30. This is the cleanest decline signal. |
| `active_days_delta` | Active days in the last 30 minus the previous 30. |




## decisions:

---

- **models:**
  we use two models here: `LightGBM` from our inductive bias about this task, as it's known to perform well on large datasets, handles categorical variables well with no encoding, highly optimized, and natively learns the multivariate relationships with the target and also the underlying information behind MNAR missing values, which would be lost with imputation.

  `Logistic Regression` is used as a baseline to justify the usage of `LightGBM`, this latter should outperform it strongly, otherwise, the task in hand would be considered simple enough that using a complex model would be just an overkill.

---  

---  

- **metrics:**  
    we use `Log Loss` (cross-entropy) as the tuning metric and primary comparison metric, as it penalizes distorted probabilities heavily, so it optimizes for calibrated models.  

    we also use `Average Precision` (AUC of Precision-Recall curve, averaged over thresholds) as the diagnostic for imbalance.  

---

---

- **class imbalance handling:**
    we avoided using models hyperparams that would handle this (class_weight for LogisticRegression and is_unbalance for LightGBM) as they multiply the minority loss by ~17x, so the models act as if the prior is 50/50, thus distorting the probabilities and breaks calibration.  

    this is a safe choice because we have enough minority samples to learn the patterns (~50k minority samples), combined with the usage of stratisfied cross validation folds.

---

- **evaluation on out-of-fold:**
    we first fit the best pipeline per model on all data and save it, to finally test it on the Round2 data. we then run `cross_val_predict` with `StratifiedKFold`, which clones the pipeline and trains 5 fresh copies on 4/5ths each, predicting the held-out 1/5th. 

    the 5 held-out prediction pieces are concatenated into one vector per user and scored once. this pooled score differs from averaging the calculated scores per fold: pooled builds a single PR curve over all ~49k positives (more stable at 5.6% churn), while mean-of-folds averages 5 noisy curves.

    concatenation is valid here because all folds share hyperparams, base rate (~5.6% via stratification), and calibration (no reweighting), and because the used metrics allow it.  
    the pooled artifact is needed for threshold/top-k tuning, calibration checks, and LR+GBM stacking, a single mean number cannot do that.

---