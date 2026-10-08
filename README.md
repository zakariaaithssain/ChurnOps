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
| `age` | Age from `bd`, kept only if it is between 10 and 90. | The raw `bd` column has outliers from -7000 to 2015. |
| `days_since_registration` | Cutoff minus registration date. | NULL if registered after the cutoff or no member row. |
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
| `last_list_price` / `last_amount_paid` | Plan list price and amount actually paid. The gap between them is the discount. |
| `last_is_autorenew` | Whether auto-renew was on. This is one of the strongest churn signals. |
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
| `secs_trend_30d_vs_90d` | The same idea for 30d versus 90d. The baseline contains the numerator, so it is the weakest of the three. |
| `secs_trend_30d_vs_prev30d` | Last 30 days of seconds divided by the previous 30. This is the cleanest decline signal. |
| `active_days_delta` | Active days in the last 30 minus the previous 30. |



