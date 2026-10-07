SELECT
    COUNT(*)                                        AS n_rows,
    COUNT(DISTINCT msno)                            AS n_users,
    COUNT(*) FILTER (WHERE days_to_expiry IS NULL)  AS null_days_to_expiry,
    MIN(days_to_expiry)                             AS min_dte,
    MAX(days_to_expiry)                             AS max_dte,
    COUNT(DISTINCT skip_rate_90d)                   AS distinct_skip_rate_90d,
    AVG(label::int)::numeric(6,4)                   AS churn_rate
FROM {schema}.train_features;