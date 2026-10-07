BEGIN;

DROP TABLE IF EXISTS train_features;

CREATE TABLE train_features AS
WITH
params AS (
  SELECT %(cutoff)s::date AS cutoff, %(win_end)s::date AS win_end
),

-- 1) user logs: single scan over the round's userlogs (already labeled-users-only, <= cutoff)
logs AS (
  SELECT
    x.msno,
    COUNT(*)                                                        AS active_days_90d,
    COUNT(*) FILTER (WHERE x.logdate > p.cutoff - 30)               AS active_days_30d,
    COUNT(*) FILTER (WHERE x.logdate > p.cutoff - 7)                AS active_days_7d,
    COUNT(*) FILTER (WHERE x.logdate >  p.cutoff - 60
                       AND x.logdate <= p.cutoff - 30)              AS active_days_prev30d,

    SUM(x.secs)                                                     AS secs_90d,
    SUM(x.secs) FILTER (WHERE x.logdate > p.cutoff - 30)            AS secs_30d,
    SUM(x.secs) FILTER (WHERE x.logdate > p.cutoff - 7)             AS secs_7d,
    SUM(x.secs) FILTER (WHERE x.logdate >  p.cutoff - 60
                          AND x.logdate <= p.cutoff - 30)           AS secs_prev30d,

    SUM(x.plays)                                                    AS plays_90d,
    SUM(x.plays) FILTER (WHERE x.logdate > p.cutoff - 30)           AS plays_30d,

    SUM(x.num25 + x.num50)::float   / NULLIF(SUM(x.plays), 0)       AS skip_rate_90d,
    SUM(x.num985 + x.num100)::float / NULLIF(SUM(x.plays), 0)       AS completion_rate_90d,
    SUM(x.numunq)::float            / NULLIF(SUM(x.plays), 0)       AS repeat_ratio_90d,
    SUM(x.secs)                     / NULLIF(SUM(x.plays), 0)       AS secs_per_play_90d,

    MAX(x.secs)                                                     AS max_secs_day_90d,
    STDDEV(x.secs)                                                  AS std_secs_90d,
    (p.cutoff - MAX(x.logdate))                                     AS days_since_last_log
  FROM (
    SELECT u.msno, u.logdate, u.num25, u.num50, u.num985, u.num100, u.numunq,
           LEAST(GREATEST(u.totalsecs, 0), 86400)                   AS secs,
           (u.num25 + u.num50 + u.num75 + u.num985 + u.num100)      AS plays
    FROM userlogs u
    CROSS JOIN params p
    WHERE u.logdate <= p.cutoff
      AND u.logdate >  p.cutoff - 90
  ) x
  CROSS JOIN params p
  GROUP BY x.msno, p.cutoff
),

-- 2) transaction history up to the cutoff
trans_agg AS (
  SELECT
    t.msno,
    COUNT(*)                                                        AS n_trans,
    COUNT(*) FILTER (WHERE t.transactiondate > p.cutoff - 90)       AS n_trans_90d,
    COUNT(*) FILTER (WHERE t.iscancel)                              AS n_cancel,
    COUNT(*) FILTER (WHERE NOT t.isautorenew)                       AS n_no_autorenew,
    COUNT(DISTINCT t.paymentmethodid)                               AS n_payment_methods,
    COUNT(DISTINCT t.paymentplandays)                               AS n_distinct_plans,
    COUNT(*) FILTER (WHERE t.actualamountpaid < t.planlistprice)    AS n_discounted,
    SUM(t.actualamountpaid)::float                                  AS total_paid,
    AVG(t.actualamountpaid)::float                                  AS avg_paid,
    (p.cutoff - MIN(t.transactiondate))                             AS days_since_first_trans
  FROM transactions t
  CROSS JOIN params p
  WHERE t.transactiondate <= p.cutoff
  GROUP BY t.msno, p.cutoff
),

-- 3) most recent transaction state
last_trans AS (
  SELECT DISTINCT ON (t.msno)
    t.msno,
    t.paymentmethodid                                               AS last_payment_method,
    t.paymentplandays                                               AS last_plan_days,
    t.planlistprice::float                                          AS last_list_price,
    t.actualamountpaid::float                                       AS last_amount_paid,
    t.isautorenew                                                   AS last_is_autorenew,
    t.iscancel                                                      AS last_is_cancel,
    (t.membershipexpiredate - p.cutoff)                             AS days_to_expiry
  FROM transactions t
  CROSS JOIN params p
  WHERE t.transactiondate <= p.cutoff
  ORDER BY t.msno, t.transactiondate DESC, t.membershipexpiredate DESC
)

-- 4) final assembly, starting from the label table
SELECT
  l.msno,
  l.ischurn::int                                                   AS label,

  m.city,
  m.registeredvia,
  m.gender,
  CASE WHEN m.bd BETWEEN 10 AND 90 THEN m.bd END                   AS age,
  CASE WHEN m.registrationinitdate <= p.cutoff
       THEN p.cutoff - m.registrationinitdate END                  AS days_since_registration,
  (m.msno IS NOT NULL)                                             AS has_member_row,

  ta.n_trans,
  ta.n_trans_90d,
  ta.n_cancel,
  ta.n_no_autorenew,
  ta.n_payment_methods, ta.n_distinct_plans, ta.n_discounted,
  ta.total_paid, ta.avg_paid, ta.days_since_first_trans,
  lt.last_payment_method, lt.last_plan_days, lt.last_list_price,
  lt.last_amount_paid, lt.last_is_autorenew, lt.last_is_cancel, lt.days_to_expiry,

  (lg.msno IS NOT NULL)                                            AS has_logs,
  COALESCE(lg.active_days_7d, 0)                                   AS active_days_7d,
  COALESCE(lg.active_days_30d, 0)                                  AS active_days_30d,
  COALESCE(lg.active_days_90d, 0)                                  AS active_days_90d,
  COALESCE(lg.active_days_prev30d, 0)                              AS active_days_prev30d,
  COALESCE(lg.secs_7d, 0)                                          AS secs_7d,
  COALESCE(lg.secs_30d, 0)                                         AS secs_30d,
  COALESCE(lg.secs_90d, 0)                                         AS secs_90d,
  COALESCE(lg.secs_prev30d, 0)                                     AS secs_prev30d,
  COALESCE(lg.plays_30d, 0)                                        AS plays_30d,
  COALESCE(lg.plays_90d, 0)                                        AS plays_90d,
  lg.skip_rate_90d, lg.completion_rate_90d, lg.repeat_ratio_90d,
  lg.secs_per_play_90d, lg.max_secs_day_90d, lg.std_secs_90d,
  lg.days_since_last_log,

  (lg.secs_7d  / 7.0)  / NULLIF(lg.secs_30d / 30.0, 0)             AS secs_trend_7d_vs_30d,
  (lg.secs_30d / 30.0) / NULLIF(lg.secs_90d / 90.0, 0)             AS secs_trend_30d_vs_90d,
  lg.secs_30d / NULLIF(lg.secs_prev30d, 0)                         AS secs_trend_30d_vs_prev30d,
  COALESCE(lg.active_days_30d, 0) - COALESCE(lg.active_days_prev30d, 0) AS active_days_delta
FROM labels l
CROSS JOIN params p
LEFT JOIN raw.members m  USING (msno)
JOIN      trans_agg   ta USING (msno)      -- inner join: drops users with no pre-cutoff history
LEFT JOIN last_trans  lt USING (msno)
LEFT JOIN logs        lg USING (msno)
WHERE lt.days_to_expiry BETWEEN 1 AND (p.win_end - p.cutoff);


ALTER TABLE train_features ADD PRIMARY KEY (msno);
ANALYZE train_features;

COMMIT;