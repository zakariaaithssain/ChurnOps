-- run before building features, per round

SELECT 'labels' AS tbl, COUNT(*) AS n, NULL::date AS min_d, NULL::date AS max_d FROM labels
UNION ALL SELECT 'transactions', COUNT(*), MIN(transactiondate), MAX(transactiondate) FROM transactions
UNION ALL SELECT 'userlogs',     COUNT(*), MIN(logdate),         MAX(logdate)         FROM userlogs;

SELECT COUNT(*) AS labeled_users,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM transactions t WHERE t.msno = l.msno)) AS with_trans,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM userlogs u    WHERE u.msno = l.msno)) AS with_logs,
       COUNT(*) FILTER (WHERE EXISTS (SELECT 1 FROM public.members m WHERE m.msno = l.msno)) AS with_member,
       AVG(ischurn::int) AS churn_rate
FROM labels l;