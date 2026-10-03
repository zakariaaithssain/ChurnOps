-- run per round, set search_path to round, raw

BEGIN;

ANALYZE labels;

-- labeled users only, nothing after the cutoff date
CREATE TABLE transactions AS
SELECT *
FROM raw.transactions_all
WHERE transactiondate <= %(cutoff)s::date
  AND msno IN (SELECT msno FROM labels);

CREATE TABLE userlogs AS
SELECT *
FROM raw.userlogs
WHERE logdate <= %(cutoff)s::date
  AND logdate >  %(cutoff)s::date - %(lookback_days)s
  AND msno IN (SELECT msno FROM labels);

ANALYZE transactions;
ANALYZE userlogs;

COMMIT;