-- Build one round's working tables. Run with search_path set to (round, raw)
-- so unqualified names (labels, transactions, userlogs) resolve to the round
-- schema while raw.* stays explicit. Params: %(cutoff)s, %(lookback_days)s
-- from config.ROUNDS. CREATE TABLE has no DROP IF EXISTS on purpose:
-- reruns fail fast instead of silently wiping round data.

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