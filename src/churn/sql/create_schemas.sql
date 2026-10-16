-- Schemas must match config.SCHEMAS (r1, r2) plus config.RAW_SCHEMA (raw).
-- Keep in sync with reset.sql and data.execute_queries.reset().
-- CREATE SCHEMA uses IF NOT EXISTS (cheap, safe to rerun).
-- CREATE TABLE has no IF NOT EXISTS on purpose: reruns fail fast so
-- raw data is never silently wiped. Only train_features may be dropped
-- (see build_features.sql).
BEGIN;


-- raw: shared source tables, loaded once via load_raw()
CREATE SCHEMA IF NOT EXISTS raw;
-- members_v3.csv: one file shared by both rounds (data/raw/members_v3.csv)
CREATE TABLE raw.members (
    msno                  CHAR(44) PRIMARY KEY,
    city                  INT,
    bd                    INT,           -- raw cleaned in the feature SQL
    gender                VARCHAR(10),
    registeredvia         INT,
    registrationinitdate  DATE
);

-- data/raw/transactions.csv
CREATE TABLE raw.transactions_v1 (
    msno                  CHAR(44) NOT NULL,
    paymentmethodid       INT,
    paymentplandays       INT,
    planlistprice         NUMERIC(10,2),
    actualamountpaid      NUMERIC(10,2),
    isautorenew           BOOLEAN,
    transactiondate       DATE,
    membershipexpiredate  DATE,
    iscancel              BOOLEAN
);
-- data/raw/test/churn_comp_refresh/transactions_v2.csv, same shape as v1
CREATE TABLE raw.transactions_v2 (LIKE raw.transactions_v1);


-- data/raw/user_logs.csv, shared by both rounds (cutoff-filtered per round).
-- user_logs_v2.csv is intentionally unused and deleted.
CREATE UNLOGGED TABLE raw.userlogs (
    msno       CHAR(44) NOT NULL,
    logdate    DATE,
    num25      INT,
    num50      INT,
    num75      INT,
    num985     INT,
    num100     INT,
    numunq     INT,
    totalsecs  DOUBLE PRECISION
);


-- union of both transaction dumps, minus exact duplicates already in v1
CREATE VIEW raw.transactions_all AS
SELECT * FROM raw.transactions_v1
UNION ALL
(SELECT * FROM raw.transactions_v2
 EXCEPT ALL
 SELECT * FROM raw.transactions_v1);


--round 1 (config ROUNDS["r1"])
CREATE SCHEMA IF NOT EXISTS r1;
CREATE TABLE r1.labels (
    msno     CHAR(44) PRIMARY KEY,
    ischurn  BOOLEAN NOT NULL
);

--round 2 (config ROUNDS["r2"])
CREATE SCHEMA IF NOT EXISTS r2;
CREATE TABLE r2.labels (LIKE r1.labels INCLUDING ALL);

COMMIT;
