-- once per schema (round)
DROP TABLE IF EXISTS labels, transactions, userlogs_raw, userlogs CASCADE;

CREATE TABLE labels (
    msno     CHAR(44) PRIMARY KEY,
    ischurn  BOOLEAN NOT NULL
);

CREATE TABLE transactions (
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


-- not users are needed (because not all of them are present in labels)
CREATE UNLOGGED TABLE userlogs_raw (
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


-- post-load

ANALYZE labels;

-- Labeled users only.
CREATE TABLE userlogs AS
SELECT u.*
FROM userlogs_raw u
JOIN labels l USING (msno);

SET maintenance_work_mem = '2GB';
SET max_parallel_maintenance_workers = 4;

CREATE INDEX transactions_msno_date_idx ON transactions (msno, transactiondate);
CREATE INDEX userlogs_msno_date_idx      ON userlogs     (msno, logdate);

ANALYZE transactions;
ANALYZE userlogs;