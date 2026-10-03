BEGIN;


-- raw
CREATE SCHEMA raw;
-- members_v3.csv: one file shared by both rounds
CREATE TABLE raw.members (
    msno                  CHAR(44) PRIMARY KEY,
    city                  INT,
    bd                    INT,           -- raw cleaned in the feature SQL
    gender                VARCHAR(10),
    registeredvia         INT,
    registrationinitdate  DATE
);

CREATE TABLE raw.transactions_v1 (       -- transactions.csv
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
CREATE TABLE raw.transactions_v2 (LIKE raw.transactions_v1);   -- transactions_v2.csv


 --to be dropped after
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


CREATE VIEW raw.transactions_all AS
SELECT * FROM raw.transactions_v1
UNION ALL
(SELECT * FROM raw.transactions_v2
 EXCEPT ALL
 SELECT * FROM raw.transactions_v1);


--round 1
 CREATE SCHEMA r1;
CREATE TABLE r1.labels (
    msno     CHAR(44) PRIMARY KEY,
    ischurn  BOOLEAN NOT NULL
);

--round 2
CREATE SCHEMA r2;
CREATE TABLE r2.labels (LIKE r1.labels INCLUDING ALL);

COMMIT;