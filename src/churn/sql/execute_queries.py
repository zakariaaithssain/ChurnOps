import argparse
import os
from datetime import date
from pathlib import Path

import psycopg
from psycopg import sql

DSN = os.getenv("CHURN_DSN", "postgresql://kkbox:postgrespass@localhost:5432/kkbox_data")

SQL_DIR = Path(__file__).parent
#sql files names
CREATE_SCHEMAS = "create_schemas.sql"
LOAD_ROUND = "load_round.sql"
RESET = "reset.sql"
BUILD_FEATURES = "build_features.sql"

DATA = Path("data/raw")
CHUNK = 16 * 1024 * 1024

# lookback_days must be >= the longest window used in build_features.sql (90 days).
ROUNDS = {
    "r1": {
        "cutoff": "2017-01-31",
        "lookback_days": 90,
        "labels": "train.csv",
        "win_end" : "2017-02-28"
        },
    "r2": {
        "cutoff": "2017-02-28",
        "lookback_days": 90,
        "labels": "test/churn_comp_refresh/train_v2.csv",
        "win_end" : "2017-03-31"
        },
}

# COPY maps columns by position: these lists must match the CSV column order.
COLS = {
    "labels":       ["msno", "ischurn"],
    "members":      ["msno", "city", "bd", "gender", "registeredvia", "registrationinitdate"],
    "transactions": ["msno", "paymentmethodid", "paymentplandays", "planlistprice",
                     "actualamountpaid", "isautorenew", "transactiondate",
                     "membershipexpiredate", "iscancel"],
    "userlogs":     ["msno", "logdate", "num25", "num50", "num75",
                     "num985", "num100", "numunq", "totalsecs"],
}

# userlogs last (huge)
RAW_LOADS = [
    ("raw.members",         "members",      "members_v3.csv"),
    ("raw.transactions_v2", "transactions", "test/churn_comp_refresh/transactions_v2.csv"),
    ("raw.transactions_v1", "transactions", "transactions.csv"),
    ("raw.userlogs",        "userlogs",     "user_logs.csv"),
]

RANGES = """
SELECT 'labels' AS tbl, COUNT(*) AS n, NULL::date AS min_d, NULL::date AS max_d FROM labels
UNION ALL SELECT 'transactions', COUNT(*), MIN(transactiondate), MAX(transactiondate) FROM transactions
UNION ALL SELECT 'userlogs',     COUNT(*), MIN(logdate),         MAX(logdate)         FROM userlogs
"""
COVERAGE = """
SELECT COUNT(*)           AS labeled_users,
       COUNT(t.msno)      AS with_trans,
       COUNT(u.msno)      AS with_logs,
       COUNT(m.msno)      AS with_member,
       AVG(l.ischurn::int) AS churn_rate
FROM labels l
LEFT JOIN (SELECT DISTINCT msno FROM transactions) t USING (msno)
LEFT JOIN (SELECT DISTINCT msno FROM userlogs)     u USING (msno)
LEFT JOIN raw.members m USING (msno)
"""



def connect(*schemas, **kwargs):
    conn = psycopg.connect(DSN, autocommit=True, **kwargs)
    path = sql.SQL(", ").join(map(sql.Identifier, schemas))
    # switch to schema
    conn.execute(sql.SQL("SET search_path TO {}").format(path))
    conn.execute("SET synchronous_commit = off")
    return conn


def run_sql_file(conn, name, params=None):
    conn.execute((SQL_DIR / name).read_text(), params)


def copy_csv(conn, table, cols, csv_path):
    q = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
        sql.Identifier(*table.split(".")),
        sql.SQL(", ").join(map(sql.Identifier, cols)))
    with conn.cursor() as cur:
        with open(csv_path, "rb") as f, cur.copy(q) as cp:
            while chunk := f.read(CHUNK):
                cp.write(chunk)
        return cur.rowcount



def init():
    with connect("public") as c:
        run_sql_file(c, CREATE_SCHEMAS)


def load_raw():
    with connect("raw") as c:
        for table, key, fname in RAW_LOADS:
            ident = sql.Identifier(*table.split("."))
            c.execute(sql.SQL("TRUNCATE {}").format(ident))      # makes retries safe
            n = copy_csv(c, table, COLS[key], DATA / fname)
            c.execute(sql.SQL("ANALYZE {}").format(ident))
            print(f"{table}: {n:,} rows  (compare with: wc -l {fname}, minus 1)")


def check_round(c, cutoff: date):
    for q in (RANGES, COVERAGE):
        cur = c.execute(q)
        print([d.name for d in cur.description])
        for row in cur.fetchall():
            print(row)
    assert c.execute("SELECT MAX(transactiondate) FROM transactions").fetchone()[0] <= cutoff
    assert c.execute("SELECT MAX(logdate) FROM userlogs").fetchone()[0] <= cutoff


def build_round(schema):
    config = ROUNDS[schema]
    with connect(schema, "raw") as c:
        c.execute("TRUNCATE labels")        
        n = copy_csv(c, "labels", COLS["labels"], DATA / config["labels"])
        print(f"{schema}.labels: {n:,} rows")
    # client-side binding: the build SQL has parameters and several statements
    with connect(schema, "raw", cursor_factory=psycopg.ClientCursor) as c:
        c.execute("SET work_mem = '512MB'")
        run_sql_file(c, LOAD_ROUND,
                     {"cutoff": config["cutoff"], "lookback_days": config["lookback_days"]})
        check_round(c, date.fromisoformat(config["cutoff"]))



def build_features(schema):
    config = ROUNDS[schema]
    with connect(schema, "raw", cursor_factory=psycopg.ClientCursor) as c:
        c.execute("SET work_mem = '512MB'")
        run_sql_file(c, BUILD_FEATURES, {"cutoff": config["cutoff"], "win_end" : config["win_end"]})


def reset():
    if input("Drops schemas raw, r1, r2 and all their data. Type 'yes': ") != "yes":
        return
    with connect("public") as c:
        run_sql_file(c, RESET)



if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("reset", "init", "load-raw"):
        sub.add_parser(name)
    sub.add_parser("round").add_argument("schema", choices=ROUNDS)
    sub.add_parser("build-features").add_argument("schema", choices=ROUNDS)
    a = p.parse_args()
    if a.cmd == "reset":
        reset()
    elif a.cmd == "init":
        init()
    elif a.cmd == "load-raw":
        load_raw()
    elif a.cmd == "round":
        build_round(a.schema)
    elif a.cmd == "build-features":
        build_features(a.schema)