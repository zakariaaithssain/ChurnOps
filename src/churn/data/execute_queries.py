import argparse
import psycopg

from datetime import date
from pathlib import Path
from psycopg import sql

from churn.config import (DATA_DSN, CHUNK, WORK_MEM, SYNCHRONOUS_COMMIT,
    CREATE_SCHEMAS, RAW_LOADS, RAW_SCHEMA, SCHEMAS,
    COLS, RAW_DATA, RANGES, COVERAGE, ROUNDS, LOAD_ROUND, BUILD_FEATURES, RESET
)


SQL_DIR = Path(__file__).resolve().parent.parent / "sql"



def connect(*schemas, **kwargs):
    conn = psycopg.connect(DATA_DSN, autocommit=True, **kwargs)
    path = sql.SQL(", ").join(map(sql.Identifier, schemas))
    # switch to schema; callers pass (round, RAW_SCHEMA) so that
    # unqualified names resolve to the round schema and raw.* stays explicit.
    conn.execute(sql.SQL("SET search_path TO {}").format(path))
    conn.execute(sql.SQL("SET synchronous_commit TO {}").format(sql.Literal(SYNCHRONOUS_COMMIT)))
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
    with connect(RAW_SCHEMA) as c:
        for table, key, fname in RAW_LOADS:
            ident = sql.Identifier(*table.split("."))
            c.execute(sql.SQL("TRUNCATE {}").format(ident))      # makes retries safe
            n = copy_csv(c, table, COLS[key], RAW_DATA / fname)
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
    with connect(schema, RAW_SCHEMA) as c:
        c.execute("TRUNCATE labels")        
        n = copy_csv(c, "labels", COLS["labels"], RAW_DATA / config["labels"])
        print(f"{schema}.labels: {n:,} rows")
    # client-side binding: the build SQL has parameters and several statements.
    # load_round.sql uses CREATE TABLE without DROP: reruns fail fast on purpose
    # so raw/round tables are never silently wiped (only train_features may be dropped).
    with connect(schema, RAW_SCHEMA, cursor_factory=psycopg.ClientCursor) as c:
        c.execute(sql.SQL("SET work_mem TO {}").format(sql.Literal(WORK_MEM)))
        run_sql_file(c, LOAD_ROUND,
                     {"cutoff": config["cutoff"], "lookback_days": config["lookback_days"]})
        check_round(c, date.fromisoformat(config["cutoff"]))



def build_features(schema):
    config = ROUNDS[schema]
    with connect(schema, RAW_SCHEMA, cursor_factory=psycopg.ClientCursor) as c:
        c.execute(sql.SQL("SET work_mem TO {}").format(sql.Literal(WORK_MEM)))
        run_sql_file(c, BUILD_FEATURES, {"cutoff": config["cutoff"], "win_end" : config["win_end"]})


def reset():
    schemas = ", ".join([RAW_SCHEMA, *SCHEMAS])
    if input(f"Drops schemas {schemas} and all their data. Type 'yes': ") != "yes":
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