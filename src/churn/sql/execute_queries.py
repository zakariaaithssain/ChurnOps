from pathlib import Path
import psycopg
from psycopg import sql


# TODO: move hardcoded creds to external injection
DSN = "postgresql://kkbox:postgrespass@localhost:5432/kkbox-data"
DATA = Path("data")
CHUNK = 16 * 1024 * 1024 # (16MB)

# COPY maps columns by position, so the explicit lists must match the CSV column order.
COLS = {
    "labels":       ["msno", "ischurn"],
    "transactions": ["msno", "paymentmethodid", "paymentplandays", "planlistprice",
                     "actualamountpaid", "isautorenew", "transactiondate",
                     "membershipexpiredate", "iscancel"],
    "userlogs_raw": ["msno", "logdate", "num25", "num50", "num75",
                     "num985", "num100", "numunq", "totalsecs"],
    "members":      ["msno", "city", "bd", "gender", "registeredvia", "registrationinitdate"],
}

FILES = {
    "r1": {"labels": "train.csv", "transactions": "transactions.csv", "userlogs_raw": "user_logs.csv"},
    "r2": {"labels": "test/churn_comp_refresh/train_v2.csv",
           "transactions": "test/churn_comp_refresh/transactions_v2.csv",
           "userlogs_raw": "test/churn_comp_refresh/user_logs_v2.csv"},
}

def connect(*schemas):
    conn = psycopg.connect(DSN, autocommit=True)   # one transaction per statement/COPY
    #change the schema
    conn.execute(sql.SQL("SET search_path TO {}").format(
        sql.SQL(", ").join(map(sql.Identifier, schemas))))
    conn.execute("SET synchronous_commit = off")
    return conn

def run_sql_file(conn, filename):
    path : Path = Path("src/churn/sql") / filename
    conn.execute(path.read_text())

def copy_csv(conn, table, csv_path):
    cols = sql.SQL(", ").join(map(sql.Identifier, COLS[table]))
    conn.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(table)))   # makes retries safe
    q = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, HEADER true)").format(
        sql.Identifier(table), cols)
    with open(csv_path, "rb") as f, conn.cursor().copy(q) as cp:
        while chunk := f.read(CHUNK):
            cp.write(chunk)

if __name__ == "__main__":
    with connect("public") as c:
        run_sql_file(c, "setup.sql")
        copy_csv(c, "members", DATA / "members_v3.csv")
        c.execute("ANALYZE public.members")

    for schema, files in FILES.items():
        with connect(schema) as c:                  # schema ONLY, so DROP can't hit public
            run_sql_file(c, "create_tables.sql")
            for table, fname in files.items():
                copy_csv(c, table, DATA / fname)
        with connect(schema, "public") as c:
            run_sql_file(c, "sql_checks.sql")