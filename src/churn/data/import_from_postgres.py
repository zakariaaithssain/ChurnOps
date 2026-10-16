import pandas as pd
import psycopg
import argparse

from typing import Literal
from psycopg import sql

from churn.config import DATA_DSN, SCHEMAS, FEATURES



def import_from_postgres(schema: Literal['r1', 'r2']):
    save_path = FEATURES[schema]
    with psycopg.connect(DATA_DSN) as conn:
        q = sql.SQL("SELECT * FROM {}.train_features").format(sql.Identifier(schema))
        cursor = conn.execute(q)
        cols = [d.name for d in cursor.description]
        save_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cursor.fetchall(), columns=cols).to_parquet(save_path)
        return save_path



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=f"import features table from Postgres schemas {SCHEMAS}.")
    parser.add_argument("schema", choices=SCHEMAS, help="target schema to import.")
    args = parser.parse_args()

    print(f"importing from schema: {args.schema}")
    filepath = import_from_postgres(args.schema)
    print(f"saved to : {filepath}")
