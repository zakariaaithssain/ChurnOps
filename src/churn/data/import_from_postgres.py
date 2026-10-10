import pandas as pd
import psycopg
import os
import argparse

from typing import Literal
from pathlib import Path

from churn.config import DATA_DSN, SCHEMAS



def import_from_postgres(schema: Literal['r1', 'r2']):
    with psycopg.connect(DATA_DSN) as conn:
        cursor = conn.execute(f"SELECT * FROM {schema}.train_features")
        cols = [d.name for d in cursor.description]
        save_path = Path(f"data/{schema}_features.parquet")
        pd.DataFrame(cursor.fetchall(), columns=cols).to_parquet(save_path)
        return save_path



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=f"import features table from Postgres schemas {SCHEMAS}.")
    parser.add_argument("schema", choices=SCHEMAS, help="target schema to import.")
    args = parser.parse_args()

    print(f"importing from schema: {args.schema}")
    filepath = import_from_postgres(args.schema)
    print(f"saved to : {filepath}")

       
