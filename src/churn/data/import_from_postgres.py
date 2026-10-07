import pandas as pd
import psycopg
import os

from typing import Literal
from pathlib import Path

#TODO: remove hardcoded schema names across files to some config file for all project scope 

Path("data").mkdir(exist_ok=True)
DSN = os.getenv("CHURN_DSN", "postgresql://kkbox:postgrespass@localhost:5432/kkbox-data")
SCHEMAS = ['r1', 'r2']


def import_from_postgres(schema: Literal['r1', 'r2']):
    with psycopg.connect(DSN) as conn:
        cursor = conn.execute(f"SELECT * FROM {schema}.train_features")
        cols = [d.name for d in cursor.description]
        save_path = Path(f"data/{schema}_features.parquet")
        pd.DataFrame(cursor.fetchall(), columns=cols).to_parquet(save_path)
        return save_path



if __name__ == "__main__": 
    for schema in SCHEMAS: 
        print(f"importing from schema: {schema}")
        filepath = import_from_postgres(schema)
        print(f"saved to : {filepath}")

       
