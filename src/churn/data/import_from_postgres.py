import pandas as pd
import psycopg
import os

from typing import Literal
from pathlib import Path



Path("data").mkdir(exist_ok=True)
DSN = os.getenv("CHURN_DSN", "postgresql://kkbox:postgrespass@localhost:5432/kkbox-data")



def import_from_postgres(schema: Literal['r1', 'r2']):
    with psycopg.connect(DSN) as conn:
        cursor = conn.execute(f"SELECT * FROM {schema}.train_features")
        cols = [d.name for d in cursor.description]
        pd.DataFrame(cursor.fetchall(), columns=cols).to_parquet(f"data/{schema}_features.parquet")


