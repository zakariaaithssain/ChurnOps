import psycopg
from pathlib import Path


QUERIES_FILE = Path("src/churn/data/create_tables.sql")
with open(QUERIES_FILE, "r") as file: 
    content = file.read()
    #psycopg cursors can't execute multiple comma separated queries
    queries = content.split(";")


with psycopg.connect(
    host = "localhost",
    dbname = "kkbox-data",
    port = "5432",
    user = "kkbox",
    password = "postgrespass"               
                ) as connection: 
    
    with connection.cursor() as cursor: 
    
        for idx, query in enumerate(queries): 
            if query.strip(): 
                try:
                    cursor.execute(query)
                except Exception as e:
                    print(f"failed to execute query {idx} - error: {e}")
