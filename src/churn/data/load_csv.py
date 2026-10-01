import psycopg


CSV_TABLE_MAP = {"Labels": "data/train.csv",
                 "Members": "data/members_v3.csv", 
                 "Transactions": "data/transactions.csv", 
                 "UserLogs": "data/user_logs.csv", 
                 }


import psycopg
from psycopg import sql

CSV_TABLE_MAP = {
    "Members": "data/members_v3.csv", 
    "Transactions": "data/transactions.csv", 
    "UserLogs": "data/user_logs.csv", 
    "Labels": "data/train.csv"
}

with psycopg.connect(
    host="localhost", 
    port="5432", 
    dbname="kkbox-data", 
    user="kkbox", 
    password="postgrespass"
) as connection: 
    with connection.cursor() as cursor: 
        for table, file_path in CSV_TABLE_MAP.items(): 
            try: 
                query = sql.SQL(
                    "COPY {table} FROM STDIN WITH (FORMAT CSV, HEADER true, DELIMITER ',')"
                ).format(table=sql.Identifier(table.lower()))
                
                with open(file_path, "r", encoding="utf-8") as f:
                    with cursor.copy(query) as copy:
                        while data := f.read(1024 * 1024):  # 1MB to memo
                            copy.write(data)
                            
                print(f"loaded table: {table}")
                
            except Exception as e: 
                print(f"table {table} - file {file_path} - error: {e}")
                connection.rollback()
