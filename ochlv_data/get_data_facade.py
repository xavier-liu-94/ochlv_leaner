import duckdb
import pandas as pd

con = duckdb.connect()


def get_data_by_sql(sql: str) -> pd.DataFrame:
    real_sql = sql.replace(
        "#ochlv_table#", 
        "read_parquet('ochlv_data/A/*.parquet')"
    )
    return con.sql(real_sql).df()


def get_data_by_sql_to_file(sql: str, file_path: str):
    real_sql = sql.replace(
        "#ochlv_table#", 
        "read_parquet('ochlv_data/A/*.parquet')"
    )
    return con.sql(real_sql).write_parquet(file_path)

def get_data_from_path(sql: str, file_path: str):
    real_sql = sql.replace(
        "#table#", 
        f"read_parquet('{file_path}')"
    )
    return con.sql(real_sql).df()