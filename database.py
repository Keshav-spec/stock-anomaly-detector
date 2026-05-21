import sqlite3
import pandas as pd
import os

os.makedirs("data", exist_ok=True)
DB_PATH = "data/stocks.db"

def save_to_db(df, table="stock_prices"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

def load_from_db(ticker, table="stock_prices"):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"SELECT * FROM {table} WHERE ticker='{ticker}'", conn)
    conn.close()
    return df