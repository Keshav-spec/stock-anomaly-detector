import sqlite3
import pandas as pd


DB_PATH = "data/stocks.db"


def load_from_db(ticker):

    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql(
        f"""
        SELECT * FROM stock_prices
        WHERE ticker='{ticker}'
        """,
        conn
    )

    conn.close()

    df["Date"] = pd.to_datetime(df["Date"])

    return df