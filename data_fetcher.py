import yfinance as yf
import pandas as pd
import os
import sqlite3

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "WIPRO.NS"]
DB_PATH = "data/stocks.db"

def fetch_historical(ticker, period="2y", interval="1d"):
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True
    )

    # Flatten multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.reset_index(inplace=True)

    df["ticker"] = ticker

    print(df.columns)

    return df

def save_to_db(df, table="stock_prices"):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()

for t in TICKERS:
    print(f"Fetching {t}...")
    df = fetch_historical(t)
    df.to_csv(f"data/raw/{t.replace('.','_')}.csv", index=False)
    save_to_db(df)
    print(f"  Saved {t}: {len(df)} rows to CSV and DB")

print("Done. Database saved at data/stocks.db")