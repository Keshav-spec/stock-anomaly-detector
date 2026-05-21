import sqlite3
import pandas as pd

conn = sqlite3.connect("data/stocks.db")
df = pd.read_sql("SELECT * FROM stock_prices LIMIT 5", conn)
conn.close()

print(df.head())               # should show Date, Open, High, Low, Close, Volume, ticker
print("Shape:", df.shape)      # should have many rows, 7-8 columns
print("Tickers:", df["ticker"].unique())  # all 4 tickers should appear
print("Nulls:\n", df.isnull().sum())      # should all be 0