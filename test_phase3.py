import sqlite3
import pandas as pd

from features import (
    add_price_features,
    add_rolling_features,
    FEATURES
)

# Connect to DB
conn = sqlite3.connect("data/stocks.db")

# Load one ticker
df = pd.read_sql(
    "SELECT * FROM stock_prices WHERE ticker='RELIANCE.NS'",
    conn
)

conn.close()

# Apply feature engineering
df = add_price_features(df)

df = add_rolling_features(df)

# Remove NaN rows created by rolling windows
df = df.dropna().reset_index(drop=True)

# Checks
print("\nShape:")
print(df.shape)

print("\nFeature Statistics:")
print(df[FEATURES].describe())

print("\nNull Values:")
print(df[FEATURES].isnull().sum())