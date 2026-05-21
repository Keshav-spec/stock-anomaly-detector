import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from features import (
    add_price_features,
    add_rolling_features
)

# Connect to database
conn = sqlite3.connect("data/stocks.db")

# Load ticker data
df = pd.read_sql(
    "SELECT * FROM stock_prices WHERE ticker='RELIANCE.NS'",
    conn
)

conn.close()

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"])

# Apply feature engineering
df = add_price_features(df)

df = add_rolling_features(df)

# Remove NaN rows
df = df.dropna().reset_index(drop=True)

# ======================
# Plotting
# ======================

fig, axes = plt.subplots(
    3,
    1,
    figsize=(14, 10),
    sharex=True
)

# Close price
axes[0].plot(
    df["Date"],
    df["Close"],
    linewidth=0.8
)

axes[0].set_title("Close Price")

# RSI
axes[1].plot(
    df["Date"],
    df["rsi"],
    linewidth=0.8,
    color="orange"
)

axes[1].axhline(
    70,
    color="red",
    linestyle="--",
    linewidth=0.5
)

axes[1].axhline(
    30,
    color="green",
    linestyle="--",
    linewidth=0.5
)

axes[1].set_title("RSI")

# Volume ratio
axes[2].bar(
    df["Date"],
    df["vol_ratio"],
    width=1,
    color="steelblue",
    alpha=0.6
)

axes[2].set_title("Volume Ratio (vs 20-day avg)")

plt.tight_layout()

plt.savefig(
    "outputs/phase3_sanity_check.png",
    dpi=150
)

plt.show()