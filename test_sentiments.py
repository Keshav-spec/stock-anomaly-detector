import sqlite3
import pandas as pd

from sentiments.news_fetcher import fetch_news
from sentiments.finbert_sentiments import get_sentiment_score

# Load stock data
conn = sqlite3.connect("data/stocks.db")

df = pd.read_sql(
    "SELECT * FROM stock_prices WHERE ticker='RELIANCE.NS'",
    conn
)

conn.close()

df["Date"] = pd.to_datetime(df["Date"])

# =================================
# LIMIT DATA FOR TESTING
# =================================

df = df.tail(10)

# =================================
# Sentiment scoring
# =================================

sentiments = []

for d in df["Date"]:

    date_str = d.strftime("%Y-%m-%d")

    print(f"\nFetching news for {date_str}")

    news = fetch_news(
        "Reliance Industries",
        date_str,
        date_str
    )

    print(news[:2])

    score = get_sentiment_score(news)

    print(f"Sentiment Score: {score}")

    sentiments.append(score)

df["sentiment"] = sentiments

print("\nFinal DataFrame:\n")

print(
    df[
        ["Date", "Close", "sentiment"]
    ].head()
)