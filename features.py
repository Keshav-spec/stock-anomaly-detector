import numpy as np

def add_price_features(df):
    df["daily_return"] = df["Close"].pct_change()
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["price_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    return df

def add_rolling_features(df):
    for w in [5, 10, 20, 50]:
        df[f"ma_{w}"] = df["Close"].rolling(w).mean()
        df[f"std_{w}"] = df["daily_return"].rolling(w).std()

    # Bollinger Bands (20-day)
    df["bb_upper"] = df["ma_20"] + 2 * df["std_20"]
    df["bb_lower"] = df["ma_20"] - 2 * df["std_20"]
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["ma_20"]

    # RSI (14-day)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss))

    # Volume spike
    df["vol_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    return df

FEATURES = [
    "daily_return", "log_return", "price_range", "gap",
    "std_5", "std_20", "bb_width", "rsi", "vol_ratio"
]

