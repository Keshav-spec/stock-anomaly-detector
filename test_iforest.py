import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

from features import (
    add_price_features,
    add_rolling_features,
    FEATURES
)

from models.isolation_forest import (
    train_isolation_forest,
    predict_anomalies
)

# ======================
# Load data
# ======================

conn = sqlite3.connect("data/stocks.db")

df = pd.read_sql(
    "SELECT * FROM stock_prices WHERE ticker='RELIANCE.NS'",
    conn
)

conn.close()

df["Date"] = pd.to_datetime(df["Date"])

# ======================
# Feature engineering
# ======================

df = add_price_features(df)

df = add_rolling_features(df)

df = df.dropna().reset_index(drop=True)

# ======================
# Train Isolation Forest
# ======================

X = df[FEATURES]

model, scaler = train_isolation_forest(
    X,
    contamination=0.02
)

preds, scores = predict_anomalies(
    X,
    model,
    scaler
)

# ======================
# Save predictions
# ======================

df["anomaly"] = preds
df["anomaly_score"] = scores

# ======================
# Plot anomalies
# ======================

anomalies = df[df["anomaly"] == -1]

plt.figure(figsize=(14,5))

plt.plot(
    df["Date"],
    df["Close"],
    linewidth=0.8,
    label="Price"
)

plt.scatter(
    anomalies["Date"],
    anomalies["Close"],
    color="red",
    s=30,
    zorder=5,
    label="Anomaly"
)

plt.title("Anomaly Detection — Isolation Forest")

plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/anomalies_iforest.png",
    dpi=150
)

plt.show()