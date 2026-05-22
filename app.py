import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import timedelta
from utils.db_loader import load_from_db
from utils.model_runner import (
    process_features,
    run_iforest,
    run_lstm
)

from sentiments.news_fetcher import fetch_news
from sentiments.finbert_sentiments import get_sentiment_score


# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Stock Anomaly Detector",
    layout="wide"
)


# ======================================================
# TITLE
# ======================================================

st.title("📈 Real-Time Stock Anomaly Detector")


# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.header("Controls")

ticker = st.sidebar.selectbox(
    "Select Stock",
    [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "WIPRO.NS"
    ]
)

model_type = st.sidebar.radio(
    "Model",
    [
        "Isolation Forest",
        "LSTM Autoencoder"
    ]
)


# ======================================================
# LOAD DATA
# ======================================================

with st.spinner("Loading stock data..."):

    df = load_from_db(ticker)

    df = process_features(df)


# ======================================================
# RUN MODEL
# ======================================================

with st.spinner("Running anomaly detection..."):

    if model_type == "Isolation Forest":

        preds, scores = run_iforest(df)

    else:

        preds, scores = run_lstm(df)

df["anomaly"] = preds

df["anomaly_score"] = scores


# ======================================================
# METRICS
# ======================================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Anomalies",
    int((df["anomaly"] == -1).sum())
)

col2.metric(
    "Anomaly Rate",
    f"{(df['anomaly'] == -1).mean() * 100:.2f}%"
)

col3.metric(
    "Latest Close",
    f"₹{df['Close'].iloc[-1]:.2f}"
)


# ======================================================
# ANOMALY CHART
# ======================================================

st.subheader(f"{ticker} Anomaly Detection")

fig = go.Figure()

# Price line
fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Close"],
        name="Price",
        line=dict(width=1.5)
    )
)

# Anomaly points
anom = df[df["anomaly"] == -1]

fig.add_trace(
    go.Scatter(
        x=anom["Date"],
        y=anom["Close"],
        mode="markers",
        name="Anomaly",
        marker=dict(
            color="red",
            size=8
        )
    )
)

fig.update_layout(
    height=600,
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark"
)

st.plotly_chart(
    fig,
    use_container_width=True
)




# ======================================================
# RECENT ANOMALIES TABLE
# ======================================================

st.subheader("🚨 Recent Anomalies")

recent_anoms = df[
    df["anomaly"] == -1
][
    [
        "Date",
        "Close",
        "Volume",
        "anomaly_score"
    ]
].tail(10)

st.dataframe(
    recent_anoms,
    use_container_width=True
)


# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown("""
### 📌 About

This project detects unusual stock market behavior using:

- Isolation Forest
- LSTM Autoencoder
- Financial sentiment analysis using FinBERT
- Interactive Streamlit dashboard

### 🛠 Tech Stack

Python · PyTorch · scikit-learn · Streamlit · Plotly · FinBERT · SQLite

Built for financial anomaly detection and exploratory market intelligence.

""")