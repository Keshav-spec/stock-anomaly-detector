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


#  
# PAGE CONFIG
#  

st.set_page_config(
    page_title="Stock Anomaly Detector",
    layout="wide"
)
st.markdown("""
<style>

/* Main app background */
.stApp {
    background-color: #0e1117;
    color: #FAFAFA;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #1c2333;
    border: 1px solid #2d3748;
    padding: 1rem;
    border-radius: 12px;
}

/* Metric labels */
[data-testid="metric-container"] label {
    color: #cbd5e1 !important;
}

/* Metric values */
[data-testid="metric-container"] div {
    color: white !important;
}

/* Headers */
h1, h2, h3, h4 {
    color: white !important;
}

/* Paragraph text */
p, div {
    color: #e5e7eb;
}

/* Insight box */
.insight-box {
    background: linear-gradient(
        135deg,
        rgba(67,97,238,0.15),
        rgba(118,75,162,0.15)
    );

    border-left: 4px solid #4361ee;

    padding: 1rem 1.5rem;

    border-radius: 10px;

    margin-top: 1rem;
    margin-bottom: 1rem;
}

</style>
""", unsafe_allow_html=True)

#  
# TITLE
#  

st.title("📈 Real-Time Stock Anomaly Detector")


#  
# SIDEBAR
#  

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


#  
# LOAD DATA
#  

with st.spinner("Loading stock data..."):

    df = load_from_db(ticker)

    df = process_features(df)


#  
# RUN MODEL
#  

with st.spinner("Running anomaly detection..."):

    if model_type == "Isolation Forest":

        preds, scores = run_iforest(df)

    else:

        preds, scores = run_lstm(df)

df["anomaly"] = preds

df["anomaly_score"] = scores


#  
# METRICS
#  

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


#  
# ANOMALY CHART
#  

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

    template="plotly_dark",

    paper_bgcolor="#0e1117",
    plot_bgcolor="#0e1117",

    font=dict(
        color="white"
    ),

    legend=dict(
        bgcolor="#111827"
    )
)
fig.update_traces(
    line=dict(color="#4361ee"),
    selector=dict(name="Price")
)

fig.update_traces(
    marker=dict(
        color="#ef233c",
        size=10
    ),
    selector=dict(name="Anomaly")
)
st.plotly_chart(
    fig,
    use_container_width=True
)

st.markdown("---")

worst_anomaly = (
    df[df["anomaly"] == -1]
    .nsmallest(1, "anomaly_score")
    .iloc[0]
)

recent_anomaly = (
    df[df["anomaly"] == -1]
    .sort_values("Date")
    .iloc[-1]
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
    <div class="insight-box">
    <h4>🔍 How to read this chart</h4>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    The blue line shows **{ticker}** closing prices.

    Red dots indicate statistically unusual trading days
    detected by the anomaly model.

    - More negative scores = stronger anomaly
    - {int((df["anomaly"]==-1).sum())} anomaly days detected
    - Dataset contains {len(df)} trading sessions
    """)

    st.markdown("</div>", unsafe_allow_html=True)

with col2:

    st.markdown("""
    <div class="insight-box">
    <h4>📌 Most Significant Anomaly</h4>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    **Date:** {str(worst_anomaly['Date'])[:10]}

    **Close Price:** ₹{worst_anomaly['Close']:.2f}

    **Anomaly Score:** {worst_anomaly['anomaly_score']:.4f}

    **Volume:** {int(worst_anomaly['Volume']):,}
    """)

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown("### 📊 Anomaly Score Distribution")

fig2 = go.Figure()

fig2.add_trace(
    go.Histogram(
        x=df[df["anomaly"] == 1]["anomaly_score"],
        name="Normal Days",
        nbinsx=40,
        marker_color="#4361ee",
        opacity=0.7
    )
)

fig2.add_trace(
    go.Histogram(
        x=df[df["anomaly"] == -1]["anomaly_score"],
        name="Anomaly Days",
        nbinsx=10,
        marker_color="#ef233c",
        opacity=0.9
    )
)

fig2.update_layout(
    barmode="overlay",

    xaxis_title="Isolation Forest Anomaly Score",

    yaxis_title="Number of Trading Days",

    paper_bgcolor="#0e1117",

    plot_bgcolor="#0e1117",

    font=dict(
        color="white"
    ),

    height=400
)

st.plotly_chart(
    fig2,
    width="stretch"
)

#  
# RECENT ANOMALIES TABLE
#  

st.subheader("🚨 Recent Anomalies")

display_df = df[
    df["anomaly"] == -1
][
    [
        "Date",
        "Close",
        "Volume",
        "anomaly_score"
    ]
].copy()

display_df["Date"] = pd.to_datetime(
    display_df["Date"]
).dt.strftime("%d %b %Y")

display_df["Close"] = display_df["Close"].apply(
    lambda x: f"₹{x:.2f}"
)

display_df["Volume"] = display_df["Volume"].apply(
    lambda x: f"{int(x):,}"
)

display_df["anomaly_score"] = display_df[
    "anomaly_score"
].apply(
    lambda x: f"{x:.4f}"
)

display_df = display_df.rename(
    columns={
        "Close": "Close Price",
        "Volume": "Volume",
        "anomaly_score": "Anomaly Score"
    }
)

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True
)


#  
# FOOTER
#  

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