# Real-Time Stock Anomaly Detector

> An end-to-end machine learning system that detects statistically unusual trading behaviour in Indian stock market data using **Isolation Forest** and **LSTM Autoencoder** models, enriched with **FinBERT** financial sentiment analysis and served through an interactive Streamlit dashboard.

**Live Demo:** [your-app.streamlit.app](https://stock-anomaly-detector.streamlit.app/)   
**Dataset:** NSE-listed stocks via Yahoo Finance API (2-year historical data)  
**Built:** May 2026

---

##  Dashboard Preview

> **Screenshot 1 — Main anomaly chart (Isolation Forest)**  


<img width="1889" height="816" alt="image" src="https://github.com/user-attachments/assets/22aee284-3515-43ef-81bb-5c73b950d171" />





> **Screenshot 2 — Chart explanation + Score distribution**  


<img width="1554" height="769" alt="image" src="https://github.com/user-attachments/assets/15d7c1a5-582d-42da-885e-a09c7440a66d" />


> **Screenshot 3 — Recent anomalies table**  


<img width="1541" height="512" alt="image" src="https://github.com/user-attachments/assets/5f407675-9509-4f4d-b606-d6ccbfefb93b" />


---

##  Project Overview

Most retail investors and analysts lack tools to quickly identify when a stock is behaving in a statistically abnormal way — whether due to earnings surprises, macroeconomic shocks, or unusual trading volume. This project builds a full anomaly detection pipeline that:

- Automatically fetches and stores 2 years of daily OHLCV data for NSE stocks
- Engineers 9 financial features including RSI, Bollinger Band width, and volume ratio
- Detects anomalous trading days using two complementary ML models
- Correlates price anomalies with financial news sentiment using FinBERT NLP
- Presents findings through an interactive, production-style Streamlit dashboard

---

##  Models Used & Why

### 1. Isolation Forest
**What it is:** An unsupervised ensemble method that detects anomalies by randomly partitioning the feature space and measuring how quickly a data point gets isolated.

**Why it was chosen for this project:**
- No labelled data is required — financial anomaly data has no ground truth "this was an anomaly" labels, making supervised models inappropriate
- Handles high-dimensional financial features well (multiple technical indicators simultaneously)
- Computationally fast, allowing near-real-time inference on new trading days
- The contamination parameter (set at 2%) gives direct control over the expected anomaly rate, aligning with financial domain knowledge that roughly 1–3% of trading days are genuinely unusual

**How it works here:** The model is trained on 9 engineered features for each stock. It assigns an anomaly score to every trading day — scores closer to -1.0 indicate stronger anomalies. A threshold of approximately -0.58 separates flagged days from normal trading sessions.

---

### 2. LSTM Autoencoder
**What it is:** A deep learning architecture that learns to compress and reconstruct sequences of normal trading patterns. Days where reconstruction error is unusually high are flagged as anomalies.

**Why it was chosen for this project:**
- Captures temporal dependencies — Isolation Forest treats each day independently, but stock market behaviour is sequential. A sudden price drop after a long uptrend is more anomalous than the same drop in a volatile period
- Better at detecting structural breaks in time series (e.g., regime changes post-earnings)
- Complements Isolation Forest: the two models flag different anomaly types, giving a richer picture

**How it works here:** Input data is reshaped into 30-day rolling windows. The encoder compresses each window into a latent representation; the decoder reconstructs it. Reconstruction MSE above mean + 3 standard deviations is flagged as an anomaly. The lower anomaly count from LSTM (2 vs 9) reflects its focus on temporal pattern breaks rather than single-day statistical extremes.

---

### Model Comparison

| Aspect | Isolation Forest | LSTM Autoencoder |
|---|---|---|
| Type | Unsupervised ensemble | Deep learning sequence model |
| Input | Single-day features | 30-day rolling windows |
| Anomaly signal | Feature-space isolation | Temporal reconstruction error |
| Speed | Fast (< 1 sec) | Slower (training ~2 min) |
| Typical anomaly rate | ~2% | ~0.5% |
| Best for | Volume/volatility spikes | Trend breaks, regime shifts |

---

##  Features Engineered

| Feature | Description | Why it matters |
|---|---|---|
| `daily_return` | Day-over-day % price change | Core signal for unusual price moves |
| `log_return` | Log of price ratio | Normalises return distribution |
| `price_range` | (High - Low) / Close | Intraday volatility indicator |
| `gap` | Open vs previous Close | Overnight news/event signal |
| `std_5`, `std_20` | Rolling return std deviation | Short and medium-term volatility |
| `bb_width` | Bollinger Band width | Market volatility regime indicator |
| `rsi` | 14-day Relative Strength Index | Overbought/oversold signal |
| `vol_ratio` | Volume vs 20-day avg volume | Unusual trading activity signal |

---

##  Sentiment Analysis Layer

Beyond price and volume, this project adds a **news sentiment dimension** using **FinBERT** — a BERT-based model specifically fine-tuned on financial text (research papers, earnings calls, financial news).

**Why FinBERT over generic sentiment models:**
General models (VADER, TextBlob) misclassify financial language. For example, "the stock beat earnings estimates" is positive in finance but "beat" might confuse a general model. FinBERT was trained on 10,000+ financial sentences and correctly handles domain-specific phrasing.

**What it adds:** Each trading day receives a sentiment score (−1 to +1) derived from news headlines. This allows cross-referencing: do price anomalies cluster on days with negative news sentiment? This correlation analysis adds a "why" dimension to what would otherwise be purely quantitative signals.

---

##  Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Data collection | `yfinance` | NSE/BSE stock data via Yahoo Finance |
| Local storage | `SQLite` + `pandas` | Cache data, avoid API rate limits |
| Feature engineering | `pandas`, `numpy` | Technical indicators pipeline |
| ML Models | `scikit-learn`, `PyTorch` | Isolation Forest + LSTM Autoencoder |
| NLP / Sentiment | `transformers` (FinBERT) | Financial news sentiment scoring |
| News data | `NewsAPI` | Financial headlines by date |
| Dashboard | `Streamlit`, `Plotly` | Interactive web application |
| Deployment | Streamlit Community Cloud | Free public hosting |

---

##  Project Structure

```
stock-anomaly-detector/
│
├── data/
│   ├── raw/                    # Per-ticker CSV files from yfinance
│   │   ├── RELIANCE_NS.csv
│   │   ├── TCS_NS.csv
│   │   ├── INFY_NS.csv
│   │   └── WIPRO_NS.csv
│   └── stocks.db               # SQLite database (all tickers combined)
│
├── models/
│   ├── isolation_forest.py     # Isolation Forest training + inference
│   ├── lstm_autoencoder.py     # LSTM Autoencoder architecture + training
│   └── iforest.pkl             # Saved model weights
│
├── outputs/
│   └── anomalies_iforest.png   # Static validation plots
│
├── screenshots/                # Dashboard screenshots for README
│
├── app.py                      # Main Streamlit dashboard
├── data_fetcher.py             # Data collection + SQLite pipeline
├── database.py                 # DB read/write utilities
├── features.py                 # Feature engineering pipeline
├── sentiment.py                # FinBERT sentiment scoring
│
├── .env                        # API keys (not committed to Git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/stock-anomaly-detector.git
cd stock-anomaly-detector
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables**

Create a `.env` file in the root directory:
```
NEWSAPI_KEY=your_newsapi_key_here
```
Get a free API key at [newsapi.org](https://newsapi.org)

**5. Fetch stock data**
```bash
python data_fetcher.py
```

**6. Launch the dashboard**
```bash
streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) in your browser.

---

##  Key Results — RELIANCE.NS

| Model | Anomalies Detected | Anomaly Rate | Notable Flagged Dates |
|---|---|---|---|
| Isolation Forest | 9 / 448 days | 2.01% | Apr 2025 crash, Jan 2026 peak reversal |
| LSTM Autoencoder | 2 / 448 days | 0.45% | May 2025 trend break |

The Isolation Forest flagged **07 Apr 2025** (₹1161 close, anomaly score −0.6332) — cross-referencing with NSE announcements confirms this coincided with a broad market selloff. The **28 Apr 2025** flag (score −0.6848, strongest anomaly) aligned with an unusually high-volume session at a local support level.

---

##  Validation Methodology

Since no labelled ground truth exists for "correct" anomalies, validation used a multi-method approach:

1. **Visual inspection** — flagged dates plotted on price charts; reviewed against visible price extremes
2. **Business logic checks** — cross-referenced top flagged dates with NSE corporate announcements and broad market events
3. **Score distribution analysis** — confirmed anomaly scores form a clearly separated left-tail distribution vs normal trading days
4. **Model comparison** — checked whether both models agree on the most extreme events (they do for the Apr 2025 crash)

---

## 🚀 Future Improvements

- Integrate real-time tick data via **Zerodha Kite API** for intraday anomaly detection
- Add **portfolio-level anomaly scoring** across all 4 tickers simultaneously
- Build an **email/WhatsApp alert system** for newly flagged days
- Expand to **Nifty 50 universe** using batch data fetching
- Add **explainability layer** showing which feature most contributed to each anomaly flag

---

##  Author

**Kesahv Sharma** — Data Analytics Student  
📧 keshavsharma71982@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is open source under the [MIT License](LICENSE).

---

*Built as part of a portfolio project. All stock data sourced from Yahoo Finance via the yfinance library for educational purposes only. This is not financial advice.*

