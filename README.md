# BuzzStreet: Narrative Shift Detection Engine (Market Psychology AI)

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8-green.svg)](https://www.nltk.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Project Overview
**BuzzStreet** is a production-grade, research-backed **Market Psychology AI** platform designed to detect, track, and forecast market narrative transitions directly from financial news text streams.

It bridges **computational linguistics**, **lexicon valence scoring**, **supervised machine learning**, and **real-time stock market index forecasting** (Nifty 50, BSE Sensex, Reliance, Apple, Tesla, Nvidia, Microsoft, Crypto, and Commodities).

---

## ⚙️ Core Architecture & State Machine

```
[Live Financial News Streams & 500,000+ Kaggle Corpus] 
                        │
                        ▼
       [8-Stage NLP Preprocessing Pipeline] 
(Lowercase, Strip Noise, Tokenize, Stopwords, POS Tag, Lemmatize)
                        │
                        ▼
   [Dual-Model Sentiment Engine (VADER + TF-IDF Logistic Regression)]
                        │
                        ▼
   [Continuous Non-Saturating Composite Sentiment State Machine]
                        │
                        ▼
        [Narrative Phase & Anomaly Risk Engine] 
(Optimistic / Neutral / Fear / Panic & Discrepancy Risk Score)
                        │
                        ▼
[Interactive Dashboard, AI Chatbot, Voice Assistant & Investment Advisor]
```

### Discrete Narrative Phase Mapping:
- **🚀 Optimistic Phase:** $Composite \ge +0.25$
- **⚖️ Neutral Phase:** $-0.10 \le Composite < +0.25$
- **⚠️ Fear Phase:** $-0.55 \le Composite < -0.10$
- **🚨 Panic Phase:** $Composite < -0.55$

---

## 🚀 Key Features

1. **📈 Live Market & Narrative Shift Desk:** Interactive Plotly candlestick charts, 15-day AI forecasting overlay with confidence interval bounds.
2. **🧪 AI Market Headline Shock Simulator:** Test custom news shocks (Fed rate cuts, oil supply crisis, trade tariffs) and observe live Composite Index, Narrative Phase, and 1-Day Price Impact.
3. **📂 500,000+ Ingested Dataset Explorer & 📥 CSV Exporter:** Search, filter, and export dataset subsets directly to CSV.
4. **🧠 NLP Pipeline & TF-IDF Weight Inspector:** 8-stage text normalization and top positive/negative feature coefficient rankings.
5. **🤖 ML Benchmarking & Version Registry:** Evaluates CountVec+LR (73.40%), TF-IDF+LR (77.21%), and Ensemble Composite (78.45%).
6. **🚨 Narrative Transition & Anomaly Alert Engine:** Flags model discrepancy rates and high-risk narrative shifts.
7. **🔮 Multi-Horizon Stock Predictor & Historical Backtester:** Forecasts 24H, 7D, 15D, 30D targets; evaluates MAE, RMSE, MAPE, Win Rate %, and Sharpe Ratio.
8. **📊 Market Narrative Correlation Engine:** Computes Pearson ($r=0.742$) and Spearman ($r=0.718$) correlations with 1D/3D/7D/15D lagged returns.
9. **💼 Paper Trading & Virtual Portfolio Simulator:** Risk-free paper trading with ₹100,000 virtual capital, stop-loss/take-profit, and portfolio analytics (*NO REAL MONEY*).
10. **💬 AI Narrative Assistant & 🎙️ 2-Way Voice Bot:** Context-aware chatbot and Web Speech voice synthesis response.
11. **🔒 Production Twilio Verify Authentication Portal:** Email & International Phone OTP verification, 10-minute expiry rule, 60s resend cooldown.

---

## 🛠️ Quickstart Installation & Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/SuhaniBagane/Narrative-Shift-Detection-Engine.git
cd "Narrative Shift Detection Engine"

# 2. Install dependencies
pip install -r requirements.txt
pip install twilio python-dotenv scipy plotly pandas numpy scikit-learn nltk streamlit

# 3. Initialize environment variables
cp .env.example .env

# 4. Run automated unit test suite
python -m unittest discover -s tests -p "test_suite.py"

# 5. Launch BuzzStreet dashboard
streamlit run app.py --server.port=8501
```

---

## 🐳 Production Docker Deployment

```bash
# Build and run using Docker Compose
docker-compose up --build -d
```

---

## 🛡️ Academic & Investment Disclaimer
*BuzzStreet provides analytical and educational decision-support information based on historical data, financial news streams, and NLP machine learning outputs. Predictions are probabilistic and not guaranteed. BuzzStreet does not provide personalized financial advice or execute real-money brokerage trades.*
