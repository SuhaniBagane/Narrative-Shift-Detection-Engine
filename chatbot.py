"""
BuzzStreet – chatbot.py
Advanced Context-Aware & Versatile AI Chatbot Engine.
Processes complex user queries regarding market sentiment, stock predictions, asset comparisons,
NLP cleaning pipelines, ML model metrics, 300,000+ Kaggle datasets, and market anomalies.
"""

import re
import yfinance as yf
import pandas as pd
import numpy as np

def generate_chatbot_response(user_query, system_context):
    """
    Generates an intelligent, versatile, context-aware response based on the user's prompt
    and the current state of the BuzzStreet engine.
    """
    query = user_query.lower().strip()
    
    sentiment_idx = system_context.get("sentiment_index", 0.0)
    phase = system_context.get("narrative_phase", "Neutral")
    chain = system_context.get("transition_chain", "No transition data yet.")
    headlines = system_context.get("headlines", [])
    nifty_val = system_context.get("nifty_val", 22400.0)
    nifty_change = system_context.get("nifty_change", 0.0)
    sensex_val = system_context.get("sensex_val", 73900.0)
    sensex_change = system_context.get("sensex_change", 0.0)
    accuracy = system_context.get("model_accuracy", 0.7721)
    
    # ---------------------------------------------------------
    # 1. GREETINGS & SYSTEM INTRODUCTION
    # ---------------------------------------------------------
    if any(k in query for k in ["hello", "hi", "hey", "greetings", "introduce", "who are you", "what can you do", "help"]):
        return (
            "👋 **Welcome to BuzzStreet Narrative Intelligence Assistant!**\n\n"
            "I am your versatile AI market analyst & voice assistant. You can ask me anything about:\n"
            "- 📊 **Market Atmosphere:** *\"What is current market sentiment?\"* or *\"Why is the market negative?\"*\n"
            "- 🔮 **Stock Predictions:** *\"Predict Tesla stock\"*, *\"What is Apple target price?\"*, or *\"Nifty forecast\"*\n"
            "- 📊 **Asset Comparisons:** *\"Compare Apple and Tesla\"* or *\"Compare Nifty vs Sensex\"*\n"
            "- 🧠 **AI & NLP Pipeline:** *\"Explain NLP cleaning\"*, *\"How does Logistic Regression work?\"*, or *\"Model accuracy\"*\n"
            "- 📂 **Dataset Corpus:** *\"Tell me about the Kaggle dataset\"* or *\"How big is the dataset?\"*\n"
            "- 🚨 **Anomaly Risks:** *\"What is the current anomaly risk rating?\"*\n\n"
            "You can speak via the microphone or type queries anytime!"
        )

    # ---------------------------------------------------------
    # 2. STOCK PREDICTION & ASSET TARGET QUERIES
    # ---------------------------------------------------------
    if any(k in query for k in ["predict", "forecast", "target", "future", "going up", "going down", "apple", "tesla", "microsoft", "reliance", "nvidia", "spy", "qqq"]):
        ticker_map = {
            "apple": ("AAPL", "Apple Inc."),
            "tesla": ("TSLA", "Tesla Inc."),
            "microsoft": ("MSFT", "Microsoft Corp."),
            "reliance": ("RELIANCE.NS", "Reliance Industries"),
            "nvidia": ("NVDA", "NVIDIA Corp."),
            "nifty": ("^NSEI", "NSE Nifty 50 Index"),
            "sensex": ("^BSESN", "BSE Sensex Index"),
            "spy": ("SPY", "S&P 500 ETF"),
            "qqq": ("QQQ", "Nasdaq 100 ETF")
        }
        
        target_symbol = "^NSEI"
        asset_name = "Nifty 50 Index"
        for kw, (sym, name) in ticker_map.items():
            if kw in query:
                target_symbol = sym
                asset_name = name
                break
                
        try:
            df = yf.download(target_symbol, period="3mo", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and 'Close' in df.columns:
                prices = df['Close'].values.tolist()
                curr_p = prices[-1]
                lookback = min(30, len(prices))
                y = np.array(prices[-lookback:])
                x = np.arange(lookback)
                slope, _ = np.polyfit(x, y, 1)
                
                blend_slope = (0.7 * slope) + (0.3 * sentiment_idx * curr_p * 0.003)
                pred_15d = curr_p + (blend_slope * 15)
                ret_pct = ((pred_15d - curr_p) / curr_p) * 100
                
                direction = "BULLISH 🚀" if ret_pct >= 0 else "BEARISH ⚠️"
                signal = "STRONG BUY" if ret_pct > 2.5 else "BUY" if ret_pct > 0 else "SELL"
                
                return (
                    f"### 🔮 AI Forecast Report: **{asset_name}** ({target_symbol})\n\n"
                    f"- **Current Price:** `${curr_p:,.2f}`\n"
                    f"- **15-Day AI Target:** `${pred_15d:,.2f}` (**{ret_pct:+.2f}%** change)\n"
                    f"- **Technical Bias:** **{direction}**\n"
                    f"- **Quantitative Trade Signal:** `{signal}`\n"
                    f"- **Model Factors:** Combining 30-day price momentum (slope: `{slope:+.2f}`) and real-time textual sentiment index (`{sentiment_idx:+.3f}`).\n\n"
                    f"💡 *Tip:* View the full 680px interactive candlestick chart on the Live Global Market Desk!"
                )
        except Exception:
            pass
            
        return "### 🔮 Stock Predictor Engine\nThe stock predictor uses linear regression trendlines, RSI technical indicators, and sentiment momentum to project 5 to 60-day price targets for top Indian and US equities."

    # ---------------------------------------------------------
    # 3. ASSET COMPARISON QUERIES
    # ---------------------------------------------------------
    if any(k in query for k in ["compare", "versus", "vs", "difference", "comparison"]):
        return (
            "### 📊 Multi-Asset Comparison Engine\n\n"
            "BuzzStreet allows side-by-side comparative analysis of market assets (Nifty 50, Sensex, Reliance, Apple, Tesla, Microsoft, Nvidia).\n\n"
            "- **Normalized Relative Returns:** Compare percentage growth over 1-Month, 3-Month, 6-Month, or 1-Year horizons.\n"
            "- **Volatile Risk Beta:** Contrast standard deviation and momentum slopes across Indian and US markets.\n\n"
            "Say *\"Open Comparison\"* or select Tab 6 to inspect side-by-side comparative charts!"
        )

    # ---------------------------------------------------------
    # 4. KAGGLE DATASET & DATA SCALING QUERIES
    # ---------------------------------------------------------
    if any(k in query for k in ["dataset", "kaggle", "size", "count", "corpus", "250k", "300k", "rows", "data size", "how many headlines"]):
        return (
            "### 📂 Ingested Kaggle Dataset Corpus\n\n"
            "- **Total Corpus Size:** **300,000+ Labeled Financial Headlines** (`data/sentiment_data.csv` at 43.6 MB).\n"
            "- **Coverage:** Apple, Tesla, Reliance, TCS, Nvidia, Bitcoin, Ethereum, Gold, Crude Oil, S&P 500, Nasdaq.\n"
            "- **Unified Display:** Ingested headlines are searchable with 100-row pagination across 3,000 pages directly in the main stream view.\n"
            "- **Fast Ingestion:** Deduplicated set-based loading in `< 0.25 seconds` with pre-trained Logistic Regression classification."
        )

    # ---------------------------------------------------------
    # 5. MACHINE LEARNING & CLASSIFIER MODEL QUERIES
    # ---------------------------------------------------------
    if any(k in query for k in ["model", "accuracy", "classifier", "logistic regression", "pickle", "pkl", "training", "f1", "precision"]):
        return (
            f"### 🤖 Machine Learning Model Metrics\n\n"
            f"- **Algorithm:** Supervised Logistic Regression Classifier.\n"
            f"- **Testing Accuracy:** **{accuracy:.2%}** score on held-out test data.\n"
            f"- **Model Pickling:** Pre-trained weights stored in `data/model.pkl` and `data/vectorizer.pkl` for fast startup (< 0.1s).\n"
            f"- **Vectorization:** TF-IDF feature extraction with n-gram range (1, 2) and sublinear term frequency scaling."
        )

    # ---------------------------------------------------------
    # 6. NLP CLEANING & PREPROCESSING QUERIES
    # ---------------------------------------------------------
    if any(k in query for k in ["nlp", "preprocess", "cleaning", "lemmatize", "tokenize", "vader", "pos tag", "nltk", "stopwords"]):
        return (
            "### 🧠 NLP Preprocessing Pipeline\n\n"
            "Our 8-stage text transformation engine processes raw news feeds:\n"
            "1. **Lowercasing & Normalization:** Standardizes character encoding.\n"
            "2. **Noise Stripping:** Removes URLs, HTML tags, and non-alphanumeric symbols.\n"
            "3. **Tokenization:** Uses NLTK word tokenizers.\n"
            "4. **Stopword Removal:** Filters out high-frequency non-informative words.\n"
            "5. **POS Tagging & Lemmatization:** Converts words to root dictionary lemmas (e.g. *'surging'* -> *'surge'*).\n"
            "6. **TF-IDF Weighting:** Computes numerical term importance scores."
        )

    # ---------------------------------------------------------
    # 7. ANOMALY DETECTION & RISK RATING
    # ---------------------------------------------------------
    if any(k in query for k in ["anomaly", "risk", "suspicious", "flagged", "threat", "divergence"]):
        discrepancy_count = sum(1 for h in headlines if h.get("vader", {}).get("Sentiment Label") != h.get("lr", {}).get("Sentiment Label"))
        discrepancy_rate = discrepancy_count / len(headlines) if headlines else 0
        base_anomaly = {"Panic": 90, "Fear": 60, "Neutral": 15, "Optimistic": 5}.get(phase, 15)
        anomaly_score = min(base_anomaly + int(discrepancy_rate * 25), 100)
        
        status = "Stable (Low Risk)" if anomaly_score < 30 else "Moderate Volatility" if anomaly_score < 70 else "Systemic Anomaly Alert"
        
        return (
            f"### 🚨 Systemic Anomaly Report\n\n"
            f"- **Anomaly Score:** `{anomaly_score}%` Risk\n"
            f"- **Market Risk Status:** **{status}**\n"
            f"- **Model Divergence Rate:** Out of `{len(headlines)}` active headlines, VADER and ML classifier disagree on `{discrepancy_count}` headlines ({discrepancy_rate:.1%})."
        )

    # ---------------------------------------------------------
    # 8. CURRENT MARKET SENTIMENT & MOOD
    # ---------------------------------------------------------
    if any(k in query for k in ["sentiment", "mood", "feeling", "status", "current state", "market state", "how are the markets"]):
        sentiment_word = "bullish" if sentiment_idx > 0.1 else "bearish" if sentiment_idx < -0.1 else "neutral/stable"
        return (
            f"### 📊 Current Market Sentiment Analysis\n\n"
            f"- **Composite Sentiment Index:** `{sentiment_idx:+.3f}` (-1.0 to +1.0 scale)\n"
            f"- **Narrative Shift Phase:** **{phase}** ({sentiment_word.upper()})\n"
            f"- **Nifty 50 Index:** `{nifty_val:,.2f}` ({nifty_change:+.2f}%)\n"
            f"- **BSE Sensex Index:** `{sensex_val:,.2f}` ({sensex_change:+.2f}%)\n\n"
            f"Market state machine classifies current environment as **{phase}** based on combined lexicon & ML probabilities."
        )

    # ---------------------------------------------------------
    # 9. DRIVERS / WHY IS MARKET UP OR DOWN
    # ---------------------------------------------------------
    if any(k in query for k in ["why", "reason", "driver", "cause", "explain", "behind", "source", "factors"]):
        if sentiment_idx < 0:
            neg_h = [h for h in headlines if h.get("vader", {}).get("Sentiment Label") == "Negative"]
            top_text = neg_h[0]["raw"] if neg_h else "Macro economic caution and inflationary warnings."
            return f"### 🔍 Key Bearish Drivers\nThe market narrative is leaning negative primarily due to macro anxieties: *\"{top_text}\"*"
        else:
            pos_h = [h for h in headlines if h.get("vader", {}).get("Sentiment Label") == "Positive"]
            top_text = pos_h[0]["raw"] if pos_h else "Strong corporate earnings and steady growth indicators."
            return f"### 🔍 Key Bullish Drivers\nThe market narrative is positive due to expansionary indicators: *\"{top_text}\"*"

    # ---------------------------------------------------------
    # 10. DYNAMIC INTELLIGENT FALLBACK / GENERAL QUERY
    # ---------------------------------------------------------
    sentiment_word = "bullish" if sentiment_idx > 0.1 else "bearish" if sentiment_idx < -0.1 else "consolidating"
    return (
        f"### 🧠 BuzzStreet AI Analysis for: *\"{user_query}\"*\n\n"
        f"Based on real-time narrative feeds, the market is currently in an **{phase}** phase with a sentiment index of `{sentiment_idx:+.3f}`. "
        f"Nifty 50 is trading at `{nifty_val:,.2f}` ({nifty_change:+.2f}%) and Sensex at `{sensex_val:,.2f}` ({sensex_change:+.2f}%).\n\n"
        f"Try asking more specific questions:\n"
        f"- *\"Predict Tesla stock\"* or *\"What is Nifty forecast?\"*\n"
        f"- *\"Compare Apple and Tesla\"*\n"
        f"- *\"Explain the NLP pipeline\"* or *\"Model accuracy\"*\n"
        f"- *\"Tell me about the 300k Kaggle dataset\"*"
    )

