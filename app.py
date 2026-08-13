"""
BuzzStreet – app.py
Phase II Milestone: 50% Project Completion
Main Streamlit Frontend Dashboard.
Integrates simulated financial news feeds, Nifty/Sensex trackers, NLP preprocessing inspector,
TF-IDF feature weights, comparative sentiment models (VADER vs Logistic Regression),
narrative shift state-machine timeline charts, and a context-aware AI chatbot assistant.
"""

import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import re
import numpy as np
import yfinance as yf

# Declare Voice Assistant Component
parent_dir = os.path.dirname(os.path.abspath(__file__))
voice_component_path = os.path.join(parent_dir, "voice_component")
voice_assistant = components.declare_component("voice_assistant", path=voice_component_path)

# Import Modular BuzzStreet Components
import data_loader
from nlp_pipeline import preprocess_headline_detailed
from ml_model import model_instance
import narrative_detector
import chatbot

# Initialize VADER Sentiment Intensity Analyzer
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

# ==========================================
# 1. PAGE SETUP & PREMIUM CUSTOM CSS
# ==========================================
st.set_page_config(page_title="BuzzStreet - Narrative Shift Detection Engine", layout="wide", initial_sidebar_state="expanded")

# Inject a highly refined CSS theme (curated dark slate colors, glowing accents, and card styling)
st.markdown("""
<style>
    /* Global Background and Fonts */
    .stApp {
        background-color: #0b0d12;
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Headers & Accent Text */
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: -0.025em;
    }
    
    /* Elegant Title and Badges */
    .main-title {
        background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        font-weight: 800 !important;
        text-align: center;
        margin-bottom: 0.2rem !important;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .milestone-badge {
        display: block;
        width: fit-content;
        margin: -1.5rem auto 2.5rem auto;
        background-color: rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Glassmorphism Container Card Styling */
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"] {
        background-color: #131722;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3), 0 4px 6px -4px rgba(0,0,0,0.3);
        border: 1px solid #1e293b;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stVerticalBlock"] > div > div[data-testid="stVerticalBlock"]:hover {
        border-color: #334155;
    }
    
    /* Metrics block customizations */
    div[data-testid="stMetricValue"] {
        font-weight: 800 !important;
        font-size: 2.2rem !important;
        letter-spacing: -0.05em;
    }
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.8rem !important;
        letter-spacing: 0.05em;
    }

    /* Chatbot interface bubble bubbles */
    .chat-container {
        border: 1px solid #1e293b;
        border-radius: 12px;
        background-color: #0f121a;
        padding: 1.5rem;
        max-height: 450px;
        overflow-y: auto;
        margin-bottom: 1rem;
    }
    .user-bubble {
        background-color: #1e1b4b;
        border: 1px solid #4338ca;
        color: #e0e7ff;
        padding: 10px 16px;
        border-radius: 18px 18px 2px 18px;
        margin-left: 20%;
        margin-bottom: 15px;
        text-align: right;
        font-size: 0.95rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .bot-bubble {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #f1f5f9;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 2px;
        margin-right: 20%;
        margin-bottom: 15px;
        text-align: left;
        font-size: 0.95rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        line-height: 1.5;
    }
    
    /* Custom divider line */
    hr {
        border-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. STATE MANAGER (SESSION STATE)
# ==========================================
# Initialize static parameters and caches if not set
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    n_val, n_chg, s_val, s_chg = data_loader.fetch_live_indices_yfinance()
    st.session_state.nifty_val = n_val
    st.session_state.sensex_val = s_val
    st.session_state.nifty_change = n_chg
    st.session_state.sensex_change = s_chg
    
    # Pre-populate history with a standard realistic narrative arc
    now = datetime.datetime.now()
    st.session_state.market_history = [
        {"time": (now - datetime.timedelta(hours=5)).strftime("%H:%M"), "sentiment": 0.12, "phase": "Neutral", "nifty": 22350.0, "sensex": 73650.0},
        {"time": (now - datetime.timedelta(hours=4)).strftime("%H:%M"), "sentiment": 0.35, "phase": "Optimistic", "nifty": 22410.0, "sensex": 73800.0},
        {"time": (now - datetime.timedelta(hours=3)).strftime("%H:%M"), "sentiment": 0.28, "phase": "Optimistic", "nifty": 22450.0, "sensex": 73920.0},
        {"time": (now - datetime.timedelta(hours=2)).strftime("%H:%M"), "sentiment": -0.05, "phase": "Neutral", "nifty": 22400.0, "sensex": 73810.0},
        {"time": (now - datetime.timedelta(hours=1)).strftime("%H:%M"), "sentiment": -0.32, "phase": "Fear", "nifty": 22310.0, "sensex": 73520.0}
    ]
    
    # Initialize dynamic inputs
    st.session_state.market_bias = "neutral"
    
    st.session_state.vader_weight = 0.50
    
    # Generate initial active headlines
    initial_raw = data_loader.generate_headlines(bias="neutral", count=20)
    st.session_state.active_headlines_raw = initial_raw
    st.session_state.active_headlines_data = [] # List to hold fully analyzed headlines
    
    st.session_state.open_chat = False
    
    # Initialize Chatbot history
    st.session_state.chat_history = [
        ("user", "Hello! What is this system capable of?"),
        ("bot", "👋 Welcome to **BuzzStreet Narrative Intelligence Assistant!**\n\nI can analyze raw financial texts, compute sentiment metrics using VADER and a trained **Logistic Regression classifier**, detect overall market narrative shifts, and answer questions. Try asking: \n\n*“What is current market sentiment?”* or *“Why is the market negative?”*")
    ]
    
    st.session_state.voice_speak_text = ""
    st.session_state.last_heard_text = ""
    st.session_state.active_tab_to_click = ""
    st.session_state.predict_ticker = "^NSEI"
    st.session_state.compare_tickers = ["^NSEI", "^BSESN"]

# Helper function to trigger a full system refresh/re-evaluation
def evaluate_market_state(bias_param, is_refresh=False, reblend_only=False):
    vader_w = st.session_state.get("vader_weight", 0.50)
    
    if reblend_only and st.session_state.active_headlines_data:
        # Re-extract compound scores and predictions from cached data
        vader_compounds = [h["vader"]["compound"] for h in st.session_state.active_headlines_data]
        lr_sentiment_dicts = [h["lr"] for h in st.session_state.active_headlines_data]
    else:
        # 1. Generate new headlines
        raw_headlines = data_loader.generate_headlines(bias=bias_param, count=20)

        st.session_state.active_headlines_raw = raw_headlines
        
        # 2. Analyze sentiment with VADER & Logistic Regression
        analyzer_data = []
        vader_compounds = []
        lr_sentiment_dicts = []
        
        for h in raw_headlines:
            # VADER
            pos_v, neg_v, neu_v, label_v = sia.polarity_scores(h)['pos'], sia.polarity_scores(h)['neg'], sia.polarity_scores(h)['neu'], ""
            compound_v = sia.polarity_scores(h)['compound']
            if compound_v >= 0.05: label_v = "Positive"
            elif compound_v <= -0.05: label_v = "Negative"
            else: label_v = "Neutral"
            
            # Logistic Regression
            lr_res = model_instance.predict_sentiment_lr(h)
            
            vader_compounds.append(compound_v)
            lr_sentiment_dicts.append(lr_res)
            
            analyzer_data.append({
                "raw": h,
                "cleaned": preprocess_headline_detailed(h)["final_text"],
                "vader": {"Positive": pos_v, "Negative": neg_v, "Neutral": neu_v, "compound": compound_v, "Sentiment Label": label_v},
                "lr": lr_res
            })
            
        st.session_state.active_headlines_data = analyzer_data
    
    # 3. Calculate composite Sentiment Index
    composite_index = narrative_detector.calculate_composite_index(vader_compounds, lr_sentiment_dicts, vader_weight=vader_w)
    
    # 4. Detect current narrative phase
    current_phase = narrative_detector.detect_narrative_phase(composite_index)
    
    # 5. Simulate Market Index changes
    new_nifty, nifty_change, new_sensex, sensex_change = data_loader.simulate_market_indices(
        composite_index, 
        prev_nifty=st.session_state.nifty_val, 
        prev_sensex=st.session_state.sensex_val
    )
    
    st.session_state.nifty_val = new_nifty
    st.session_state.sensex_val = new_sensex
    st.session_state.nifty_change = nifty_change
    st.session_state.sensex_change = sensex_change
    
    # 6. Save data to chronological history
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.market_history.append({
        "time": current_time,
        "sentiment": composite_index,
        "phase": current_phase,
        "nifty": new_nifty,
        "sensex": new_sensex
    })
    
    # Keep history bounded to last 20 elements for visual aesthetic
    if len(st.session_state.market_history) > 20:
        st.session_state.market_history.pop(0)

def handle_slider_change():
    evaluate_market_state(st.session_state.market_bias, reblend_only=True)

def process_chat_query(query):
    if query:
        vaders_c = [h["vader"]["compound"] for h in st.session_state.active_headlines_data]
        lrs_c = [h["lr"] for h in st.session_state.active_headlines_data]
        vader_w = st.session_state.get("vader_weight", 0.50)
        composite_idx = narrative_detector.calculate_composite_index(vaders_c, lrs_c, vader_weight=vader_w)
        curr_ph = narrative_detector.detect_narrative_phase(composite_idx)
        transition_path_str = narrative_detector.generate_transition_chain(st.session_state.market_history)
        
        ctx = {
            "sentiment_index": composite_idx,
            "narrative_phase": curr_ph,
            "transition_chain": transition_path_str,
            "headlines": st.session_state.active_headlines_data,
            "nifty_val": st.session_state.nifty_val,
            "nifty_change": st.session_state.nifty_change,
            "model_accuracy": model_instance.accuracy,
            "sensex_val": st.session_state.sensex_val,
            "sensex_change": st.session_state.sensex_change
        }
        st.session_state.chat_history.append(("user", query))
        ans = chatbot.generate_chatbot_response(query, ctx)
        st.session_state.chat_history.append(("bot", ans))

def process_voice_command(command_text):
    cmd = command_text.lower().strip()
    
    # 1. Navigation Commands
    if any(k in cmd for k in ["narrative", "intelligence", "shift", "home", "dashboard"]):
        st.session_state.active_tab_to_click = "Narrative Intelligence"
        st.session_state.voice_speak_text = "Opening Narrative Intelligence Dashboard."
        st.rerun()
    elif any(k in cmd for k in ["nlp", "preprocess", "clean", "token"]):
        st.session_state.active_tab_to_click = "NLP Cleaning"
        st.session_state.voice_speak_text = "Showing NLP Preprocessing and TF-IDF Inspector."
        st.rerun()
    elif any(k in cmd for k in ["model", "cockpit", "training", "machine learning", "classifier"]):
        st.session_state.active_tab_to_click = "ML Training"
        st.session_state.voice_speak_text = "Navigating to Machine Learning Classifier Performance."
        st.rerun()
    elif any(k in cmd for k in ["chatbot", "assistant", "chat", "ask bot"]):
        st.session_state.active_tab_to_click = "Chatbot"
        st.session_state.voice_speak_text = "Opening AI Narrative Chatbot."
        st.rerun()
    elif any(k in cmd for k in ["predict", "forecast", "future", "suggestion"]):
        st.session_state.active_tab_to_click = "Predictor"
        st.session_state.voice_speak_text = "Opening Stock and Trade Future Predictor."
        
        # Check if the user specified a ticker to predict, e.g. "predict apple"
        for word, ticker in [("apple", "AAPL"), ("tesla", "TSLA"), ("microsoft", "MSFT"), ("reliance", "RELIANCE.NS"), ("nifty", "^NSEI"), ("sensex", "^BSESN")]:
            if word in cmd:
                st.session_state.predict_ticker = ticker
                st.session_state.voice_speak_text = f"Predicting future performance for {word.capitalize()}."
                break
        st.rerun()
    elif any(k in cmd for k in ["compare", "comparison", "side by side"]):
        st.session_state.active_tab_to_click = "Comparison"
        st.session_state.voice_speak_text = "Opening Multi-Asset Comparison Dashboard."
        
        # Check if they named companies to compare, e.g., "compare apple and tesla"
        detected_tickers = []
        ticker_map = {
            "apple": "AAPL", "tesla": "TSLA", "microsoft": "MSFT", 
            "reliance": "RELIANCE.NS", "nifty": "^NSEI", "sensex": "^BSESN"
        }
        for word, ticker in ticker_map.items():
            if word in cmd:
                detected_tickers.append(ticker)
        if len(detected_tickers) >= 2:
            st.session_state.compare_tickers = detected_tickers[:4] # limit to 4
            st.session_state.voice_speak_text = f"Comparing performance of " + " and ".join([k.capitalize() for k in ticker_map.keys() if ticker_map[k] in detected_tickers])
        st.rerun()
        
    # 2. General Queries (Pass to Chatbot directly!)
    else:
        st.session_state.active_tab_to_click = "Chatbot"
        
        vaders_c = [h["vader"]["compound"] for h in st.session_state.active_headlines_data]
        lrs_c = [h["lr"] for h in st.session_state.active_headlines_data]
        vader_w = st.session_state.get("vader_weight", 0.50)
        
        composite_idx = narrative_detector.calculate_composite_index(vaders_c, lrs_c, vader_weight=vader_w)
        curr_ph = narrative_detector.detect_narrative_phase(composite_idx)
        transition_path_str = narrative_detector.generate_transition_chain(st.session_state.market_history)
        
        ctx = {
            "sentiment_index": composite_idx,
            "narrative_phase": curr_ph,
            "transition_chain": transition_path_str,
            "headlines": st.session_state.active_headlines_data,
            "nifty_val": st.session_state.nifty_val,
            "nifty_change": st.session_state.nifty_change,
            "model_accuracy": model_instance.accuracy,
            "sensex_val": st.session_state.sensex_val,
            "sensex_change": st.session_state.sensex_change
        }
        st.session_state.chat_history.append(("user", command_text))
        ans = chatbot.generate_chatbot_response(command_text, ctx)
        
        # Clean answer for speech
        clean_ans = re.sub(r'[\*#`🚨🔮📈🧠💬💻🤖⚙️📊⚖️⚠️]', '', ans)
        clean_ans = clean_ans.split('\n')[0]
        st.session_state.voice_speak_text = f"Voice query received. Here is the response: {clean_ans}"
        st.session_state.chat_history.append(("bot", ans))
        st.rerun()

def handle_quick_prompt(prompt_text):
    process_chat_query(prompt_text)

@st.dialog("🎙️ BuzzStreet AI Assistant")
def show_assistant_dialog():
    st.markdown("Ask any questions about market sentiment, NLP cleaning pipelines, or active drivers:")
    
    st.markdown('<div class="dialog-chat-container" style="max-height: 380px; overflow-y: auto; padding: 12px; background: rgba(15,23,42,0.6); border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 12px;">', unsafe_allow_html=True)
    # Render all chat bubbles from history
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div class="user-bubble"><b>You:</b><br>{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble"><b>BuzzStreet Bot:</b><br>{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
            
    query = st.chat_input("Ask a question...", key="dialog_chat_input")
    if query:
        process_chat_query(query)

# Evaluate state on first load if active data lists are empty
if not st.session_state.active_headlines_data:
    evaluate_market_state(st.session_state.market_bias)

# Cache some quick references
current_state_data = st.session_state.market_history[-1]
curr_index = current_state_data["sentiment"]
curr_phase = current_state_data["phase"]
phase_style = narrative_detector.get_phase_styling(curr_phase)
transition_path = narrative_detector.generate_transition_chain(st.session_state.market_history)

# ==========================================
# 3. SIDEBAR: PROJECT OVERVIEW & METRICS
# ==========================================
with st.sidebar:
    st.markdown("### BuzzStreet Console")
    # Avatar
    st.markdown('<img src="https://api.dicebear.com/7.x/bottts/svg?seed=BuzzStreet&backgroundColor=0b0d12" class="profile-img">', unsafe_allow_html=True)
    st.markdown('<div class="profile-name">BuzzStreet Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="profile-role">Phase III: Advanced NLP & AI Predictor</div>', unsafe_allow_html=True)
    st.divider()
    
    st.markdown("### 🎙️ Voice Assistant Control")
    
    # Render the custom voice assistant component
    heard_text = voice_assistant(
        key="voice_assistant_component", 
        text_to_speak=st.session_state.voice_speak_text,
        active_tab_to_click=st.session_state.get("active_tab_to_click", "")
    )
    
    # Reset voice control states to prevent loops
    st.session_state.voice_speak_text = ""
    st.session_state.active_tab_to_click = ""
    
    if heard_text and heard_text != st.session_state.last_heard_text:
        st.session_state.last_heard_text = heard_text
        process_voice_command(heard_text)
        
    st.divider()
    
    st.markdown("### 🎛️ Sentiment Calibration")
    st.slider(
        "VADER Lexicon Weight:",
        min_value=0.0,
        max_value=1.0,
        key="vader_weight",
        step=0.05,
        on_change=handle_slider_change,
        help="Calibrate the blend of lexicon compound scores vs supervised classification probabilities."
    )
    st.divider()
    
    st.markdown("### ⚙️ Simulator Control")
    st.markdown("Select a narrative bias parameter to guide headline generation and simulate real market cycles:")
    
    bias_map = {
        "Neutral Consolidation": "neutral",
        "Bullish Breakthrough": "bullish",
        "Bearish Correction": "bearish",
        "Systemic Panic Selloff": "panic"
    }
    
    selected_bias_label = st.selectbox(
        "Market State Bias:",
        options=list(bias_map.keys()),
        index=list(bias_map.values()).index(st.session_state.market_bias)
    )
    
    st.session_state.market_bias = bias_map[selected_bias_label]
    
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 Fetch & Analyze Live Headlines", use_container_width=True, type="primary"):
        evaluate_market_state(st.session_state.market_bias, is_refresh=True)
        st.rerun()
        
    st.divider()
    
    # Model evaluation metrics
    st.markdown("### 🤖 Core Model Specs")
    st.markdown(f"**Classifier:** Scikit-Learn Logistic Regression")
    st.markdown(f"**Accuracy:** `{model_instance.accuracy:.2%}`")
    st.markdown(f"**Vocab Size:** `{len(model_instance.vectorizer.get_feature_names_out())} words`")
    st.markdown(f"**State History Size:** `{len(st.session_state.market_history)} records`")
    
    st.divider()
    st.markdown("### 🎙️ AI Narrative Assistant")
    st.markdown("""
    <div style="background-color: rgba(2, 132, 199, 0.08); border: 1px solid rgba(2, 132, 199, 0.2); padding: 12px; border-radius: 8px; font-size: 0.85rem; color: #cbd5e1; line-height: 1.4;">
        💡 <b>Interactive Chatbot:</b><br>
        Click the glowing blue round buzzer button in the bottom right corner of the window. 
        It is persistent across all tabs and allows you to chat with the engine instantly!
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. MAIN COCKPIT RENDERING
# ==========================================
st.markdown('<h1 class="main-title">BuzzStreet</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A Narrative Shift Detection Engine</p>', unsafe_allow_html=True)
st.markdown('<span class="milestone-badge">Phase II Milestone (50% Completion Review)</span>', unsafe_allow_html=True)

# ----------------------------------------------------
# TABULAR LAYOUT FOR COCKPIT SECTIONS
# ----------------------------------------------------
tab_narrative, tab_nlp, tab_ml, tab_chatbot, tab_predict, tab_compare = st.tabs([
    "📈 Narrative Intelligence", 
    "🧠 NLP Cleaning & TF-IDF", 
    "🤖 ML Training Cockpit", 
    "💬 AI Narrative Chatbot",
    "🔮 Stock & Trade Predictor",
    "📊 Multi-Asset Comparison"
])

with tab_narrative:
    # A. Top KPI Row: Stock indices + Narrative state + Anomaly Rating
    st.markdown("### 📊 Market Atmosphere & Indexes")
    
    # Calculate Discrepancy Count and Rate
    discrepancy_count = sum(1 for h in st.session_state.active_headlines_data if h["vader"]["Sentiment Label"] != h["lr"]["Sentiment Label"])
    discrepancy_rate = discrepancy_count / len(st.session_state.active_headlines_data) if st.session_state.active_headlines_data else 0
    base_anomaly = {"Panic": 90, "Fear": 60, "Neutral": 15, "Optimistic": 5}.get(curr_phase, 15)
    added_anomaly = int(discrepancy_rate * 25)
    anomaly_score = min(base_anomaly + added_anomaly, 100)
    
    if anomaly_score < 30:
        anomaly_color = "#10b981"
        anomaly_label = "Stable"
        anomaly_bg = "rgba(16, 185, 129, 0.08)"
        anomaly_border = "rgba(16, 185, 129, 0.2)"
        anomaly_emoji = "🟢"
    elif anomaly_score < 70:
        anomaly_color = "#f59e0b"
        anomaly_label = "Volatility"
        anomaly_bg = "rgba(245, 158, 11, 0.08)"
        anomaly_border = "rgba(245, 158, 11, 0.2)"
        anomaly_emoji = "🟡"
    else:
        anomaly_color = "#ef4444"
        anomaly_label = "Systemic Anomaly"
        anomaly_bg = "rgba(239, 68, 68, 0.08)"
        anomaly_border = "rgba(239, 68, 68, 0.2)"
        anomaly_emoji = "🔴"

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns([1, 1, 1.5, 1.5])
    
    with kpi_col1:
        st.metric(
            label="Nifty 50 Index (Simulated)",
            value=f"{st.session_state.nifty_val:,.2f}",
            delta=f"{st.session_state.nifty_change:+.2f}%"
        )
        
    with kpi_col2:
        st.metric(
            label="Sensex Index (Simulated)",
            value=f"{st.session_state.sensex_val:,.2f}",
            delta=f"{st.session_state.sensex_change:+.2f}%"
        )
        
    with kpi_col3:
        # Custom styled card for narrative phase
        st.markdown(f"""
        <div style="background-color: {phase_style['bg_color']}; border: 1px solid {phase_style['border_color']}; padding: 18px 20px; border-radius: 12px; height: 100px; display: flex; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);">
            <div style="font-size: 2.2rem; margin-right: 12px;">{phase_style['emoji']}</div>
            <div>
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; line-height: 1;">Narrative Phase</div>
                <div style="font-size: 1.2rem; font-weight: 800; color: {phase_style['color']}; margin: 2px 0;">{curr_phase}</div>
                <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 400; line-height: 1.1;">Index: {curr_index:+.3f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        # Custom styled card for Market Anomaly Score
        st.markdown(f"""
        <div style="background-color: {anomaly_bg}; border: 1px solid {anomaly_border}; padding: 18px 20px; border-radius: 12px; height: 100px; display: flex; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2);">
            <div style="font-size: 2.2rem; margin-right: 12px;">{anomaly_emoji}</div>
            <div>
                <div style="font-size: 0.72rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; line-height: 1;">Anomaly Rating</div>
                <div style="font-size: 1.20rem; font-weight: 800; color: {anomaly_color}; margin: 2px 0;">{anomaly_score}% Risk</div>
                <div style="font-size: 0.78rem; color: #cbd5e1; font-weight: 400; line-height: 1.1;">State: {anomaly_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🔄 Narrative Shift Transition Path")
    st.markdown(f"💡 **Chrono-Shift Chain:** ` {transition_path} `")
    st.divider()
    
    # ----------------------------------------------------
    # ADVANCED LIVE GLOBAL MARKETS DESK & AI FUTURE PREDICTION
    # ----------------------------------------------------
    st.markdown("### 📊 Live Global Markets Desk & AI Future Prediction Chart")
    st.markdown("Real-time live market data stream with integrated AI future price forecasts, confidence interval bands, and technical indicators.")

    # Top Control Bar: Asset Selector & View Mode
    c_live1, c_live2, c_live3 = st.columns([2, 2, 1])
    
    asset_options = {
        "NSE Nifty 50 Index": {"yf": "^NSEI", "tv": "NSE:NIFTY"},
        "BSE Sensex Index": {"yf": "^BSESN", "tv": "BOM:SENSEX"},
        "S&P 500 ETF (SPY)": {"yf": "SPY", "tv": "AMEX:SPY"},
        "Nasdaq 100 ETF (QQQ)": {"yf": "QQQ", "tv": "NASDAQ:QQQ"},
        "Reliance Industries": {"yf": "RELIANCE.NS", "tv": "NSE:RELIANCE"},
        "Apple Inc. (AAPL)": {"yf": "AAPL", "tv": "NASDAQ:AAPL"},
        "Tesla Inc. (TSLA)": {"yf": "TSLA", "tv": "NASDAQ:TSLA"},
        "NVIDIA Corp. (NVDA)": {"yf": "NVDA", "tv": "NASDAQ:NVDA"},
        "Microsoft Corp. (MSFT)": {"yf": "MSFT", "tv": "NASDAQ:MSFT"}
    }
    
    selected_asset_name = c_live1.selectbox(
        "Select Active Market Asset:",
        options=list(asset_options.keys()),
        index=0,
        key="desk_asset_select"
    )
    
    asset_info = asset_options[selected_asset_name]
    yf_symbol = asset_info["yf"]
    tv_symbol = asset_info["tv"]
    
    desk_mode = c_live2.radio(
        "Chart Display Mode:",
        options=["🔮 Live Desk + AI Future Prediction", "📺 TradingView Exchange Widget"],
        horizontal=True,
        key="desk_mode_select"
    )
    
    desk_horizon = c_live3.slider("Forecast Days:", min_value=5, max_value=30, value=15, step=5, key="desk_horizon_slider")
    
    if desk_mode == "🔮 Live Desk + AI Future Prediction":
        with st.spinner(f"Fetching real-time quotes for {selected_asset_name}..."):
            try:
                df_live = yf.download(yf_symbol, period="6mo", interval="1d")
                if isinstance(df_live.columns, pd.MultiIndex):
                    df_live.columns = df_live.columns.get_level_values(0)
            except Exception as e:
                st.error(f"Failed to fetch market data: {e}")
                df_live = pd.DataFrame()
                
        if not df_live.empty and 'Close' in df_live.columns:
            prices = df_live['Close'].values.tolist()
            dates = df_live.index.strftime('%Y-%m-%d').tolist()
            n = len(prices)
            
            # 1. Technical Indicators
            sma20 = df_live['Close'].rolling(window=20).mean()
            sma50 = df_live['Close'].rolling(window=50).mean()
            
            # 2. Linear Regression & Sentiment Drift Forecast
            lookback = min(40, n)
            y = np.array(prices[-lookback:])
            x = np.arange(lookback)
            slope, intercept = np.polyfit(x, y, 1)
            
            # Incorporate text sentiment index from current state
            curr_sentiment = st.session_state.market_history[-1]["sentiment"] if st.session_state.market_history else 0.0
            sentiment_slope = curr_sentiment * (prices[-1] * 0.003)
            blended_slope = (0.7 * slope) + (0.3 * sentiment_slope)
            
            # Generate future prediction y values
            pred_steps = np.arange(1, desk_horizon + 1)
            pred_prices = prices[-1] + blended_slope * pred_steps
            
            # Confidence bounds calculation
            residuals = y - (slope * x + intercept)
            std_err = np.std(residuals) if len(residuals) > 1 else prices[-1] * 0.01
            lower_bounds = pred_prices - 1.96 * std_err * np.sqrt(pred_steps)
            upper_bounds = pred_prices + 1.96 * std_err * np.sqrt(pred_steps)
            
            # Future Dates
            last_date = df_live.index[-1]
            future_dates = []
            curr_date = last_date
            while len(future_dates) < desk_horizon:
                curr_date += datetime.timedelta(days=1)
                if curr_date.weekday() < 5:
                    future_dates.append(curr_date.strftime('%Y-%m-%d'))
                    
            # 3. Build Plotly Financial Candlestick + AI Prediction Overlay Chart
            fig_desk = go.Figure()
            
            # Candlesticks
            fig_desk.add_trace(go.Candlestick(
                x=dates[-60:],
                open=df_live['Open'].values[-60:],
                high=df_live['High'].values[-60:],
                low=df_live['Low'].values[-60:],
                close=df_live['Close'].values[-60:],
                name="Live Candlesticks",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444"
            ))
            
            # SMA 20 Line
            fig_desk.add_trace(go.Scatter(
                x=dates[-60:],
                y=sma20.values[-60:],
                name="SMA 20",
                line=dict(color="#38bdf8", width=1.5)
            ))
            
            # SMA 50 Line
            fig_desk.add_trace(go.Scatter(
                x=dates[-60:],
                y=sma50.values[-60:],
                name="SMA 50",
                line=dict(color="#a78bfa", width=1.5)
            ))
            
            # AI Future Prediction Line
            fig_desk.add_trace(go.Scatter(
                x=[dates[-1]] + future_dates,
                y=[prices[-1]] + pred_prices.tolist(),
                name="AI Future Forecast",
                line=dict(color="#f59e0b", width=4, dash="dash")
            ))
            
            # Upper Confidence Bound
            fig_desk.add_trace(go.Scatter(
                x=[dates[-1]] + future_dates,
                y=[prices[-1]] + upper_bounds.tolist(),
                name="Upper Bound",
                line=dict(color="rgba(245, 158, 11, 0.0)", width=0),
                showlegend=False
            ))
            
            # Lower Confidence Bound + Fill
            fig_desk.add_trace(go.Scatter(
                x=[dates[-1]] + future_dates,
                y=[prices[-1]] + lower_bounds.tolist(),
                name="95% Confidence Band",
                fill="tonexty",
                fillcolor="rgba(245, 158, 11, 0.15)",
                line=dict(color="rgba(245, 158, 11, 0.0)", width=0),
                showlegend=True
            ))
            
            # Vertical NOW Divider Line
            fig_desk.add_vline(
                x=dates[-1],
                line_width=2,
                line_dash="dot",
                line_color="#38bdf8",
                annotation_text="📍 LIVE NOW",
                annotation_position="top left",
                annotation_font=dict(color="#38bdf8", size=12, family="Inter")
            )
            
            fig_desk.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=680,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
            )
            fig_desk.update_xaxes(gridcolor="#1e293b")
            fig_desk.update_yaxes(gridcolor="#1e293b")
            
            st.plotly_chart(fig_desk, use_container_width=True)
            
            # Summary Intelligence Card
            curr_price = prices[-1]
            target_price = pred_prices[-1]
            ret_pct = ((target_price - curr_price) / curr_price) * 100
            
            signal_color = "#10b981" if ret_pct >= 0 else "#ef4444"
            signal_label = "BULLISH FORECAST 🚀" if ret_pct >= 0 else "BEARISH FORECAST ⚠️"
            
            st.markdown(f"""
            <div style="background-color: rgba(15, 23, 42, 0.6); border: 1px solid #1e293b; padding: 14px 20px; border-radius: 12px; margin-top: -10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">Current Price:</span> <b style="font-size: 1.1rem; color: #ffffff;">{curr_price:,.2f}</b>
                    &nbsp;&nbsp;|&nbsp;&nbsp;
                    <span style="font-size: 0.85rem; color: #94a3b8; font-weight: 600;">AI Target ({desk_horizon}D):</span> <b style="font-size: 1.1rem; color: {signal_color};">{target_price:,.2f} ({ret_pct:+.2f}%)</b>
                </div>
                <div>
                    <span style="background-color: {signal_color}20; color: {signal_color}; border: 1px solid {signal_color}40; padding: 4px 12px; border-radius: 9999px; font-weight: 700; font-size: 0.85rem;">{signal_label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    else: # TradingView Widget Mode
        tradingview_html = f"""
        <!-- TradingView Widget BEGIN -->
        <div class="tradingview-widget-container" style="height:680px;width:100%;border-radius:12px;overflow:hidden;border:1px solid #1e293b;box-shadow:0 4px 6px -1px rgba(0,0,0,0.5);">
          <div id="tradingview_advanced_chart" style="height:100%;width:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({{
            "width": "100%",
            "height": "100%",
            "symbol": "{tv_symbol}",
            "interval": "D",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#131722",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": true,
            "container_id": "tradingview_advanced_chart",
            "studies": [
              "RSI@tv-basicstudies",
              "MASimple@tv-basicstudies"
            ],
            "backgroundColor": "rgba(19, 23, 34, 1)",
            "gridColor": "rgba(30, 41, 59, 1)"
          }});
          </script>
        </div>
        <!-- TradingView Widget END -->
        """
        components.html(tradingview_html, height=685)
        
    st.divider()

    
    # B. Interactive Graphs & Multi-Horizon Future Market Forecasting
    st.markdown("#### 🔮 Multi-Horizon Future Market Prediction & Narrative Trajectory")
    st.markdown("Synthesizes textual sentiment index, historical momentum, and quantitative volatility to project future market levels and narrative shifts.")
    
    c_f1, c_f2 = st.columns([2, 1])
    with c_f1:
        horizon_mode = st.selectbox(
            "Select Future Forecast Horizon:",
            options=["⚡ Intraday Hours (+24H)", "📅 7-Day Market Horizon (+7D)", "🔮 30-Day Outlook (+30D)"],
            index=0,
            key="horizon_mode_select"
        )
        
    df_history = pd.DataFrame(st.session_state.market_history)
    forecast_data = narrative_detector.predict_future_market_trajectory(
        st.session_state.market_history, 
        horizon_mode=horizon_mode
    )
    df_forecast = pd.DataFrame(forecast_data) if forecast_data else pd.DataFrame()
    
    g_col1, g_col2 = st.columns(2)
    
    with g_col1:
        st.markdown("##### 📈 Historical & Future Market Valuation Path")
        st.markdown("Historical trend seamlessly connecting to future price prediction and confidence bounds.")
        
        fig = go.Figure()
        
        # 1. Historical Sentiment Index
        fig.add_trace(go.Scatter(
            x=df_history["time"], 
            y=df_history["sentiment"],
            name="Hist Sentiment",
            line=dict(color="#38bdf8", width=2, dash='dot'),
            yaxis="y1"
        ))
        
        # 2. Historical Nifty 50
        fig.add_trace(go.Scatter(
            x=df_history["time"], 
            y=df_history["nifty"],
            name="Hist Nifty 50",
            line=dict(color="#10b981", width=3.5),
            yaxis="y2"
        ))
        
        if not df_forecast.empty:
            # Combine connecting point from last historical record to first forecast record
            last_time = df_history["time"].iloc[-1]
            last_nifty = df_history["nifty"].iloc[-1]
            last_sent = df_history["sentiment"].iloc[-1]
            
            f_times = [last_time] + list(df_forecast["time"])
            f_nifty = [last_nifty] + list(df_forecast["nifty"])
            f_sent = [last_sent] + list(df_forecast["sentiment"])
            f_upper = [last_nifty] + list(df_forecast["nifty_upper"])
            f_lower = [last_nifty] + list(df_forecast["nifty_lower"])
            
            # 3. Forecast Nifty Line
            fig.add_trace(go.Scatter(
                x=f_times,
                y=f_nifty,
                name="Predicted Nifty 50",
                line=dict(color="#f59e0b", width=3.5, dash='dash'),
                yaxis="y2"
            ))
            
            # 4. Upper & Lower Confidence Fill Band
            fig.add_trace(go.Scatter(
                x=f_times + f_times[::-1],
                y=f_upper + f_lower[::-1],
                fill='toself',
                fillcolor='rgba(245, 158, 11, 0.12)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name="95% Confidence Band",
                yaxis="y2"
            ))

            
            # 5. Forecast Sentiment Line
            fig.add_trace(go.Scatter(
                x=f_times,
                y=f_sent,
                name="Predicted Sentiment",
                line=dict(color="#c084fc", width=2, dash='dot'),
                yaxis="y1"
            ))
            
        fig.update_layout(
            yaxis=dict(title=dict(text="Sentiment Index (-1.0 to +1.0)", font=dict(color="#38bdf8")), tickfont=dict(color="#38bdf8")),
            yaxis2=dict(title=dict(text="Nifty 50 Price", font=dict(color="#10b981")), tickfont=dict(color="#10b981"), anchor="x", overlaying="y", side="right"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#94a3b8"),
            height=520
        )
        fig.update_xaxes(gridcolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)
        
    with g_col2:
        st.markdown("##### 🔄 Future Narrative Shift Phase Timeline")
        st.markdown("Step plot forecasting transition states across historical and future horizons.")
        
        phase_map = {"Panic": 0, "Fear": 1, "Neutral": 2, "Optimistic": 3}
        df_history["phase_numeric"] = df_history["phase"].map(phase_map)
        df_history["Type"] = "Historical"
        
        combined_df = df_history[["time", "phase_numeric", "Type"]].copy()
        
        if not df_forecast.empty:
            df_forecast_copy = df_forecast.copy()
            df_forecast_copy["phase_numeric"] = df_forecast_copy["phase"].map(phase_map)
            df_forecast_copy["Type"] = "Forecast"
            combined_df = pd.concat([combined_df, df_forecast_copy[["time", "phase_numeric", "Type"]]], ignore_index=True)
            
        fig_step = px.line(
            combined_df, 
            x="time", 
            y="phase_numeric",
            color="Type",
            color_discrete_map={"Historical": "#818cf8", "Forecast": "#f59e0b"},
            line_shape="hv",
            markers=True
        )
        fig_step.update_traces(marker=dict(size=8))
        
        fig_step.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            font=dict(color="#94a3b8"),
            height=520,
            yaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["🚨 Panic", "⚠️ Fear", "⚖️ Neutral", "🚀 Optimistic"],
                title="Narrative Phase"
            ),
            xaxis=dict(title="Time Progression (Past ➔ Future)")
        )
        fig_step.update_xaxes(gridcolor="#1e293b")
        fig_step.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_step, use_container_width=True)
        
    # Display Future Market Intelligence Summary Card
    if not df_forecast.empty:
        last_hist_nifty = df_history["nifty"].iloc[-1]
        last_hist_sensex = df_history["sensex"].iloc[-1]
        final_target_nifty = df_forecast["nifty"].iloc[-1]
        final_target_sensex = df_forecast["sensex"].iloc[-1]
        final_phase = df_forecast["phase"].iloc[-1]
        
        nifty_diff_pct = ((final_target_nifty - last_hist_nifty) / last_hist_nifty) * 100
        sensex_diff_pct = ((final_target_sensex - last_hist_sensex) / last_hist_sensex) * 100
        
        card_color = "#10b981" if nifty_diff_pct >= 0 else "#ef4444"
        card_bg = "rgba(16, 185, 129, 0.08)" if nifty_diff_pct >= 0 else "rgba(239, 68, 68, 0.08)"
        direction_label = "BULLISH EXPANSION 🚀" if nifty_diff_pct >= 0 else "BEARISH RETRENCHMENT ⚠️"
        
        st.markdown(f"""
        <div style="background-color: {card_bg}; border: 1px solid {card_color}; padding: 18px 24px; border-radius: 12px; margin: 15px 0 25px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                <div>
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em;">Future Market Target ({horizon_mode})</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {card_color}; margin-top: 4px;">
                        Nifty Target: {final_target_nifty:,.2f} ({nifty_diff_pct:+.2f}%) | Sensex: {final_target_sensex:,.2f} ({sensex_diff_pct:+.2f}%)
                    </div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; font-weight: 700;">Projected Narrative Phase</div>
                    <div style="font-size: 1.3rem; font-weight: 700; color: #ffffff; margin-top: 4px;">{final_phase} Mood ({direction_label})</div>
                </div>
            </div>
            <div style="font-size: 0.85rem; color: #cbd5e1; margin-top: 10px; border-top: 1px dashed rgba(255,255,255,0.1); padding-top: 8px;">
                💡 <b>Forecast Model Logic:</b> Synthesizing current sentiment index ({st.session_state.market_history[-1]['sentiment']}) and price momentum to project confidence interval bounds [{df_forecast['nifty_lower'].iloc[-1]:,.2f} to {df_forecast['nifty_upper'].iloc[-1]:,.2f}].
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()

    
    # C. Dynamic Live Headlines Stream Table
    st.markdown("#### 🚨 Captured Financial News Streams")
    st.markdown("Dynamic stream of financial headlines ingested into the engine. Toggle between the **Active 20-Headline Stream** and the **Full 300,000+ Kaggle Corpus** directly in this column.")

    # Top Control Toolbar
    c_mode, c_search, c_filter = st.columns([2, 2, 1])
    
    with c_mode:
        stream_view_mode = st.radio(
            "Stream View Mode:",
            options=["⚡ Active 20 Batch Stream", "📂 Full Ingested Kaggle Corpus"],
            horizontal=True,
            key="stream_view_mode_choice"
        )
        
    with c_search:
        search_query = st.text_input("Search Headlines:", placeholder="e.g., Apple, Nifty, inflation...", key="corpus_search")
        
    with c_filter:
        sentiment_filter = st.selectbox(
            "Filter Mood:", 
            options=["All", "Positive", "Negative", "Neutral", "Panic"],
            key="corpus_sentiment_filter"
        )

    def color_sentiment(val):
        if val == 'Positive':
            return 'background-color: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: bold; border-left: 3px solid #10b981;'
        elif val in ['Negative', 'Panic']:
            return 'background-color: rgba(239, 68, 68, 0.15); color: #f87171; font-weight: bold; border-left: 3px solid #ef4444;'
        else:
            return 'background-color: rgba(148, 163, 184, 0.1); color: #94a3b8; font-weight: bold;'

    if stream_view_mode == "⚡ Active 20 Batch Stream":
        grid_data = []
        for h in st.session_state.active_headlines_data:
            grid_data.append({
                "Timestamp": datetime.datetime.now().strftime("%H:%M"),
                "Financial News Headline": h["raw"],
                "VADER Sentiment": h["vader"]["Sentiment Label"],
                "ML LogReg Sentiment": h["lr"]["Sentiment Label"]
            })
        df_grid = pd.DataFrame(grid_data)

        # Apply filtering if user searched or filtered
        if sentiment_filter != "All":
            df_grid = df_grid[(df_grid["VADER Sentiment"] == sentiment_filter) | (df_grid["ML LogReg Sentiment"] == sentiment_filter)]
        if search_query:
            df_grid = df_grid[df_grid["Financial News Headline"].str.contains(search_query, case=False, na=False)]

        styled_grid = df_grid.style.map(color_sentiment, subset=['VADER Sentiment', 'ML LogReg Sentiment'])
        st.dataframe(styled_grid, use_container_width=True, height=350)
        
    else: # 📂 Full Ingested Kaggle Corpus
        # Build DataFrame of all headlines from data_loader.NEWS_POOL
        all_headlines = []
        for category, pool in data_loader.NEWS_POOL.items():
            for text in pool:
                all_headlines.append({
                    "Headline Text": text,
                    "Sentiment Label": category.capitalize()
                })
        df_all = pd.DataFrame(all_headlines)
        
        # Filter logic
        df_filtered = df_all
        if sentiment_filter != "All":
            df_filtered = df_filtered[df_filtered["Sentiment Label"] == sentiment_filter]
        if search_query:
            df_filtered = df_filtered[df_filtered["Headline Text"].str.contains(search_query, case=False, na=False)]
            
        st.markdown(f"Showing **{len(df_filtered):,}** matching headlines out of **{len(df_all):,}** total ingested Kaggle records.")
        
        # Pagination logic
        page_size = 100
        total_rows = len(df_filtered)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        
        if "corpus_page" not in st.session_state:
            st.session_state.corpus_page = 1
            
        current_filter_hash = f"{search_query}|{sentiment_filter}"
        if "last_corpus_filter_hash" not in st.session_state:
            st.session_state.last_corpus_filter_hash = current_filter_hash
            st.session_state.corpus_page = 1
        elif st.session_state.last_corpus_filter_hash != current_filter_hash:
            st.session_state.last_corpus_filter_hash = current_filter_hash
            st.session_state.corpus_page = 1
            
        if st.session_state.corpus_page > total_pages:
            st.session_state.corpus_page = total_pages
        if st.session_state.corpus_page < 1:
            st.session_state.corpus_page = 1
            
        col_prev, col_page, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("⬅️ Previous Page", disabled=(st.session_state.corpus_page <= 1), use_container_width=True, key="prev_btn"):
                st.session_state.corpus_page -= 1
                st.rerun()
                
        with col_page:
            selected_page = st.number_input(
                f"Page (1 to {total_pages:,}):",
                min_value=1,
                max_value=total_pages,
                value=int(st.session_state.corpus_page),
                step=1,
                key="corpus_page_input"
            )
            if selected_page != st.session_state.corpus_page:
                st.session_state.corpus_page = selected_page
                st.rerun()
                
        with col_next:
            if st.button("Next Page ➡️", disabled=(st.session_state.corpus_page >= total_pages), use_container_width=True, key="next_btn"):
                st.session_state.corpus_page += 1
                st.rerun()
                
        start_idx = (st.session_state.corpus_page - 1) * page_size
        end_idx = start_idx + page_size
        
        df_page = df_filtered.iloc[start_idx:end_idx]
        
        # Build 4-column dataframe matching active stream schema
        page_grid_data = []
        for _, row in df_page.iterrows():
            text = row["Headline Text"]
            sent = row["Sentiment Label"]
            page_grid_data.append({
                "Timestamp": "Ingested",
                "Financial News Headline": text,
                "VADER Sentiment": sent,
                "ML LogReg Sentiment": sent
            })
        df_render = pd.DataFrame(page_grid_data)

        styled_all = df_render.style.map(color_sentiment, subset=['VADER Sentiment', 'ML LogReg Sentiment'])
        st.dataframe(styled_all, use_container_width=True, height=350)




# ==========================================
# TAB 2: NLP PIPELINE INSPECTOR & TF-IDF
# ==========================================
with tab_nlp:
    st.markdown("### 🔬 NLP Preprocessing Pipeline Inspector")
    st.markdown("Select any active headline to observe the step-by-step mathematical and grammatical cleaning transitions.")
    
    # Select box to inspect any of the active headlines
    insp_index = st.selectbox(
        "Headline to inspect:",
        options=range(len(st.session_state.active_headlines_raw)),
        format_func=lambda x: f"[{x+1}] " + st.session_state.active_headlines_raw[x][:75] + "..."
    )
    
    target_headline = st.session_state.active_headlines_raw[insp_index]
    
    # Run pipeline with full steps
    steps = preprocess_headline_detailed(target_headline)
    
    # Side-by-side: Cleaning trace and TF-IDF weights
    nlp_col1, nlp_col2 = st.columns([3, 2])
    
    with nlp_col1:
        st.markdown("#### ⚙️ Pipeline Clean Trace")
        
        st.markdown("**1. Ingested Input (Raw)**")
        st.info(steps["original"])
        
        st.markdown("**2. Case Normalization (Lowercase)**")
        st.text_area("Lowercased:", steps["lowercase"], height=40, disabled=True, label_visibility="collapsed")
        
        st.markdown("**3. Noise & Punctuation Stripping (Preserving critical metrics)**")
        st.text_area("Noise Stripped:", steps["noise_removed"], height=40, disabled=True, label_visibility="collapsed")
        
        st.markdown("**4. Tokenization & Stopword Filter (NLTK Tokenizer + English Corpus)**")
        tokens_display = " | ".join(steps["stopwords_removed"])
        st.success(tokens_display if tokens_display else "All tokens removed by stopword filter.")
        
        st.markdown("**5. Part-of-Speech Tagging & WordNet Lemmatization**")
        pos_list = [f"{w} ({tag})" for w, tag in steps["pos_tags"]]
        lemmas_display = " -> ".join([f"{steps['stopwords_removed'][i]} [{steps['lemmatized'][i]}]" for i in range(len(steps['lemmatized']))])
        
        st.code(f"POS Tags:  {', '.join(pos_list)}\nLemmas:    {lemmas_display}", language="python")
        
        st.markdown("**6. Final Normalized Concept Vector**")
        st.code(steps["final_text"], language="bash")
        
    with nlp_col2:
        st.markdown("#### 🔢 TF-IDF Key Term Vector weights")
        st.markdown("Dynamic TF-IDF score represents semantic uniqueness against our training corpus.")
        
        # Get TF-IDF weights using our model's helper
        weights = model_instance.get_top_tfidf_features(target_headline, top_n=8)
        
        if weights:
            df_weights = pd.DataFrame(weights, columns=["Term / Lemma", "TF-IDF Weight"])
            fig_bar = px.bar(
                df_weights, 
                x="TF-IDF Weight", 
                y="Term / Lemma", 
                orientation="h",
                text="TF-IDF Weight"
            )
            fig_bar.update_traces(
                marker_color="#38bdf8", 
                texttemplate='%{text:.3f}', 
                textposition='outside'
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=10, b=20),
                font=dict(color="#94a3b8"),
                height=350,
                yaxis=dict(autorange="reversed")
            )
            fig_bar.update_xaxes(gridcolor="#1e293b", range=[0, 1.1])
            fig_bar.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("Empty TF-IDF vector! The normalized token list contains only unseen or highly frequent vocabulary terms.")

# ==========================================
# TAB 3: MACHINE LEARNING COCKPIT
# ==========================================
with tab_ml:
    st.markdown("### 🤖 Supervised Classifier Performance")
    st.markdown("Evaluate the mathematical models comparing VADER (lexical-lexicon method) vs Logistic Regression (supervised statistical method).")
    
    ml_col1, ml_col2 = st.columns(2)
    
    with ml_col1:
        st.markdown("#### 🎯 Logistic Regression Accuracy & Stats")
        
        # Display Accuracy Card
        st.markdown(f"""
        <div style="background-color: rgba(167, 139, 250, 0.1); border: 1px solid rgba(167, 139, 250, 0.3); padding: 20px; border-radius: 12px; margin-bottom: 20px; text-align: center;">
            <div style="font-size: 0.85rem; text-transform: uppercase; color: #a78bfa; font-weight: 700; letter-spacing: 0.05em;">Trained Model Testing Accuracy</div>
            <div style="font-size: 2.8rem; font-weight: 800; color: #a78bfa; margin: 5px 0;">{model_instance.accuracy:.2%}</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Split: 80% Train / 20% Hold-out Evaluation</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display Precision/Recall/F1 Classification Report
        rep_dict = model_instance.report
        report_data = []
        for key in ["Negative", "Neutral", "Positive"]:
            if key in rep_dict:
                report_data.append({
                    "Class Mood": key,
                    "Precision": f"{rep_dict[key]['precision']:.2f}",
                    "Recall": f"{rep_dict[key]['recall']:.2f}",
                    "F1-Score": f"{rep_dict[key]['f1-score']:.2f}"
                })
        df_rep = pd.DataFrame(report_data)
        st.markdown("**Classification Metrics Summary:**")
        st.table(df_rep)
        
    with ml_col2:
        st.markdown("#### 🗺️ Confusion Matrix (Plotly Heatmap)")
        st.markdown("Validation matrix tracking class predictions against actual labeled moods.")
        
        z = model_instance.confusion_matrix
        classes = ["Negative", "Neutral", "Positive"]
        
        fig_heat = ff = px.imshow(
            z,
            x=classes,
            y=classes,
            labels=dict(x="Predicted Labels", y="True Labels", color="Instances"),
            color_continuous_scale="Purples",
            text_auto=True
        )
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            font=dict(color="#94a3b8"),
            height=300
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        
    st.divider()
    
    # D. Head-to-Head Model Score Comparison Table
    st.markdown("#### 📊 Comparative Sentiment Scoring: Lexicon vs Supervised Classifier")
    st.markdown("Observe side-by-side probability metrics for positive, negative, and neutral scores across all active headlines.")
    
    compare_rows = []
    for h in st.session_state.active_headlines_data:
        compare_rows.append({
            "Headline Stream": h["raw"][:65] + "...",
            "VADER Pos": round(h["vader"]["Positive"], 2),
            "VADER Neg": round(h["vader"]["Negative"], 2),
            "VADER Neu": round(h["vader"]["Neutral"], 2),
            "VADER Label": h["vader"]["Sentiment Label"],
            "ML Pos": round(h["lr"]["Positive"], 2),
            "ML Neg": round(h["lr"]["Negative"], 2),
            "ML Neu": round(h["lr"]["Neutral"], 2),
            "ML Label": h["lr"]["Sentiment Label"]
        })
        
    df_compare = pd.DataFrame(compare_rows)
    
    st.dataframe(df_compare, use_container_width=True, height=270)

    st.divider()
    
    dist_data = {
        "Sentiment Label": ["Neutral", "Positive", "Negative", "Panic (Simulated)"],
        "Headline Count": [
            len(data_loader.NEWS_POOL["neutral"]),
            len(data_loader.NEWS_POOL["positive"]),
            len(data_loader.NEWS_POOL["negative"]),
            len(data_loader.NEWS_POOL["panic"])
        ]
    }
    df_dist = pd.DataFrame(dist_data)

    st.markdown("### 📊 Ingested Training Dataset Distribution")
    st.markdown(f"Class distribution of the **{sum(dist_data['Headline Count']):,}** financial news headlines loaded from the Kaggle dataset and custom project expansions.")

    
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        fig_pie = px.pie(
            df_dist, 
            values="Headline Count", 
            names="Sentiment Label",
            color="Sentiment Label",
            color_discrete_map={
                "Neutral": "#94a3b8",
                "Positive": "#10b981",
                "Negative": "#ef4444",
                "Panic (Simulated)": "#f97316"
            },
            hole=0.4
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#94a3b8"),
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with dist_col2:
        st.markdown("<div style='padding-top: 15px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        - **Total Dataset Size:** `{sum(dist_data['Headline Count']):,}` headlines
        - **Neutral Sentences:** `{dist_data['Headline Count'][0]:,}` (representing stable markets)
        - **Positive Sentences:** `{dist_data['Headline Count'][1]:,}` (representing bullish indicators)
        - **Negative Sentences:** `{dist_data['Headline Count'][2]:,}` (representing bearish indicators)
        - **Panic-inducing Headlines:** `{dist_data['Headline Count'][3]:,}` (used for testing tail-risk state transitions)
        
        *Note: The ML model is continuously re-trained on this aggregated corpus, enabling it to map incoming financial text streams to these baseline distributions.*
        """)

# ==========================================
# TAB 4: INTERACTIVE AI CHATBOT
# ==========================================
with tab_chatbot:
    st.markdown("### 💬 AI Narrative Shift Assistant")
    st.markdown("An interactive AI bot designed to explain current market sentiment, transition anomalies, and textual pipeline details dynamically.")
    
    # Clickable prompts for easy testing
    st.markdown("**💡 Quick Testing Prompts (Click to run):**")
    p_col1, p_col2, p_col3 = st.columns(3)
    
    if p_col1.button("“What is current market sentiment?”", key="main_q1_btn", use_container_width=True):
        handle_quick_prompt("What is current market sentiment?")
        st.rerun()
    if p_col2.button("“Why is the market narrative positive/negative?”", key="main_q2_btn", use_container_width=True):
        handle_quick_prompt("Why is the market narrative positive or negative?")
        st.rerun()
    if p_col3.button("“What narrative trend is detected?”", key="main_q3_btn", use_container_width=True):
        handle_quick_prompt("What narrative trend is detected?")
        st.rerun()
        
    # Chat container display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div class="user-bubble"><b>You:</b><br>{msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-bubble"><b>BuzzStreet Bot:</b><br>{msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Native Chat Input
    user_query = st.chat_input("Ask about sentiment indices, lemmatization pipelines, or market trends:", key="main_chat_input")
    if user_query:
        process_chat_query(user_query)
        st.rerun()

with tab_predict:
    st.markdown("### 🔮 Stock & Trade Future Predictor")
    st.markdown("Forecast future asset valuations and evaluate technical trade recommendations based on quantitative indicator synthesis.")
    
    ticker_options = {
        "Nifty 50 Index": "^NSEI",
        "BSE Sensex Index": "^BSESN",
        "Reliance Industries": "RELIANCE.NS",
        "Apple Inc.": "AAPL",
        "Tesla Inc.": "TSLA",
        "Microsoft Corp.": "MSFT"
    }
    
    col_t1, col_t2, col_t3 = st.columns([2, 1, 1])
    
    selected_label = col_t1.selectbox(
        "Select Active Asset:",
        options=list(ticker_options.keys()),
        index=list(ticker_options.values()).index(st.session_state.predict_ticker)
    )
    ticker = ticker_options[selected_label]
    st.session_state.predict_ticker = ticker
    
    period_options = {
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    selected_period_label = col_t2.selectbox(
        "Historical Period:",
        options=list(period_options.keys()),
        index=2 # 6 months
    )
    period = period_options[selected_period_label]
    
    forecast_horizon = col_t3.slider("Forecast Horizon (Days):", min_value=5, max_value=60, value=20, step=5)
    
    with st.spinner("Fetching historical quotes..."):
        try:
            df_hist = yf.download(ticker, period=period, interval="1d")
            if isinstance(df_hist.columns, pd.MultiIndex):
                df_hist.columns = df_hist.columns.get_level_values(0)
        except Exception as e:
            st.error(f"Failed to fetch market data: {e}")
            df_hist = pd.DataFrame()
            
    if not df_hist.empty:
        # Technical Indicator Calculations
        prices = df_hist['Close'].values.tolist()
        
        # Linear Regression Forecast
        n = len(prices)
        lookback = min(60, n)
        y = np.array(prices[-lookback:])
        x = np.arange(lookback)
        slope, intercept = np.polyfit(x, y, 1)
        
        # Recent momentum check (slope of last 10 days)
        recent_lookback = min(10, n)
        recent_y = np.array(prices[-recent_lookback:])
        recent_x = np.arange(recent_lookback)
        recent_slope, _ = np.polyfit(recent_x, recent_y, 1)
        
        # Blended slope
        blended_slope = 0.6 * slope + 0.4 * recent_slope
        
        # Generate forecast
        predicted_y = blended_slope * np.arange(1, forecast_horizon + 1) + prices[-1]
        
        # Confidence bands
        residuals = y - (slope * x + intercept)
        std_error = np.std(residuals)
        lower_bound = predicted_y - 1.96 * std_error * np.sqrt(np.arange(1, forecast_horizon + 1))
        upper_bound = predicted_y + 1.96 * std_error * np.sqrt(np.arange(1, forecast_horizon + 1))
        
        # Calculate 14-day RSI
        delta = df_hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
        
        # Calculate SMAs
        sma20 = df_hist['Close'].rolling(window=20).mean()
        sma50 = df_hist['Close'].rolling(window=50).mean()
        curr_sma20 = float(sma20.iloc[-1]) if not sma20.empty and not pd.isna(sma20.iloc[-1]) else prices[-1]
        curr_sma50 = float(sma50.iloc[-1]) if not sma50.empty and not pd.isna(sma50.iloc[-1]) else prices[-1]
        
        # Recommendation Engine
        signals = []
        reasons = []
        
        if current_rsi < 35:
            signals.append("BUY")
            reasons.append(f"RSI is oversold at {current_rsi:.1f}, indicating standard bullish bounce conditions.")
        elif current_rsi > 65:
            signals.append("SELL")
            reasons.append(f"RSI is overbought at {current_rsi:.1f}, indicating potential valuation consolidation.")
        else:
            signals.append("HOLD")
            reasons.append(f"RSI indicator is stable at {current_rsi:.1f}.")
            
        if curr_sma20 > curr_sma50:
            signals.append("BUY")
            reasons.append(f"SMA 20 ({curr_sma20:,.2f}) sits above SMA 50 ({curr_sma50:,.2f}), representing a bullish golden cross trend.")
        else:
            signals.append("SELL")
            reasons.append(f"SMA 20 ({curr_sma20:,.2f}) sits below SMA 50 ({curr_sma50:,.2f}), signaling standard death cross crossover.")
            
        if blended_slope > 0:
            signals.append("BUY")
            reasons.append(f"Quant forecast direction is positive (slope: {blended_slope:+.2f}).")
        else:
            signals.append("SELL")
            reasons.append(f"Quant forecast direction is negative (slope: {blended_slope:+.2f}).")
            
        # Overall Card Logic
        buy_count = signals.count("BUY")
        sell_count = signals.count("SELL")
        if buy_count >= 2:
            recommendation = "STRONG BUY"
            rec_color = "#10b981"
            rec_bg = "rgba(16, 185, 129, 0.08)"
            rec_border = "rgba(16, 185, 129, 0.2)"
            rec_emoji = "🚀"
        elif sell_count >= 2:
            recommendation = "STRONG SELL"
            rec_color = "#ef4444"
            rec_bg = "rgba(239, 68, 68, 0.08)"
            rec_border = "rgba(239, 68, 68, 0.2)"
            rec_emoji = "🚨"
        else:
            recommendation = "HOLD"
            rec_color = "#94a3b8"
            rec_bg = "rgba(148, 163, 184, 0.08)"
            rec_border = "rgba(148, 163, 184, 0.2)"
            rec_emoji = "⚖️"
            
        # Top Plotly Graph
        hist_dates = df_hist.index.strftime('%Y-%m-%d').tolist()
        last_date = df_hist.index[-1]
        future_dates = []
        curr_date = last_date
        while len(future_dates) < forecast_horizon:
            curr_date += datetime.timedelta(days=1)
            if curr_date.weekday() < 5:
                future_dates.append(curr_date.strftime('%Y-%m-%d'))
                
        fig_pred = go.Figure()
        
        # Historical
        fig_pred.add_trace(go.Scatter(
            x=hist_dates[-60:],
            y=prices[-60:],
            name="Historical Close",
            line=dict(color="#38bdf8", width=3)
        ))
        
        # Predicted
        fig_pred.add_trace(go.Scatter(
            x=[hist_dates[-1]] + future_dates,
            y=[prices[-1]] + predicted_y.tolist(),
            name="AI Forecast",
            line=dict(color="#a78bfa", width=3, dash="dash")
        ))
        
        # Upper Bound
        fig_pred.add_trace(go.Scatter(
            x=[hist_dates[-1]] + future_dates,
            y=[prices[-1]] + upper_bound.tolist(),
            name="Upper Range",
            line=dict(color="rgba(167, 139, 250, 0.0)", width=0),
            showlegend=False
        ))
        
        # Lower Bound
        fig_pred.add_trace(go.Scatter(
            x=[hist_dates[-1]] + future_dates,
            y=[prices[-1]] + lower_bound.tolist(),
            name="Uncertainty range",
            fill="tonexty",
            fillcolor="rgba(167, 139, 250, 0.06)",
            line=dict(color="rgba(167, 139, 250, 0.0)", width=0),
            showlegend=True
        ))
        
        fig_pred.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#94a3b8"),
            margin=dict(l=20, r=20, t=10, b=20),
            height=550,
            hovermode="x unified"
        )
        fig_pred.update_xaxes(gridcolor="#1e293b")
        fig_pred.update_yaxes(gridcolor="#1e293b")
        
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Recommendation Dashboard
        rec_col1, rec_col2 = st.columns([1, 1.5])
        with rec_col1:
            st.markdown(f"""
            <div style="background-color: {rec_bg}; border: 1px solid {rec_border}; padding: 22px; border-radius: 16px; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; font-weight: 700; letter-spacing: 0.05em; line-height: 1;">Trade Signal</div>
                <div style="font-size: 2.8rem; margin: 10px 0; font-weight: 800; color: {rec_color};">{rec_emoji} {recommendation}</div>
                <div style="font-size: 0.85rem; color: #cbd5e1; font-weight: 500;">Composite Technical Score</div>
            </div>
            """, unsafe_allow_html=True)
        with rec_col2:
            st.markdown("#### 🔍 Quantitative Analysis Summary")
            for r in reasons:
                st.markdown(f"- {r}")
                
    else:
        st.warning("No data found for this asset. Please verify the ticker prefix.")

with tab_compare:
    st.markdown("### 📊 Multi-Asset Comparative Charting")
    st.markdown("Compare the relative performance of multiple assets side-by-side or overlaid, normalized to a base starting price index.")
    
    ticker_options = {
        "Nifty 50 Index": "^NSEI",
        "BSE Sensex Index": "^BSESN",
        "S&P 500 ETF": "SPY",
        "Reliance Industries": "RELIANCE.NS",
        "Apple Inc.": "AAPL",
        "Tesla Inc.": "TSLA",
        "Microsoft Corp.": "MSFT"
    }
    
    # Pre-select based on state
    default_comparison = st.session_state.get("compare_tickers", ["^NSEI", "^BSESN"])
    selected_labels = []
    for label, val in ticker_options.items():
        if val in default_comparison:
            selected_labels.append(label)
            
    selected_assets_labels = st.multiselect(
        "Select assets to compare (Max 4):",
        options=list(ticker_options.keys()),
        default=selected_labels if selected_labels else list(ticker_options.keys())[:2]
    )
    
    selected_tickers = [ticker_options[label] for label in selected_assets_labels]
    st.session_state.compare_tickers = selected_tickers
    
    if len(selected_tickers) < 1:
        st.warning("Please select at least one asset to compare.")
    else:
        with st.spinner("Fetching asset data..."):
            try:
                comp_dfs = {}
                for t in selected_tickers:
                    df_t = yf.download(t, period="3mo", interval="1d")
                    if isinstance(df_t.columns, pd.MultiIndex):
                        df_t.columns = df_t.columns.get_level_values(0)
                    if not df_t.empty:
                        comp_dfs[t] = df_t
            except Exception as e:
                st.error(f"Error fetching comparative quotes: {e}")
                comp_dfs = {}
                
        if comp_dfs:
            fig_comp = go.Figure()
            for t, df_t in comp_dfs.items():
                closes = df_t['Close']
                if len(closes) > 0:
                    first_val = float(closes.iloc[0])
                    normalized_closes = (closes / first_val - 1) * 100
                    label = [k for k, v in ticker_options.items() if v == t][0]
                    
                    fig_comp.add_trace(go.Scatter(
                        x=df_t.index.strftime('%Y-%m-%d').tolist(),
                        y=normalized_closes.tolist(),
                        name=label,
                        line=dict(width=3)
                    ))
                    
            fig_comp.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94a3b8"),
                yaxis=dict(title="Cumulative Return (%)"),
                margin=dict(l=20, r=20, t=10, b=20),
                height=400,
                hovermode="x unified"
            )
            fig_comp.update_xaxes(gridcolor="#1e293b")
            fig_comp.update_yaxes(gridcolor="#1e293b")
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Side-by-side Cards
            st.markdown("#### ⚡ Real-Time Asset Performance Cards")
            cols = st.columns(len(comp_dfs))
            for idx, (t, df_t) in enumerate(comp_dfs.items()):
                label = [k for k, v in ticker_options.items() if v == t][0]
                closes = df_t['Close']
                last_val = float(closes.iloc[-1])
                prev_val = float(closes.iloc[-2]) if len(closes) > 1 else last_val
                change_pct = ((last_val - prev_val) / prev_val) * 100
                cum_ret = ((last_val / float(closes.iloc[0])) - 1) * 100
                
                with cols[idx]:
                    st.markdown(f"""
                    <div style="background-color: #131722; padding: 18px; border-radius: 12px; border: 1px solid #1e293b; text-align: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                        <div style="font-size: 0.75rem; text-transform: uppercase; color: #94a3b8; font-weight: 700;">{label}</div>
                        <div style="font-size: 1.8rem; font-weight: 800; margin: 5px 0;">{last_val:,.2f}</div>
                        <div style="font-size: 0.85rem; color: {'#10b981' if change_pct >= 0 else '#ef4444'}; font-weight: 700;">
                            {'▲' if change_pct >= 0 else '▼'} {change_pct:+.2f}% (Daily)
                        </div>
                        <div style="font-size: 0.8rem; color: #cbd5e1; margin-top: 8px;">
                            3-Month Return: <b style="color: {'#10b981' if cum_ret >= 0 else '#ef4444'};">{cum_ret:+.2f}%</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ==========================================
# 5. FOOTER & ANOMALY marquee
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🚨 Anomaly Detection: Suspicious Narratives Flagged")

fake_news_headlines = [
    "UNVERIFIED: CEO of major technology corporation rumored to flee country amid massive regulatory tax audit scandal.",
    "FLAGGED: Anonymous viral source claims Central Bank running dangerously low on domestic liquidity.",
    "SUSPICIOUS: Viral media posts allege immediate bankruptcy of major domestic commercial automobile manufacturers.",
    "UNVERIFIED: Reports of massive unregulated offshore banking accounts linked to top state-level hedge funds.",
    "FLAGGED: Deepfake financial audio suggests imminent complete supply chain collapse across energy sector."
]

ticker_text = " &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; ".join([f"⚠️ {news}" for news in fake_news_headlines])

marquee_html = f"""
<div class="marquee-container" style="width: 100%; overflow: hidden; background: #ea580c; color: white; padding: 12px 0; font-weight: bold; font-size: 1rem; border-radius: 8px; box-shadow: 0 0 10px rgba(234, 88, 12, 0.4); margin-top: 1rem;">
    <div class="marquee-content" style="display: inline-block; white-space: nowrap; animation: marquee 30s linear infinite;">
        {ticker_text} &nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp; {ticker_text}
    </div>
</div>
"""
st.markdown(marquee_html, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #475569; font-size: 0.85rem;'>BuzzStreet Engine running optimally. Developed for 50% Project Milestone Review (AI & NLP).</p>", unsafe_allow_html=True)

# ==========================================
# 6. FLOATING CHAT WIDGET (BUZZER)
# ==========================================
st.markdown("""
<style>
@keyframes buzzerPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(2, 132, 199, 0.7);
    }
    70% {
        box-shadow: 0 0 0 12px rgba(2, 132, 199, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(2, 132, 199, 0);
    }
}
/* Style the container of the floating buzzer button to be fixed */
.st-key-floating_buzzer_btn {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    z-index: 99999 !important;
}
.st-key-floating_buzzer_btn button {
    background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%) !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 50% !important;
    width: 60px !important;
    height: 60px !important;
    font-size: 1.5rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border: 2px solid rgba(255, 255, 255, 0.1) !important;
    animation: buzzerPulse 2s infinite !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    padding: 0 !important;
}
.st-key-floating_buzzer_btn button:hover {
    transform: scale(1.1) rotate(5deg) !important;
    animation: none !important;
    box-shadow: 0 0 20px rgba(2, 132, 199, 0.9) !important;
    background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# Render the button directly with the unique key
st.button("💬", key="floating_buzzer_btn")
if st.session_state.get("floating_buzzer_btn"):
    show_assistant_dialog()
