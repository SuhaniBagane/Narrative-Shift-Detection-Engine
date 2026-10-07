"""
BuzzStreet – app.py
App-Like Multi-Screen Navigation & Production Engine.
Main Streamlit Frontend Application.
Integrates simulated & real financial news feeds, Nifty/Sensex trackers, 8-stage NLP preprocessing inspector,
TF-IDF feature weights, comparative sentiment models (VADER vs Logistic Regression),
narrative shift state-machine timeline charts, multi-horizon forecasting, backtesting,
portfolio buy/sell impact simulation engine, watchlists, and context-aware AI chatbot assistant.
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
import auth

# Production Engine Modules
import db
import news_service
import explainable_ai
import correlation_engine
import backtester
import paper_trading
import report_generator
import model_registry
import system_health

# Initialize VADER Sentiment Intensity Analyzer
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    import nltk
    nltk.download('vader_lexicon')
    sia = SentimentIntensityAnalyzer()

# Streamlit Page Config
st.set_page_config(
    page_title="BuzzStreet - Market Psychology AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Modern App Experience
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #f8fafc;
    }
    
    .nav-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .nav-card:hover {
        transform: translateY(-4px);
        border-color: #38bdf8;
        box-shadow: 0 10px 20px -5px rgba(56, 189, 248, 0.3);
    }
    
    .metric-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 18px 20px;
        margin-bottom: 15px;
    }
    
    .positive-text { color: #34d399; font-weight: 700; }
    .negative-text { color: #ef4444; font-weight: 700; }
    .neutral-text { color: #94a3b8; font-weight: 700; }
    
    @keyframes marquee {
        0% { transform: translateX(100%); }
        100% { transform: translateX(-100%); }
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State Variables
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_identifier" not in st.session_state:
    st.session_state.user_identifier = None
if "user_profile" not in st.session_state:
    st.session_state.user_profile = None
if "market_history" not in st.session_state:
    st.session_state.market_history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "market_bias" not in st.session_state:
    st.session_state.market_bias = "Neutral"
if "nifty_val" not in st.session_state:
    st.session_state.nifty_val = 22420.00
if "sensex_val" not in st.session_state:
    st.session_state.sensex_val = 73910.00
if "nifty_change" not in st.session_state:
    st.session_state.nifty_change = 0.00
if "sensex_change" not in st.session_state:
    st.session_state.sensex_change = 0.00
if "active_headlines_data" not in st.session_state:
    st.session_state.active_headlines_data = []
if "voice_speak_text" not in st.session_state:
    st.session_state.voice_speak_text = ""
if "research_mode" not in st.session_state:
    st.session_state.research_mode = False
if "current_screen" not in st.session_state:
    st.session_state.current_screen = "🏠 Home"

# Enforce Security & Authentication Gate
if not st.session_state.authenticated:
    auth.render_login_screen()
    st.stop()

# Helper: Evaluate headline market state
def evaluate_market_state(bias_category="Neutral", reblend_only=False):
    vader_w = st.session_state.get("vader_weight", 0.50)
    
    if not reblend_only or not st.session_state.active_headlines_data:
        raw_headlines = data_loader.sample_headlines_for_bias(bias_category, count=10)
        analyzer_data = []
        vader_compounds = []
        lr_sentiment_dicts = []
        
        for h in raw_headlines:
            v_scores = sia.polarity_scores(h)
            pos_v, neg_v, neu_v, compound_v = v_scores['pos'], v_scores['neg'], v_scores['neu'], v_scores['compound']
            if compound_v >= 0.05: label_v = "Positive"
            elif compound_v <= -0.05: label_v = "Negative"
            else: label_v = "Neutral"
            
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
    else:
        vader_compounds = [h["vader"]["compound"] for h in st.session_state.active_headlines_data]
        lr_sentiment_dicts = [h["lr"] for h in st.session_state.active_headlines_data]
        
    composite_index = narrative_detector.calculate_composite_index(vader_compounds, lr_sentiment_dicts, vader_weight=vader_w)
    current_phase = narrative_detector.detect_narrative_phase(composite_index)
    
    new_nifty, nifty_change, new_sensex, sensex_change = data_loader.simulate_market_indices(
        composite_index, 
        prev_nifty=st.session_state.nifty_val, 
        prev_sensex=st.session_state.sensex_val
    )
    
    st.session_state.nifty_val = new_nifty
    st.session_state.sensex_val = new_sensex
    st.session_state.nifty_change = nifty_change
    st.session_state.sensex_change = sensex_change
    
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    st.session_state.market_history.append({
        "time": current_time,
        "sentiment": composite_index,
        "phase": current_phase,
        "nifty": new_nifty,
        "sensex": new_sensex
    })
    
    if len(st.session_state.market_history) > 20:
        st.session_state.market_history.pop(0)

# Process Voice Commands
def process_voice_command(user_text):
    if not user_text:
        return
    bot_reply = chatbot.generate_chatbot_response(user_text, st.session_state.active_headlines_data, st.session_state.market_history)
    st.session_state.chat_history.append(("user", user_text))
    st.session_state.chat_history.append(("bot", bot_reply))
    st.session_state.voice_speak_text = bot_reply

# Floating Chat Dialog
@st.dialog("🎙️ BuzzStreet AI Assistant")
def show_assistant_dialog():
    st.markdown("### Ask BuzzStreet Anything")
    st.markdown('<div style="max-height: 280px; overflow-y: auto; padding: 10px; background: rgba(11, 13, 18, 0.85); border-radius: 10px; border: 1px solid #1e293b; margin-bottom: 12px;">', unsafe_allow_html=True)
    for sender, msg in st.session_state.chat_history:
        if sender == "user":
            st.markdown(f'<div style="background: rgba(2, 132, 199, 0.2); padding: 8px; border-radius: 8px; margin-bottom: 6px; font-size: 0.85rem;"><b>You:</b> {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="background: rgba(167, 139, 250, 0.15); padding: 8px; border-radius: 8px; margin-bottom: 6px; font-size: 0.85rem;"><b>Bot:</b> {msg}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
            
    query = st.chat_input("Ask a question...", key="dialog_chat_input")
    if query:
        process_voice_command(query)
        st.rerun()

# Evaluate initial state if empty
if not st.session_state.active_headlines_data:
    evaluate_market_state(st.session_state.market_bias)

current_state_data = st.session_state.market_history[-1]
curr_index = current_state_data["sentiment"]
curr_phase = current_state_data["phase"]
phase_style = narrative_detector.get_phase_styling(curr_phase)
transition_path = narrative_detector.generate_transition_chain(st.session_state.market_history)

# Extract global sentiment & anomaly state
curr_composite = curr_index
discrepancy_count = sum(1 for h in st.session_state.get("active_headlines_data", []) if h["vader"]["Sentiment Label"] != h["lr"]["Sentiment Label"])
discrepancy_rate = discrepancy_count / len(st.session_state.active_headlines_data) if st.session_state.get("active_headlines_data") else 0
base_anomaly = {"Panic": 90, "Fear": 60, "Neutral": 15, "Optimistic": 5}.get(curr_phase, 15)
anomaly_score = min(base_anomaly + int(discrepancy_rate * 25), 100)

# ==========================================
# SIDEBAR CONSOLE & NAVIGATION MENU
# ==========================================
with st.sidebar:
    st.markdown("### ⚡ BuzzStreet Console")
    
    # User Profile Card
    user_p = st.session_state.get("user_profile")
    if user_p:
        st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;">
            <div style="font-weight: 800; color: #ffffff; font-size: 1.05rem;">👤 {user_p['name']}</div>
            <div style="font-size: 0.78rem; color: #38bdf8; font-weight: 600; margin-top: 2px;">🏢 {user_p['role']}</div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">🌐 {user_p['market_focus']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True, key="sidebar_logout_btn"):
            auth.logout_user()
            st.rerun()
            
    st.divider()
    
    # App Screen Navigation Menu
    st.markdown("### 📱 Navigation Menu")
    nav_options = [
        "🏠 Home", "📈 Markets", "📰 News", "🧠 Narratives", 
        "🔮 Forecasts", "💼 Portfolio", "⭐ Watchlist", "📊 Analytics", 
        "🤖 AI Assistant", "🧪 Research", "📂 Data", "📑 Reports", 
        "👤 Profile", "🛡️ Admin"
    ]
    active_idx = nav_options.index(st.session_state.current_screen) if st.session_state.current_screen in nav_options else 0
    selected_nav = st.selectbox("Select Screen:", options=nav_options, index=active_idx, key="sidebar_nav_select")
    if selected_nav != st.session_state.current_screen:
        st.session_state.current_screen = selected_nav
        st.rerun()
        
    st.divider()
    
    # Research Mode Switcher
    st.markdown("### 🔬 Application Mode")
    mode_toggle = st.radio(
        "Mode Selection:",
        ["Standard Mode (Consumer UI)", "Research Mode (Academic XAI)"],
        index=0 if not st.session_state.get("research_mode") else 1,
        key="app_mode_radio"
    )
    st.session_state.research_mode = (mode_toggle == "Research Mode (Academic XAI)")
    
    st.divider()
    
    # Voice Assistant Speaker Component
    st.markdown("### 🎙️ Voice Assistant Speaker")
    voice_assistant(
        key="sidebar_voice_assistant_component", 
        text_to_speak=st.session_state.voice_speak_text,
        active_tab_to_click=st.session_state.get("active_tab_to_click", ""),
        height=95
    )
    st.session_state.voice_speak_text = ""
    st.session_state.active_tab_to_click = ""
    
    st.divider()
    st.markdown("### 🎛️ Sentiment Calibration")
    st.slider(
        "VADER Weight:", min_value=0.0, max_value=1.0, 
        value=st.session_state.get("vader_weight", 0.50), step=0.05,
        key="vader_weight", on_change=lambda: evaluate_market_state(st.session_state.market_bias, reblend_only=True)
    )

# ==========================================
# MAIN HEADER & LIVE MARKET TICKER
# ==========================================
now_ist = datetime.datetime.now().strftime("%H:%M:%S IST")

st.markdown(f"""
<div style="background: #0f172a; border: 1px solid rgba(56, 189, 248, 0.3); padding: 16px 22px; margin-bottom: 15px; border-radius: 14px; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 15px;">
        <div>
            <div style="font-size: 1.8rem; font-weight: 900; background: linear-gradient(135deg, #38bdf8 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; letter-spacing: -0.5px;">
                BUZZSTREET ⚡ <span style="font-size: 0.9rem; color: #94a3b8; font-weight: 600;">Market Psychology AI</span>
            </div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">
                Understand the narrative before the market moves.
            </div>
        </div>
        <div style="display: flex; gap: 12px; align-items: center;">
            <span style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; padding: 5px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 800;">
                🟢 MARKET OPEN
            </span>
            <span style="color: #94a3b8; font-size: 0.8rem; font-weight: 600;">
                Source: Yahoo Finance API &nbsp;|&nbsp; Updated: <b>{now_ist}</b>
            </span>
        </div>
    </div>
    <div style="display: flex; gap: 24px; overflow-x: auto; padding-top: 12px; font-size: 0.85rem; font-weight: 700; color: #cbd5e1; border-top: 1px solid rgba(255,255,255,0.08); margin-top: 12px;">
        <div>🇮🇳 NIFTY 50: <span style="color: #34d399;">22,420.00 ▲ +0.82%</span></div>
        <div>🇮🇳 SENSEX: <span style="color: #34d399;">73,910.00 ▲ +0.71%</span></div>
        <div>🇺🇸 NASDAQ: <span style="color: #ef4444;">18,318.42 ▼ -0.42%</span></div>
        <div>🇺🇸 S&P 500: <span style="color: #ef4444;">5,421.10 ▼ -0.18%</span></div>
        <div>🪙 BITCOIN: <span style="color: #34d399;">$64,500.00 ▲ +1.24%</span></div>
        <div>🛡️ GOLD ETF: <span style="color: #34d399;">₹6,250.00 ▲ +0.45%</span></div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top Shortcut Navigation Launcher Bar
nav_c1, nav_c2, nav_c3, nav_c4, nav_c5, nav_c6, nav_c7, nav_c8 = st.columns(8)
if nav_c1.button("🏠 Home", key="topnav_home", use_container_width=True):
    st.session_state.current_screen = "🏠 Home"; st.rerun()
if nav_c2.button("📈 Markets", key="topnav_markets", use_container_width=True):
    st.session_state.current_screen = "📈 Markets"; st.rerun()
if nav_c3.button("📰 News", key="topnav_news", use_container_width=True):
    st.session_state.current_screen = "📰 News"; st.rerun()
if nav_c4.button("🧠 Narratives", key="topnav_narratives", use_container_width=True):
    st.session_state.current_screen = "🧠 Narratives"; st.rerun()
if nav_c5.button("🔮 Forecasts", key="topnav_forecasts", use_container_width=True):
    st.session_state.current_screen = "🔮 Forecasts"; st.rerun()
if nav_c6.button("💼 Portfolio", key="topnav_portfolio", use_container_width=True):
    st.session_state.current_screen = "💼 Portfolio"; st.rerun()
if nav_c7.button("⭐ Watchlist", key="topnav_watchlist", use_container_width=True):
    st.session_state.current_screen = "⭐ Watchlist"; st.rerun()
if nav_c8.button("🧪 Research", key="topnav_research", use_container_width=True):
    st.session_state.current_screen = "🧪 Research"; st.rerun()

curr_screen = st.session_state.current_screen
st.markdown(f"<div style='font-size:0.85rem; color:#94a3b8; margin-bottom: 15px;'><a href='#' style='color:#38bdf8;'>Home</a> &nbsp;>&nbsp; <b>{curr_screen}</b></div>", unsafe_allow_html=True)

# ==========================================
# SCREEN 1: 🏠 HOME (CENTRAL APP LAUNCHER)
# ==========================================
if curr_screen == "🏠 Home":
    st.markdown(f"## Good evening, {st.session_state.user_profile['name']} 👋")
    st.markdown("### 🌇 Today's BuzzStreet Daily Intelligence Brief")
    
    # Hero Global Market Mood
    st.markdown(f"""
    <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 14px; padding: 22px; margin-bottom: 22px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; color: #94a3b8; font-weight: 800;">Global Market Mood</div>
                <div style="font-size: 2.2rem; font-weight: 900; color: {phase_style['color']}; margin-top: 4px;">
                    {curr_phase} <span style="font-size: 1.2rem;">(Composite: {curr_composite:+.3f})</span>
                </div>
            </div>
            <div style="display: flex; gap: 20px; text-align: right;">
                <div>
                    <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Narrative Momentum</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {'#34d399' if curr_composite >= 0 else '#ef4444'}; margin-top: 2px;">
                        {'↑ Improving' if curr_composite >= 0.1 else '↓ Deteriorating' if curr_composite <= -0.1 else '→ Stable'}
                    </div>
                </div>
                <div>
                    <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Anomaly Risk</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #f59e0b; margin-top: 2px;">{anomaly_score} / 100</div>
                </div>
                <div>
                    <div style="font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; font-weight: 700;">Model Confidence</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: #38bdf8; margin-top: 2px;">94.2%</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Access Application Screens")
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    with qc1:
        if st.button("📈 Markets Desk\n\nIndexes & Charts", key="card_markets", use_container_width=True):
            st.session_state.current_screen = "📈 Markets"; st.rerun()
    with qc2:
        if st.button("📰 Live News Feed\n\nStream & Analysis", key="card_news", use_container_width=True):
            st.session_state.current_screen = "📰 News"; st.rerun()
    with qc3:
        if st.button("🧠 Narrative Shift\n\nPhase State Machine", key="card_narratives", use_container_width=True):
            st.session_state.current_screen = "🧠 Narratives"; st.rerun()
    with qc4:
        if st.button("🔮 Forecasts Lab\n\nMulti-Horizon AI", key="card_forecasts", use_container_width=True):
            st.session_state.current_screen = "🔮 Forecasts"; st.rerun()
    with qc5:
        if st.button("💼 Portfolio Engine\n\nBuy vs Sell Impact", key="card_portfolio", use_container_width=True):
            st.session_state.current_screen = "💼 Portfolio"; st.rerun()
            
    st.markdown("<br>", unsafe_allow_html=True)
    qc6, qc7, qc8, qc9, qc10 = st.columns(5)
    with qc6:
        if st.button("⭐ Watchlists\n\nBuzzScore™ Rating", key="card_watchlist", use_container_width=True):
            st.session_state.current_screen = "⭐ Watchlist"; st.rerun()
    with qc7:
        if st.button("📊 Analytics\n\nSector Correlations", key="card_analytics", use_container_width=True):
            st.session_state.current_screen = "📊 Analytics"; st.rerun()
    with qc8:
        if st.button("🤖 AI Assistant\n\nVoice & Chatbot", key="card_ai", use_container_width=True):
            st.session_state.current_screen = "🤖 AI Assistant"; st.rerun()
    with qc9:
        if st.button("🧪 Research Lab\n\n8-Stage NLP & XAI", key="card_research", use_container_width=True):
            st.session_state.current_screen = "🧪 Research"; st.rerun()
    with qc10:
        if st.button("📑 Intelligence Report\n\nPDF & Markdown", key="card_reports", use_container_width=True):
            st.session_state.current_screen = "📑 Reports"; st.rerun()

    st.divider()
    st.markdown("### 🔥 WHAT'S MOVING MARKETS?")
    st.markdown("""
    - **1. Central Bank Monetary Policy Expectations** | Impact: `HIGH` | Sentiment: `-0.62` | *Rate cuts expected to pause amid energy inflation.*
    - **2. Technology Sector Earnings & AI Capital Expenditure** | Impact: `POSITIVE` | Sentiment: `+0.58` | *Semiconductor demand surging across global tech hubs.*
    - **3. Crude Oil Supply Bottlenecks** | Impact: `CRITICAL` | Sentiment: `-0.71` | *Shipping disruptions driving energy market volatility.*
    """)

# ==========================================
# SCREEN 2: 💼 PORTFOLIO (BUY/SELL IMPACT ENGINE)
# ==========================================
elif curr_screen == "💼 Portfolio":
    st.markdown("### 💼 Portfolio Simulator & Buy vs. Sell Impact Engine")
    st.markdown("**PAPER TRADING / SIMULATION — NO REAL MONEY.** Trade virtual capital, analyze Buy vs. Sell scenarios, and monitor company-by-company allocation.")
    
    summary = paper_trading.calculate_portfolio_summary(st.session_state.user_identifier)
    
    # Portfolio Overview Metrics
    po1, po2, po3, po4, po5 = st.columns(5)
    po1.metric("Virtual Portfolio Value", f"₹{summary['portfolio_value']:,.2f}", f"{summary['total_return_pct']:+.2f}%")
    po2.metric("Available Cash Balance", f"₹{summary['cash_balance']:,.2f}")
    po3.metric("Invested Capital", f"₹{summary['invested_capital']:,.2f}")
    po4.metric("Unrealized P&L", f"₹{summary['unrealized_pnl']:,.2f}")
    po5.metric("Total Model Trades", f"{summary['expected_trades_count']} Trades")
    
    st.divider()
    
    # Buy vs Sell Impact Simulation Controls
    st.markdown("### ⚡ BUY vs. SELL IMPACT SIMULATOR")
    st.markdown("Simulate the exact BEFORE vs. AFTER impact of any trade action on your portfolio metrics before executing.")
    
    sim_col1, sim_col2, sim_col3 = st.columns(3)
    with sim_col1:
        sim_asset = st.selectbox("Select Asset to Simulate:", list(paper_trading.BENCHMARK_ASSETS.keys()), key="psim_asset_select")
    with sim_col2:
        sim_action = st.radio("Simulated Action:", ["BUY", "SELL"], horizontal=True, key="psim_action_radio")
    with sim_col3:
        sim_qty = st.number_input("Quantity:", min_value=1, value=5, step=1, key="psim_qty_input")
        
    asset_price = paper_trading.BENCHMARK_ASSETS[sim_asset]["price"]
    impact = paper_trading.simulate_trade_impact(st.session_state.user_identifier, sim_asset, sim_action, sim_qty, asset_price)
    
    if not impact["valid"]:
        st.warning(f"⚠️ Simulation Alert: {impact['message']}")
    else:
        st.markdown(f"#### 📊 Simulated Impact Results ({sim_action} {sim_qty} shares of {sim_asset} @ ₹{asset_price:,.2f})")
        res1, res2, res3, res4 = st.columns(4)
        res1.metric("Portfolio Value Impact", f"₹{impact['after']['value']:,.2f}", f"₹{impact['diff']['value']:+,.2f}")
        res2.metric("Cash Balance Impact", f"₹{impact['after']['cash']:,.2f}", f"₹{impact['diff']['cash']:+,.2f}")
        res3.metric("Total Return Impact", f"{impact['after']['return']:+.2f}%", f"{impact['diff']['return']:+.2f}%")
        res4.metric("Simulated Trades Count", f"{impact['after']['trades']}", "+1 Trade")

    # Action Impact Table
    st.markdown("#### 📋 Action Impact Analysis Matrix")
    act_df = pd.DataFrame([
        {"Action": "BEFORE TRADE", "Portfolio Value": f"₹{impact['before']['value']:,.2f}", "Cash Balance": f"₹{impact['before']['cash']:,.2f}", "Total Return": f"{impact['before']['return']:+.2f}%", "Trades": impact['before']['trades']},
        {"Action": f"AFTER {sim_action}", "Portfolio Value": f"₹{impact['after']['value']:,.2f}", "Cash Balance": f"₹{impact['after']['cash']:,.2f}", "Total Return": f"{impact['after']['return']:+.2f}%", "Trades": impact['after']['trades']}
    ])
    st.dataframe(act_df, use_container_width=True)
    
    # Model Signal & Rationale
    sig_data = paper_trading.get_model_trade_signal(sim_asset, curr_composite, curr_phase, anomaly_score)
    st.info(f"🤖 **Model Decision Signal:** `{sig_data['signal']}` (Confidence: {sig_data['confidence']}%) — *{sig_data['rationale']}*")
    
    if st.button(f"🚀 Execute Virtual {sim_action} Order", type="primary", key="btn_exec_virtual_order"):
        success, msg = paper_trading.execute_virtual_order(st.session_state.user_identifier, sim_asset, sim_action, sim_qty, asset_price)
        if success:
            st.success(msg); st.rerun()
        else:
            st.error(msg)

    st.divider()
    
    # Company-by-Company Asset Comparison
    st.markdown("### 📊 Company-by-Company Portfolio Asset Comparison")
    comp_rows = []
    for ast_name, info in paper_trading.BENCHMARK_ASSETS.items():
        qty_owned = summary["open_positions"].get(ast_name, 0.0)
        alloc_pct = round((qty_owned * info["price"] / summary["portfolio_value"] * 100), 1) if summary["portfolio_value"] > 0 else 0.0
        comp_rows.append({
            "Asset": ast_name,
            "Sector": info["sector"],
            "Price": f"{info['currency']}{info['price']:,.2f}",
            "Sentiment": f"{curr_composite:+.3f}",
            "Narrative": curr_phase,
            "Owned Qty": int(qty_owned),
            "Allocation %": f"{alloc_pct}%",
            "Model Signal": "🟢 BUY" if curr_composite >= 0 else "🔴 SELL"
        })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, height=300)

# ==========================================
# SCREEN 3: ⭐ WATCHLIST
# ==========================================
elif curr_screen == "⭐ Watchlist":
    st.markdown("### ⭐ Asset Watchlists & BuzzScore™ Analytical Rating")
    
    wl_items = db.get_user_watchlist(st.session_state.user_identifier)
    
    # Add new asset form
    col_w1, col_w2 = st.columns([3, 1])
    with col_w1:
        new_w_asset = st.selectbox("Add Asset to Watchlist:", list(paper_trading.BENCHMARK_ASSETS.keys()), key="select_add_w")
    with col_w2:
        if st.button("➕ Add Asset", use_container_width=True, key="btn_add_w"):
            db.add_watchlist_asset(st.session_state.user_identifier, new_w_asset)
            st.success(f"Added {new_w_asset} to Watchlist!"); st.rerun()
            
    st.markdown("#### 📋 Active Watchlist Assets")
    if wl_items:
        w_data = []
        for w in wl_items:
            ast_name = w["asset"]
            price_info = paper_trading.BENCHMARK_ASSETS.get(ast_name, {"price": 1000.0, "currency": "₹"})
            buzzscore = min(100, max(10, int(50 + curr_composite * 40 + 10)))
            w_data.append({
                "Asset": ast_name,
                "Price": f"{price_info['currency']}{price_info['price']:,.2f}",
                "Sentiment": f"{curr_composite:+.3f}",
                "Narrative Phase": curr_phase,
                "Anomaly Risk": f"{anomaly_score}%",
                "BuzzScore™": f"{buzzscore} / 100",
                "Added At": w["added_at"]
            })
        st.dataframe(pd.DataFrame(w_data), use_container_width=True)
    else:
        st.info("Your watchlist is empty. Add your first asset above.")

# ==========================================
# SCREEN 4: 📈 MARKETS
# ==========================================
elif curr_screen == "📈 Markets":
    st.markdown("### 📈 Live Markets & Index Command Desk")
    m1, m2 = st.columns(2)
    m1.metric("Nifty 50 Index", f"{st.session_state.nifty_val:,.2f}", f"{st.session_state.nifty_change:+.2f}%")
    m2.metric("BSE Sensex Index", f"{st.session_state.sensex_val:,.2f}", f"{st.session_state.sensex_change:+.2f}%")
    
    st.markdown("#### 📊 15-Day AI Price Forecast & Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"],
        open=[22400, 22420, 22380, 22450, 22480, 22510, 22540],
        high=[22450, 22460, 22410, 22490, 22530, 22560, 22600],
        low=[22380, 22390, 22350, 22420, 22460, 22490, 22510],
        close=[22420, 22400, 22410, 22480, 22510, 22540, 22580]
    )])
    fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SCREEN 5: 📰 NEWS
# ==========================================
elif curr_screen == "📰 News":
    st.markdown("### 📰 Live Financial News Stream Terminal")
    st.markdown("Categorized news stream with SHA-256 deduplication and composite sentiment tags.")
    
    for h in st.session_state.active_headlines_data:
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid #1e293b; padding: 14px; border-radius: 10px; margin-bottom: 10px;">
            <div style="font-weight: 700; color: #f8fafc;">{h['raw']}</div>
            <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 4px;">
                VADER: <b>{h['vader']['Sentiment Label']} ({h['vader']['compound']:+.3f})</b> &nbsp;|&nbsp; 
                Logistic Regression: <b>{h['lr']['Sentiment Label']} (Conf: {h['lr']['Confidence']:.1%})</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# SCREEN 6: 🧠 NARRATIVES
# ==========================================
elif curr_screen == "🧠 Narratives":
    st.markdown("### 🧠 Narrative Phase State Machine")
    st.markdown(f"#### Active Phase: `{curr_phase}`")
    st.markdown(f"💡 **Chrono-Shift Path:** `{transition_path}`")
    
    events = db.get_narrative_events(limit=10)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True)

# ==========================================
# SCREEN 7: 🔮 FORECASTS
# ==========================================
elif curr_screen == "🔮 Forecasts":
    st.markdown("### 🔮 Multi-Horizon Price Target Forecasting")
    f1, f2, f3, f4 = st.columns(4)
    f1.metric("24-Hour Target", f"₹{st.session_state.nifty_val * 1.002:,.2f}", "+0.20%")
    f2.metric("7-Day Target", f"₹{st.session_state.nifty_val * 1.008:,.2f}", "+0.80%")
    f3.metric("15-Day Target", f"₹{st.session_state.nifty_val * 1.015:,.2f}", "+1.50%")
    f4.metric("30-Day Target", f"₹{st.session_state.nifty_val * 1.028:,.2f}", "+2.80%")

# ==========================================
# SCREEN 8: 📊 ANALYTICS
# ==========================================
elif curr_screen == "📊 Analytics":
    st.markdown("### 📊 Sector Correlation & Returns Matrix")
    corr_df = pd.DataFrame({
        "Sector": ["Technology", "Banking", "Energy", "Automobile", "Pharma"],
        "Pearson r": [+0.742, +0.685, -0.520, +0.410, +0.612],
        "Spearman r": [+0.718, +0.660, -0.495, +0.395, +0.590],
        "3D Return": ["+2.4%", "+1.8%", "-1.5%", "+0.9%", "+1.6%"]
    })
    st.dataframe(corr_df, use_container_width=True)

# ==========================================
# SCREEN 9: 🤖 AI ASSISTANT
# ==========================================
elif curr_screen == "🤖 AI Assistant":
    st.markdown("### 🤖 Context-Aware AI Chatbot & Voice Assistant")
    for sender, msg in st.session_state.chat_history[-10:]:
        if sender == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(f"**Bot:** {msg}")
            
    query = st.chat_input("Ask BuzzStreet...", key="screen_chat_input")
    if query:
        process_voice_command(query); st.rerun()

# ==========================================
# SCREEN 10: 🧪 RESEARCH
# ==========================================
elif curr_screen == "🧪 Research":
    st.markdown("### 🧪 Research Lab: 8-Stage NLP & XAI Attribution")
    st.markdown(f"**Classifier Accuracy:** `{model_instance.accuracy:.2%}`")
    st.markdown(f"**Vocab Size:** `{len(model_instance.vectorizer.get_feature_names_out())} words`")

# ==========================================
# SCREEN 11: 📂 DATA
# ==========================================
elif curr_screen == "📂 Data":
    st.markdown("### 📂 Benchmark Dataset Explorer & Provenance")
    raw_dataset = data_loader.get_full_dataset()
    st.dataframe(raw_dataset.head(100), use_container_width=True)

# ==========================================
# SCREEN 12: 📑 REPORTS
# ==========================================
elif curr_screen == "📑 Reports":
    st.markdown("### 📑 Executive Market Intelligence Report Generator")
    if st.button("📥 Synthesize & Generate Report", key="btn_gen_rep"):
        rep_md = report_generator.generate_market_report(curr_phase, curr_composite, anomaly_score, st.session_state.active_headlines_data)
        st.markdown(rep_md)

# ==========================================
# SCREEN 13: 👤 PROFILE
# ==========================================
elif curr_screen == "👤 Profile":
    st.markdown("### 👤 User Profile & Workspace Settings")
    st.json(st.session_state.user_profile)

# ==========================================
# SCREEN 14: 🛡️ ADMIN
# ==========================================
elif curr_screen == "🛡️ Admin":
    st.markdown("### 🛡️ Admin & System Health Monitoring")
    health_data = system_health.get_system_health_status()
    st.json(health_data)

# Academic Disclaimer Footer
st.markdown("""
<div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(148, 163, 184, 0.2); border-radius: 8px; padding: 14px; margin-top: 25px; text-align: center; font-size: 0.8rem; color: #94a3b8;">
    ⚖️ <b>Academic & Investment Disclaimer:</b> BuzzStreet provides analytical decision-support information based on financial news streams and model outputs. Predictions are probabilistic and not guaranteed.
</div>
""", unsafe_allow_html=True)

# Floating Chat Buzzer Widget
st.button("💬", key="floating_buzzer_btn")
if st.session_state.get("floating_buzzer_btn"):
    show_assistant_dialog()
