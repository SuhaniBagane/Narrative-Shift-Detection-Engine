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
    st.session_state.nifty_val = 22420.00
    st.session_state.sensex_val = 73910.00
    st.session_state.nifty_change = 0.00
    st.session_state.sensex_change = 0.00
    
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
    initial_raw = data_loader.generate_headlines(bias="neutral", count=8)
    st.session_state.active_headlines_raw = initial_raw
    st.session_state.active_headlines_data = [] # List to hold fully analyzed headlines
    
    st.session_state.open_chat = False
    
    # Initialize Chatbot history
    st.session_state.chat_history = [
        ("user", "Hello! What is this system capable of?"),
        ("bot", "👋 Welcome to **BuzzStreet Narrative Intelligence Assistant!**\n\nI can analyze raw financial texts, compute sentiment metrics using VADER and a trained **Logistic Regression classifier**, detect overall market narrative shifts, and answer questions. Try asking: \n\n*“What is current market sentiment?”* or *“Why is the market negative?”*")
    ]

# Helper function to trigger a full system refresh/re-evaluation
def evaluate_market_state(bias_param, is_refresh=False, reblend_only=False):
    vader_w = st.session_state.get("vader_weight", 0.50)
    
    if reblend_only and st.session_state.active_headlines_data:
        # Re-extract compound scores and predictions from cached data
        vader_compounds = [h["vader"]["compound"] for h in st.session_state.active_headlines_data]
        lr_sentiment_dicts = [h["lr"] for h in st.session_state.active_headlines_data]
    else:
        # 1. Generate new headlines
        raw_headlines = data_loader.generate_headlines(bias=bias_param, count=8)
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
    st.markdown('<div class="profile-role">Phase II: Intermediate NLP AI</div>', unsafe_allow_html=True)
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
tab_narrative, tab_nlp, tab_ml, tab_chatbot = st.tabs([
    "📈 Narrative Intelligence", 
    "🧠 NLP Cleaning & TF-IDF", 
    "🤖 ML Training Cockpit", 
    "💬 AI Narrative Chatbot"
])

# ==========================================
# TAB 1: NARRATIVE INTELLIGENCE
# ==========================================
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
    # ADVANCED LIVE GLOBAL MARKETS DESK (TradingView Embed)
    # ----------------------------------------------------
    st.markdown("### 📊 Live Global Markets Desk")
    st.markdown("Real-time streaming feeds directly from TradingView global exchanges. Toggle active assets to analyze visual price trends and volume actions:")
    
    symbol_dict = {
        "NSE Nifty 50 (Index)": "NSE:NIFTY",
        "BSE Sensex (Index)": "BOM:SENSEX",
        "S&P 500 ETF (SPY)": "AMEX:SPY",
        "Reliance Industries (RELIANCE)": "NSE:RELIANCE",
        "Apple Inc. (AAPL)": "NASDAQ:AAPL",
        "Tesla Inc. (TSLA)": "NASDAQ:TSLA"
    }
    
    tv_symbol_label = st.selectbox(
        "Select Active Live Asset Feed:",
        options=list(symbol_dict.keys()),
        index=0
    )
    tv_symbol = symbol_dict[tv_symbol_label]
    
    # Dynamic high-fidelity advanced chart widget configuration matching HSL dark cockpit style
    tradingview_html = f"""
    <!-- TradingView Widget BEGIN -->
    <div class="tradingview-widget-container" style="height:480px;width:100%;border-radius:12px;overflow:hidden;border:1px solid #1e293b;box-shadow:0 4px 6px -1px rgba(0,0,0,0.5);">
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
    
    components.html(tradingview_html, height=485)
    st.divider()
    
    # B. Interactive Graphs (Sentiment trend & Narrative timeline side-by-side)
    g_col1, g_col2 = st.columns(2)
    
    df_history = pd.DataFrame(st.session_state.market_history)
    
    with g_col1:
        st.markdown("#### 📈 Composite Sentiment & Nifty 50 Trend")
        st.markdown("Dual axis view of textual sentiment driving actual index valuations over time.")
        
        fig = go.Figure()
        # Add sentiment line
        fig.add_trace(go.Scatter(
            x=df_history["time"], 
            y=df_history["sentiment"],
            name="Sentiment Index",
            line=dict(color="#38bdf8", width=3, dash='dot'),
            yaxis="y1"
        ))
        # Add Nifty line
        fig.add_trace(go.Scatter(
            x=df_history["time"], 
            y=df_history["nifty"],
            name="Nifty 50",
            line=dict(color="#10b981", width=4),
            yaxis="y2"
        ))
        
        # Configure layout with dual Y axes
        fig.update_layout(
            yaxis=dict(title=dict(text="Sentiment Index (-1.0 to +1.0)", font=dict(color="#38bdf8")), tickfont=dict(color="#38bdf8")),
            yaxis2=dict(title=dict(text="Nifty 50 Price", font=dict(color="#10b981")), tickfont=dict(color="#10b981"), anchor="x", overlaying="y", side="right"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            font=dict(color="#94a3b8"),
            height=320
        )
        fig.update_xaxes(gridcolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, use_container_width=True)
        
    with g_col2:
        st.markdown("#### 🔄 Narrative Shift Phase Timeline")
        st.markdown("Step plot demonstrating discrete categorical transitions between Optimistic, Neutral, Fear, and Panic phases.")
        
        # Map categorical phases to numeric values for plotting steps
        phase_map = {"Panic": 0, "Fear": 1, "Neutral": 2, "Optimistic": 3}
        df_history["phase_numeric"] = df_history["phase"].map(phase_map)
        
        fig_step = px.line(
            df_history, 
            x="time", 
            y="phase_numeric", 
            line_shape="hv",
            markers=True
        )
        fig_step.update_traces(line=dict(color="#a78bfa", width=3), marker=dict(size=8, color="#818cf8"))
        
        fig_step.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=10, b=20),
            font=dict(color="#94a3b8"),
            height=320,
            yaxis=dict(
                tickvals=[0, 1, 2, 3],
                ticktext=["🚨 Panic", "⚠️ Fear", "⚖️ Neutral", "🚀 Optimistic"],
                title="Narrative Phase"
            ),
            xaxis=dict(title="Simulated Time Ticks")
        )
        fig_step.update_xaxes(gridcolor="#1e293b")
        fig_step.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_step, use_container_width=True)
        
    st.divider()
    
    # C. Dynamic Live Headlines Stream Table
    st.markdown("#### 🚨 Captured Financial News Streams")
    st.markdown("The underlying dynamic stream of headlines ingested by the system. Use the sidebar controls to alter this stream.")
    
    grid_data = []
    for h in st.session_state.active_headlines_data:
        grid_data.append({
            "Timestamp": datetime.datetime.now().strftime("%H:%M"),
            "Financial News Headline": h["raw"],
            "VADER Sentiment": h["vader"]["Sentiment Label"],
            "ML LogReg Sentiment": h["lr"]["Sentiment Label"]
        })
    df_grid = pd.DataFrame(grid_data)
    
    def color_sentiment(val):
        if val == 'Positive':
            return 'background-color: rgba(16, 185, 129, 0.15); color: #34d399; font-weight: bold; border-left: 3px solid #10b981;'
        elif val == 'Negative':
            return 'background-color: rgba(239, 68, 68, 0.15); color: #f87171; font-weight: bold; border-left: 3px solid #ef4444;'
        else:
            return 'background-color: rgba(148, 163, 184, 0.1); color: #94a3b8; font-weight: bold;'
            
    styled_grid = df_grid.style.map(color_sentiment, subset=['VADER Sentiment', 'ML LogReg Sentiment'])
    st.dataframe(styled_grid, use_container_width=True, height=290)

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
    
    # Apply stylings
    def color_agree(row):
        is_agree = row['VADER Label'] == row['ML Label']
        color = 'background-color: rgba(16, 185, 129, 0.05); border-right: 2px solid #10b981;' if is_agree else 'background-color: rgba(239, 68, 68, 0.05); border-right: 2px solid #ef4444;'
        return [color] * len(row)
        
    st.dataframe(df_compare, use_container_width=True, height=270)

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
