"""
BuzzStreet – chatbot.py
Context-Aware AI Chatbot Assistant.
Processes user queries regarding market sentiment, narrative shifts, and key drivers,
dynamically responding using real-time system context (headlines, scores, state history).
"""

import re
from ml_model import model_instance

def generate_chatbot_response(user_query, system_context):
    """
    Generates a context-aware response based on the user's question and the current
    state of the BuzzStreet engine.
    
    system_context expects a dictionary:
    {
        "sentiment_index": float,
        "narrative_phase": str,
        "transition_chain": str,
        "headlines": list of dicts (with 'raw', 'vader', 'lr', 'cleaned'),
        "nifty_val": float,
        "nifty_change": float,
        "model_accuracy": float
    }
    """
    query = user_query.lower().strip()
    
    sentiment_idx = system_context.get("sentiment_index", 0.0)
    phase = system_context.get("narrative_phase", "Neutral")
    chain = system_context.get("transition_chain", "No transition data yet.")
    headlines = system_context.get("headlines", [])
    nifty_val = system_context.get("nifty_val", 22400.0)
    nifty_change = system_context.get("nifty_change", 0.0)
    accuracy = system_context.get("model_accuracy", 0.85)
    
    # 1. GREETINGS & INTRO
    if any(k in query for k in ["hello", "hi", "hey", "greetings", "introduce yourself", "who are you"]):
        return (
            "👋 **Hello! I am the BuzzStreet Narrative Intelligence Assistant.**\n\n"
            "I analyze real-time financial narrative feeds, model predictions, and NLP cleaning pipelines. "
            "Here is what you can ask me:\n"
            "- 📊 **Market Sentiment:** *\"What is the current market mood?\"* or *\"Current sentiment?\"*\n"
            "- 🔍 **Market Drivers:** *\"Why is the market narrative positive/negative?\"* or *\"What is causing this?\"*\n"
            "- 📈 **Price Details:** *\"What is the Nifty value?\"* or *\"Show me stock indexes\"*\n"
            "- 🔄 **Narrative Trends:** *\"What narrative trend is detected?\"* or *\"Show the transition chain\"*\n"
            "- 🧠 **AI Architecture:** *\"How does the NLP pipeline work?\"* or *\"Tell me about the ML model\"*\n"
            "- 🚨 **Anomaly Status:** *\"What is the current anomaly risk score?\"*\n"
            "- 📰 **Active Streams:** *\"List the current headlines\"*"
        )

    # 2. STOCK PRICES & INDICES
    if any(k in query for k in ["nifty", "sensex", "price", "valuation", "value", "stock index", "index value", "market indices"]):
        return (
            f"### 📈 Live Index Valuations\n\n"
            f"- **Nifty 50 Index:** `{nifty_val:,.2f}` ({nifty_change:+.2f}% change)\n"
            f"- **BSE Sensex Index:** `{system_context.get('sensex_val', 73900.0):,.2f}` ({system_context.get('sensex_change', 0.0):+.2f}% change)\n\n"
            f"These indices are dynamically updated based on the **composite sentiment index** (currently **{sentiment_idx:+.3f}**)."
        )

    # 3. ANOMALY DETECTION & RISK
    if any(k in query for k in ["anomaly", "risk", "suspicious", "flagged", "threat"]):
        # Recalculate anomaly score
        discrepancy_count = sum(1 for h in headlines if h["vader"]["Sentiment Label"] != h["lr"]["Sentiment Label"])
        discrepancy_rate = discrepancy_count / len(headlines) if headlines else 0
        base_anomaly = {"Panic": 90, "Fear": 60, "Neutral": 15, "Optimistic": 5}.get(phase, 15)
        anomaly_score = min(base_anomaly + int(discrepancy_rate * 25), 100)
        
        status = "Stable (Low Risk)" if anomaly_score < 30 else "Moderate Volatility" if anomaly_score < 70 else "Systemic Anomaly Alert (High Risk)"
        emoji = "🟢" if anomaly_score < 30 else "🟡" if anomaly_score < 70 else "🔴"
        
        return (
            f"### 🚨 Systemic Anomaly Report\n\n"
            f"- **Risk Score:** `{anomaly_score}%` {emoji}\n"
            f"- **Alert Status:** **{status}**\n"
            f"- **Model Divergence:** Out of `{len(headlines)}` active headlines, VADER and Logistic Regression classifier disagree on `{discrepancy_count}` headlines ({discrepancy_rate:.1%} model divergence rate).\n\n"
            "High divergence rates combined with bearish narrative shifts trigger anomaly warnings in the marquee ticker at the bottom."
        )

    # 4. ACTIVE HEADLINES LIST
    if any(k in query for k in ["headlines", "headline", "news", "list headlines", "show news", "streams"]):
        response = f"### 📰 Active Captured News Stream ({len(headlines)} items)\n\n"
        for i, h in enumerate(headlines[:6]):
            response += f"{i+1}. **\"{h['raw']}\"**\n"
            response += f"   - *VADER label:* `{h['vader']['Sentiment Label']}` | *ML Classifier label:* `{h['lr']['Sentiment Label']}`\n"
        return response

    # 5. CURRENT MARKET SENTIMENT QUERY
    if any(k in query for k in ["sentiment", "mood", "feeling", "status", "current state", "market state", "how is the market", "how are the markets", "index score"]):
        sentiment_word = "bullish" if sentiment_idx > 0.1 else "bearish" if sentiment_idx < -0.1 else "flat/stable"
        response = (
            f"### 📊 Current Market Sentiment Analysis\n\n"
            f"The **composite sentiment index** is currently at **{sentiment_idx:+.3f}**, which represents a **{phase}** narrative phase.\n\n"
            f"Overall, the market mood is leaning **{sentiment_word}**. "
        )
        if phase == "Optimistic":
            response += "Investors are actively buying on growth narratives and strong corporate balance sheets."
        elif phase == "Neutral":
            response += "The market is in consolidation. Buyers and sellers are in equilibrium awaiting key policy indicators."
        elif phase == "Fear":
            response += "Cautious sentiment prevails. Hedging activity has increased as macro worries start creeping in."
        elif phase == "Panic":
            response += "Severe risk-off sentiment! Investors are dumping equities as high-risk anomalies and liquidity flags trigger."
            
        response += f"\n\n*Nifty 50* is trading at **{nifty_val:,.2f}** ({nifty_change:+.2f}% change in the last live tick)."
        return response
        
    # 6. WHY IS THE MARKET NEGATIVE / POSITIVE
    if any(k in query for k in ["why", "reason", "driver", "cause", "explain the narrative", "what happened", "behind", "source", "factors", "factor"]):
        if sentiment_idx < 0:
            # Look for negative headlines
            neg_headlines = [h for h in headlines if h["vader"]["Sentiment Label"] == "Negative" or h["lr"]["Sentiment Label"] == "Negative"]
            if neg_headlines:
                top_neg = neg_headlines[:3]
                response = (
                    f"### 🔍 Key Bearish Drivers Detected\n\n"
                    f"The current narrative is negative primarily due to macro anxieties and risk-off flags. "
                    f"Here are the top negative headlines our NLP pipeline is analyzing:\n\n"
                )
                for h in top_neg:
                    # Get high TF-IDF terms to show AI explanation
                    features = model_instance.get_top_tfidf_features(h["raw"], top_n=3)
                    feats_str = ", ".join([f"*{f[0]}*" for f in features])
                    response += f"- **Headline:** \"{h['raw']}\"\n  *NLP Core Tokens:* {feats_str}\n"
            else:
                response = "The sentiment is leaning negative, but no specific high-severity headlines were captured in this batch. General regulatory or macro plateaus might be the cause."
        else:
            # Look for positive headlines
            pos_headlines = [h for h in headlines if h["vader"]["Sentiment Label"] == "Positive" or h["lr"]["Sentiment Label"] == "Positive"]
            if pos_headlines:
                top_pos = pos_headlines[:3]
                response = (
                    f"### 🔍 Key Bullish Drivers Detected\n\n"
                    f"The narrative is positive because of expansionary indicators. Our NLP parser has flagged these key positive headlines:\n\n"
                )
                for h in top_pos:
                    features = model_instance.get_top_tfidf_features(h["raw"], top_n=3)
                    feats_str = ", ".join([f"*{f[0]}*" for f in features])
                    response += f"- **Headline:** \"{h['raw']}\"\n  *NLP Core Tokens:* {feats_str}\n"
            else:
                response = "The sentiment is positive, driven by quiet, low-volatility neutral-positive drift. No heavy growth catalysts are currently visible."
        return response
        
    # 7. NARRATIVE TREND / TRANSITION
    if any(k in query for k in ["trend", "narrative trend", "transition", "shift", "history", "path", "movement", "cycle", "past", "timeline"]):
        response = (
            f"### 🔄 Narrative Shift Timeline\n\n"
            f"The BuzzStreet state machine has detected the following narrative path over the session:\n\n"
            f"✨ **` {chain} `**\n\n"
        )
        if "Panic" in chain:
            response += "⚠️ **Warning:** The transition chain indicates a descent into panic conditions. Market liquidity should be closely monitored."
        elif "Fear" in chain and not "Optimistic" in chain[-5:]:
            response += "📉 **Trend:** The narrative trend shows rising risk aversion. Sentiment is degrading over time."
        elif "Optimistic" in chain and not "Panic" in chain[-5:]:
            response += "🚀 **Trend:** Bullish momentum is solid. Market narrative has successfully broken out of fear/panic cycles."
        else:
            response += "⚖️ **Trend:** The narrative is cycling within stable, range-bound parameters."
        return response
        
    # 8. EXPLAIN NLP / TF-IDF / LOGISTIC REGRESSION (AI pipeline questions)
    if any(k in query for k in ["nlp", "preprocess", "pipeline", "tfidf", "tf-idf", "logistic regression", "machine learning", "model", "accuracy", "classifier", "lemmatization", "tokenization", "cleaning"]):
        return (
            f"### 🧠 BuzzStreet Core AI Pipeline Details\n\n"
            f"1. **NLP Cleaning:** We lowercase the raw financial headline, strip URLs and HTML noise, tokenize via NLTK, remove stopwords, and perform **POS-tagged Lemmatization** to extract root concepts (e.g., 'surging' -> 'surge').\n"
            f"2. **Feature Extraction:** We map tokens to numerical weights using **TF-IDF Vectorization** which penalizes common terms and boosts rare, high-information terms.\n"
            f"3. **Logistic Regression:** Trained on-the-fly on a labeled corpus with **{accuracy:.1%} testing accuracy**. It computes probabilities for positive, negative, and neutral sentiments.\n"
            f"4. **VADER vs ML comparison:** We contrast rule-based lexical scoring (VADER) with statistical machine learning (Logistic Regression) to avoid false sentiment signals."
        )

    # 9. GENERAL / FALLBACK
    return (
        f"👋 **I am the BuzzStreet Narrative Intelligence Assistant.**\n\n"
        f"I didn't quite capture the keyword for your query. Try asking:\n"
        f"- *“What is the current market sentiment?”* to see overall mood index and index values.\n"
        f"- *“Why is the market positive/negative?”* to inspect the core headlines and high-scoring TF-IDF words driving the narrative.\n"
        f"- *“What narrative trend is detected?”* to review the transition history path.\n"
        f"- *“How does the NLP pipeline work?”* to check details about lemmatization and the TF-IDF feature space."
    )
