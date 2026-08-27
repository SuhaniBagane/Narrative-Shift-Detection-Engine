"""
BuzzStreet – narrative_detector.py
Narrative Shift Detection Engine.
Tracks aggregate sentiment trends and classifies market narrative phases
(Optimistic, Neutral, Fear, Panic) using a state-machine logic.
Formats narrative transition paths.
"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Global VADER instance
try:
    _sia_instance = SentimentIntensityAnalyzer()
except Exception:
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    _sia_instance = SentimentIntensityAnalyzer()

def analyze_vader_sentiment(text):
    """
    Computes VADER polarity scores for a given headline.
    """
    scores = _sia_instance.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        label = "Positive"
    elif compound <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return {
        "Compound Score": round(compound, 4),
        "Sentiment Label": label,
        "Scores": scores
    }

def predict_lr_sentiment(text, vectorizer, lr_model):
    """
    Predicts sentiment probabilities for a headline using TF-IDF + Logistic Regression.
    """
    from nlp_pipeline import preprocess_headline_detailed
    p_res = preprocess_headline_detailed(text)
    clean_str = " ".join(p_res["clean_tokens"])
    vec = vectorizer.transform([clean_str])
    probs = lr_model.predict_proba(vec)[0]
    classes = list(lr_model.classes_)
    prob_dict = dict(zip(classes, probs))
    
    pos_p = prob_dict.get("Positive", 0.0)
    neg_p = prob_dict.get("Negative", 0.0)
    max_label = max(prob_dict, key=prob_dict.get)
    
    return {
        "Positive Prob": pos_p,
        "Negative Prob": neg_p,
        "Sentiment Label": max_label,
        "Probabilities": prob_dict
    }

def detect_narrative_phase(sentiment_index):
    """
    Classifies the current market mood based on the composite Sentiment Index.
    sentiment_index ranges from -1.0 (extremely bearish) to +1.0 (extremely bullish).
    """
    if sentiment_index >= 0.25:
        return "Optimistic"
    elif sentiment_index >= -0.10:
        return "Neutral"
    elif sentiment_index >= -0.55:
        return "Fear"
    else:
        return "Panic"

def get_phase_styling(phase):
    """
    Returns hex colors and brief descriptions for each narrative phase.
    """
    styling = {
        "Optimistic": {
            "color": "#00e676", # Vibrant Green
            "bg_color": "rgba(0, 230, 118, 0.1)",
            "border_color": "#00e676",
            "emoji": "🚀",
            "desc": "High confidence, expansionary sentiment, strong buy volumes."
        },
        "Neutral": {
            "color": "#9e9e9e", # Gray
            "bg_color": "rgba(158, 158, 158, 0.1)",
            "border_color": "#9e9e9e",
            "emoji": "⚖️",
            "desc": "Steady, consolidative trading, matching supply and demand."
        },
        "Fear": {
            "color": "#ff9100", # Orange
            "bg_color": "rgba(255, 145, 0, 0.1)",
            "border_color": "#ff9100",
            "emoji": "⚠️",
            "desc": "Rising caution, hedging behavior, sell-on-rise stance."
        },
        "Panic": {
            "color": "#ff1744", # Deep Red
            "bg_color": "rgba(255, 23, 68, 0.1)",
            "border_color": "#ff1744",
            "emoji": "🚨",
            "desc": "Capitulation, extreme risk aversion, heavy margin call pressure."
        }
    }
    return styling.get(phase, styling["Neutral"])

def calculate_composite_index(vader_compound_scores, lr_sentiment_dicts, vader_weight=0.5):
    """
    Computes a composite market sentiment index by combining:
    1. VADER average compound score.
    2. Logistic Regression average net probability (Positive Prob - Negative Prob).
    Both sit between -1.0 and +1.0. Returns a single composite score.
    """
    if not vader_compound_scores or not lr_sentiment_dicts:
        return 0.0
        
    avg_vader = sum(vader_compound_scores) / len(vader_compound_scores)
    
    lr_net_scores = [d["Positive"] - d["Negative"] for d in lr_sentiment_dicts]
    avg_lr = sum(lr_net_scores) / len(lr_net_scores)
    
    # Custom weighted combination
    composite_index = (vader_weight * avg_vader) + ((1.0 - vader_weight) * avg_lr)
    return round(composite_index, 3)

def generate_transition_chain(history):
    """
    Takes a history of phase detections (list of dicts with 'phase')
    and outputs a formatted string representing the transition path.
    Example: Optimistic → Neutral → Fear
    """
    if not history:
        return "No narrative data available."
        
    phases = [item["phase"] for item in history]
    
    # Deduplicate consecutive identical states to show clean chronological shifts
    deduped_phases = []
    for p in phases:
        if not deduped_phases or deduped_phases[-1] != p:
            deduped_phases.append(p)
            
    # Format with nice arrow indicators
    return " → ".join(deduped_phases)

import numpy as np

def predict_future_market_trajectory(history, horizon_mode="Intraday Hours (+24H)", steps=8):
    """
    Predicts future market valuation trajectory (Nifty 50, Sensex, Sentiment, Narrative Phase)
    and computes upper/lower confidence bands for future horizons:
    - "Intraday Hours (+24H)": Hourly increments (+3h, +6h, +9h, +12h, etc.)
    - "7-Day Market Horizon (+7D)": Daily increments (Day +1, Day +2, Day +3, etc.)
    - "30-Day Outlook (+30D)": Multi-day increments (Day +4, Day +8, Day +12, etc.)
    """
    if not history:
        return []
        
    last_record = history[-1]
    curr_nifty = last_record.get("nifty", 22400.0)
    curr_sensex = last_record.get("sensex", 73850.0)
    curr_sentiment = last_record.get("sentiment", 0.0)
    
    # Calculate price momentum from history
    nifty_prices = [h.get("nifty", curr_nifty) for h in history]
    if len(nifty_prices) > 1:
        price_change_pct = (nifty_prices[-1] - nifty_prices[0]) / nifty_prices[0]
    else:
        price_change_pct = 0.0
        
    # Blended drift slope per step: driven by sentiment index (65%) + historical price momentum (35%)
    sentiment_drift = curr_sentiment * 0.004
    momentum_drift = price_change_pct * 0.05
    step_drift = (0.65 * sentiment_drift) + (0.35 * momentum_drift)
    
    # Base standard deviation error for confidence bounds
    std_dev_nifty = curr_nifty * 0.003
    std_dev_sensex = curr_sensex * 0.003
    
    forecast_records = []
    
    for i in range(1, steps + 1):
        # Calculate expected change with slight decay factor to avoid divergence
        decay = 0.95 ** (i - 1)
        cumulative_factor = 1.0 + (step_drift * i * decay)
        
        pred_nifty = round(curr_nifty * cumulative_factor, 2)
        pred_sensex = round(curr_sensex * cumulative_factor, 2)
        
        # Expanding confidence bands
        uncertainty = np.sqrt(i) * 1.5
        nifty_lower = round(pred_nifty - (std_dev_nifty * uncertainty), 2)
        nifty_upper = round(pred_nifty + (std_dev_nifty * uncertainty), 2)
        
        sensex_lower = round(pred_sensex - (std_dev_sensex * uncertainty), 2)
        sensex_upper = round(pred_sensex + (std_dev_sensex * uncertainty), 2)
        
        # Projected future sentiment (drifts toward current mood)
        pred_sentiment = round(max(-1.0, min(1.0, curr_sentiment * (0.92 ** i))), 3)
        pred_phase = detect_narrative_phase(pred_sentiment)
        
        # Generate future label
        if horizon_mode == "Intraday Hours (+24H)":
            label = f"+{i*3}h"
        elif horizon_mode == "7-Day Market Horizon (+7D)":
            label = f"Day +{i}"
        else: # 30-Day Outlook
            label = f"Day +{i*4}"
            
        forecast_records.append({
            "time": label,
            "sentiment": pred_sentiment,
            "phase": pred_phase,
            "nifty": pred_nifty,
            "nifty_lower": nifty_lower,
            "nifty_upper": nifty_upper,
            "sensex": pred_sensex,
            "sensex_lower": sensex_lower,
            "sensex_upper": sensex_upper,
            "is_future": True
        })
        
    return forecast_records

