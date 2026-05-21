"""
BuzzStreet – narrative_detector.py
Narrative Shift Detection Engine.
Tracks aggregate sentiment trends and classifies market narrative phases
(Optimistic, Neutral, Fear, Panic) using a state-machine logic.
Formats narrative transition paths.
"""

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
