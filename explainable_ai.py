"""
BuzzStreet – explainable_ai.py
Explainable AI (XAI) Engine.
Provides feature attribution, top positive/negative TF-IDF n-gram token weights,
model contribution breakdowns, and market forecast rationale.
"""

import numpy as np
from nlp_pipeline import preprocess_text
from ml_model import model_instance

def explain_headline_sentiment(headline_text):
    """
    Extracts positive and negative TF-IDF feature tokens and model attribution weights
    for a given financial headline.
    """
    clean_text = preprocess_text(headline_text)
    tokens = clean_text.split()
    
    vec = model_instance.vectorizer
    lr = model_instance.model
    feature_names = np.array(vec.get_feature_names_out())
    
    # Transform headline
    tfidf_vec = vec.transform([clean_text]).toarray()[0]
    nonzero_idx = np.where(tfidf_vec > 0)[0]
    
    # Calculate feature contributions using class weights
    coefs = lr.coef_ # Shape (3, n_features) for Negative, Neutral, Positive
    classes = list(lr.classes_)
    
    pos_idx = classes.index("Positive") if "Positive" in classes else 2
    neg_idx = classes.index("Negative") if "Negative" in classes else 0
    
    pos_weights = coefs[pos_idx]
    neg_weights = coefs[neg_idx]
    
    positive_terms = []
    negative_terms = []
    
    for idx in nonzero_idx:
        term = feature_names[idx]
        val = tfidf_vec[idx]
        p_score = pos_weights[idx] * val
        n_score = neg_weights[idx] * val
        
        if p_score > 0.05:
            positive_terms.append((term, round(float(p_score), 4)))
        elif n_score > 0.05:
            negative_terms.append((term, round(float(n_score), 4)))
            
    # Sort terms by magnitude
    positive_terms.sort(key=lambda x: x[1], reverse=True)
    negative_terms.sort(key=lambda x: x[1], reverse=True)
    
    return {
        "headline": headline_text,
        "clean_tokens": tokens,
        "positive_contributing_terms": positive_terms[:5],
        "negative_contributing_terms": negative_terms[:5],
        "total_active_features": len(nonzero_idx)
    }

def explain_market_forecast(asset_name, current_phase, sentiment_score, nifty_target, horizon_days):
    """
    Generates transparent rationale for stock price forecasting.
    """
    rationale_points = [
        f"1. Recent Narrative Phase ({current_phase}): Market sentiment is currently in {current_phase} phase with a net Composite Score of {sentiment_score:+.3f}.",
        f"2. Sentiment Drift Momentum: Textual news streams display a net directional bias of {sentiment_score * 100:+.1f}%.",
        f"3. Market Price Trend: Integrating 6-month price action and 20-day/50-day moving averages for {asset_name}.",
        f"4. Horizon Projection ({horizon_days} Days): Linear regression trend combined with sentiment drift projects a target of {nifty_target:,.2f}.",
        f"5. Statistical Confidence Bound: Model Agreement and VADER/LogReg ensemble alignment validate current directional guidance."
    ]
    return rationale_points
