"""
BuzzStreet – data_loader.py
Simulated news and market data loader. Provides realistic, dynamic financial headlines
and generates correlated Nifty 50 and Sensex index movements to simulate live streams.
"""

import random
import datetime

# Large curated pool of realistic financial headlines divided by sentiment/narrative
NEWS_POOL = {
    "positive": [
        "Tech sector sees unprecedented growth in Q4, driving indices higher.",
        "New government subsidy boosts renewable energy stocks across the board.",
        "Retail sales exceed expectations ahead of holiday season, showing consumer strength.",
        "Mega-merger between two telecom giants approved, synergies expected.",
        "Startups see a major surge in venture capital funding as confidence returns.",
        "Corporate tax cuts lead to stock market rally and corporate expansions.",
        "Consumer spending hits all-time high in July, beating economist forecasts.",
        "Groundbreaking AI model released, sending technology stocks soaring.",
        "Dividends increased for top blue-chip companies amid strong cash flows.",
        "E-commerce platforms report 300% YoY growth in active customer base.",
        "Logistics companies report easing of supply chain bottlenecks worldwide.",
        "Inflation drops below 2% target, markets celebrate possible rate cuts.",
        "Major airline reports record passenger numbers, sparking travel sector rebound.",
        "Breakthrough in solid-state batteries drives automobile shares up.",
        "Green energy initiatives create thousands of new jobs, boosting manufacturing.",
        "Cryptocurrency markets rebound sharply after regulatory clarity.",
        "FDI inflows rise by 15% as foreign investors find domestic markets highly attractive.",
        "Excellent earnings reports from banking sector lift financial index.",
        "Manufacturing PMI climbs to a 3-year high, indicating robust industrial growth.",
        "Exports touch record highs as global demand for pharmaceutical products surges."
    ],
    "negative": [
        "Inflation hits record high, sparking recession fears and market selloffs.",
        "Central bank announces unexpected interest rate hike of 50 basis points.",
        "Supply chain crisis worsens as port strikes continue indefinitely.",
        "Global semiconductor shortage halts automobile production globally.",
        "Tech giants face severe antitrust lawsuits in the European Union.",
        "Unemployment claims rise unexpectedly this month, dampening consumer confidence.",
        "Widespread crop failure drives agricultural and food prices up.",
        "Trade war concerns escalate with new tariffs on imported goods.",
        "Healthcare stocks plummet after failed clinical trials of blockbuster drug.",
        "Major data breach exposes millions of users' personal info, dragging tech sector.",
        "Investor confidence drops as corporate earnings disappoint in Q3.",
        "European markets sink amid geopolitical and political instability.",
        "Bond yields spike to multi-year highs, making equities less attractive.",
        "Crude oil prices surge above $95 a barrel, threatening to spark further inflation.",
        "Credit rating agency downgrades economic outlook from stable to negative.",
        "Fiscal deficit widens as government expenditure outpaces revenue collections.",
        "Industrial output contracts by 1.2% in negative surprise to analysts.",
        "Corporate bond defaults reach highest level in five years, signaling distress."
    ],
    "panic": [
        "IMMEDIATE LIQUIDITY CRISIS: Major investment bank halts withdrawals.",
        "Global stock markets trigger circuit breakers as panic selling intensifies.",
        "Sovereign debt default feared as talks between nation and IMF collapse.",
        "CRASH PREDICTION: Leading economists warn of imminent financial system meltdown.",
        "Cryptocurrency exchange hacked, $5 billion in user assets completely lost.",
        "Severe geopolitical conflict escalates, halting trade in key global shipping lanes.",
        "Massive bank run reported at regional banks, retail investors panic.",
        "Real estate bubble burst imminent as mortgage default rate doubles overnight.",
        "Currency value crashes by 10% in a single session against the dollar.",
        "Hyperinflation warnings triggered as wholesale prices surge by 40%."
    ],
    "neutral": [
        "Federal Reserve to hold meeting on monetary policy next week.",
        "Automakers shift focus gradually towards electric vehicle production.",
        "Congress debates new infrastructure spending bill proposals.",
        "Global trade volume remains steady in the third quarter.",
        "Tech companies adapt to new user privacy regulations.",
        "Consumer behavior shifts gradually towards sustainable products.",
        "Minimum wage debate continues in several state legislatures.",
        "Energy consumption patterns plateau with steady hybrid work models.",
        "Real estate prices plateau after a year of rapid growth.",
        "Investors await upcoming corporate quarterly earnings reports.",
        "Regulatory changes proposed for the local gig economy.",
        "Demographic shifts impact long-term retail market trends.",
        "International travel routes reopen fully to post-pandemic levels.",
        "Central banks explore the creation of sovereign digital currencies.",
        "New tax filing rules come into effect this fiscal year.",
        "Commodities market experiences standard seasonal volatility.",
        "Annual financial audit reports published for major PSUs.",
        "National statistics office to release GDP figures on Friday."
    ]
}

# Dynamically extend NEWS_POOL with the Kaggle CSV dataset if available
import os
import pandas as pd
csv_path = "data/sentiment_data.csv"
if os.path.exists(csv_path):
    try:
        df = pd.read_csv(csv_path)
        if 'Sentence' in df.columns and 'Sentiment' in df.columns:
            df = df.dropna(subset=['Sentence', 'Sentiment'])
            for _, row in df.iterrows():
                sent = str(row['Sentiment']).strip().lower()
                text = str(row['Sentence']).strip()
                if sent in NEWS_POOL:
                    if text not in NEWS_POOL[sent]:
                        NEWS_POOL[sent].append(text)
    except Exception as e:
        print(f"Warning: Failed to load news pool CSV {csv_path}: {e}")

def generate_headlines(bias="neutral", count=10):
    """
    Generates a batch of news headlines based on the current market narrative bias.
    bias can be: "bullish" (optimistic), "bearish" (fear), "panic", or "neutral"
    """
    headlines = []
    
    # Configure sampling probabilities based on bias
    if bias == "bullish":
        weights = {"positive": 0.70, "negative": 0.05, "panic": 0.00, "neutral": 0.25}
    elif bias == "bearish":
        weights = {"positive": 0.05, "negative": 0.65, "panic": 0.10, "neutral": 0.20}
    elif bias == "panic":
        weights = {"positive": 0.00, "negative": 0.30, "panic": 0.60, "neutral": 0.10}
    else:  # neutral
        weights = {"positive": 0.20, "negative": 0.20, "panic": 0.00, "neutral": 0.60}
        
    categories = list(weights.keys())
    probs = list(weights.values())
    
    # Pick counts for each category
    chosen_categories = random.choices(categories, weights=probs, k=count)
    
    used_headlines = set()
    for cat in chosen_categories:
        # Avoid duplicate headlines in a single batch
        available = [h for h in NEWS_POOL[cat] if h not in used_headlines]
        if not available:
            available = NEWS_POOL[cat]
            
        headline = random.choice(available)
        used_headlines.add(headline)
        headlines.append(headline)
        
    return headlines

def simulate_market_indices(sentiment_score, prev_nifty=22400.0, prev_sensex=73850.0):
    """
    Calculates the new values of Nifty 50 and Sensex based on the headline sentiment score.
    sentiment_score is between -1 (extremely bearish) and +1 (extremely bullish).
    Returns new nifty, nifty_change_pct, new_sensex, sensex_change_pct
    """
    # Fluctuation base is driven by the sentiment score plus some random noise
    noise = random.uniform(-0.002, 0.002)
    change_pct = (sentiment_score * 0.012) + noise
    
    # Bound the daily change to realistic levels (-3% to +3% in a single tick/day)
    change_pct = max(-0.03, min(0.03, change_pct))
    
    new_nifty = prev_nifty * (1.0 + change_pct)
    new_sensex = prev_sensex * (1.0 + change_pct)
    
    nifty_change = new_nifty - prev_nifty
    sensex_change = new_sensex - prev_sensex
    
    return round(new_nifty, 2), round(change_pct * 100, 2), round(new_sensex, 2), round(change_pct * 100, 2)

def fetch_live_indices_yfinance():
    """
    Fetches actual real-time prices and daily change percentages
    for Nifty 50 (^NSEI) and BSE Sensex (^BSESN) from Yahoo Finance.
    Falls back to hardcoded default indices if connection fails.
    """
    import yfinance as yf
    try:
        nifty = yf.Ticker('^NSEI')
        nifty_p = float(nifty.fast_info['last_price'])
        nifty_pc = float(nifty.fast_info['previous_close'])
        nifty_change = ((nifty_p - nifty_pc) / nifty_pc) * 100
        
        sensex = yf.Ticker('^BSESN')
        sensex_p = float(sensex.fast_info['last_price'])
        sensex_pc = float(sensex.fast_info['previous_close'])
        sensex_change = ((sensex_p - sensex_pc) / sensex_pc) * 100
        
        return round(nifty_p, 2), round(nifty_change, 2), round(sensex_p, 2), round(sensex_change, 2)
    except Exception as e:
        print(f"Warning: Failed to fetch live Yahoo Finance data: {e}")
        # Default fallback values
        return 22420.00, 0.00, 73910.00, 0.00

