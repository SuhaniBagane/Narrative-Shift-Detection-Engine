"""
BuzzStreet – news_service.py
Real-Time News Ingestion & Processing Pipeline.
Architecture: News Source ➔ Fetch ➔ Validate ➔ Normalize ➔ Deduplicate (SHA256) ➔ NLP Processing ➔ Sentiment ➔ Narrative Phase ➔ Database.
"""

import hashlib
import datetime
import pandas as pd
import data_loader
import narrative_detector
from db import get_connection

def compute_headline_hash(headline):
    """Generates a SHA256 hash for deduplication."""
    return hashlib.sha256(headline.strip().lower().encode("utf-8")).hexdigest()

def ingest_and_process_headlines(headlines_list, source="Live Feed Stream", affected_asset="Nifty 50"):
    """
    Validates, normalizes, deduplicates, and processes a batch of headlines.
    Stores clean articles and sentiment records in the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    processed_records = []
    
    for item in headlines_list:
        if isinstance(item, str):
            raw_text = item
        elif isinstance(item, dict):
            raw_text = item.get("raw") or item.get("headline") or ""
        else:
            continue
            
        headline = raw_text.strip()
        if not headline or len(headline) < 10:
            continue # Validation filter
            
        dedup_hash = compute_headline_hash(headline)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert into news_articles (IGONE on duplicate hash)
        try:
            cursor.execute("""
            INSERT OR IGNORE INTO news_articles 
            (source, headline, url, published_at, ingested_at, affected_asset, raw_text, duplicate_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (source, headline, f"https://finance.yahoo.com/search?q={affected_asset}", now_str, now_str, affected_asset, headline, dedup_hash))
            
            article_id = cursor.lastrowid
        except Exception:
            article_id = None
            
        processed_records.append({
            "headline": headline,
            "hash": dedup_hash,
            "article_id": article_id,
            "ingested_at": now_str
        })
        
    conn.commit()
    conn.close()
    return processed_records

def get_ingested_news_summary():
    """Retrieves ingestion metrics for system health monitoring."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as total FROM news_articles")
    total_articles = cursor.fetchone()["total"]
    
    cursor.execute("SELECT COUNT(*) as today_count FROM news_articles WHERE date(ingested_at) = date('now')")
    today_articles = cursor.fetchone()["today_count"]
    
    conn.close()
    return {
        "total_articles": total_articles,
        "today_articles": today_articles
    }
