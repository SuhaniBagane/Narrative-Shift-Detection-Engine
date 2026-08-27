"""
BuzzStreet – db.py
Persistent Database Architecture & Schema Manager.
Provides SQLite/SQLAlchemy schema management and ORM models for:
- Users & Authentication Metadata
- User Profiles & Onboarding Preferences
- Ingested Financial News Articles
- Dual-Model Sentiment Results & Narrative Scores
- Historical Market Data (Tickers & Price Action)
- Detected Narrative Shift Events
- Multi-Horizon Price Predictions & Forecasts
- Paper Trading Accounts & Transaction Logs
- Watchlist Assets
- Model Version Registry
"""

import os
import sqlite3
import datetime
import json
from pathlib import Path

# Database path configuration
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "buzzstreet.db")

def get_connection():
    """Returns a SQLite connection with dict row factory."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Creates database tables and indexes if they do not exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # 1. Users Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        phone_or_email TEXT UNIQUE NOT NULL,
        auth_metadata TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # 2. User Profiles Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id INTEGER PRIMARY KEY,
        full_name TEXT NOT NULL,
        trader_role TEXT NOT NULL,
        market_focus TEXT NOT NULL,
        alert_preferences TEXT,
        risk_preferences TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    """)

    # 3. News Articles Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS news_articles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT NOT NULL,
        headline TEXT NOT NULL,
        url TEXT,
        published_at TIMESTAMP,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        affected_asset TEXT DEFAULT 'Nifty 50',
        raw_text TEXT,
        duplicate_hash TEXT UNIQUE
    )
    """)

    # 4. Sentiment Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id INTEGER,
        vader_score REAL NOT NULL,
        logistic_score REAL NOT NULL,
        composite_score REAL NOT NULL,
        narrative_phase TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_agreement REAL NOT NULL,
        anomaly_score REAL NOT NULL,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (article_id) REFERENCES news_articles(id)
    )
    """)

    # 5. Market Data Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS market_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        open REAL,
        high REAL,
        low REAL,
        close REAL,
        volume REAL,
        daily_return REAL,
        UNIQUE(asset, timestamp)
    )
    """)

    # 6. Narrative Events Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS narrative_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        previous_phase TEXT NOT NULL,
        current_phase TEXT NOT NULL,
        old_composite REAL NOT NULL,
        new_composite REAL NOT NULL,
        shift_magnitude REAL NOT NULL,
        model_agreement REAL NOT NULL,
        confidence REAL NOT NULL,
        anomaly_score REAL NOT NULL,
        affected_assets TEXT NOT NULL
    )
    """)

    # 7. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        asset TEXT NOT NULL,
        prediction_horizon TEXT NOT NULL,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        predicted_price REAL NOT NULL,
        lower_bound REAL NOT NULL,
        upper_bound REAL NOT NULL,
        predicted_direction TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_version TEXT DEFAULT 'v2.1'
    )
    """)

    # 8. Paper Trades Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS paper_trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_identifier TEXT NOT NULL,
        asset TEXT NOT NULL,
        action TEXT NOT NULL,
        quantity REAL NOT NULL,
        execution_price REAL NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        stop_loss REAL,
        take_profit REAL,
        status TEXT DEFAULT 'OPEN'
    )
    """)

    # 9. Watchlist Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS watchlist (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_identifier TEXT NOT NULL,
        asset TEXT NOT NULL,
        added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_identifier, asset)
    )
    """)

    # 10. Model Registry Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        version TEXT PRIMARY KEY,
        training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        dataset_version TEXT NOT NULL,
        preprocessing_config TEXT NOT NULL,
        hyperparameters TEXT NOT NULL,
        accuracy REAL NOT NULL,
        precision_score REAL NOT NULL,
        recall_score REAL NOT NULL,
        f1_score REAL NOT NULL,
        model_file_ref TEXT NOT NULL
    )
    """)

    # Create Performance Indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_asset_ts ON market_data(asset, timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON narrative_events(timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_user ON paper_trades(user_identifier);")

    conn.commit()
    conn.close()

# Initialize DB on import
init_db()

# --- HELPER DATABASE CRUD OPERATIONS ---

def register_user(phone_or_email, auth_metadata=""):
    """Registers or updates user login."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO users (phone_or_email, auth_metadata, created_at, last_login)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(phone_or_email) DO UPDATE SET last_login = ?
    """, (phone_or_email, auth_metadata, now, now, now))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return user_id

def save_user_profile(user_identifier, full_name, trader_role, market_focus, alert_pref="", risk_pref=""):
    """Saves or updates user profile."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO user_profiles (user_id, full_name, trader_role, market_focus, alert_preferences, risk_preferences)
    VALUES (
        (SELECT id FROM users WHERE phone_or_email = ? LIMIT 1),
        ?, ?, ?, ?, ?
    )
    ON CONFLICT(user_id) DO UPDATE SET
        full_name=excluded.full_name,
        trader_role=excluded.trader_role,
        market_focus=excluded.market_focus,
        alert_preferences=excluded.alert_preferences,
        risk_preferences=excluded.risk_preferences
    """, (user_identifier, full_name, trader_role, market_focus, alert_pref, risk_pref))
    conn.commit()
    conn.close()

def record_narrative_event(prev_phase, curr_phase, old_comp, new_comp, shift_mag, agreement, conf, anomaly, assets):
    """Records a narrative shift transition event."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO narrative_events 
    (timestamp, previous_phase, current_phase, old_composite, new_composite, shift_magnitude, model_agreement, confidence, anomaly_score, affected_assets)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now, prev_phase, curr_phase, old_comp, new_comp, shift_mag, agreement, conf, anomaly, assets))
    conn.commit()
    event_id = cursor.lastrowid
    conn.close()
    return event_id

def get_narrative_events(limit=50):
    """Retrieves recent narrative shift events."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM narrative_events ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def save_paper_trade(user_identifier, asset, action, quantity, price, stop_loss=None, take_profit=None):
    """Executes and records a paper trade."""
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO paper_trades (user_identifier, asset, action, quantity, execution_price, timestamp, stop_loss, take_profit)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_identifier, asset, action, quantity, price, now, stop_loss, take_profit))
    conn.commit()
    conn.close()

def get_user_paper_trades(user_identifier):
    """Fetches user paper trades."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM paper_trades WHERE user_identifier = ? ORDER BY id DESC", (user_identifier,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_user_watchlist(user_identifier):
    """Fetches user watchlist assets."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT asset FROM watchlist WHERE user_identifier = ?", (user_identifier,))
    rows = [row["asset"] for row in cursor.fetchall()]
    conn.close()
    return rows if rows else ["Nifty 50", "BSE Sensex", "Reliance Industries", "Apple Inc.", "Tesla Inc.", "NVIDIA Corp."]

def add_watchlist_asset(user_identifier, asset):
    """Adds an asset to user watchlist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT OR IGNORE INTO watchlist (user_identifier, asset) VALUES (?, ?)", (user_identifier, asset))
    conn.commit()
    conn.close()

def remove_watchlist_asset(user_identifier, asset):
    """Removes an asset from user watchlist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM watchlist WHERE user_identifier = ? AND asset = ?", (user_identifier, asset))
    conn.commit()
    conn.close()
