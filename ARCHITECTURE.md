# BuzzStreet System Architecture Documentation

## Overview
BuzzStreet is architected as a modular, decoupled full-stack Market Psychology AI system.

```
+-----------------------------------------------------------------------+
|                           Streamlit UI Layer                          |
| (11 Reorganized Cockpit Tabs, Plotly Charts, AI Chatbot, Voice Bot)   |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                      Authentication & Security Gate                   |
|  (auth.py - Twilio Verify API / E.164 Phone OTP / 10-Min Expiry)      |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                     Core Analytics Engine Layer                       |
|  - nlp_pipeline.py (8-Stage Text Cleaning & Lemmatization)            |
|  - ml_model.py (TF-IDF Vectorizer + Logistic Regression)              |
|  - narrative_detector.py (VADER + NC-SSM Composite State Machine)    |
|  - explainable_ai.py (TF-IDF Term Feature Attribution)                |
|  - correlation_engine.py (Pearson/Spearman & Lagged Returns)          |
|  - backtester.py (MAE, RMSE, Win Rate %, Sharpe Ratio, Max Drawdown)  |
|  - paper_trading.py (Virtual Order Book & Portfolio Manager)          |
|  - report_generator.py (Executive Report Generator)                   |
|  - model_registry.py (Model Version Registry)                         |
|  - system_health.py (Admin Component Diagnostics & Telemetry)        |
+-----------------------------------------------------------------------+
                                   │
                                   ▼
+-----------------------------------------------------------------------+
|                       Persistent Database Layer                       |
|  (db.py - SQLite Engine with Indexing: Users, News, Events, Trades)   |
+-----------------------------------------------------------------------+
```

## Module Summary
- **`app.py`**: Main cockpit entry point managing Streamlit layout, session state, and 11 reorganized tabs.
- **`auth.py`**: Production SMS/Email authentication module using Twilio Verify API.
- **`db.py`**: Persistent database architecture with 10 structured tables and indexes.
- **`nlp_pipeline.py`**: 8-stage text normalization pipeline using NLTK WordNetLemmatizer.
- **`ml_model.py`**: Supervised Multinomial Logistic Regression model trained on 300,000 headlines.
- **`narrative_detector.py`**: Dual-model sentiment scoring and 4-phase state machine mapping.
- **`explainable_ai.py`**: XAI feature attribution and prediction rationale generator.
- **`correlation_engine.py`**: Statistical Pearson/Spearman correlation and post-shift return analyzer.
- **`backtester.py`**: Empirical backtesting and forecast validation engine.
- **`paper_trading.py`**: Simulation-only paper trading and portfolio analytics engine.
- **`report_generator.py`**: Executive market intelligence markdown report synthesizer.
- **`model_registry.py`**: Lightweight model version specification registry.
- **`system_health.py`**: Real-time diagnostic service exposing `/api/health` status.
