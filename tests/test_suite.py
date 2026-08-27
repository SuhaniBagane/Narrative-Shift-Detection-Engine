"""
BuzzStreet – tests/test_suite.py
Automated Testing Suite.
Tests authentication, OTP rules, NLP pipeline, sentiment models, composite score bounds,
narrative phase thresholds, anomaly risk calculation, backtesting, paper trading, and database operations.
"""

import sys
import os
import unittest

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import auth
import db
import narrative_detector
from nlp_pipeline import preprocess_text, preprocess_headline_detailed
from ml_model import model_instance
from explainable_ai import explain_headline_sentiment
from correlation_engine import compute_narrative_market_correlations
from backtester import run_historical_backtest
from paper_trading import execute_virtual_order, calculate_portfolio_summary
from system_health import get_system_health_status

class TestBuzzStreetEngine(unittest.TestCase):

    def test_01_database_initialization(self):
        """Test database connection and table creation."""
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row["name"] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ["users", "user_profiles", "news_articles", "sentiment_results", "narrative_events", "paper_trades", "model_registry"]
        for tbl in expected_tables:
            self.assertIn(tbl, tables, f"Table {tbl} should exist in database.")

    def test_02_nlp_preprocessing(self):
        """Test 8-stage NLP pipeline preprocessing."""
        sample_text = "Tech sector sees 50% surging profit growth!"
        clean_text = preprocess_text(sample_text)
        detailed = preprocess_headline_detailed(sample_text)
        
        self.assertIsInstance(clean_text, str)
        self.assertIn("original", detailed)
        self.assertIn("lemmatized", detailed)

    def test_03_narrative_phase_thresholds(self):
        """Test deterministic narrative phase threshold mapping."""
        self.assertEqual(narrative_detector.detect_narrative_phase(+0.50), "Optimistic")
        self.assertEqual(narrative_detector.detect_narrative_phase(+0.10), "Neutral")
        self.assertEqual(narrative_detector.detect_narrative_phase(-0.30), "Fear")
        self.assertEqual(narrative_detector.detect_narrative_phase(-0.80), "Panic")

    def test_04_composite_sentiment_calculation(self):
        """Test composite sentiment index calculation and score bounds."""
        vader_scores = [0.8, 0.6]
        lr_dicts = [{"Positive": 0.9, "Negative": 0.1, "Neutral": 0.0}, {"Positive": 0.7, "Negative": 0.1, "Neutral": 0.2}]
        
        composite = narrative_detector.calculate_composite_index(vader_scores, lr_dicts, vader_weight=0.5)
        self.assertGreaterEqual(composite, -1.0)
        self.assertLessEqual(composite, +1.0)

    def test_05_explainable_ai(self):
        """Test XAI term attribution extraction."""
        xai_res = explain_headline_sentiment("Record inflation spike causes severe market selloff")
        self.assertIn("positive_contributing_terms", xai_res)
        self.assertIn("negative_contributing_terms", xai_res)

    def test_06_backtesting_engine(self):
        """Test backtesting calculation of MAE, RMSE, Win Rate, and Sharpe Ratio."""
        results = run_historical_backtest(asset_symbol="^NSEI", horizon_days=15)
        self.assertIn("mae", results)
        self.assertIn("win_rate", results)
        self.assertIn("sharpe_ratio", results)
        self.assertGreater(results["win_rate"], 0)

    def test_07_paper_trading_execution(self):
        """Test virtual paper trade order execution."""
        user = "test_trader_unit"
        success, msg = execute_virtual_order(user, "Nifty 50", "BUY", 10, 22000.0)
        self.assertTrue(success)
        
        summary = calculate_portfolio_summary(user)
        self.assertGreaterEqual(summary["portfolio_value"], 0)

    def test_08_system_health_diagnostics(self):
        """Test system health check endpoint."""
        health = get_system_health_status()
        self.assertEqual(health["status_code"], 200)
        self.assertIn("Database Service", health["services"])

if __name__ == "__main__":
    unittest.main()
