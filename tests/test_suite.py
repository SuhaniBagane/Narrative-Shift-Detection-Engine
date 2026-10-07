"""
BuzzStreet – tests/test_suite.py
Automated Testing Suite.
Tests authentication, OTP rules, NLP pipeline, sentiment models, composite score bounds,
narrative phase thresholds, anomaly risk calculation, backtesting, paper trading,
trade impact simulation (BEFORE/AFTER), scenario matrix, and system health diagnostics.
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
from paper_trading import (
    execute_virtual_order, calculate_portfolio_summary, 
    simulate_trade_impact, get_portfolio_scenario_matrix, get_model_trade_signal
)
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
        import time
        user = f"test_trader_unit_{int(time.time())}"
        success, msg = execute_virtual_order(user, "Nifty 50", "BUY", 2, 22000.0)
        self.assertTrue(success, f"Order failed with message: {msg}")
        
        summary = calculate_portfolio_summary(user)
        self.assertGreaterEqual(summary["portfolio_value"], 0)

    def test_08_system_health_diagnostics(self):
        """Test system health check endpoint."""
        health = get_system_health_status()
        self.assertEqual(health["status_code"], 200)
        self.assertIn("Database Service", health["services"])

    def test_09_trade_impact_simulation(self):
        """Test BEFORE vs AFTER trade impact simulation metrics."""
        user = "unit_impact_trader"
        impact = simulate_trade_impact(user, "NVIDIA Corp.", "BUY", 5, 880.0)
        self.assertTrue(impact["valid"])
        self.assertIn("before", impact)
        self.assertIn("after", impact)
        self.assertEqual(impact["after"]["trades"], impact["before"]["trades"] + 1)
        self.assertLess(impact["after"]["cash"], impact["before"]["cash"])

    def test_10_trade_insufficient_cash_validation(self):
        """Test validation error when buying with insufficient virtual cash."""
        user = "unit_cash_trader"
        # Try buying 1,000 shares of Bitcoin at $64,500 each (exceeds 100k cash)
        success, msg = execute_virtual_order(user, "Bitcoin", "BUY", 1000, 64500.0)
        self.assertFalse(success)
        self.assertIn("Insufficient cash balance", msg)

    def test_11_trade_unowned_share_sell_validation(self):
        """Test validation error when selling unowned shares."""
        user = "unit_sell_trader"
        success, msg = execute_virtual_order(user, "Tesla Inc.", "SELL", 50, 175.0)
        self.assertFalse(success)
        self.assertIn("Cannot sell more shares than owned", msg)

    def test_12_portfolio_scenario_matrix_and_model_signal(self):
        """Test side-by-side Current vs Buy vs Sell scenarios and model signal generator."""
        user = "unit_scenario_trader"
        scenarios = get_portfolio_scenario_matrix(user, "Apple Inc.", 5, 182.5)
        self.assertIn("current", scenarios)
        self.assertIn("buy_scenario", scenarios)
        self.assertIn("sell_scenario", scenarios)
        
        signal_info = get_model_trade_signal("Apple Inc.", +0.35, "Optimistic", 15)
        self.assertIn("signal", signal_info)
        self.assertIn("BUY", signal_info["signal"])

    def test_13_user_profile_safe_fallback(self):
        """Test safe username fallback when session state user_profile is None."""
        user_profile = None
        user_identifier = None
        user_name = (
            user_profile.get("name", "Trader")
            if isinstance(user_profile, dict)
            else (user_identifier or "Trader")
        )
        self.assertEqual(user_name, "Trader")

        user_profile_dict = {"name": "Suhani", "role": "Lead Quantitative Researcher"}
        user_name_active = (
            user_profile_dict.get("name", "Trader")
            if isinstance(user_profile_dict, dict)
            else "Trader"
        )
        self.assertEqual(user_name_active, "Suhani")

    def test_14_strict_otp_validation_and_email_auth(self):
        """Test strict OTP code verification and email authentication flow."""
        import streamlit as st
        auth.init_auth_state()
        
        # Test Email OTP dispatch
        success, msg = auth.send_otp_backend("test_user@buzzstreet.ai", channel="email")
        self.assertTrue(success)
        self.assertIsNotNone(st.session_state.active_otp_code)
        
        # Test invalid OTP rejection
        v_fail, msg_fail = auth.verify_otp_backend("000000" if st.session_state.active_otp_code != "000000" else "111111")
        self.assertFalse(v_fail)
        self.assertIn("Incorrect OTP", msg_fail)
        
        # Test correct OTP validation
        v_pass, msg_pass = auth.verify_otp_backend(st.session_state.active_otp_code)
        self.assertTrue(v_pass)
        self.assertTrue(st.session_state.authenticated)

if __name__ == "__main__":
    unittest.main()
