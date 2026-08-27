# BuzzStreet Automated Testing Documentation

## Test Suite Overview (`tests/test_suite.py`)
The automated test suite verifies 8 core subsystems:

1. **`test_01_database_initialization`**: Verifies SQLite table creation and indexes.
2. **`test_02_nlp_preprocessing`**: Tests 8-stage text normalization and WordNet lemmatization.
3. **`test_03_narrative_phase_thresholds`**: Tests deterministic mapping of composite scores to narrative phases.
4. **`test_04_composite_sentiment_calculation`**: Verifies score bounds $[-1.0, +1.0]$.
5. **`test_05_explainable_ai`**: Tests XAI TF-IDF term feature attribution.
6. **`test_06_backtesting_engine`**: Tests calculation of MAE, RMSE, Win Rate %, and Sharpe Ratio.
7. **`test_07_paper_trading_execution`**: Tests virtual order book execution and portfolio valuation.
8. **`test_08_system_health_diagnostics`**: Tests component health monitoring.

## Running Tests
```bash
python -m unittest discover -s tests -p "test_suite.py"
```
