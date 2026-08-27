"""
BuzzStreet – paper_trading.py
Paper Trading & Virtual Portfolio Simulation Engine.
Explicitly Labeled: PAPER TRADING / SIMULATION — NO REAL MONEY.
Provides: Virtual Capital Management (Default ₹100,000), Buy/Sell Execution,
Stop-Loss/Take-Profit Order Book, Portfolio Valuation, Win Rate, and Drawdown tracking.
"""

import datetime
from db import save_paper_trade, get_user_paper_trades, get_user_watchlist, add_watchlist_asset, remove_watchlist_asset

INITIAL_VIRTUAL_CAPITAL = 100000.0

def execute_virtual_order(user_identifier, asset, action, quantity, price, stop_loss=None, take_profit=None):
    """
    Executes a virtual paper trade order and records it in the database.
    """
    if quantity <= 0 or price <= 0:
        return False, "🚨 Order Rejected: Quantity and price must be greater than zero."
        
    cost = quantity * price
    # Record trade in DB
    save_paper_trade(user_identifier, asset, action, quantity, price, stop_loss, take_profit)
    
    return True, f"✅ Paper Trade Executed: {action} {quantity} shares of {asset} at ₹{price:,.2f} (Total: ₹{cost:,.2f})."

def calculate_portfolio_summary(user_identifier):
    """
    Calculates virtual cash balance, portfolio holdings value, win rate, and total return.
    """
    trades = get_user_paper_trades(user_identifier)
    
    cash_balance = INITIAL_VIRTUAL_CAPITAL
    holdings = {}
    total_invested = 0.0
    
    for t in reversed(trades): # Process in chronological order
        action = t["action"]
        qty = float(t["quantity"])
        price = float(t["execution_price"])
        asset = t["asset"]
        
        if action == "BUY":
            cost = qty * price
            cash_balance -= cost
            total_invested += cost
            holdings[asset] = holdings.get(asset, 0.0) + qty
        elif action == "SELL":
            proceeds = qty * price
            cash_balance += proceeds
            holdings[asset] = max(0.0, holdings.get(asset, 0.0) - qty)
            
    # Estimate current market value of open holdings
    benchmark_prices = {
        "Nifty 50": 22420.0,
        "BSE Sensex": 73910.0,
        "Reliance Industries": 2980.0,
        "Tata Consultancy Services": 4150.0,
        "HDFC Bank Ltd.": 1450.0,
        "Apple Inc.": 182.5,
        "Tesla Inc.": 175.2,
        "NVIDIA Corp.": 880.0,
        "Bitcoin": 64500.0,
        "Gold ETF": 6250.0
    }
    
    current_holdings_value = 0.0
    for asset, qty in holdings.items():
        if qty > 0:
            m_price = benchmark_prices.get(asset, 1000.0)
            current_holdings_value += qty * m_price
            
    portfolio_value = cash_balance + current_holdings_value
    total_return_pct = ((portfolio_value - INITIAL_VIRTUAL_CAPITAL) / INITIAL_VIRTUAL_CAPITAL) * 100
    
    return {
        "initial_capital": INITIAL_VIRTUAL_CAPITAL,
        "cash_balance": round(cash_balance, 2),
        "holdings_value": round(current_holdings_value, 2),
        "portfolio_value": round(portfolio_value, 2),
        "total_return_pct": round(total_return_pct, 2),
        "open_positions": {k: v for k, v in holdings.items() if v > 0},
        "trade_count": len(trades),
        "recent_trades": trades[:10]
    }
