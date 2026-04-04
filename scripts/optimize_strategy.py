#!/usr/bin/env python3
"""
Grid search optimizer for trading strategies.
Tests multiple parameter combinations to find the best performing one.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from itertools import product

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cli.backtest.engine import BacktestEngine
from trading_cli.strategy.strategy_factory import create_trading_strategy
from trading_cli.data.market import fetch_ohlcv_yfinance

# Configure logging
logging.basicConfig(level=logging.WARNING)

def optimize_mean_reversion(symbols, days=180):
    # Fetch data once
    ohlcv_data = {}
    for symbol in symbols:
        df = fetch_ohlcv_yfinance(symbol, days=days)
        if not df.empty:
            ohlcv_data[symbol] = df
            
    if not ohlcv_data:
        print("No data fetched.")
        return

    # Parameter grid
    rsi_oversold_vals = [5, 10, 15, 20]
    rsi_overbought_vals = [70, 80, 85, 90]
    bb_std_vals = [1.0, 1.5, 2.0, 2.5]
    
    results = []
    
    combinations = list(product(rsi_oversold_vals, rsi_overbought_vals, bb_std_vals))
    print(f"Testing {len(combinations)} combinations across {len(ohlcv_data)} symbols...")
    
    for rsi_os, rsi_ob, bb_std in combinations:
        config = {
            "strategy_id": "mean_reversion",
            "rsi_oversold": rsi_os,
            "rsi_overbought": rsi_ob,
            "bb_std": bb_std,
            "risk_pct": 0.02,
        }
        
        total_return = 0
        total_sharpe = 0
        total_win_rate = 0
        total_trades = 0
        
        for symbol, ohlcv in ohlcv_data.items():
            strategy = create_trading_strategy(config)
            engine = BacktestEngine(config=config, use_sentiment=False, strategy=strategy)
            res = engine.run(symbol, ohlcv)
            
            total_return += res.total_return_pct
            total_sharpe += res.sharpe_ratio
            total_win_rate += res.win_rate
            total_trades += res.total_trades
            
        avg_return = total_return / len(ohlcv_data)
        avg_sharpe = total_sharpe / len(ohlcv_data)
        avg_win_rate = total_win_rate / len(ohlcv_data)
        
        results.append({
            "rsi_os": rsi_os,
            "rsi_ob": rsi_ob,
            "bb_std": bb_std,
            "avg_return": avg_return,
            "avg_sharpe": avg_sharpe,
            "avg_win_rate": avg_win_rate,
            "total_trades": total_trades
        })
        
    # Sort results
    df = pd.DataFrame(results)
    best = df.sort_values("avg_return", ascending=False).head(10)
    
    print("\n--- Top 10 Configurations ---")
    print(best)
    
    return best

if __name__ == "__main__":
    optimize_mean_reversion(["AAPL", "MSFT", "NVDA", "TSLA", "AMD", "COIN"], days=180)
