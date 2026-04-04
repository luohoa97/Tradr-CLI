#!/usr/bin/env python3
"""
Multi-stock backtesting script for strategy evolution.
Tests one or more strategies across multiple symbols and timeframes.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cli.backtest.engine import BacktestEngine
from trading_cli.strategy.strategy_factory import create_trading_strategy, available_strategies
from trading_cli.data.market import fetch_ohlcv_yfinance

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "COIN", "MARA"]
DEFAULT_DAYS = 365

def run_multi_backtest(symbols, strategy_ids, days=DEFAULT_DAYS, config=None):
    if config is None:
        config = {
            "signal_buy_threshold": 0.2,
            "signal_sell_threshold": -0.15,
            "risk_pct": 0.02,
            "stop_loss_pct": 0.05,
        }
    
    results = []
    
    print(f"{'Symbol':<8} | {'Strategy':<15} | {'Return %':>10} | {'Sharpe':>8} | {'Win%':>6} | {'Trades':>6}")
    print("-" * 70)
    
    for symbol in symbols:
        # Fetch data once per symbol
        ohlcv = fetch_ohlcv_yfinance(symbol, days=days)
        if ohlcv.empty:
            print(f"Failed to fetch data for {symbol}")
            continue
            
        for strategy_id in strategy_ids:
            # Create strategy
            strat_config = config.copy()
            strat_config["strategy_id"] = strategy_id
            strategy = create_trading_strategy(strat_config)
            
            # Run backtest
            engine = BacktestEngine(
                config=strat_config,
                use_sentiment=False,  # Skip sentiment for pure technical baseline
                strategy=strategy
            )
            
            res = engine.run(symbol, ohlcv, initial_capital=100_000.0)
            
            print(f"{symbol:<8} | {strategy_id:<15} | {res.total_return_pct:>9.2f}% | {res.sharpe_ratio:>8.2f} | {res.win_rate:>5.1f}% | {res.total_trades:>6}")
            
            results.append({
                "symbol": symbol,
                "strategy": strategy_id,
                "return_pct": res.total_return_pct,
                "sharpe": res.sharpe_ratio,
                "win_rate": res.win_rate,
                "trades": res.total_trades,
                "max_drawdown": res.max_drawdown_pct
            })
            
    # Aggregate results by strategy
    df = pd.DataFrame(results)
    if not df.empty:
        summary = df.groupby("strategy").agg({
            "return_pct": ["mean", "std"],
            "sharpe": "mean",
            "win_rate": "mean",
            "trades": "sum"
        })
        print("\n--- Summary ---")
        print(summary)
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--strategies", nargs="+", default=["hybrid", "mean_reversion", "momentum", "trend_following"])
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS)
    args = parser.parse_args()
    
    run_multi_backtest(args.symbols, args.strategies, args.days)
