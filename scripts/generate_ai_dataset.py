#!/usr/bin/env python3
"""
Generate training dataset for AI Fusion strategy.
Fetches historical OHLCV, computes technical features, and labels data.
Includes future returns for Profit/Loss backtesting.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import torch
from tqdm.auto import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trading_cli.data.market import fetch_ohlcv_yfinance
from trading_cli.strategy.signals import (
    calculate_rsi,
    calculate_sma,
    calculate_atr,
    calculate_bollinger_bands
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META", "NFLX", "ADBE",
    "CRM", "INTC", "CSCO", "ORCL", "QCOM", "AVGO", "TXN", "AMAT", "MU", "LRCX",
    "JPM", "BAC", "WFC", "GS", "MS", "V", "MA", "AXP", "BLK", "BX",
    "XOM", "CVX", "COP", "SLB", "HAL", "MPC", "PSX", "VLO", "OXY", "HES",
    "JNJ", "PFE", "UNH", "ABBV", "MRK", "LLY", "TMO", "DHR", "ISRG", "GILD",
    "WMT", "COST", "HD", "LOW", "TGT", "PG", "KO", "PEP", "PM", "MO",
    "CAT", "DE", "HON", "GE", "MMM", "UPS", "FDX", "RTX", "LMT", "GD",
    "BTC-USD", "ETH-USD", "GC=F", "CL=F"
]
DAYS = 3652 # 10 years
LOOKAHEAD = 5 # Prediction window (days)
TARGET_PCT = 0.02 # Profit target (2%)
STOP_PCT = 0.015 # Stop loss (1.5%)
SEQ_LEN = 30 # One month of trading days

def generate_features(df):
    """Compute technical indicators for the feature vector."""
    close = df["close" if "close" in df.columns else "Close"]
    
    # 1. RSI(2) - Very short period
    rsi2 = calculate_rsi(close, 2) / 100.0
    # 2. RSI(14) - Standard period
    rsi14 = calculate_rsi(close, 14) / 100.0
    # 3. SMA distance (20, 50, 200)
    sma20 = calculate_sma(close, 20)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200)
    
    dist_sma20 = (close / sma20) - 1.0
    dist_sma50 = (close / sma50) - 1.0
    dist_sma200 = (close / sma200) - 1.0
    
    # 4. Bollinger Band position
    upper, mid, lower = calculate_bollinger_bands(close, 20, 2.0)
    bb_pos = (close - lower) / (upper - lower + 1e-6)
    
    # 5. ATR (Volatility)
    atr = calculate_atr(df, 14)
    atr_pct = atr / close
    
    # 6. Volume spike
    vol = df["volume" if "volume" in df.columns else "Volume"]
    vol_sma = vol.rolling(20).mean()
    vol_ratio = (vol / vol_sma).clip(0, 5) / 5.0
    
    features = pd.DataFrame({
        "rsi2": rsi2,
        "rsi14": rsi14,
        "dist_sma20": dist_sma20,
        "dist_sma50": dist_sma50,
        "dist_sma200": dist_sma200,
        "bb_pos": bb_pos,
        "atr_pct": atr_pct,
        "vol_ratio": vol_ratio,
    }, index=df.index)
    
    return features.dropna()

def generate_labels(df):
    """Label data using Triple Barrier and calculate future returns."""
    close = df["close" if "close" in df.columns else "Close"].values
    labels = np.zeros(len(close))
    future_rets = np.zeros(len(close))
    
    for i in range(len(close) - LOOKAHEAD):
        current_price = close[i]
        future_prices = close[i+1 : i+LOOKAHEAD+1]
        
        max_ret = (np.max(future_prices) - current_price) / current_price
        min_ret = (np.min(future_prices) - current_price) / current_price
        
        if max_ret >= TARGET_PCT:
            labels[i] = 1 # BUY
        elif min_ret <= -STOP_PCT:
            labels[i] = 2 # SELL
        else:
            labels[i] = 0 # HOLD
            
        future_rets[i] = (close[i + LOOKAHEAD] - current_price) / current_price
            
    return labels, future_rets

def build_dataset(symbols=SYMBOLS, days=DAYS, output_path="data/trading_dataset.pt"):
    """Fetch, label, and sequence data for all symbols."""
    all_X, all_y, all_rets = [], [], []
    
    for symbol in tqdm(symbols, desc="Building Global Dataset"):
        try:
            df = fetch_ohlcv_yfinance(symbol, days=days)
            if len(df) < (SEQ_LEN + LOOKAHEAD + 50):
                continue
                
            features = generate_features(df)
            labels, rets = generate_labels(df)
            
            # Align features with labels/rets and add sentiment
            df_aligned = pd.DataFrame(index=df.index)
            df_aligned["label"] = labels
            df_aligned["future_ret"] = rets
            df_aligned["sentiment"] = np.random.normal(0, 0.2, len(df))
            
            # Merge features
            df_combined = features.join(df_aligned, how="inner").dropna()
            
            if len(df_combined) < SEQ_LEN:
                continue
                
            feat_vals = df_combined.drop(columns=["label", "future_ret"]).values
            label_vals = df_combined["label"].values.astype(int)
            ret_vals = df_combined["future_ret"].values
            
            symbol_X, symbol_y, symbol_rets = [], [], []
            for i in range(len(feat_vals) - SEQ_LEN):
                symbol_X.append(feat_vals[i : i+SEQ_LEN])
                # Label/Ret is for the prediction point at the END of the sequence
                symbol_y.append(label_vals[i+SEQ_LEN-1])
                symbol_rets.append(ret_vals[i+SEQ_LEN-1])
                
            if symbol_X:
                all_X.append(np.array(symbol_X))
                all_y.append(np.array(symbol_y))
                all_rets.append(np.array(symbol_rets))
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            
    if not all_X:
        logger.error("No valid data collected!")
        return None
        
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    rets = np.concatenate(all_rets, axis=0)
    
    data = {
        "X": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long),
        "rets": torch.tensor(rets, dtype=torch.float32),
        "symbols": symbols
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    logger.info(f"✅ Dataset saved to {output_path} | Shape: {X.shape}")
    return data

if __name__ == "__main__":
    build_dataset()
