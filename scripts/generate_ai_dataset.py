#!/usr/bin/env python3
"""
Generate training dataset for AI Fusion strategy.
Fetches historical OHLCV, computes technical features, and labels data.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
import torch
from datetime import datetime, timedelta

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
    "BTC-USD", "ETH-USD", "GC=F", "CL=F" # Crypto and Commodities for diversity
]
DAYS = 3652 # 10 years
LOOKAHEAD = 5 # Prediction window (days)
TARGET_PCT = 0.02 # Profit target (2%)
STOP_PCT = 0.015 # Stop loss (1.5%)

def generate_features(df):
    """Compute technical indicators for the feature vector."""
    close = df["close" if "close" in df.columns else "Close"]
    high = df["high" if "high" in df.columns else "High"]
    low = df["low" if "low" in df.columns else "Low"]
    
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
    
    # 6. Volume spike (Ratio to SMA 20)
    vol = df["volume" if "volume" in df.columns else "Volume"]
    vol_sma = vol.rolling(20).mean()
    vol_ratio = (vol / vol_sma).clip(0, 5) / 5.0 # Normalized 0-1
    
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
    
    # Ensure all columns are 1D (should be Series already after flatten in market.py)
    for col in features.columns:
        if isinstance(features[col], pd.DataFrame):
            features[col] = features[col].squeeze()
            
    return features

def generate_labels(df):
    """Label data using Triple Barrier: 1=Buy, 2=Sell, 0=Hold."""
    close = df["close" if "close" in df.columns else "Close"].values
    labels = np.zeros(len(close))
    
    for i in range(len(close) - LOOKAHEAD):
        current_price = close[i]
        future_prices = close[i+1 : i+LOOKAHEAD+1]
        
        # Look ahead for profit target or stop loss
        max_ret = (np.max(future_prices) - current_price) / current_price
        min_ret = (np.min(future_prices) - current_price) / current_price
        
        if max_ret >= TARGET_PCT:
            labels[i] = 1 # BUY
        elif min_ret <= -STOP_PCT:
            labels[i] = 2 # SELL
        else:
            labels[i] = 0 # HOLD
            
    return labels

SEQ_LEN = 30 # One month of trading days

def build_dataset(symbols=SYMBOLS, days=DAYS, output_path="data/trading_dataset.pt"):
    """
    Programmatically build the sequence dataset.
    Used by local scripts and the Hugging Face Cloud trainer.
    """
    all_features = []
    all_labels = []
    
    for symbol in symbols:
        logger.info("Fetching data for %s", symbol)
        df = fetch_ohlcv_yfinance(symbol, days=days)
        total_days = len(df)
        if df.empty or total_days < (days // 2): # Ensure we have enough data
            logger.warning("Skipping %s: Insufficient history (%d < %d)", symbol, total_days, days // 2)
            continue
            
        features = generate_features(df)
        labels = generate_labels(df)
        
        # Sentiment simulation
        sentiment = np.random.normal(0, 0.2, len(features))
        features["sentiment"] = sentiment
        
        # Combine and drop NaN
        features["label"] = labels
        features = features.dropna()
        
        if len(features) < (SEQ_LEN + 100):
            logger.warning("Skipping %s: Too few valid samples after dropna (%d < %d)", symbol, len(features), SEQ_LEN + 100)
            continue
            
        # Create sequences
        feat_vals = features.drop(columns=["label"]).values
        label_vals = features["label"].values
        
        symbol_features = []
        symbol_labels = []
        
        for i in range(len(feat_vals) - SEQ_LEN):
            # Window of features: [i : i + SEQ_LEN]
            # Label is for the LAST day in the window
            symbol_features.append(feat_vals[i : i+SEQ_LEN])
            symbol_labels.append(label_vals[i+SEQ_LEN-1])
            
        all_features.append(np.array(symbol_features))
        all_labels.append(np.array(symbol_labels))
        
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Save as PyTorch dataset
    data = {
        "X": torch.tensor(X, dtype=torch.float32),
        "y": torch.tensor(y, dtype=torch.long)
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(data, output_path)
    logger.info("Sequence dataset saved to %s. Shape: %s", output_path, X.shape)
    return data

if __name__ == "__main__":
    build_dataset()
