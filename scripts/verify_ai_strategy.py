import pandas as pd
import torch
import logging
from trading_cli.strategy.adapters.ai_fusion import AIFusionStrategy
from trading_cli.data.market import fetch_ohlcv_yfinance

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ai_fusion():
    symbol = "AAPL"
    logger.info(f"Testing AI Fusion Strategy for {symbol}...")
    
    # 1. Fetch data
    df = fetch_ohlcv_yfinance(symbol, days=250)
    if df.empty:
        logger.error("Failed to fetch data")
        return
    
    # 2. Instantiate strategy
    strategy = AIFusionStrategy()
    
    # 3. Generate signal
    # Note: sentiment_score is optional, defaults to 0.0
    result = strategy.generate_signal(symbol, df, sentiment_score=0.1)
    
    # 4. Print result
    logger.info("Signal Result:")
    logger.info(f"  Symbol:     {result.symbol}")
    logger.info(f"  Action:     {result.action}")
    logger.info(f"  Confidence: {result.confidence:.2%}")
    logger.info(f"  Reason:     {result.reason}")
    
    if result.metadata:
        logger.info(f"  Metadata:   {result.metadata}")

if __name__ == "__main__":
    test_ai_fusion()
