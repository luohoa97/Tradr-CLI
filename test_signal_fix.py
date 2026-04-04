#!/usr/bin/env python
"""Quick test to verify signal generation works without errors."""

import pandas as pd
import numpy as np
from trading_cli.strategy.signals import (
    volume_score,
    calculate_atr,
    sma_crossover_score,
    rsi_score,
    bollinger_score,
    ema_score,
    technical_score,
    generate_signal,
)

# Create sample OHLCV data
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ohlcv = pd.DataFrame({
    'Date': dates,
    'Open': np.random.uniform(100, 200, 100),
    'High': np.random.uniform(150, 250, 100),
    'Low': np.random.uniform(90, 190, 100),
    'Close': np.random.uniform(100, 200, 100),
    'Volume': np.random.randint(1000000, 10000000, 100),
})

print("Testing individual score functions...")

# Test volume_score
try:
    vol = volume_score(ohlcv)
    print(f"✓ volume_score: {vol:.3f}")
except Exception as e:
    print(f"✗ volume_score FAILED: {e}")

# Test calculate_atr
try:
    atr = calculate_atr(ohlcv)
    print(f"✓ calculate_atr: {atr.iloc[-1]:.3f}")
except Exception as e:
    print(f"✗ calculate_atr FAILED: {e}")

# Test sma_crossover_score
try:
    sma = sma_crossover_score(ohlcv)
    print(f"✓ sma_crossover_score: {sma:.3f}")
except Exception as e:
    print(f"✗ sma_crossover_score FAILED: {e}")

# Test rsi_score
try:
    rsi = rsi_score(ohlcv)
    print(f"✓ rsi_score: {rsi:.3f}")
except Exception as e:
    print(f"✗ rsi_score FAILED: {e}")

# Test bollinger_score
try:
    bb = bollinger_score(ohlcv)
    print(f"✓ bollinger_score: {bb:.3f}")
except Exception as e:
    print(f"✗ bollinger_score FAILED: {e}")

# Test ema_score
try:
    ema = ema_score(ohlcv)
    print(f"✓ ema_score: {ema:.3f}")
except Exception as e:
    print(f"✗ ema_score FAILED: {e}")

# Test technical_score
try:
    tech = technical_score(ohlcv)
    print(f"✓ technical_score: {tech:.3f}")
except Exception as e:
    print(f"✗ technical_score FAILED: {e}")

# Test generate_signal
try:
    signal = generate_signal(
        symbol="AAPL",
        ohlcv=ohlcv,
        sentiment_score=0.5,
        tech_weight=0.6,
        sent_weight=0.4,
    )
    print(f"\n✓ generate_signal:")
    print(f"  Symbol: {signal['symbol']}")
    print(f"  Action: {signal['action']}")
    print(f"  Confidence: {signal['confidence']:.3f}")
    print(f"  Hybrid Score: {signal['hybrid_score']:.3f}")
    print(f"  Reason: {signal['reason']}")
except Exception as e:
    print(f"\n✗ generate_signal FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests completed!")
