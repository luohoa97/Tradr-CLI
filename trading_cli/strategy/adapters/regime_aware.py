"""Regime-Aware Hybrid Strategy.

Dynamically switches between Trend Following and Mean Reversion based on
market volatility and trend strength (Regime Detection).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import register_strategy
from trading_cli.strategy.signals import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_atr,
    calculate_sma,
    calculate_ema,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_strategy
class RegimeAwareStrategy(StrategyAdapter):
    """Regime-Aware Strategy.

    Detects if the market is 'Trending' or 'Ranging' and uses the
    appropriate sub-strategy.
    - Trending: Donchian Breakout + MACD/EMA
    - Ranging: RSI(2) + Bollinger Bands
    - Sentiment: Acts as a final confirmation filter.
    """

    @property
    def strategy_id(self) -> str:
        return "regime_aware"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Regime-Aware Hybrid",
            description=(
                "Detects market regime (Trending vs Ranging). "
                "Uses breakouts in trending markets and mean-reversion in ranges. "
                "Applies a sentiment filter to increase conviction."
            ),
            params_schema={
                "volatility_threshold": {"type": "float", "default": 0.015, "desc": "ATR/Price threshold for volatility"},
                "trend_threshold": {"type": "float", "default": 0.02, "desc": "SMA distance as % of price to consider trending"},
                "rsi_period": {"type": "int", "default": 2, "desc": "RSI period for mean reversion"},
                "bb_window": {"type": "int", "default": 20, "desc": "Bollinger window"},
                "sentiment_threshold": {"type": "float", "default": 0.2, "desc": "Min sentiment for confirmation"},
            },
        )

    def generate_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        sentiment_score: float = 0.0,
        positions: list | None = None,
        **kwargs,
    ) -> SignalResult:
        config = self.config
        close_col = "close" if "close" in ohlcv.columns else "Close"
        high_col = "high" if "high" in ohlcv.columns else "High"
        low_col = "low" if "low" in ohlcv.columns else "Low"

        if close_col not in ohlcv.columns:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "missing data")

        closes = ohlcv[close_col]
        if len(closes) < 50:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "insufficient data")

        current_price = closes.iloc[-1]
        
        # 1. Regime Detection
        # ATR / Price as volatility measure
        atr = calculate_atr(ohlcv, 14).iloc[-1]
        atr_pct = (atr / current_price) if current_price > 0 else 0
        
        # SMA Distance as Trend Strength measure
        sma20 = calculate_sma(closes, 20).iloc[-1]
        sma50 = calculate_sma(closes, 50).iloc[-1]
        trend_dist = abs(sma20 - sma50) / sma50 if sma50 > 0 else 0
        
        vol_threshold = config.get("volatility_threshold", 0.015)
        trend_threshold = config.get("trend_threshold", 0.02)
        
        is_trending = trend_dist > trend_threshold or atr_pct > vol_threshold
        
        # 2. Strategy Logic
        in_position = any(p.symbol == symbol for p in (positions or []))
        reason_parts = []
        
        if is_trending:
            # TREND FOLLOWING (Breakout)
            regime = "TRENDING"
            # Donchian Breakout (from previous highs)
            prev_highs = ohlcv[high_col].shift(1)
            donchian_high = prev_highs.rolling(20).max().iloc[-1]
            prev_lows = ohlcv[low_col].shift(1)
            donchian_low = prev_lows.rolling(10).min().iloc[-1]
            
            if current_price >= donchian_high and not in_position:
                action = "BUY"
                score = 0.7
                reason_parts.append(f"Trending Breakout (${current_price:.2f} >= ${donchian_high:.2f})")
            elif current_price <= donchian_low and in_position:
                action = "SELL"
                score = -0.7
                reason_parts.append(f"Trending Breakdown (${current_price:.2f} <= ${donchian_low:.2f})")
            else:
                action = "HOLD"
                score = 0.0
                reason_parts.append("Trending (No Breakout)")
        else:
            # RANGE (Mean Reversion)
            regime = "RANGING"
            rsi = calculate_rsi(closes, config.get("rsi_period", 2)).iloc[-1]
            upper, middle, lower = calculate_bollinger_bands(closes, config.get("bb_window", 20), 2.0)
            l_band = lower.iloc[-1]
            m_band = middle.iloc[-1]
            
            if rsi < 15 and current_price <= l_band and not in_position:
                action = "BUY"
                score = 0.6
                reason_parts.append(f"Range Oversold (RSI={rsi:.0f}, Price <= BB Lower)")
            elif (rsi > 80 or current_price >= m_band) and in_position:
                action = "SELL"
                score = -0.6
                reason_parts.append(f"Range Mean Reversion (RSI={rsi:.0f} or Price >= BB Middle)")
            else:
                action = "HOLD"
                score = 0.0
                reason_parts.append("Ranging (No Signal)")

        # 3. Sentiment Filter
        sent_threshold = config.get("sentiment_threshold", 0.2)
        if action == "BUY" and sentiment_score < -sent_threshold:
            # Strong negative sentiment cancels buy
            action = "HOLD"
            reason_parts.append(f"Buy cancelled by Sentiment ({sentiment_score:+.2f})")
            score = 0.1
        elif action == "SELL" and sentiment_score > sent_threshold:
            # Strong positive sentiment delays sell (unless profit taking)
            # We'll keep the sell for now as safety first, or just lower confidence
            score *= 0.5
            reason_parts.append(f"Sell conviction lowered by Sentiment ({sentiment_score:+.2f})")

        return SignalResult(
            symbol=symbol,
            action=action,
            confidence=abs(score),
            score=score,
            reason=f"[{regime}] " + " + ".join(reason_parts),
            metadata={"regime": regime, "vol": atr_pct, "trend": trend_dist}
        )
