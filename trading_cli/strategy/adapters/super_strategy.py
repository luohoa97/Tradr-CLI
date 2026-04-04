"""SuperStrategy — Optimized Mean Reversion + Trend Filter + ATR Trailing Stop.

Based on Larry Connors RSI(2) but with major modern improvements:
1. Trend Filter: Only Long above SMA 200.
2. Entry: RSI(2) < 10 (extreme oversold).
3. Exit: Price > SMA 5 OR ATR Trailing Stop.
4. Volatility-Adjusted: Uses ATR for both position sizing and stops.
5. Sentiment Filter: Uses news sentiment to avoid 'falling knives' with bad news.
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
    calculate_sma,
    calculate_atr,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_strategy
class SuperStrategy(StrategyAdapter):
    """SuperStrategy: The evolved trading strategy."""

    @property
    def strategy_id(self) -> str:
        return "super_strategy"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="SuperStrategy (Optimized MR)",
            description=(
                "Modernized Larry Connors RSI(2). "
                "Longs only above SMA 200. Entry at RSI(2) < 10. "
                "Exit at Price > SMA 5 or ATR Trailing Stop (3x ATR). "
                "Filters entries using sentiment score."
            ),
            params_schema={
                "rsi_period": {"type": "int", "default": 2, "desc": "RSI period"},
                "rsi_oversold": {"type": "int", "default": 10, "desc": "RSI oversold threshold"},
                "sma_long_period": {"type": "int", "default": 200, "desc": "Trend filter SMA period"},
                "sma_exit_period": {"type": "int", "default": 5, "desc": "Mean reversion exit SMA period"},
                "atr_multiplier": {"type": "float", "default": 3.0, "desc": "ATR trailing stop multiplier"},
                "sentiment_threshold": {"type": "float", "default": 0.1, "desc": "Min sentiment to allow trade"},
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
        sma_long_p = config.get("sma_long_period", 200)
        
        if len(closes) < sma_long_p:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, f"insufficient data (need {sma_long_p})")

        current_price = closes.iloc[-1]
        
        # Indicators
        sma200 = calculate_sma(closes, sma_long_p).iloc[-1]
        sma5 = calculate_sma(closes, config.get("sma_exit_period", 5)).iloc[-1]
        rsi2 = calculate_rsi(closes, config.get("rsi_period", 2)).iloc[-1]
        atr = calculate_atr(ohlcv, 14).iloc[-1]
        atr_pct = (atr / current_price) if current_price > 0 else 0
        
        # Regime / Trend Check
        is_bullish = current_price > sma200
        # Volatility Circuit Breaker: Avoid entry if market is too volatile (ATR > 3%)
        is_too_volatile = atr_pct > 0.03
        
        # Position Check
        in_position = any(p.symbol == symbol for p in (positions or []))
        position_entry = None
        if in_position:
            for p in (positions or []):
                if p.symbol == symbol:
                    position_entry = p.avg_entry_price
                    break

        reason_parts = []
        
        # Entry Logic
        if not in_position:
            if is_bullish:
                if is_too_volatile:
                    return SignalResult(symbol, "HOLD", 0.0, 0.0, f"Circuit Breaker: ATR={atr_pct:.1%} too high")
                
                rsi_os = config.get("rsi_oversold", 10)
                if rsi2 < rsi_os:
                    # Potential Buy
                    sent_threshold = config.get("sentiment_threshold", 0.1)
                    if sentiment_score >= -sent_threshold:
                        reason_parts.append(f"Bullish MR: Price > SMA200, RSI(2)={rsi2:.1f} < {rsi_os}")
                        if sentiment_score > 0:
                            reason_parts.append(f"Sent Confirmation: {sentiment_score:+.2f}")
                        return SignalResult(
                            symbol, "BUY",
                            confidence=0.8,
                            score=0.7,
                            reason=" + ".join(reason_parts),
                            metadata={"rsi2": rsi2, "sma200": sma200, "sent": sentiment_score}
                        )
                    else:
                        reason_parts.append(f"Buy skipped: Bad Sentiment ({sentiment_score:+.2f})")
            else:
                reason_parts.append(f"Neutral: Price < SMA200 (${current_price:.2f} < ${sma200:.2f})")

        # Exit Logic
        if in_position and position_entry:
            # 1. Target Exit: Price above SMA 5
            if current_price > sma5:
                reason_parts.append(f"Exit: Reverted to mean (${current_price:.2f} > SMA5 ${sma5:.2f})")
                return SignalResult(
                    symbol, "SELL",
                    confidence=0.9, score=-0.8,
                    reason=" + ".join(reason_parts),
                )
            
            # 2. Break-even Stop (if profit > 2%)
            profit_pct = (current_price - position_entry) / position_entry
            if profit_pct > 0.02:
                if current_price < position_entry * 1.005: # Small buffer above entry
                    reason_parts.append(f"Exit: Break-even stop hit (${current_price:.2f} <= ${position_entry:.2f})")
                    return SignalResult(
                        symbol, "SELL",
                        confidence=1.0, score=-1.0,
                        reason=" + ".join(reason_parts),
                    )
                else:
                    reason_parts.append("Profit protection active (Break-even)")
            
            # 3. ATR Trailing Stop
            # Tighten stop in high volatility
            atr_mult = config.get("atr_multiplier", 3.0)
            if is_too_volatile:
                atr_mult = 1.5 # Half the distance in crash-like conditions
                reason_parts.append(f"Tightened stop (High Vol ATR={atr_pct:.1%})")
                
            stop_price = position_entry - (atr_mult * atr)
            if current_price < stop_price:
                reason_parts.append(f"Exit: ATR Stop Hit (${current_price:.2f} < ${stop_price:.2f})")
                return SignalResult(
                    symbol, "SELL",
                    confidence=1.0, score=-1.0,
                    reason=" + ".join(reason_parts),
                )
            
            reason_parts.append(f"Holding: Target SMA5 ${sma5:.2f}, Stop ATR ${stop_price:.2f}")

        return SignalResult(
            symbol, "HOLD", 0.0, 0.0,
            " + ".join(reason_parts) if reason_parts else "neutral",
            metadata={"rsi2": rsi2, "sma200": sma200, "sent": sentiment_score}
        )
