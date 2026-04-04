"""Trend Following strategy — Donchian Channel Breakout (Turtle Trading).

Entry: Close > 20-day high (upper Donchian channel)
Exit: Close < 10-day low (lower Donchian channel)
Filter: ATR filter to skip sideways/low-volatility markets

Proven characteristics:
  - Win rate: 30–45%
  - Gain/Loss ratio: >2:1 (often ~4:1)
  - Few trades, large winners, quick losers
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import register_strategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_strategy
class TrendFollowingStrategy(StrategyAdapter):
    """Donchian Channel Breakout (Turtle Trading) with ATR filter.

    This is one of the most proven trend-following systems. It doesn't
    try to predict — it simply follows price breakouts with strict rules.
    """

    @property
    def strategy_id(self) -> str:
        return "trend_following"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Trend Following (Donchian Breakout)",
            description=(
                "Entry: Price breaks above 20-day Donchian high. "
                "Exit: Price breaks below 10-day Donchian low. "
                "Filter: Skip when ATR/volatility is too low (sideways market). "
                "Low win rate (~35%) but high reward ratio (>3:1)."
            ),
            params_schema={
                "entry_period": {"type": "int", "default": 20, "desc": "Donchian entry lookback (high breakout)"},
                "exit_period": {"type": "int", "default": 10, "desc": "Donchian exit lookback (low breakdown)"},
                "atr_period": {"type": "int", "default": 20, "desc": "ATR period for volatility filter"},
                "atr_multiplier": {"type": "float", "default": 0.5, "desc": "Minimum ATR as % of price to trade (filter sideways)"},
                "signal_buy_threshold": {"type": "float", "default": 0.0, "desc": "Not used — signals are binary"},
                "signal_sell_threshold": {"type": "float", "default": 0.0, "desc": "Not used — signals are binary"},
            },
        )

    def generate_signal(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        sentiment_score: float = 0.0,
        prices: dict[str, float] | None = None,
        positions: list | None = None,
        portfolio_value: float = 0.0,
        cash: float = 0.0,
        **kwargs,
    ) -> SignalResult:
        config = self.config
        close_col = "close" if "close" in ohlcv.columns else "Close"
        high_col = "high" if "high" in ohlcv.columns else "High"
        low_col = "low" if "low" in ohlcv.columns else "Low"

        if close_col not in ohlcv.columns or high_col not in ohlcv.columns or low_col not in ohlcv.columns:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "missing data")

        closes = ohlcv[close_col]
        highs = ohlcv[high_col]
        lows = ohlcv[low_col]

        entry_period = config.get("entry_period", 20)
        exit_period = config.get("exit_period", 10)
        atr_period = config.get("atr_period", 20)
        atr_min_pct = config.get("atr_multiplier", 0.5) / 100.0

        if len(closes) < entry_period + 5:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "insufficient data")

        # Donchian channels
        donchian_high = highs.rolling(window=entry_period, min_periods=entry_period).max().iloc[-1]
        donchian_low = lows.rolling(window=exit_period, min_periods=exit_period).min().iloc[-1]
        current_price = closes.iloc[-1]

        # ATR filter — skip if market is too quiet (sideways)
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period, min_periods=atr_period).mean().iloc[-1]
        atr_pct = atr / current_price if current_price > 0 else 0

        # Check if we're already in a position
        in_position = any(p.symbol == symbol for p in (positions or []))

        reason_parts = []

        # ATR filter
        if atr_pct < atr_min_pct:
            reason_parts.append(f"ATR={atr_pct:.1%} (too quiet)")
            if in_position:
                # If already in position, don't force exit on low vol
                pass
            else:
                return SignalResult(
                    symbol, "HOLD", 0.0, 0.0,
                    " + ".join(reason_parts) if reason_parts else "low volatility filter",
                    metadata={"donchian_high": donchian_high, "donchian_low": donchian_low, "atr": atr},
                )

        # Entry signal: breakout above 20-day high
        if current_price >= donchian_high and not in_position:
            score = min(1.0, (current_price - donchian_high) / (atr or 1.0))
            reason_parts.append(f"Breakout ${current_price:.2f} >= ${donchian_high:.2f}")
            if atr > 0:
                reason_parts.append(f"ATR={atr:.2f} ({atr_pct:.1%})")
            return SignalResult(
                symbol, "BUY",
                confidence=min(1.0, score + 0.3),
                score=score,
                reason=" + ".join(reason_parts),
                metadata={"donchian_high": donchian_high, "donchian_low": donchian_low, "atr": atr},
            )

        # Exit signal: breakdown below 10-day low
        if current_price <= donchian_low and in_position:
            score = -min(1.0, (donchian_low - current_price) / (atr or 1.0))
            reason_parts.append(f"Breakdown ${current_price:.2f} <= ${donchian_low:.2f}")
            return SignalResult(
                symbol, "SELL",
                confidence=min(1.0, abs(score) + 0.3),
                score=score,
                reason=" + ".join(reason_parts),
                metadata={"donchian_high": donchian_high, "donchian_low": donchian_low, "atr": atr},
            )

        # Neutral
        if in_position:
            reason_parts.append(f"In position, hold (${donchian_low:.2f} exit)")
        else:
            reason_parts.append(f"No breakout (${current_price:.2f} < ${donchian_high:.2f})")
        return SignalResult(
            symbol, "HOLD", 0.0, 0.0,
            " + ".join(reason_parts),
            metadata={"donchian_high": donchian_high, "donchian_low": donchian_low, "atr": atr},
        )
