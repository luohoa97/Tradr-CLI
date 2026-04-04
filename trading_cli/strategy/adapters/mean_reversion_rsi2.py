"""Mean Reversion strategy — RSI(2) + Bollinger Bands.

Entry: RSI(2) < 10 AND price below lower Bollinger Band
Exit: RSI(2) > 80 OR price crosses above middle Bollinger Band
Stop: 2x ATR below entry

Proven characteristics:
  - Win rate: 60–75%
  - Gain/Loss ratio: ~1:1
  - High frequency, small consistent gains
  - Main risk: severe drawdowns during strong trends
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


def calculate_rsi_fast(prices: pd.Series, period: int = 2) -> pd.Series:
    """RSI with very short period for mean reversion (classic Larry Connors approach)."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


@register_strategy
class MeanReversionStrategy(StrategyAdapter):
    """RSI(2) + Bollinger Bands mean reversion.

    Buys when price is extremely oversold (RSI(2) < 10 AND below lower BB).
    Sells when price reverts to mean (RSI(2) > 80 OR above middle BB).

    This is a high-win-rate strategy that works well in ranging markets.
    """

    @property
    def strategy_id(self) -> str:
        return "mean_reversion"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Mean Reversion (RSI(2) + Bollinger)",
            description=(
                "Entry: RSI(2) < 10 AND price below lower Bollinger Band (20,2). "
                "Exit: RSI(2) > 80 OR price crosses above middle BB. "
                "High win rate (~65%) with small consistent gains. "
                "Risk: drawdowns during sustained trends."
            ),
            params_schema={
                "rsi_period": {"type": "int", "default": 2, "desc": "RSI period (short = more sensitive)"},
                "rsi_oversold": {"type": "int", "default": 10, "desc": "RSI oversold threshold (buy)"},
                "rsi_overbought": {"type": "int", "default": 80, "desc": "RSI overbought threshold (sell)"},
                "bb_window": {"type": "int", "default": 20, "desc": "Bollinger Bands window"},
                "bb_std": {"type": "float", "default": 2.0, "desc": "Bollinger Bands std multiplier"},
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

        if close_col not in ohlcv.columns:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "missing data")

        closes = ohlcv[close_col]

        rsi_period = config.get("rsi_period", 2)
        rsi_oversold = config.get("rsi_oversold", 10)
        rsi_overbought = config.get("rsi_overbought", 80)
        bb_window = config.get("bb_window", 20)
        bb_std = config.get("bb_std", 2.0)

        if len(closes) < bb_window + 5:
            return SignalResult(symbol, "HOLD", 0.0, 0.0, "insufficient data")

        current_price = closes.iloc[-1]

        # RSI(2) — very short period for extreme oversold detection
        rsi = calculate_rsi_fast(closes, rsi_period).iloc[-1]

        # Bollinger Bands
        sma = closes.rolling(window=bb_window, min_periods=bb_window).mean()
        std = closes.rolling(window=bb_window, min_periods=bb_window).std()
        upper = sma.iloc[-1] + bb_std * std.iloc[-1]
        middle = sma.iloc[-1]
        lower = sma.iloc[-1] - bb_std * std.iloc[-1]

        # Check if we're in a position
        in_position = any(p.symbol == symbol for p in (positions or []))
        # Get position info for stop-loss check
        position_entry = None
        for p in (positions or []):
            if p.symbol == symbol:
                position_entry = p.avg_entry_price
                break

        reason_parts = []

        # Stop-loss: 2x ATR below entry
        if in_position and position_entry:
            high_col = "high" if "high" in ohlcv.columns else "High"
            low_col = "low" if "low" in ohlcv.columns else "Low"
            if high_col in ohlcv.columns and low_col in ohlcv.columns:
                tr1 = ohlcv[high_col] - ohlcv[low_col]
                prev_close = closes.shift(1)
                tr2 = (ohlcv[high_col] - prev_close).abs()
                tr3 = (ohlcv[low_col] - prev_close).abs()
                atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
                stop_price = position_entry - 2 * atr
                if current_price <= stop_price:
                    return SignalResult(
                        symbol, "SELL",
                        confidence=0.9,
                        score=-0.8,
                        reason=f"Stop-loss ${stop_price:.2f} hit (entry ${position_entry:.2f})",
                        metadata={"rsi": rsi, "bb_lower": lower, "bb_middle": middle, "stop": stop_price},
                    )

        # Entry: RSI(2) < oversold AND price below lower BB
        if not in_position:
            if rsi <= rsi_oversold and current_price <= lower:
                # Distance from lower band as score (deeper = stronger signal)
                band_width = upper - lower
                depth = (lower - current_price) / (band_width or 1.0)
                score = min(1.0, 0.5 + depth)
                reason_parts.append(f"RSI(2)={rsi:.0f} (oversold)")
                reason_parts.append(f"Price ${current_price:.2f} <= BB lower ${lower:.2f}")
                return SignalResult(
                    symbol, "BUY",
                    confidence=min(1.0, score + 0.3),
                    score=score,
                    reason=" + ".join(reason_parts),
                    metadata={"rsi": rsi, "bb_lower": lower, "bb_middle": middle, "bb_upper": upper},
                )
            elif rsi <= rsi_oversold:
                reason_parts.append(f"RSI(2)={rsi:.0f} but price above lower BB")
            elif current_price <= lower:
                reason_parts.append(f"Price at lower BB but RSI(2)={rsi:.0f} not oversold")
            else:
                reason_parts.append(f"RSI(2)={rsi:.0f}, no extreme oversold")

        # Exit: RSI(2) > overbought OR price crosses above middle BB
        if in_position:
            if rsi >= rsi_overbought:
                reason_parts.append(f"RSI(2)={rsi:.0f} (overbought)")
                return SignalResult(
                    symbol, "SELL",
                    confidence=0.8,
                    score=-0.5,
                    reason=" + ".join(reason_parts),
                    metadata={"rsi": rsi, "bb_lower": lower, "bb_middle": middle, "bb_upper": upper},
                )
            elif current_price >= middle:
                reason_parts.append(f"Price ${current_price:.2f} >= BB middle ${middle:.2f}")
                return SignalResult(
                    symbol, "SELL",
                    confidence=0.6,
                    score=-0.3,
                    reason=" + ".join(reason_parts),
                    metadata={"rsi": rsi, "bb_lower": lower, "bb_middle": middle, "bb_upper": upper},
                )
            else:
                reason_parts.append(f"Holding (RSI={rsi:.0f}, exit at middle BB ${middle:.2f})")

        return SignalResult(
            symbol, "HOLD", 0.0, 0.0,
            " + ".join(reason_parts) if reason_parts else "neutral",
            metadata={"rsi": rsi, "bb_lower": lower, "bb_middle": middle, "bb_upper": upper},
        )
