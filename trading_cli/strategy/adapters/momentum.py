"""Momentum strategy — trades based on trend-following momentum indicators."""

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
class MomentumStrategy(StrategyAdapter):
    """Pure momentum strategy — rides trends using ROC and MACD.

    Enters long when price momentum is strong and positive, exits when
    momentum weakens or reverses. Ignores sentiment entirely.
    """

    @property
    def strategy_id(self) -> str:
        return "momentum"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Momentum (Trend Following)",
            description=(
                "Uses Rate of Change (ROC) and MACD histogram to identify and trade "
                "strong trends. Buys on accelerating upside momentum, sells on "
                "deceleration or reversal. No sentiment analysis."
            ),
            params_schema={
                "roc_period": {"type": "int", "default": 14, "desc": "Rate of Change lookback"},
                "macd_fast": {"type": "int", "default": 12, "desc": "MACD fast EMA"},
                "macd_slow": {"type": "int", "default": 26, "desc": "MACD slow EMA"},
                "macd_signal": {"type": "int", "default": 9, "desc": "MACD signal line"},
                "momentum_threshold": {"type": "float", "default": 0.3, "desc": "ROC threshold for entry"},
                "signal_buy_threshold": {"type": "float", "default": 0.4, "desc": "Combined score buy threshold"},
                "signal_sell_threshold": {"type": "float", "default": -0.3, "desc": "Combined score sell threshold"},
            },
        )

    @staticmethod
    def _calculate_roc(closes: pd.Series, period: int) -> pd.Series:
        """Rate of Change: (close - close[n]) / close[n]."""
        return closes.pct_change(periods=period)

    @staticmethod
    def _calculate_macd_hist(
        closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.Series:
        """MACD histogram = MACD line - Signal line."""
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

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
        closes = self._safe_close(ohlcv)

        roc_period = config.get("roc_period", 14)
        macd_fast = config.get("macd_fast", 12)
        macd_slow = config.get("macd_slow", 26)
        macd_signal = config.get("macd_signal", 9)
        momentum_threshold = config.get("momentum_threshold", 0.3)

        if len(closes) < max(macd_slow, roc_period) + 5:
            return SignalResult(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                score=0.0,
                reason="Insufficient data for momentum",
            )

        # Rate of Change — measures % price change over lookback
        roc = self._calculate_roc(closes, roc_period).iloc[-1]
        # Normalize ROC to roughly [-1, 1] — clamp at ±10%
        roc_score = float(max(-1.0, min(1.0, roc * 10)))

        # MACD histogram — positive & rising = bullish momentum
        macd_hist = self._calculate_macd_hist(closes, macd_fast, macd_slow, macd_signal)
        macd_val = macd_hist.iloc[-1]
        macd_prev = macd_hist.iloc[-2] if len(macd_hist) > 1 else 0.0

        # MACD signal: positive histogram + rising = +1, negative + falling = -1
        if macd_val > 0 and macd_val > macd_prev:
            macd_score = 1.0
        elif macd_val < 0 and macd_val < macd_prev:
            macd_score = -1.0
        elif macd_val > 0:
            macd_score = 0.3
        elif macd_val < 0:
            macd_score = -0.3
        else:
            macd_score = 0.0

        # Combined momentum score (ROC 60% + MACD 40%)
        combined = 0.6 * roc_score + 0.4 * macd_score

        buy_threshold = config.get("signal_buy_threshold", 0.4)
        sell_threshold = config.get("signal_sell_threshold", -0.3)

        if combined >= buy_threshold and roc_score >= momentum_threshold:
            action = "BUY"
        elif combined <= sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        reason_parts = [f"ROC={roc_score:+.2f}", f"MACD={'↑' if macd_score > 0 else '↓'}"]
        reason = " + ".join(reason_parts)

        return SignalResult(
            symbol=symbol,
            action=action,
            confidence=abs(combined),
            score=combined,
            reason=reason,
            metadata={
                "roc": roc_score,
                "macd_histogram": float(macd_val),
                "macd_score": macd_score,
            },
        )
