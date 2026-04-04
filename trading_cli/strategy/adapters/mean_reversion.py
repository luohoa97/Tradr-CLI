"""Mean-reversion strategy — fades extreme moves using Bollinger Bands and RSI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import register_strategy
from trading_cli.strategy.signals import calculate_rsi, calculate_bollinger_bands

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_strategy
class MeanReversionStrategy(StrategyAdapter):
    """Mean-reversion — buys dips, sells rips.

    Enters when price deviates significantly from its mean (oversold/overbought)
    and exits when it reverts. Uses Bollinger Bands + RSI confluence.
    """

    @property
    def strategy_id(self) -> str:
        return "mean_reversion"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Mean Reversion",
            description=(
                "Fades extreme moves. Buys when price touches lower Bollinger Band "
                "and RSI is oversold; sells when price touches upper band and RSI "
                "is overbought. Expects price to revert toward the mean."
            ),
            params_schema={
                "bb_window": {"type": "int", "default": 20, "desc": "Bollinger Bands window"},
                "bb_std": {"type": "float", "default": 2.0, "desc": "Bollinger Bands std multiplier"},
                "rsi_period": {"type": "int", "default": 14, "desc": "RSI period"},
                "rsi_oversold": {"type": "int", "default": 30, "desc": "RSI oversold threshold"},
                "rsi_overbought": {"type": "int", "default": 70, "desc": "RSI overbought threshold"},
                "signal_buy_threshold": {"type": "float", "default": 0.15, "desc": "Combined score buy threshold"},
                "signal_sell_threshold": {"type": "float", "default": -0.15, "desc": "Combined score sell threshold"},
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
        closes = self._safe_close(ohlcv)

        bb_window = config.get("bb_window", 20)
        bb_std = config.get("bb_std", 2.0)
        rsi_period = config.get("rsi_period", 14)
        rsi_oversold = config.get("rsi_oversold", 30)
        rsi_overbought = config.get("rsi_overbought", 70)

        if len(closes) < max(bb_window, rsi_period) + 2:
            return SignalResult(
                symbol=symbol,
                action="HOLD",
                confidence=0.0,
                score=0.0,
                reason="Insufficient data for mean-reversion",
            )

        # Bollinger Bands
        upper, middle, lower = calculate_bollinger_bands(closes, bb_window, bb_std)
        last_close = closes.iloc[-1]
        last_upper = upper.iloc[-1]
        last_lower = lower.iloc[-1]
        bandwidth = last_upper - last_lower

        if bandwidth == 0:
            bb_position = 0.0
        else:
            # 0.0 = at lower band, 0.5 = middle, 1.0 = upper
            bb_position = (last_close - last_lower) / bandwidth

        # BB score: near lower band → bullish (expect bounce), near upper → bearish
        if bb_position <= 0.05:
            bb_score = 1.0  # At or below lower band — strong buy signal
        elif bb_position <= 0.2:
            bb_score = 0.5
        elif bb_position >= 0.95:
            bb_score = -1.0  # At or above upper band — strong sell signal
        elif bb_position >= 0.8:
            bb_score = -0.5
        else:
            bb_score = 0.0

        # RSI
        rsi = calculate_rsi(closes, rsi_period).iloc[-1]

        if rsi <= rsi_oversold:
            rsi_score = 1.0  # Oversold — expect bounce
        elif rsi <= rsi_oversold + 10:
            rsi_score = 0.5
        elif rsi >= rsi_overbought:
            rsi_score = -1.0  # Overbought — expect pullback
        elif rsi >= rsi_overbought - 10:
            rsi_score = -0.5
        else:
            rsi_score = 0.0

        # Combined: equal-weight BB + RSI
        combined = 0.5 * bb_score + 0.5 * rsi_score

        buy_threshold = config.get("signal_buy_threshold", 0.15)
        sell_threshold = config.get("signal_sell_threshold", -0.15)

        if combined >= buy_threshold:
            action = "BUY"
        elif combined <= sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        reason = f"BB={'↓' if bb_score > 0 else '↑'} (pos={bb_position:.2f}) + RSI={rsi:.0f}"

        return SignalResult(
            symbol=symbol,
            action=action,
            confidence=abs(combined),
            score=combined,
            reason=reason,
            metadata={
                "bb_position": bb_position,
                "bb_score": bb_score,
                "rsi": float(rsi),
                "rsi_score": rsi_score,
                "bb_upper": float(last_upper),
                "bb_middle": float(middle.iloc[-1]),
                "bb_lower": float(last_lower),
            },
        )
