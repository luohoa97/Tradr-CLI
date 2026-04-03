"""Sentiment-driven strategy — trades purely on news sentiment signals."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import register_strategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_strategy
class SentimentStrategy(StrategyAdapter):
    """Pure sentiment strategy — trades on news-driven signals only.

    Ignores technical indicators completely. Buys on strong positive sentiment,
    sells on negative sentiment. Useful for event-driven trading (earnings,
    product launches, executive changes).
    """

    @property
    def strategy_id(self) -> str:
        return "sentiment"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Sentiment-Driven (News-Based)",
            description=(
                "Trades purely on news sentiment analysis. Buys when aggregated "
                "sentiment is strongly positive, sells when negative. No technical "
                "indicators — relies entirely on FinBERT news classification and "
                "time-decay weighted sentiment aggregation."
            ),
            params_schema={
                "sentiment_buy_threshold": {
                    "type": "float",
                    "default": 0.4,
                    "desc": "Sentiment score for buy signal",
                },
                "sentiment_sell_threshold": {
                    "type": "float",
                    "default": -0.3,
                    "desc": "Sentiment score for sell signal",
                },
                "sentiment_half_life_hours": {
                    "type": "float",
                    "default": 24.0,
                    "desc": "Time decay for sentiment relevance",
                },
                "require_volume_confirm": {
                    "type": "bool",
                    "default": False,
                    "desc": "Require above-average volume to confirm signal",
                },
                "volume_window": {
                    "type": "int",
                    "default": 20,
                    "desc": "Volume SMA lookback for confirmation",
                },
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

        buy_threshold = config.get("sentiment_buy_threshold", 0.4)
        sell_threshold = config.get("sentiment_sell_threshold", -0.3)
        require_volume = config.get("require_volume_confirm", False)

        # Sentiment score is the primary signal
        combined = sentiment_score

        # Optional volume confirmation
        volume_confirmed = True
        if require_volume and len(ohlcv) > 0:
            vol_col = "Volume" if "Volume" in ohlcv.columns else "volume"
            if vol_col in ohlcv.columns and len(ohlcv) >= config.get("volume_window", 20):
                vol_window = config.get("volume_window", 20)
                vol_sma = ohlcv[vol_col].rolling(window=vol_window).mean().iloc[-1]
                current_vol = ohlcv[vol_col].iloc[-1]
                if vol_sma > 0:
                    volume_ratio = current_vol / vol_sma
                    volume_confirmed = volume_ratio >= 1.2  # 20% above average

        if combined >= buy_threshold and volume_confirmed:
            action = "BUY"
        elif combined <= sell_threshold and volume_confirmed:
            action = "SELL"
        else:
            action = "HOLD"

        reason_parts = [f"sent={sentiment_score:+.2f}"]
        if require_volume:
            reason_parts.append(f"vol={'✓' if volume_confirmed else '✗'}")
        reason = " + ".join(reason_parts)

        return SignalResult(
            symbol=symbol,
            action=action,
            confidence=abs(combined),
            score=combined,
            reason=reason,
            metadata={
                "sentiment_score": sentiment_score,
                "volume_confirmed": volume_confirmed,
            },
        )
