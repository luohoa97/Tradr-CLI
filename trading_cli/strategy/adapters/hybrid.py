"""Hybrid strategy — combines technical indicators with sentiment analysis."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import register_strategy
from trading_cli.strategy.signals import (
    generate_signal,
    technical_score,
    sma_crossover_score,
    rsi_score,
    bollinger_score,
    ema_score,
    volume_score,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@register_strategy
class HybridStrategy(StrategyAdapter):
    """Default strategy: weighted blend of technical + sentiment.

    This is the original strategy from the codebase, wrapped in the
    adapter interface for consistency.
    """

    @property
    def strategy_id(self) -> str:
        return "hybrid"

    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="Hybrid (Technical + Sentiment)",
            description=(
                "Combines technical indicators (SMA crossover, RSI, Bollinger Bands, "
                "EMA, Volume) with news sentiment analysis using configurable weights."
            ),
            params_schema={
                "sma_short": {"type": "int", "default": 20, "desc": "Short SMA period"},
                "sma_long": {"type": "int", "default": 50, "desc": "Long SMA period"},
                "rsi_period": {"type": "int", "default": 14, "desc": "RSI period"},
                "bb_window": {"type": "int", "default": 20, "desc": "Bollinger Bands window"},
                "bb_std": {"type": "float", "default": 2.0, "desc": "Bollinger Bands std multiplier"},
                "ema_fast": {"type": "int", "default": 12, "desc": "Fast EMA period"},
                "ema_slow": {"type": "int", "default": 26, "desc": "Slow EMA period"},
                "vol_window": {"type": "int", "default": 20, "desc": "Volume SMA window"},
                "tech_weight": {"type": "float", "default": 0.6, "desc": "Weight for technical score"},
                "sent_weight": {"type": "float", "default": 0.4, "desc": "Weight for sentiment score"},
                "signal_buy_threshold": {"type": "float", "default": 0.15, "desc": "Buy signal threshold"},
                "signal_sell_threshold": {"type": "float", "default": -0.15, "desc": "Sell signal threshold"},
                "weight_sma": {"type": "float", "default": 0.25, "desc": "SMA indicator weight"},
                "weight_rsi": {"type": "float", "default": 0.25, "desc": "RSI indicator weight"},
                "weight_bb": {"type": "float", "default": 0.20, "desc": "Bollinger Bands indicator weight"},
                "weight_ema": {"type": "float", "default": 0.15, "desc": "EMA indicator weight"},
                "weight_volume": {"type": "float", "default": 0.15, "desc": "Volume indicator weight"},
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

        # Technical indicator weights from config
        tech_indicator_weights = {
            "sma": config.get("weight_sma", 0.25),
            "rsi": config.get("weight_rsi", 0.25),
            "bb": config.get("weight_bb", 0.20),
            "ema": config.get("weight_ema", 0.15),
            "volume": config.get("weight_volume", 0.15),
        }

        # Use the existing generate_signal function
        signal = generate_signal(
            symbol=symbol,
            ohlcv=ohlcv,
            sentiment_score=sentiment_score,
            buy_threshold=config.get("signal_buy_threshold", 0.15),
            sell_threshold=config.get("signal_sell_threshold", -0.15),
            sma_short=config.get("sma_short", 20),
            sma_long=config.get("sma_long", 50),
            rsi_period=config.get("rsi_period", 14),
            bb_window=config.get("bb_window", 20),
            bb_std=config.get("bb_std", 2.0),
            ema_fast=config.get("ema_fast", 12),
            ema_slow=config.get("ema_slow", 26),
            vol_window=config.get("vol_window", 20),
            tech_weight=config.get("tech_weight", 0.6),
            sent_weight=config.get("sent_weight", 0.4),
            tech_indicator_weights=tech_indicator_weights,
        )

        # Compute individual indicator scores for metadata
        metadata = {
            "sma_score": sma_crossover_score(ohlcv, config.get("sma_short", 20), config.get("sma_long", 50)),
            "rsi_score": rsi_score(ohlcv, config.get("rsi_period", 14)),
            "bb_score": bollinger_score(ohlcv, config.get("bb_window", 20), config.get("bb_std", 2.0)),
            "ema_score": ema_score(ohlcv, config.get("ema_fast", 12), config.get("ema_slow", 26)),
            "volume_score": volume_score(ohlcv, config.get("vol_window", 20)),
        }

        return SignalResult(
            symbol=symbol,
            action=signal["action"],
            confidence=signal["confidence"],
            score=signal["hybrid_score"],
            reason=signal["reason"],
            metadata=metadata,
        )
