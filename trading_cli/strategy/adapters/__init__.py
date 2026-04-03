"""Strategy adapters — pluggable trading strategy implementations."""

from trading_cli.strategy.adapters.base import SignalResult, StrategyAdapter, StrategyInfo
from trading_cli.strategy.adapters.registry import (
    create_strategy,
    get_strategy,
    list_strategies,
    register_strategy,
)

# Import all strategy implementations to trigger registration
from trading_cli.strategy.adapters.hybrid import HybridStrategy
from trading_cli.strategy.adapters.momentum import MomentumStrategy
from trading_cli.strategy.adapters.mean_reversion import MeanReversionStrategy
from trading_cli.strategy.adapters.sentiment_driven import SentimentStrategy

__all__ = [
    "StrategyAdapter",
    "StrategyInfo",
    "SignalResult",
    "create_strategy",
    "get_strategy",
    "list_strategies",
    "register_strategy",
    "HybridStrategy",
    "MomentumStrategy",
    "MeanReversionStrategy",
    "SentimentStrategy",
]
