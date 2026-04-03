"""Strategy adapter registry — discovers and instantiates strategy adapters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trading_cli.strategy.adapters.base import StrategyAdapter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global registry of available strategy adapters
_STRATEGY_ADAPTERS: dict[str, type[StrategyAdapter]] = {}


def register_strategy(cls: type[StrategyAdapter]) -> type[StrategyAdapter]:
    """Decorator to register a strategy adapter class.

    Usage:
        @register_strategy
        class HybridStrategy(StrategyAdapter):
            ...
    """
    try:
        instance = cls.__new__(cls)
        strategy_id = (
            cls.strategy_id.fget(instance)
            if hasattr(cls.strategy_id, "fget")
            else getattr(cls, "strategy_id", None)
        )
        if strategy_id:
            _STRATEGY_ADAPTERS[strategy_id] = cls
            logger.debug("Registered strategy: %s", strategy_id)
    except Exception:
        # Fallback: use class name lowercase
        strategy_id = cls.__name__.lower().replace("strategy", "")
        _STRATEGY_ADAPTERS[strategy_id] = cls
        logger.debug("Registered strategy (fallback): %s", strategy_id)
    return cls


def get_strategy(strategy_id: str) -> type[StrategyAdapter] | None:
    """Get strategy class by ID."""
    return _STRATEGY_ADAPTERS.get(strategy_id)


def list_strategies() -> list[str]:
    """List all registered strategy IDs."""
    return list(_STRATEGY_ADAPTERS.keys())


def create_strategy(strategy_id: str, config: dict) -> StrategyAdapter:
    """Create a strategy adapter instance from config.

    Args:
        strategy_id: Strategy identifier ('hybrid', 'momentum', 'mean_reversion', ...).
        config: Configuration dict with strategy-specific parameters.

    Returns:
        StrategyAdapter instance.

    Raises:
        ValueError: If strategy_id is not registered.
    """
    strategy_class = get_strategy(strategy_id)
    if strategy_class is None:
        available = list_strategies()
        raise ValueError(
            f"Unknown strategy: '{strategy_id}'. "
            f"Available strategies: {available}"
        )
    return strategy_class(config)
