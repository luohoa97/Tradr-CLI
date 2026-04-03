"""Strategy adapter factory — auto-detects and creates strategy adapters."""

from __future__ import annotations

import logging

from trading_cli.strategy.adapters.base import StrategyAdapter
from trading_cli.strategy.adapters.registry import create_strategy, list_strategies

logger = logging.getLogger(__name__)

# Default strategy when none specified
DEFAULT_STRATEGY = "hybrid"


def create_trading_strategy(config: dict) -> StrategyAdapter:
    """Create a strategy adapter based on config settings.

    Auto-detection logic:
    1. If ``strategy_id`` is set in config, use that.
    2. Otherwise fall back to the default (``hybrid``).

    Args:
        config: Trading configuration dict. May contain:
            - strategy_id: Strategy identifier ('hybrid', 'momentum',
              'mean_reversion', 'sentiment').
            - Plus any strategy-specific parameters.

    Returns:
        StrategyAdapter instance.
    """
    strategy_id = config.get("strategy_id", DEFAULT_STRATEGY)

    try:
        strategy = create_strategy(strategy_id, config)
        logger.info("Using strategy: %s", strategy_id)
        return strategy
    except ValueError as exc:
        logger.warning(
            "Strategy '%s' not found (%s). Falling back to '%s'.",
            strategy_id,
            exc,
            DEFAULT_STRATEGY,
        )
        return create_strategy(DEFAULT_STRATEGY, config)


def available_strategies() -> list[str]:
    """List all available strategy IDs."""
    return list_strategies()
