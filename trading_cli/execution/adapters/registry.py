"""Adapter registry — discovers and instantiates trading adapters."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trading_cli.execution.adapters.base import TradingAdapter

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global registry of available adapters
_ADAPTERS: dict[str, type[TradingAdapter]] = {}


def register_adapter(adapter_class: type[TradingAdapter]) -> type[TradingAdapter]:
    """Decorator to register an adapter class.

    Usage:
        @register_adapter
        class AlpacaAdapter(TradingAdapter):
            ...
    """
    # Instantiate temporarily to get adapter_id
    # We assume adapter_id is a class property or can be called without args
    try:
        instance = adapter_class.__new__(adapter_class)
        adapter_id = adapter_class.adapter_id.fget(instance) if hasattr(adapter_class.adapter_id, 'fget') else getattr(adapter_class, 'adapter_id', None)
        if adapter_id:
            _ADAPTERS[adapter_id] = adapter_class
            logger.debug("Registered adapter: %s", adapter_id)
    except Exception:
        # Fallback: use class name lowercase
        adapter_id = adapter_class.__name__.lower().replace("adapter", "")
        _ADAPTERS[adapter_id] = adapter_class
        logger.debug("Registered adapter (fallback): %s", adapter_id)
    return adapter_class


def get_adapter(adapter_id: str) -> type[TradingAdapter] | None:
    """Get adapter class by ID."""
    return _ADAPTERS.get(adapter_id)


def list_adapters() -> list[str]:
    """List all registered adapter IDs."""
    return list(_ADAPTERS.keys())


def create_adapter(adapter_id: str, config: dict) -> TradingAdapter:
    """Create an adapter instance from config.

    Args:
        adapter_id: Adapter identifier ('alpaca', 'binance', 'kraken', 'demo').
        config: Configuration dict with API keys and settings.

    Returns:
        TradingAdapter instance.

    Raises:
        ValueError: If adapter_id is not registered.
    """
    adapter_class = get_adapter(adapter_id)
    if adapter_class is None:
        available = list_adapters()
        raise ValueError(
            f"Unknown adapter: '{adapter_id}'. "
            f"Available adapters: {available}"
        )
    return adapter_class(config)
