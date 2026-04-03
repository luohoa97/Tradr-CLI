"""Adapter factory — creates the appropriate adapter from config."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from trading_cli.execution.adapters import (
    TradingAdapter,
    create_adapter,
    list_adapters,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def create_trading_adapter(config: dict) -> TradingAdapter:
    """Create a trading adapter based on config.

    Priority:
    1. If Alpaca keys are set → AlpacaAdapter
    2. Otherwise → YFinanceAdapter (demo mode)

    You can override by setting `adapter_id` in config to:
    - 'alpaca': Force Alpaca (will fallback to demo if no keys)
    - 'yfinance': Force yFinance demo
    - 'binance': Binance crypto (requires ccxt)
    - 'kraken': Kraken crypto (requires ccxt)
    """
    adapter_id = config.get("adapter_id", None)

    if adapter_id is None:
        # Auto-detect based on available keys
        if config.get("alpaca_api_key") and config.get("alpaca_api_secret"):
            adapter_id = "alpaca"
        else:
            adapter_id = "yfinance"

    try:
        adapter = create_adapter(adapter_id, config)
        logger.info("Created adapter: %s (demo=%s)", adapter.adapter_id, adapter.is_demo_mode)
        return adapter
    except ValueError as exc:
        logger.error("Failed to create adapter '%s': %s", adapter_id, exc)
        logger.info("Available adapters: %s", list_adapters())
        # Fallback to yfinance demo
        return create_adapter("yfinance", config)
