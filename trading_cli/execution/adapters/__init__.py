"""Trading platform adapters — unified interface for different exchanges."""

from trading_cli.execution.adapters.base import TradingAdapter
from trading_cli.execution.adapters.registry import AdapterRegistry

__all__ = ["TradingAdapter", "AdapterRegistry"]
