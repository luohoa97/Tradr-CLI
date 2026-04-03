"""Trading platform adapters — unified interface for different exchanges."""

from trading_cli.execution.adapters.base import (
    AccountInfo,
    MarketClock,
    OrderResult,
    Position,
    TradingAdapter,
)
from trading_cli.execution.adapters.registry import (
    create_adapter,
    get_adapter,
    list_adapters,
    register_adapter,
)

# Import all adapter implementations to trigger registration
from trading_cli.execution.adapters.alpaca import AlpacaAdapter
from trading_cli.execution.adapters.yfinance import YFinanceAdapter
from trading_cli.execution.adapters.binance import BinanceAdapter
from trading_cli.execution.adapters.kraken import KrakenAdapter

__all__ = [
    "TradingAdapter",
    "AccountInfo",
    "MarketClock",
    "OrderResult",
    "Position",
    "create_adapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
    "AlpacaAdapter",
    "YFinanceAdapter",
    "BinanceAdapter",
    "KrakenAdapter",
]
