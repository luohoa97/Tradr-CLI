"""Base adapter interface — all exchange adapters must implement this."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Unified position object across all exchanges."""

    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    unrealized_pl: float
    unrealized_plpc: float
    market_value: float
    side: str = "long"


@dataclass
class AccountInfo:
    """Unified account info across all exchanges."""

    equity: float
    cash: float
    buying_power: float
    portfolio_value: float


@dataclass
class OrderResult:
    """Unified order result across all exchanges."""

    order_id: str
    symbol: str
    action: str  # BUY or SELL
    qty: int
    status: str  # filled, rejected, pending, etc.
    filled_price: float | None = None


@dataclass
class MarketClock:
    """Market hours info."""

    is_open: bool
    next_open: str
    next_close: str


class TradingAdapter(ABC):
    """Abstract base class for all trading platform adapters.

    Implement this class to add support for new exchanges (Binance, Kraken, etc.).
    Each adapter handles:
    - Account info retrieval
    - Position management
    - Order execution
    - Market data (OHLCV, quotes)
    - Market clock
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """Unique identifier for this adapter (e.g., 'alpaca', 'binance', 'kraken')."""
        ...

    @property
    @abstractmethod
    def supports_paper_trading(self) -> bool:
        """Whether this adapter supports paper/demo trading."""
        ...

    @property
    @abstractmethod
    def is_demo_mode(self) -> bool:
        """True if running in demo/mock mode (no real API connection)."""
        ...

    # ── Account & Positions ───────────────────────────────────────────────────

    @abstractmethod
    def get_account(self) -> AccountInfo:
        """Get account balance and buying power."""
        ...

    @abstractmethod
    def get_positions(self) -> list[Position]:
        """Get all open positions."""
        ...

    # ── Orders ────────────────────────────────────────────────────────────────

    @abstractmethod
    def submit_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        """Submit a market order.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USD').
            qty: Number of shares/units.
            side: 'BUY' or 'SELL'.

        Returns:
            OrderResult with status and fill details.
        """
        ...

    @abstractmethod
    def close_position(self, symbol: str) -> OrderResult | None:
        """Close an existing position at market price.

        Returns None if no position exists for the symbol.
        """
        ...

    # ── Market Data ───────────────────────────────────────────────────────────

    @abstractmethod
    def fetch_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Fetch historical OHLCV bars.

        Returns DataFrame with columns: Open, High, Low, Close, Volume.
        Index should be datetime.
        """
        ...

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> float | None:
        """Get latest trade price for a symbol."""
        ...

    def get_latest_quotes_batch(self, symbols: list[str]) -> dict[str, float]:
        """Get latest prices for multiple symbols (batch optimized).

        Override if the exchange supports batch requests.
        Default implementation calls get_latest_quote for each symbol.
        """
        prices: dict[str, float] = {}
        for sym in symbols:
            price = self.get_latest_quote(sym)
            if price is not None:
                prices[sym] = price
        return prices

    # ── Market Info ───────────────────────────────────────────────────────────

    @abstractmethod
    def get_market_clock(self) -> MarketClock:
        """Get market open/closed status and next open/close times."""
        ...

    # ── News (optional) ───────────────────────────────────────────────────────

    def fetch_news(self, symbol: str, max_articles: int = 50,
                   days_ago: int = 0) -> list[tuple[str, float]]:
        """Fetch news headlines with timestamps.

        Returns list of (headline, unix_timestamp) tuples.
        Override if the exchange provides news data.
        Default returns empty list.
        """
        return []

    # ── Utilities ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} adapter_id={self.adapter_id} demo={self.is_demo_mode}>"
