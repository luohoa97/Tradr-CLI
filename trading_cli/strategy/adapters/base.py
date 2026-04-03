"""Strategy adapter base — unified interface for trading strategies."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from trading_cli.execution.adapters.base import TradingAdapter

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Unified trading signal output from any strategy."""

    symbol: str
    action: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 - 1.0
    score: float  # Raw strategy score (typically -1.0 to +1.0)
    reason: str
    metadata: dict = field(default_factory=dict)
    """Extra strategy-specific data (e.g. individual indicator values)."""


@dataclass
class StrategyInfo:
    """Metadata describing a strategy adapter."""

    name: str
    description: str
    version: str = "1.0.0"
    author: str = ""
    params_schema: dict = field(default_factory=dict)
    """JSON-schema-like dict describing configurable parameters."""


class StrategyAdapter(ABC):
    """Abstract base for all trading strategies.

    Subclasses implement different approaches (hybrid, momentum, mean-reversion,
    sentiment-driven, etc.) while exposing a unified interface for signal
    generation and backtesting.

    Required properties
    -------------------
    * ``strategy_id`` — unique string identifier (e.g. ``"hybrid"``).
    """

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or {}

    # ── Subclass responsibilities ─────────────────────────────────────────
    @property
    @abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier for this strategy."""
        ...

    @abstractmethod
    def info(self) -> StrategyInfo:
        """Return strategy metadata."""
        ...

    @abstractmethod
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
        """Produce a trading signal.

        Parameters
        ----------
        symbol :
            Ticker symbol.
        ohlcv :
            Historical OHLCV dataframe.
        sentiment_score :
            Pre-computed sentiment score (−1.0 … +1.0).
        prices :
            Latest price map for watchlist symbols.
        positions :
            Current open positions.
        portfolio_value :
            Total portfolio value.
        cash :
            Available cash.

        Returns
        -------
        SignalResult with action, confidence, and reason.
        """
        ...

    # ── Optional hooks ────────────────────────────────────────────────────

    def validate_config(self, config: dict) -> list[str]:
        """Return list of validation errors (empty = OK)."""
        return []

    def on_trade_executed(
        self,
        symbol: str,
        action: str,
        price: float,
        qty: int,
        result: SignalResult,
    ) -> None:
        """Called after a trade based on this strategy is executed."""
        pass

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _safe_close(ohlcv: pd.DataFrame) -> pd.Series:
        """Get close-price series regardless of column naming."""
        if "Close" in ohlcv.columns:
            return ohlcv["Close"]
        if "close" in ohlcv.columns:
            return ohlcv["close"]
        return pd.Series(dtype=float)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.strategy_id}>"
