"""yFinance adapter — free market data with mock trading."""

from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from trading_cli.execution.adapters.base import (
    AccountInfo,
    MarketClock,
    OrderResult,
    Position,
    TradingAdapter,
)
from trading_cli.execution.adapters.registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter
class YFinanceAdapter(TradingAdapter):
    """yFinance adapter for free market data with simulated trading.

    Provides:
    - Real OHLCV data from Yahoo Finance
    - Real latest quotes from Yahoo Finance
    - Simulated account and positions (demo mode)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._cash = config.get("initial_cash", 100_000.0)
        self._positions: dict[str, dict] = {}
        self._order_counter = 1000
        self._base_prices = {
            "AAPL": 175.0, "TSLA": 245.0, "NVDA": 875.0,
            "MSFT": 415.0, "AMZN": 185.0, "GOOGL": 175.0,
            "META": 510.0, "SPY": 520.0,
        }
        logger.info("YFinanceAdapter initialized in demo mode")

    @property
    def adapter_id(self) -> str:
        return "yfinance"

    @property
    def supports_paper_trading(self) -> bool:
        return True  # Simulated trading

    @property
    def is_demo_mode(self) -> bool:
        return True

    # ── Account & Positions ───────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        portfolio = sum(
            p["qty"] * self._get_mock_price(sym)
            for sym, p in self._positions.items()
        )
        equity = self._cash + portfolio
        return AccountInfo(
            equity=equity,
            cash=self._cash,
            buying_power=self._cash * 4,
            portfolio_value=equity,
        )

    def get_positions(self) -> list[Position]:
        positions = []
        for sym, p in self._positions.items():
            cp = self._get_mock_price(sym)
            ep = p["avg_price"]
            pl = (cp - ep) * p["qty"]
            plpc = (cp - ep) / ep if ep else 0.0
            positions.append(
                Position(sym, p["qty"], ep, cp, pl, plpc, cp * p["qty"])
            )
        return positions

    # ── Orders ───────────────────────────────────────────────────────────────

    def submit_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        price = self._get_mock_price(symbol)
        self._order_counter += 1
        order_id = f"YF-{self._order_counter}"

        if side.upper() == "BUY":
            cost = price * qty
            if cost > self._cash:
                return OrderResult(order_id, symbol, side, qty, "rejected")
            self._cash -= cost
            if symbol in self._positions:
                p = self._positions[symbol]
                total_qty = p["qty"] + qty
                p["avg_price"] = (p["avg_price"] * p["qty"] + price * qty) / total_qty
                p["qty"] = total_qty
            else:
                self._positions[symbol] = {"qty": qty, "avg_price": price}
        else:  # SELL
            if symbol not in self._positions or self._positions[symbol]["qty"] < qty:
                return OrderResult(order_id, symbol, side, qty, "rejected")
            self._cash += price * qty
            self._positions[symbol]["qty"] -= qty
            if self._positions[symbol]["qty"] == 0:
                del self._positions[symbol]

        return OrderResult(order_id, symbol, side, qty, "filled", price)

    def close_position(self, symbol: str) -> OrderResult | None:
        if symbol not in self._positions:
            return None
        qty = self._positions[symbol]["qty"]
        return self.submit_market_order(symbol, qty, "SELL")

    def _get_mock_price(self, symbol: str) -> float:
        """Get a mock price with small random walk for realism."""
        base = self._base_prices.get(symbol, 100.0)
        noise = random.gauss(0, base * 0.002)
        return round(max(1.0, base + noise), 2)

    # ── Market Data ───────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        """Fetch OHLCV from yfinance."""
        try:
            import yfinance as yf
            period = f"{min(days, 730)}d"
            df = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
            if df.empty:
                return pd.DataFrame()
            return df.tail(days)
        except Exception as exc:
            logger.error("yfinance fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> float | None:
        """Get latest price from yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            price = getattr(info, "last_price", None) or getattr(info, "regularMarketPrice", None)
            if price:
                return float(price)
            hist = ticker.history(period="2d", interval="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
            return None
        except Exception as exc:
            logger.warning("yfinance latest quote failed for %s: %s", symbol, exc)
            return None

    # ── Market Info ───────────────────────────────────────────────────────────

    def get_market_clock(self) -> MarketClock:
        now = datetime.now(tz=timezone.utc)
        # Mock: market open weekdays 9:30–16:00 ET (UTC-5)
        hour_et = (now.hour - 5) % 24
        is_open = now.weekday() < 5 and 9 <= hour_et < 16
        return MarketClock(
            is_open=is_open,
            next_open="09:30 ET",
            next_close="16:00 ET",
        )
