"""Kraken adapter stub — crypto trading via Kraken API.

This is a stub implementation. To enable:
1. Install ccxt: `uv add ccxt`
2. Add your Kraken API keys to config
3. Implement the TODO sections below
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

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
class KrakenAdapter(TradingAdapter):
    """Kraken adapter for cryptocurrency trading.

    Requires: ccxt library (`uv add ccxt`)
    Config keys:
        kraken_api_key: Your Kraken API key
        kraken_api_secret: Your Kraken API secret
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._api_key = config.get("kraken_api_key", "")
        self._api_secret = config.get("kraken_api_secret", "")
        self._demo = not (self._api_key and self._api_secret)

        if self._demo:
            logger.info("KrakenAdapter: no API keys found, stub mode only")
            return

        try:
            import ccxt
            self._exchange = ccxt.kraken({
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
            })
            logger.info("KrakenAdapter connected")
        except ImportError:
            logger.warning("ccxt not installed. Run: uv add ccxt")
            self._demo = True
            self._exchange = None
        except Exception as exc:
            logger.error("Failed to connect to Kraken: %s", exc)
            self._demo = True
            self._exchange = None

    @property
    def adapter_id(self) -> str:
        return "kraken"

    @property
    def supports_paper_trading(self) -> bool:
        return False  # Kraken doesn't have testnet

    @property
    def is_demo_mode(self) -> bool:
        return self._demo

    # ── Account & Positions ───────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        if self._demo or not self._exchange:
            return AccountInfo(
                equity=100000.0,
                cash=100000.0,
                buying_power=100000.0,
                portfolio_value=100000.0,
            )
        # TODO: Implement real account fetch
        balance = self._exchange.fetch_balance()
        cash = float(balance.get("USD", {}).get("free", 0))
        return AccountInfo(
            equity=cash,
            cash=cash,
            buying_power=cash,
            portfolio_value=cash,
        )

    def get_positions(self) -> list[Position]:
        if self._demo or not self._exchange:
            return []
        # TODO: Implement real position fetch
        positions = []
        balance = self._exchange.fetch_balance()
        for currency, amount_info in balance.items():
            if isinstance(amount_info, dict) and amount_info.get("total", 0) > 0:
                if currency in ("free", "used", "total", "info"):
                    continue
                total = amount_info.get("total", 0)
                positions.append(
                    Position(
                        symbol=f"{currency}/USD",
                        qty=total,
                        avg_entry_price=0.0,
                        current_price=0.0,
                        unrealized_pl=0.0,
                        unrealized_plpc=0.0,
                        market_value=0.0,
                        side="long",
                    )
                )
        return positions

    # ── Orders ──────────────────────────────────────────────────────────────

    def submit_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        if self._demo or not self._exchange:
            return OrderResult(
                order_id=f"KRAKEN-DEMO-{datetime.now().timestamp()}",
                symbol=symbol,
                action=side,
                qty=qty,
                status="filled",
                filled_price=0.0,
            )
        # TODO: Implement real order submission
        try:
            order = self._exchange.create_market_order(symbol, side.lower(), qty)
            return OrderResult(
                order_id=order.get("id", "unknown"),
                symbol=symbol,
                action=side,
                qty=qty,
                status=order.get("status", "filled"),
                filled_price=float(order.get("average") or order.get("price") or 0),
            )
        except Exception as exc:
            logger.error("Kraken order failed for %s %s %d: %s", side, symbol, qty, exc)
            raise

    def close_position(self, symbol: str) -> OrderResult | None:
        if self._demo or not self._exchange:
            return None
        # TODO: Implement position close
        return None

    # ── Market Data ───────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        if self._demo or not self._exchange:
            return pd.DataFrame()
        try:
            ohlcv = self._exchange.fetch_ohlcv(symbol, timeframe="1d", limit=days)
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Kraken OHLCV fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> float | None:
        if self._demo or not self._exchange:
            return None
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last") or 0)
        except Exception as exc:
            logger.warning("Kraken quote failed for %s: %s", symbol, exc)
            return None

    # ── Market Info ───────────────────────────────────────────────────────────

    def get_market_clock(self) -> MarketClock:
        # Crypto markets are 24/7
        return MarketClock(
            is_open=True,
            next_open="24/7",
            next_close="24/7",
        )

    # ── News ──────────────────────────────────────────────────────────────────

    def fetch_news(self, symbol: str, max_articles: int = 50,
                   days_ago: int = 0) -> list[tuple[str, float]]:
        # Kraken doesn't provide news via API
        return []
