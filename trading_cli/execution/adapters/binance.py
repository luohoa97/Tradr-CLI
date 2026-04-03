"""Binance adapter stub — crypto trading via Binance API.

This is a stub implementation. To enable:
1. Install ccxt: `uv add ccxt`
2. Add your Binance API keys to config
3. Implement the TODO sections below
"""

from __future__ import annotations

import logging
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
class BinanceAdapter(TradingAdapter):
    """Binance adapter for cryptocurrency trading.

    Requires: ccxt library (`uv add ccxt`)
    Config keys:
        binance_api_key: Your Binance API key
        binance_api_secret: Your Binance API secret
        binance_sandbox: Use sandbox/testnet (default: False)
    """

    def __init__(self, config: dict) -> None:
        self._config = config
        self._api_key = config.get("binance_api_key", "")
        self._api_secret = config.get("binance_api_secret", "")
        self._sandbox = config.get("binance_sandbox", False)
        self._demo = not (self._api_key and self._api_secret)

        if self._demo:
            logger.info("BinanceAdapter: no API keys found, stub mode only")
            return

        try:
            import ccxt
            self._exchange = ccxt.binance({
                "apiKey": self._api_key,
                "secret": self._api_secret,
                "enableRateLimit": True,
            })
            if self._sandbox:
                self._exchange.set_sandbox_mode(True)
            logger.info("BinanceAdapter connected (sandbox=%s)", self._sandbox)
        except ImportError:
            logger.warning("ccxt not installed. Run: uv add ccxt")
            self._demo = True
            self._exchange = None
        except Exception as exc:
            logger.error("Failed to connect to Binance: %s", exc)
            self._demo = True
            self._exchange = None

    @property
    def adapter_id(self) -> str:
        return "binance"

    @property
    def supports_paper_trading(self) -> bool:
        return self._sandbox  # Binance testnet

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
        # TODO: Implement real account fetch using self._exchange.fetch_balance()
        balance = self._exchange.fetch_balance()
        # Extract USDT balance as cash equivalent
        cash = float(balance.get("USDT", {}).get("free", 0))
        return AccountInfo(
            equity=cash,  # Simplified
            cash=cash,
            buying_power=cash,
            portfolio_value=cash,
        )

    def get_positions(self) -> list[Position]:
        if self._demo or not self._exchange:
            return []
        # TODO: Implement real position fetch
        # For crypto, positions are balances with non-zero amounts
        positions = []
        balance = self._exchange.fetch_balance()
        for currency, amount_info in balance.items():
            if isinstance(amount_info, dict) and amount_info.get("total", 0) > 0:
                if currency in ("free", "used", "total", "info"):
                    continue
                total = amount_info.get("total", 0)
                positions.append(
                    Position(
                        symbol=f"{currency}/USDT",
                        qty=total,
                        avg_entry_price=0.0,  # TODO: Track entry prices
                        current_price=0.0,  # TODO: Fetch current price
                        unrealized_pl=0.0,
                        unrealized_plpc=0.0,
                        market_value=0.0,
                        side="long",
                    )
                )
        return positions

    # ── Orders ───────────────────────────────────────────────────────────────

    def submit_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        if self._demo or not self._exchange:
            return OrderResult(
                order_id=f"BINANCE-DEMO-{datetime.now().timestamp()}",
                symbol=symbol,
                action=side,
                qty=qty,
                status="filled",
                filled_price=0.0,
            )
        # TODO: Implement real order submission
        try:
            # Convert to ccxt format: 'BTC/USDT'
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
            logger.error("Binance order failed for %s %s %d: %s", side, symbol, qty, exc)
            raise

    def close_position(self, symbol: str) -> OrderResult | None:
        if self._demo or not self._exchange:
            return None
        # TODO: Implement position close
        # Need to look up current position qty and sell all
        return None

    # ── Market Data ───────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        if self._demo or not self._exchange:
            return pd.DataFrame()
        try:
            # Binance uses 'BTC/USDT' format
            ohlcv = self._exchange.fetch_ohlcv(symbol, timeframe="1d", limit=days)
            df = pd.DataFrame(
                ohlcv,
                columns=["timestamp", "Open", "High", "Low", "Close", "Volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as exc:
            logger.warning("Binance OHLCV fetch failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> float | None:
        if self._demo or not self._exchange:
            return None
        try:
            ticker = self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last") or 0)
        except Exception as exc:
            logger.warning("Binance quote failed for %s: %s", symbol, exc)
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
        # Binance doesn't provide news via API
        return []
