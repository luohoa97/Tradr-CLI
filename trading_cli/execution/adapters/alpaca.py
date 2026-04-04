"""Alpaca adapter — real Alpaca API for stocks."""

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
class AlpacaAdapter(TradingAdapter):
    """Alpaca Markets adapter for US equities (paper & live trading)."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._api_key = config.get("alpaca_api_key", "")
        self._api_secret = config.get("alpaca_api_secret", "")
        self._paper = config.get("alpaca_paper", True)
        self._demo = not (self._api_key and self._api_secret)

        if self._demo:
            logger.info("AlpacaAdapter: no API keys found, running in demo mode")
            return

        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.historical.news import NewsClient

            self._trading_client = TradingClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
                paper=self._paper,
            )
            self._historical_client = StockHistoricalDataClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
            self._news_client = NewsClient(
                api_key=self._api_key,
                secret_key=self._api_secret,
            )
            logger.info("AlpacaAdapter connected (paper=%s)", self._paper)
        except ImportError as exc:
            raise RuntimeError("alpaca-py not installed. Run: uv add alpaca-py") from exc
        except Exception as exc:
            logger.error("Failed to connect to Alpaca: %s", exc)
            self._demo = True

    @property
    def adapter_id(self) -> str:
        return "alpaca"

    @property
    def supports_paper_trading(self) -> bool:
        return True

    @property
    def is_demo_mode(self) -> bool:
        return self._demo

    # ── Account & Positions ───────────────────────────────────────────────────

    def get_account(self) -> AccountInfo:
        if self._demo:
            return AccountInfo(
                equity=100000.0,
                cash=100000.0,
                buying_power=400000.0,
                portfolio_value=100000.0,
            )
        acct = self._trading_client.get_account()
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value),
        )

    def get_positions(self) -> list[Position]:
        if self._demo:
            return []
        raw = self._trading_client.get_all_positions()
        out = []
        for p in raw:
            out.append(
                Position(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    current_price=float(p.current_price),
                    unrealized_pl=float(p.unrealized_pl),
                    unrealized_plpc=float(p.unrealized_plpc),
                    market_value=float(p.market_value),
                    side=str(p.side),
                )
            )
        return out

    # ── Orders ───────────────────────────────────────────────────────────────

    def submit_market_order(self, symbol: str, qty: int, side: str) -> OrderResult:
        if self._demo:
            return OrderResult(
                order_id=f"DEMO-{datetime.now().timestamp()}",
                symbol=symbol,
                action=side,
                qty=qty,
                status="filled",
                filled_price=100.0,  # Mock price
            )

        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=TimeInForce.DAY,
        )
        try:
            order = self._trading_client.submit_order(order_data=req)
            filled_price = float(order.filled_avg_price) if order.filled_avg_price else None
            return OrderResult(
                order_id=str(order.id),
                symbol=symbol,
                action=side,
                qty=qty,
                status=str(order.status),
                filled_price=filled_price,
            )
        except Exception as exc:
            logger.error("Order submission failed for %s %s %d: %s", side, symbol, qty, exc)
            raise

    def close_position(self, symbol: str) -> OrderResult | None:
        if self._demo:
            return None
        try:
            response = self._trading_client.close_position(symbol)
            return OrderResult(
                order_id=str(response.id),
                symbol=symbol,
                action="SELL",
                qty=int(float(response.qty or 0)),
                status=str(response.status),
            )
        except Exception as exc:
            logger.error("Close position failed for %s: %s", symbol, exc)
            return None

    # ── Market Data ───────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, days: int = 90) -> pd.DataFrame:
        if self._demo:
            # Fallback to yfinance in demo mode
            from trading_cli.data.market import fetch_ohlcv_yfinance
            return fetch_ohlcv_yfinance(symbol, days)

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            end = datetime.now(tz=timezone.utc)
            start = end - timedelta(days=days + 10)  # extra buffer for weekends

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                feed="iex",
            )
            bars = self._historical_client.get_stock_bars(request)
            df = bars.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else df
            df.index = pd.to_datetime(df.index, utc=True)
            df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                     "close": "Close", "volume": "Volume"})
            return df.tail(days)
        except Exception as exc:
            logger.warning("Alpaca OHLCV fetch failed for %s: %s — falling back to yfinance", symbol, exc)
            from trading_cli.data.market import fetch_ohlcv_yfinance
            return fetch_ohlcv_yfinance(symbol, days)

    def get_latest_quote(self, symbol: str) -> float | None:
        if self._demo:
            return None
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
            trades = self._historical_client.get_stock_latest_trade(req)
            return float(trades[symbol].price)
        except Exception as exc:
            logger.warning("Alpaca latest quote failed for %s: %s", symbol, exc)
            return None

    def get_latest_quotes_batch(self, symbols: list[str]) -> dict[str, float]:
        if self._demo:
            return {}
        try:
            from alpaca.data.requests import StockLatestTradeRequest

            req = StockLatestTradeRequest(symbol_or_symbols=symbols, feed="iex")
            trades = self._historical_client.get_stock_latest_trade(req)
            return {sym: float(trade.price) for sym, trade in trades.items()}
        except Exception as exc:
            logger.warning("Batch Alpaca quote failed: %s", exc)
            return {}

    # ── Market Info ───────────────────────────────────────────────────────────

    def get_market_clock(self) -> MarketClock:
        if self._demo:
            now = datetime.now(tz=timezone.utc)
            hour_et = (now.hour - 5) % 24
            is_open = now.weekday() < 5 and 9 <= hour_et < 16
            return MarketClock(
                is_open=is_open,
                next_open="09:30 ET",
                next_close="16:00 ET",
            )
        try:
            clock = self._trading_client.get_clock()
            return MarketClock(
                is_open=clock.is_open,
                next_open=str(clock.next_open),
                next_close=str(clock.next_close),
            )
        except Exception as exc:
            logger.warning("get_market_clock failed: %s", exc)
            return MarketClock(is_open=False, next_open="Unknown", next_close="Unknown")

    # ── News ──────────────────────────────────────────────────────────────────

    def fetch_news(self, symbol: str, max_articles: int = 50,
                   days_ago: int = 0) -> list[tuple[str, float]]:
        if self._demo or not hasattr(self, '_news_client') or self._news_client is None:
            return []

        try:
            from alpaca.data.requests import NewsRequest

            now = datetime.now(tz=timezone.utc)
            target_date = now - timedelta(days=days_ago)
            day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            day_end = target_date.replace(hour=23, minute=59, second=59)

            request = NewsRequest(
                symbols=symbol,
                start=day_start,
                end=day_end,
                limit=min(max_articles, 100),
            )
            response = self._news_client.get_news(request)
            items = getattr(response, "news", response) if response else []

            headlines: list[tuple[str, float]] = []
            for item in items:
                title = getattr(item, "headline", "") or getattr(item, "title", "")
                if not title:
                    continue
                created = getattr(item, "created_at", None) or getattr(item, "updated_at", None)
                if created:
                    import pandas as pd
                    ts = pd.Timestamp(created).timestamp() if isinstance(created, str) else float(created)
                else:
                    ts = now.timestamp()
                headlines.append((title, float(ts)))

            return headlines
        except Exception as exc:
            logger.warning("Alpaca news fetch failed for %s: %s", symbol, exc)
            return []

    # ── Asset Search ──────────────────────────────────────────────────────────

    def get_all_assets(self) -> list[dict[str, str]]:
        """Fetch all available assets with their symbols and company names.
        
        Returns:
            List of dicts with 'symbol' and 'name' keys.
        """
        if self._demo:
            # Return a basic hardcoded list for demo mode
            return [
                {"symbol": "AAPL", "name": "Apple Inc."},
                {"symbol": "TSLA", "name": "Tesla Inc."},
                {"symbol": "NVDA", "name": "NVIDIA Corporation"},
                {"symbol": "MSFT", "name": "Microsoft Corporation"},
                {"symbol": "AMZN", "name": "Amazon.com Inc."},
                {"symbol": "GOOGL", "name": "Alphabet Inc. Class A"},
                {"symbol": "META", "name": "Meta Platforms Inc."},
                {"symbol": "SPY", "name": "SPDR S&P 500 ETF Trust"},
            ]
        
        try:
            from alpaca.trading.requests import GetAssetsRequest
            from alpaca.trading.enums import AssetStatus, AssetClass
            
            # Get all active US equity assets
            request = GetAssetsRequest(
                status=AssetStatus.ACTIVE,
                asset_class=AssetClass.US_EQUITY,
            )
            assets = self._trading_client.get_all_assets(request)
            
            return [
                {"symbol": asset.symbol, "name": asset.name}
                for asset in assets
                if asset.tradable
            ]
        except Exception as exc:
            logger.warning("Failed to fetch assets: %s", exc)
            return []
