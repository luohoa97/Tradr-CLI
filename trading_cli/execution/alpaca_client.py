"""Alpaca API wrapper — paper trading + market data."""
 
from __future__ import annotations
 
import logging
import random
from datetime import datetime, timezone
from typing import Any
 
logger = logging.getLogger(__name__)
 
 
class Position:
    """Unified position object (real or mock)."""
 
    def __init__(
        self,
        symbol: str,
        qty: float,
        avg_entry_price: float,
        current_price: float,
        unrealized_pl: float,
        unrealized_plpc: float,
        market_value: float,
        side: str = "long",
    ):
        self.symbol = symbol
        self.qty = qty
        self.avg_entry_price = avg_entry_price
        self.current_price = current_price
        self.unrealized_pl = unrealized_pl
        self.unrealized_plpc = unrealized_plpc
        self.market_value = market_value
        self.side = side
 
 
class AccountInfo:
    def __init__(self, equity: float, cash: float, buying_power: float, portfolio_value: float):
        self.equity = equity
        self.cash = cash
        self.buying_power = buying_power
        self.portfolio_value = portfolio_value
 
 
class OrderResult:
    def __init__(self, order_id: str, symbol: str, action: str, qty: int,
                 status: str, filled_price: float | None = None):
        self.order_id = order_id
        self.symbol = symbol
        self.action = action
        self.qty = qty
        self.status = status
        self.filled_price = filled_price
 
 
# ── Mock client for demo mode ──────────────────────────────────────────────────
 
class MockAlpacaClient:
    """Simulated Alpaca client for demo mode (no API keys required)."""
 
    def __init__(self) -> None:
        self.demo_mode = True
        self._cash = 100_000.0
        self._positions: dict[str, dict] = {}
        self._order_counter = 1000
        self._base_prices = {
            "AAPL": 175.0, "TSLA": 245.0, "NVDA": 875.0,
            "MSFT": 415.0, "AMZN": 185.0, "GOOGL": 175.0,
            "META": 510.0, "SPY": 520.0,
        }
        logger.info("MockAlpacaClient initialized in demo mode")
 
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
 
    def get_market_clock(self) -> dict:
        now = datetime.now(tz=timezone.utc)
        # Mock: market open weekdays 9:30–16:00 ET (UTC-5)
        hour_et = (now.hour - 5) % 24
        is_open = now.weekday() < 5 and 9 <= hour_et < 16
        return {"is_open": is_open, "next_open": "09:30 ET", "next_close": "16:00 ET"}
 
    def submit_market_order(
        self, symbol: str, qty: int, side: str
    ) -> OrderResult:
        price = self._get_mock_price(symbol)
        self._order_counter += 1
        order_id = f"MOCK-{self._order_counter}"
 
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
        base = self._base_prices.get(symbol, 100.0)
        # Small random walk so prices feel live
        noise = random.gauss(0, base * 0.002)
        return round(max(1.0, base + noise), 2)
 
    def historical_client(self) -> None:
        return None
 
 
# ── Real Alpaca client ─────────────────────────────────────────────────────────
 
class AlpacaClient:
    """Wraps alpaca-py SDK for paper trading."""
 
    def __init__(self, api_key: str, api_secret: str, paper: bool = True) -> None:
        self.demo_mode = False
        self._paper = paper
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
 
            self._trading_client = TradingClient(
                api_key=api_key,
                secret_key=api_secret,
                paper=paper,
            )
            self.historical_client = StockHistoricalDataClient(
                api_key=api_key,
                secret_key=api_secret,
            )
            logger.info("AlpacaClient connected (paper=%s)", paper)
        except ImportError as exc:
            raise RuntimeError("alpaca-py not installed. Run: uv add alpaca-py") from exc
 
    def get_account(self) -> AccountInfo:
        acct = self._trading_client.get_account()
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            portfolio_value=float(acct.portfolio_value),
        )
 
    def get_positions(self) -> list[Position]:
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
 
    def get_market_clock(self) -> dict:
        try:
            clock = self._trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": str(clock.next_open),
                "next_close": str(clock.next_close),
            }
        except Exception as exc:
            logger.warning("get_market_clock failed: %s", exc)
            return {"is_open": False, "next_open": "Unknown", "next_close": "Unknown"}
 
    def submit_market_order(
        self, symbol: str, qty: int, side: str
    ) -> OrderResult:
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
 
 
def create_client(config: dict) -> AlpacaClient | MockAlpacaClient:
    """Factory: return real AlpacaClient or MockAlpacaClient based on config."""
    key = config.get("alpaca_api_key", "")
    secret = config.get("alpaca_api_secret", "")
    if key and secret:
        try:
            return AlpacaClient(key, secret, paper=config.get("alpaca_paper", True))
        except Exception as exc:
            logger.error("Failed to create AlpacaClient: %s — falling back to demo mode", exc)
    return MockAlpacaClient()
