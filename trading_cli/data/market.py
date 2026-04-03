"""Market data fetching — Alpaca historical bars with yfinance fallback."""
 
from __future__ import annotations
 
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
 
import pandas as pd
 
if TYPE_CHECKING:
    from trading_cli.execution.alpaca_client import AlpacaClient
 
logger = logging.getLogger(__name__)
 
 
def fetch_ohlcv_alpaca(
    client: "AlpacaClient",
    symbol: str,
    days: int = 90,
) -> pd.DataFrame:
    """Fetch OHLCV bars from Alpaca historical data API."""
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
        bars = client.historical_client.get_stock_bars(request)
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0) if symbol in df.index.get_level_values(0) else df
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        return df.tail(days)
    except Exception as exc:
        logger.warning("Alpaca OHLCV fetch failed for %s: %s — falling back to yfinance", symbol, exc)
        return fetch_ohlcv_yfinance(symbol, days)
 
 
def fetch_ohlcv_yfinance(symbol: str, days: int = 90) -> pd.DataFrame:
    """Fetch OHLCV bars from yfinance."""
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
 
 
def get_latest_quote_alpaca(client: "AlpacaClient", symbol: str) -> float | None:
    """Get latest trade price from Alpaca."""
    try:
        from alpaca.data.requests import StockLatestTradeRequest
 
        req = StockLatestTradeRequest(symbol_or_symbols=symbol, feed="iex")
        trades = client.historical_client.get_stock_latest_trade(req)
        return float(trades[symbol].price)
    except Exception as exc:
        logger.warning("Alpaca latest quote failed for %s: %s", symbol, exc)
        return None
 
 
def get_latest_quote_yfinance(symbol: str) -> float | None:
    """Get latest price from yfinance (free tier fallback)."""
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
 
 
def get_latest_quotes_batch(
    client: "AlpacaClient | None",
    symbols: list[str],
) -> dict[str, float]:
    """Return {symbol: price} dict for multiple symbols."""
    prices: dict[str, float] = {}
    if client and not client.demo_mode:
        try:
            from alpaca.data.requests import StockLatestTradeRequest
 
            req = StockLatestTradeRequest(symbol_or_symbols=symbols, feed="iex")
            trades = client.historical_client.get_stock_latest_trade(req)
            for sym, trade in trades.items():
                prices[sym] = float(trade.price)
            return prices
        except Exception as exc:
            logger.warning("Batch Alpaca quote failed: %s — falling back", exc)
 
    # yfinance fallback
    for sym in symbols:
        price = get_latest_quote_yfinance(sym)
        if price:
            prices[sym] = price
        time.sleep(0.2)  # avoid hammering
    return prices
