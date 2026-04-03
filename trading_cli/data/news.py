"""News headline fetching — Alpaca News API (historical) with yfinance fallback."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


# ── Alpaca News API (historical, date-aware) ───────────────────────────────────

def fetch_headlines_alpaca(
    api_key: str,
    api_secret: str,
    symbol: str,
    start: datetime | None = None,
    end: datetime | None = None,
    max_articles: int = 50,
) -> list[tuple[str, float]]:
    """Fetch headlines via Alpaca News API with optional date range.

    Returns list of (headline: str, unix_timestamp: float) tuples.
    Supports historical backtesting by specifying start/end dates.
    """
    if not api_key or not api_secret:
        return []
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(api_key=api_key, secret_key=api_secret)

        now = datetime.now(tz=timezone.utc)
        if end is None:
            end = now
        if start is None:
            start = end - timedelta(days=7)

        request = NewsRequest(
            symbols=symbol,
            start=start,
            end=end,
            limit=min(max_articles, 100),  # Alpaca max is 100 per page
        )
        response = client.get_news(request)
        items = getattr(response, "news", response) if response else []

        headlines: list[tuple[str, float]] = []
        for item in items:
            title = getattr(item, "headline", "") or getattr(item, "title", "")
            if not title:
                continue
            created = getattr(item, "created_at", None) or getattr(item, "updated_at", None)
            if created:
                if isinstance(created, str):
                    ts = pd.Timestamp(created).timestamp()
                elif isinstance(created, (int, float)):
                    ts = float(created)
                else:
                    ts = pd.Timestamp(created).timestamp()
            else:
                ts = now.timestamp()
            headlines.append((title, float(ts)))

        logger.debug("Alpaca News: got %d headlines for %s (%s to %s)",
                      len(headlines), symbol, start, end)
        return headlines
    except Exception as exc:
        logger.warning("Alpaca News fetch failed for %s: %s", symbol, exc)
        return []


def fetch_headlines_yfinance(symbol: str, max_articles: int = 20) -> list[str]:
    """Fetch headlines from yfinance built-in news feed."""
    try:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        news = ticker.news or []
        headlines = []
        for item in news[:max_articles]:
            title = item.get("title") or (item.get("content", {}) or {}).get("title", "")
            if title:
                headlines.append(title)
        logger.debug("yfinance news: got %d headlines for %s", len(headlines), symbol)
        return headlines
    except Exception as exc:
        logger.warning("yfinance news failed for %s: %s", symbol, exc)
        return []


# ── Unified fetcher ───────────────────────────────────────────────────────────

def fetch_headlines(
    symbol: str,
    max_articles: int = 20,
) -> list[str]:
    """Fetch headlines, using yfinance (Alpaca news returns tuples, not plain strings)."""
    return fetch_headlines_yfinance(symbol, max_articles)


def fetch_headlines_with_timestamps(
    symbol: str,
    days_ago: int = 0,
    alpaca_key: str = "",
    alpaca_secret: str = "",
    max_articles: int = 50,
) -> list[tuple[str, float]]:
    """Fetch headlines with Unix timestamps for temporal weighting.

    For backtesting: pass days_ago > 0 to get news from a specific historical date.
    Returns list of (headline: str, unix_timestamp: float) tuples.

    Priority: Alpaca (supports historical dates) > yfinance.
    """
    now = datetime.now(tz=timezone.utc)
    target_date = now - timedelta(days=days_ago)

    # Try Alpaca first (only supports historical if API keys are set)
    if alpaca_key and alpaca_secret:
        # Alpaca can fetch news for any historical date in range
        day_start = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start.replace(hour=23, minute=59, second=59)
        headlines = fetch_headlines_alpaca(alpaca_key, alpaca_secret, symbol,
                                           start=day_start, end=day_end,
                                           max_articles=max_articles)
        if headlines:
            return headlines

    # yfinance fallback (no timestamp info, approximate)
    headlines = fetch_headlines_yfinance(symbol, max_articles)
    now_ts = now.timestamp()
    return [(h, now_ts - (i * 3600)) for i, h in enumerate(headlines)]
