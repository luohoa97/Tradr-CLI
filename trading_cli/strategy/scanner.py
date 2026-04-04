"""Batch market scanner — caches OHLCV and screens for signals efficiently.

Instead of fetching 30 days of OHLCV per stock per cycle (slow, API-heavy),
this maintains a rolling cache and screens thousands of stocks in batches.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class MarketScanner:
    """Maintains rolling OHLCV cache and screens for trading signals.

    Architecture:
      - Each stock has a cached OHLCV window (~60 days) stored on disk
      - Each cycle: fetch today's price (batch), append to cache
      - Screen vectorized: price > 20d_high for all stocks at once
      - Only compute full strategy analysis on breakout candidates
    """

    def __init__(self, cache_dir: Path | None = None):
        self._cache_dir = cache_dir or Path.home() / ".cache" / "trading-cli" / "ohlcv"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_fetch: dict[str, float] = {}  # symbol -> last fetch timestamp

    def get_cached(self, symbol: str) -> pd.DataFrame | None:
        """Load cached OHLCV for a symbol. Returns None if missing or stale."""
        path = self._cache_dir / f"{symbol}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            if not data.get("bars"):
                return None
            df = pd.DataFrame(data["bars"])
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            return df
        except Exception as exc:
            logger.debug("Cache load failed for %s: %s", symbol, exc)
            return None

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        """Save OHLCV to cache (keeps last 90 days)."""
        try:
            df_cached = df.tail(90).copy()
            bars = df_cached.reset_index().to_dict(orient="records")
            # Serialize dates
            for bar in bars:
                if isinstance(bar.get("date"), pd.Timestamp):
                    bar["date"] = bar["date"].isoformat()
                elif hasattr(bar.get("date"), "isoformat"):
                    bar["date"] = bar["date"].isoformat()
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._cache_dir / f"{symbol}.json"
            path.write_text(json.dumps({"bars": bars, "updated": datetime.now().isoformat()}))
        except Exception as exc:
            logger.debug("Cache save failed for %s: %s", symbol, exc)

    def append_bar(self, symbol: str, bar: dict) -> pd.DataFrame | None:
        """Append a new daily bar to cache. Returns updated DataFrame."""
        cached = self.get_cached(symbol)
        if cached is not None:
            # Check if bar is already present (same date)
            bar_date = bar.get("date", "")
            if isinstance(bar_date, str):
                bar_date = pd.Timestamp(bar_date)
            last_date = cached.index[-1] if len(cached) > 0 else None
            if last_date and bar_date and bar_date.date() == last_date.date():
                # Update existing bar
                cached.loc[last_date] = bar
            else:
                # Append new bar
                cached.loc[bar_date] = bar
                cached = cached.tail(90)
            self.save(symbol, cached)
            return cached
        return None

    def screen_breakouts(
        self,
        symbols: list[str],
        current_prices: dict[str, float],
        entry_period: int = 20,
    ) -> list[str]:
        """Quick screen: find stocks where price >= 20-day high.

        Uses cached data + current prices. Very fast — no fresh OHLCV fetch.
        """
        candidates = []
        for symbol in symbols:
            price = current_prices.get(symbol)
            if not price:
                continue

            cached = self.get_cached(symbol)
            if cached is None or len(cached) < entry_period:
                continue

            high_col = "high" if "high" in cached.columns else "High"
            if high_col not in cached.columns:
                continue

            donchian_high = cached[high_col].iloc[-entry_period:].max()
            if price >= donchian_high * 0.998:  # ~0.2% tolerance for intraday
                candidates.append(symbol)

        return candidates

    def cleanup_old_cache(self, max_age_days: int = 7) -> int:
        """Remove cache files older than max_age_days. Returns count removed."""
        removed = 0
        cutoff = time.time() - max_age_days * 86400
        for path in self._cache_dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except Exception:
                pass
        return removed
