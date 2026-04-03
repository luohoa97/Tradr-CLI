"""Trading signal generation — hybrid technical + sentiment pipeline."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Technical indicators ───────────────────────────────────────────────────────

def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window=window, min_periods=1).mean()


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average using span parameter."""
    return prices.ewm(span=span, adjust=False).mean()


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def calculate_bollinger_bands(
    prices: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns (upper_band, middle_band, lower_band).
    Middle = SMA(window), bands = middle ± num_std * std.
    """
    middle = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def calculate_atr(ohlcv: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — measures volatility."""
    high = ohlcv["High"] if "High" in ohlcv.columns else ohlcv.get("high", pd.Series())
    low = ohlcv["Low"] if "Low" in ohlcv.columns else ohlcv.get("low", pd.Series())
    close = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.get("close", pd.Series())

    if high.empty or low.empty or close.empty:
        return pd.Series(dtype=float)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()


def calculate_volume_sma(volume: pd.Series, window: int = 20) -> pd.Series:
    """Simple moving average of volume."""
    return volume.rolling(window=window, min_periods=1).mean()
 
 
# ── Component scores ───────────────────────────────────────────────────────────

def sma_crossover_score(ohlcv: pd.DataFrame, short: int = 20, long_: int = 50) -> float:
    """
    SMA crossover: +1.0 (bullish) if short > long, -1.0 if short < long.
    Returns 0.0 when insufficient data.
    """
    closes = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.get("close", pd.Series())
    if closes.empty or len(closes) < max(short, long_):
        return 0.0
    sma_s = calculate_sma(closes, short).iloc[-1]
    sma_l = calculate_sma(closes, long_).iloc[-1]
    if sma_l == 0:
        return 0.0
    raw = (sma_s - sma_l) / sma_l  # percent difference
    # Clamp to [-1, +1] — anything > 5% treated as max signal
    return float(max(-1.0, min(1.0, raw * 20)))


def rsi_score(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """
    RSI oversold (<30) → +1.0, overbought (>70) → -1.0, else linear interpolation.
    """
    closes = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.get("close", pd.Series())
    if closes.empty or len(closes) < period:
        return 0.0
    rsi = calculate_rsi(closes, period).iloc[-1]
    if rsi <= 30:
        return 1.0
    if rsi >= 70:
        return -1.0
    # Linear interpolation: 30→+1.0, 50→0.0, 70→-1.0
    return float((50.0 - rsi) / 20.0)


def bollinger_score(ohlcv: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> float:
    """
    Bollinger Bands: price near lower band → +1.0 (oversold), near upper → -1.0.
    Returns 0.0 when insufficient data.
    """
    closes = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.get("close", pd.Series())
    if closes.empty or len(closes) < window:
        return 0.0
    upper, middle, lower = calculate_bollinger_bands(closes, window, num_std)
    last_close = closes.iloc[-1]
    last_upper = upper.iloc[-1]
    last_lower = lower.iloc[-1]
    bandwidth = last_upper - last_lower
    if bandwidth == 0:
        return 0.0
    # Position within bands: 0.0 = lower, 0.5 = middle, 1.0 = upper
    position = (last_close - last_lower) / bandwidth
    # Map to [-1, +1]: lower band → +1.0, upper band → -1.0
    return float(1.0 - 2.0 * position)


def ema_score(ohlcv: pd.DataFrame, fast: int = 12, slow: int = 26) -> float:
    """
    EMA crossover: fast > slow → +1.0 (bullish), fast < slow → -1.0.
    Returns 0.0 when insufficient data.
    """
    closes = ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.get("close", pd.Series())
    if closes.empty or len(closes) < slow:
        return 0.0
    ema_fast = calculate_ema(closes, fast).iloc[-1]
    ema_slow = calculate_ema(closes, slow).iloc[-1]
    if ema_slow == 0:
        return 0.0
    raw = (ema_fast - ema_slow) / ema_slow
    return float(max(-1.0, min(1.0, raw * 20)))


def volume_score(ohlcv: pd.DataFrame, window: int = 20) -> float:
    """
    Volume spike detection: volume > 1.5x SMA → confirmation boost.
    Returns score based on how much volume exceeds average.
    """
    vol_col = "Volume" if "Volume" in ohlcv.columns else ohlcv.get("volume", pd.Series())
    if vol_col.empty or len(vol_col) < window:
        return 0.0
    vol_sma = calculate_volume_sma(vol_col, window)
    if vol_sma.empty or vol_sma.iloc[-1] == 0:
        return 0.0
    ratio = vol_col.iloc[-1] / vol_sma.iloc[-1]
    # Ratio > 1.5x = bullish confirmation, < 0.5x = weak signal
    if ratio >= 1.5:
        return min(1.0, (ratio - 1.0) / 1.0)  # 1.5→0.5, 2.0→1.0
    elif ratio <= 0.5:
        return max(-0.5, (ratio - 1.0) / 1.0)  # 0.5→-0.5
    return 0.0
 
 
def technical_score(
    ohlcv: pd.DataFrame,
    sma_short: int = 20,
    sma_long: int = 50,
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    ema_fast: int = 12,
    ema_slow: int = 26,
    vol_window: int = 20,
    weights: dict[str, float] | None = None,
) -> float:
    """
    Combined technical score with configurable weights.

    Default weights: SMA=0.25, RSI=0.25, BB=0.20, EMA=0.15, Volume=0.15
    """
    if weights is None:
        weights = {
            "sma": 0.25,
            "rsi": 0.25,
            "bb": 0.20,
            "ema": 0.15,
            "volume": 0.15,
        }

    sma = sma_crossover_score(ohlcv, sma_short, sma_long)
    rsi = rsi_score(ohlcv, rsi_period)
    bb = bollinger_score(ohlcv, bb_window, bb_std)
    ema = ema_score(ohlcv, ema_fast, ema_slow)
    vol = volume_score(ohlcv, vol_window)

    total_weight = sum(weights.values())
    if total_weight == 0:
        return 0.0

    return float(
        (weights.get("sma", 0) * sma +
         weights.get("rsi", 0) * rsi +
         weights.get("bb", 0) * bb +
         weights.get("ema", 0) * ema +
         weights.get("volume", 0) * vol) / total_weight
    )
 
 
# ── Signal generation ──────────────────────────────────────────────────────────

def generate_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    sentiment_score: float,
    buy_threshold: float = 0.5,
    sell_threshold: float = -0.3,
    sma_short: int = 20,
    sma_long: int = 50,
    rsi_period: int = 14,
    bb_window: int = 20,
    bb_std: float = 2.0,
    ema_fast: int = 12,
    ema_slow: int = 26,
    vol_window: int = 20,
    tech_weight: float = 0.6,
    sent_weight: float = 0.4,
    tech_indicator_weights: dict[str, float] | None = None,
) -> dict:
    """
    Hybrid signal: tech_weight * technical + sent_weight * sentiment.

    Technical indicators: SMA, RSI, Bollinger Bands, EMA, Volume.
    All weights configurable via config file.

    Returns:
        {
            "symbol": str,
            "action": "BUY" | "SELL" | "HOLD",
            "confidence": float,      # |hybrid_score|
            "hybrid_score": float,    # [-1, +1]
            "technical_score": float,
            "sentiment_score": float,
            "reason": str,
        }
    """
    tech = technical_score(
        ohlcv, sma_short, sma_long, rsi_period,
        bb_window, bb_std, ema_fast, ema_slow, vol_window,
        tech_indicator_weights,
    )
    hybrid = tech_weight * tech + sent_weight * sentiment_score

    # Compute individual scores for reason string
    sma_s = sma_crossover_score(ohlcv, sma_short, sma_long)
    rsi_s = rsi_score(ohlcv, rsi_period)
    bb_s = bollinger_score(ohlcv, bb_window, bb_std)
    ema_s = ema_score(ohlcv, ema_fast, ema_slow)
    vol_s = volume_score(ohlcv, vol_window)

    # Build human-readable reason
    parts = []
    if abs(sma_s) > 0.1:
        parts.append(f"SMA{'↑' if sma_s > 0 else '↓'}")
    if abs(rsi_s) > 0.1:
        rsi_val = calculate_rsi(
            ohlcv["Close"] if "Close" in ohlcv.columns else ohlcv.get("close", pd.Series()),
            rsi_period,
        ).iloc[-1] if not ohlcv.empty else 50
        parts.append(f"RSI={rsi_val:.0f}")
    if abs(bb_s) > 0.1:
        parts.append(f"BB{'↓' if bb_s > 0 else '↑'}")  # bb_s>0 means price near lower band
    if abs(ema_s) > 0.1:
        parts.append(f"EMA{'↑' if ema_s > 0 else '↓'}")
    if abs(vol_s) > 0.1:
        parts.append(f"Vol{'↑' if vol_s > 0 else '↓'}")
    if abs(sentiment_score) > 0.1:
        parts.append(f"sent={sentiment_score:+.2f}")

    reason = " + ".join(parts) if parts else "neutral signals"

    if hybrid >= buy_threshold:
        action = "BUY"
    elif hybrid <= sell_threshold:
        action = "SELL"
    else:
        action = "HOLD"

    logger.debug(
        "%s signal=%s hybrid=%.3f tech=%.3f sent=%.3f",
        symbol, action, hybrid, tech, sentiment_score,
    )
    return {
        "symbol": symbol,
        "action": action,
        "confidence": abs(hybrid),
        "hybrid_score": hybrid,
        "technical_score": tech,
        "sentiment_score": sentiment_score,
        "reason": reason,
    }
