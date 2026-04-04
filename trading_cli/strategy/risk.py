"""Risk management — position sizing, stop-loss, drawdown checks."""
 
from __future__ import annotations
 
import logging
import math
 
logger = logging.getLogger(__name__)
 
 
def check_market_regime(
    spy_ohlcv: pd.DataFrame,
    period: int = 200,
) -> str:
    """
    Determine if the broad market is Bullish or Bearish.
    Uses SPY or QQQ 200-day SMA as a proxy.
    """
    if spy_ohlcv.empty or len(spy_ohlcv) < period:
        return "UNKNOWN"
    
    close_col = "close" if "close" in spy_ohlcv.columns else "Close"
    closes = spy_ohlcv[close_col]
    sma = closes.rolling(window=period).mean().iloc[-1]
    current = closes.iloc[-1]
    
    return "BULLISH" if current > sma else "BEARISH"


def calculate_position_size(
    portfolio_value: float,
    price: float,
    risk_pct: float = 0.02,
    max_position_pct: float = 0.10,
) -> int:
    """
    Calculate number of shares to buy.
 
    risk_pct:         fraction of portfolio to risk per trade (default 2%)
    max_position_pct: cap single position at X% of portfolio
    Returns at least 1 share, never more than the cap.
    """
    if price <= 0 or portfolio_value <= 0:
        return 0
    
    # Calculate shares based on risk budget
    risk_budget = portfolio_value * risk_pct
    shares_by_risk = math.floor(risk_budget / price)
    
    # Calculate shares based on portfolio cap
    max_budget = portfolio_value * max_position_pct
    max_shares = math.floor(max_budget / price)
    
    # Use the smaller of the two, but at least 1 if we can afford it
    shares = min(shares_by_risk, max_shares)
    if shares <= 0 and max_shares > 0:
        shares = 1
        
    logger.debug(
        "Position size: portfolio=%.0f price=%.2f risk_pct=%.2f max_pos_pct=%.2f → %d shares",
        portfolio_value, price, risk_pct, max_position_pct, shares,
    )
    return shares
 
 
def check_stop_loss(
    entry_price: float,
    current_price: float,
    threshold: float = 0.05,
) -> bool:
    """True if position has fallen more than `threshold` from entry (long only)."""
    if entry_price <= 0:
        return False
    loss_pct = (entry_price - current_price) / entry_price
    return loss_pct >= threshold
 
 
def check_max_drawdown(
    portfolio_values: list[float],
    max_dd: float = 0.15,
) -> bool:
    """
    True if the portfolio has drawn down more than `max_dd` from its peak.
    Expects a time-ordered list of portfolio values.
    """
    if len(portfolio_values) < 2:
        return False
    peak = max(portfolio_values)
    current = portfolio_values[-1]
    if peak == 0:
        return False
    drawdown = (peak - current) / peak
    return drawdown >= max_dd
 
 
def validate_buy(
    symbol: str,
    price: float,
    qty: int,
    cash: float,
    positions: dict,
    max_positions: int = 10,
) -> tuple[bool, str]:
    """Check if a BUY order is valid."""
    cost = price * qty
    if cash < cost:
        return False, f"Insufficient cash: need ${cost:.2f}, have ${cash:.2f}"
    if len(positions) >= max_positions and symbol not in positions:
        return False, f"Max positions ({max_positions}) reached"
    return True, "OK"
 
 
def validate_sell(
    symbol: str,
    qty: int,
    positions: dict,
) -> tuple[bool, str]:
    """Check if a SELL order is valid."""
    pos = positions.get(symbol)
    if not pos:
        return False, f"No position in {symbol}"
    held = pos.get("qty", 0) if isinstance(pos, dict) else getattr(pos, "qty", 0)
    if held < qty:
        return False, f"Hold {held} shares, cannot sell {qty}"
    return True, "OK"
