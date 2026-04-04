"""Configuration management — stores settings in ~/.config/trading-cli/config.toml."""
 
from __future__ import annotations
 
import toml
from pathlib import Path
 
CONFIG_DIR = Path("~/.config/trading-cli").expanduser()
CONFIG_PATH = CONFIG_DIR / "config.toml"
DB_PATH = CONFIG_DIR / "trades.db"
 
DEFAULT_CONFIG: dict = {
    "alpaca_api_key": "",
    "alpaca_api_secret": "",
    "alpaca_paper": True,
    "adapter_id": "alpaca",
    "auto_trading": False,
    "sentiment_model": "finbert",
    "risk_pct": 0.02,
    "max_drawdown": 0.15,
    "stop_loss_pct": 0.05,
    "max_positions": 10,
    "default_symbols": ["AAPL", "TSLA", "NVDA"],
    "poll_interval_prices": 30,
    "poll_interval_news": 900,
    "poll_interval_signals": 300,
    "poll_interval_positions": 60,
    "initial_cash": 100000.0,
    "finbert_batch_size": 50,
    "sma_short": 20,
    "sma_long": 50,
    "rsi_period": 14,
    "signal_buy_threshold": 0.5,
    "signal_sell_threshold": -0.3,
    "position_size_warning": 1000.0,
    # ── Strategy weights ──────────────────────────────────────────────────────
    "tech_weight": 0.6,
    "sent_weight": 0.4,
    # ── Technical indicator weights ───────────────────────────────────────────
    "weight_sma": 0.25,
    "weight_rsi": 0.25,
    "weight_bb": 0.20,
    "weight_ema": 0.15,
    "weight_volume": 0.15,
    # ── Bollinger Bands ───────────────────────────────────────────────────────
    "bb_window": 20,
    "bb_std": 2.0,
    # ── EMA periods ───────────────────────────────────────────────────────────
    "ema_fast": 12,
    "ema_slow": 26,
    # ── Volume SMA window ─────────────────────────────────────────────────────
    "volume_window": 20,
    # ── Sentiment event weights ───────────────────────────────────────────────
    "event_weight_earnings": 1.5,
    "event_weight_executive": 1.3,
    "event_weight_product": 1.2,
    "event_weight_macro": 1.4,
    "event_weight_generic": 0.8,
    "sentiment_half_life_hours": 24.0,
}
 
 
def load_config() -> dict:
    """Load config from disk, creating defaults if absent."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        save_config(DEFAULT_CONFIG)
        return dict(DEFAULT_CONFIG)
    with open(CONFIG_PATH) as f:
        on_disk = toml.load(f)
    merged = dict(DEFAULT_CONFIG)
    merged.update(on_disk)
    return merged
 
 
def save_config(config: dict) -> None:
    """Persist config to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        toml.dump(config, f)
 
 
def get_db_path() -> Path:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return DB_PATH
 
 
def is_demo_mode(config: dict) -> bool:
    """True if Alpaca keys are not configured."""
    return not (config.get("alpaca_api_key") and config.get("alpaca_api_secret"))
