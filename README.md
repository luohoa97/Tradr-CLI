# trading-cli
 
A full-screen TUI AI trading application powered by **FinBERT** sentiment analysis and **Alpaca** paper trading.
 
```
┌─────────────────────────────────────────────────────────┐
│ TRADING CLI - Paper Trading Mode    Cash: $98,234.50    │
├─────────────────────────────────────────────────────────┤
│ [1] Dashboard  [2] Watchlist  [3] Portfolio             │
│ [4] Trades     [5] Sentiment  [6] Config  [q] Quit      │
├─────────────────────────────────────────────────────────┤
│ MARKET STATUS: ● OPEN   Last Updated: 14:23:45 EST      │
└─────────────────────────────────────────────────────────┘
```
 
---
 
## Features
 
| Feature | Details |
|---|---|
| Full-screen TUI | Textual-based, single command launch |
| FinBERT sentiment | Local inference, ProsusAI/finbert |
| Paper trading | Alpaca paper API (or built-in demo mode) |
| Live prices | Alpaca market data + yfinance fallback |
| Hybrid signals | 0.6 × technical + 0.4 × sentiment |
| Persistent state | SQLite (trades, watchlist, sentiment cache) |
| Demo mode | Works without any API keys |
 
---
 
## Quick Start
 
### 1. Install uv (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
 
### 2. Clone and install
```bash
git clone https://github.com/luohoa97/ai-trading.git
cd ai-trading
uv sync
```
 
### 3. Run
```bash
uv run trading-cli
```
 
On first launch, FinBERT (~500 MB) downloads from HuggingFace and is cached locally.
The app starts in **Demo Mode** automatically if no Alpaca keys are configured.
 
---
 
## Alpaca Paper Trading Setup (optional)
 
1. Sign up at [alpaca.markets](https://alpaca.markets) — free, no credit card needed
2. Generate paper trading API keys in the Alpaca dashboard
3. Open Config in the app (`6`), enter your keys, press `Ctrl+S`
 
The app always uses paper trading endpoints — no real money is ever at risk.
 
---
 
## Configuration
 
Config file: `~/.config/trading-cli/config.toml`
 
```toml
alpaca_api_key    = "PKxxxxxxxxxxxx"
alpaca_api_secret = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
alpaca_paper      = true
 
# Risk management
risk_pct        = 0.02    # 2% of portfolio per trade
max_drawdown    = 0.15    # halt trading at 15% drawdown
stop_loss_pct   = 0.05    # 5% stop-loss per position
max_positions   = 10
 
# Signal thresholds (hybrid score: -1 to +1)
signal_buy_threshold  =  0.5
signal_sell_threshold = -0.3

# Poll intervals (seconds)
poll_interval_prices    = 30
poll_interval_news      = 900
poll_interval_signals   = 300
poll_interval_positions = 60
```
 
---
 
## Keyboard Shortcuts
 
| Key | Action |
|---|---|
| `1`–`6` | Switch screens |
| `q` / `Ctrl+C` | Quit |
| `r` | Refresh current screen |
| `a` | Add symbol (Watchlist) |
| `d` | Delete selected symbol (Watchlist) |
| `x` | Close position (Portfolio) |
| `e` | Export trades to CSV (Trades) |
| `f` | Focus filter (Trades) |
| `Enter` | Submit symbol / confirm action |
| `Ctrl+S` | Save config (Config screen) |
 
---
 
## Screens
 
**1 — Dashboard**: Account balance, market status, live positions, real-time signal log.
 
**2 — Watchlist**: Add/remove symbols. See live prices, sentiment score, and BUY/SELL/HOLD signal per symbol.
 
**3 — Portfolio**: Full position detail from Alpaca. Press `x` to close a position via market order.
 
**4 — Trades**: Scrollable history with Alpaca `order_id`. Press `e` to export CSV.
 
**5 — Sentiment**: Type any symbol, press Enter — see FinBERT scores per headline and an aggregated gauge.
 
**6 — Config**: Edit API keys, thresholds, risk limits, toggle auto-trading.
 
---
 
## Trading Strategy
 
**Signal = 0.6 × technical + 0.4 × sentiment**
 
| Component | Calculation |
|---|---|
| `technical_score` | 0.5 × SMA crossover (20/50) + 0.5 × RSI(14) |
| `sentiment_score` | FinBERT weighted average on latest news |
| BUY | hybrid > +0.50 |
| SELL | hybrid < −0.30 |
 
In **manual mode** (default), signals appear in the log for review.
In **auto-trading mode** (Config → toggle), market orders are submitted automatically.
 
---
 
## Project Structure
 
```
trading_cli/
├── __main__.py           # Entry point: uv run trading-cli
├── app.py                # Textual App, workers, screen routing
├── config.py             # Load/save ~/.config/trading-cli/config.toml
├── screens/
│   ├── dashboard.py      # Screen 1 — main dashboard
│   ├── watchlist.py      # Screen 2 — symbol watchlist
│   ├── portfolio.py      # Screen 3 — positions & P&L
│   ├── trades.py         # Screen 4 — trade history
│   ├── sentiment.py      # Screen 5 — FinBERT analysis
│   └── config_screen.py  # Screen 6 — settings editor
├── widgets/
│   ├── positions_table.py  # Reusable P&L table
│   ├── signal_log.py       # Scrolling signal feed
│   └── sentiment_gauge.py  # Visual [-1, +1] gauge
├── sentiment/
│   ├── finbert.py          # Singleton model, batch inference, cache
│   └── aggregator.py       # Score aggregation + gauge renderer
├── strategy/
│   ├── signals.py          # SMA + RSI + sentiment hybrid signal
│   └── risk.py             # Position sizing, stop-loss, drawdown
├── execution/
│   └── alpaca_client.py    # Real AlpacaClient + MockAlpacaClient
└── data/
    ├── market.py           # OHLCV via Alpaca / yfinance
    ├── news.py             # Headlines via Alpaca News / yfinance
    └── db.py               # SQLite schema + all queries
```
 
---
 
## Database
 
Location: `~/.config/trading-cli/trades.db`
 
| Table | Contents |
|---|---|
| `trades` | Every executed order with Alpaca `order_id` |
| `signals` | Every generated signal (executed or not) |
| `watchlist` | Monitored symbols |
| `sentiment_cache` | MD5(headline) → label + score |
| `price_history` | OHLCV bars per symbol |
 
---
 
## Development
 
```bash
# Run app
uv run trading-cli
 
# Live logs
tail -f ~/.config/trading-cli/app.log
 
# Reset state
rm ~/.config/trading-cli/trades.db
rm ~/.config/trading-cli/config.toml
```# ai-trading
