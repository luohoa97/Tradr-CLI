"""
Main Textual application — screen routing, background workers, reactive state.
"""
 
from __future__ import annotations
 
import asyncio
import logging
import time
from datetime import datetime
from typing import Any
 
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, Label, ProgressBar, Static, LoadingIndicator, DataTable
from textual.containers import Vertical, Center
from textual import work

from trading_cli.widgets.ordered_footer import OrderedFooter

logger = logging.getLogger(__name__)
 
 
# ── Splash / loading screen ────────────────────────────────────────────────────
 
class SplashScreen(Screen):
    """Shown while FinBERT loads and Alpaca connects."""
 
    def __init__(self, status_messages: list[str] | None = None) -> None:
        super().__init__()
        self._messages = status_messages or []
 
    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="splash-inner"):
                yield Label(
                    "[bold cyan]TRADING CLI[/bold cyan]\n"
                    "[dim]AI-Powered Paper Trading[/dim]",
                    id="splash-title",
                )
                yield LoadingIndicator(id="splash-spinner")
                yield Label("Initialising…", id="splash-status")
 
    def set_status(self, msg: str) -> None:
        try:
            self.query_one("#splash-status", Label).update(msg)
        except Exception:
            pass
 
 
# ── Order confirmation modal ───────────────────────────────────────────────────
 
class OrderConfirmScreen(Screen):
    """Modal: confirm a BUY/SELL order before submitting."""
 
    def __init__(self, symbol: str, action: str, qty: int, price: float, reason: str) -> None:
        super().__init__()
        self._symbol = symbol
        self._action = action
        self._qty = qty
        self._price = price
        self._reason = reason
 
    def compose(self) -> ComposeResult:
        from textual.widgets import Button
        from textual.containers import Grid
 
        action_style = "green" if self._action == "BUY" else "red"
        with Grid(id="order-grid"):
            yield Label(
                f"[bold {action_style}]{self._action} {self._qty} {self._symbol}[/bold {action_style}]\n"
                f"Price: ~${self._price:.2f}   Est. value: ${self._qty * self._price:,.2f}\n"
                f"Reason: {self._reason}",
                id="order-msg",
            )
            from textual.containers import Horizontal
            with Horizontal(id="order-buttons"):
                yield Button("Execute", id="btn-exec", variant="success" if self._action == "BUY" else "error")
                yield Button("Cancel", id="btn-cancel", variant="default")
 
    def on_button_pressed(self, event) -> None:
        self.dismiss(event.button.id == "btn-exec")
 
 
# ── Main App ───────────────────────────────────────────────────────────────────
 
class TradingApp(App):
    """Full-screen TUI trading application."""
 
    CSS = """
    Screen {
        background: $surface;
    }
    #splash-inner {
        align: center middle;
        width: 60;
        height: auto;
        padding: 2 4;
        border: double $primary;
    }
    #splash-title {
        text-align: center;
        margin-bottom: 1;
    }
    #splash-status {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    #account-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    #main-split {
        height: 1fr;
    }
    #left-pane {
        width: 50%;
        border-right: solid $primary-darken-2;
        padding: 0 1;
    }
    #right-pane {
        width: 50%;
        padding: 0 1;
    }
    #signals-label, #positions-label {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    #signal-log {
        height: 1fr;
    }
    .config-label {
        width: 30;
        content-align: right middle;
        padding-right: 1;
    }
    .config-input {
        width: 40;
    }
    .config-select {
        width: 40;
    }
    .strategy-info {
        height: 3;
        padding: 0 1 0 31;
        color: $text-muted;
        text-style: italic;
    }
    #config-buttons {
        margin-top: 1;
        height: 3;
    }
    #order-grid {
        align: center middle;
        width: 60;
        height: auto;
        border: thick $error;
        padding: 2;
        background: $surface;
    }
    #order-msg {
        margin-bottom: 1;
    }
    #order-buttons {
        height: 3;
    }
    #confirm-grid {
        align: center middle;
        width: 55;
        height: auto;
        border: thick $warning;
        padding: 2;
        background: $surface;
    }
    #confirm-buttons {
        margin-top: 1;
        height: 3;
    }
    #wl-input-row {
        height: 3;
    }
    #wl-help, #sent-help, #trades-help {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }
    #sent-input-row {
        height: 3;
    }
    #sent-gauge {
        height: 2;
        padding: 0 1;
    }
    #sent-summary {
        height: 2;
        padding: 0 1;
    }
    #wl-table, #trades-table, #sent-table, #portfolio-table {
        height: 1fr;
    }
    #portfolio-summary {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    #trades-filter-row {
        height: 3;
    }
    #auto-trade-row {
        height: 3;
        margin-top: 1;
    }
    """
 
    BINDINGS = [
        Binding("1", "show_dashboard", "Dashboard", show=True, id="nav_dashboard"),
        Binding("2", "show_watchlist", "Watchlist", show=True, id="nav_watchlist"),
        Binding("3", "show_portfolio", "Portfolio", show=True, id="nav_portfolio"),
        Binding("4", "show_trades", "Trades", show=True, id="nav_trades"),
        Binding("5", "show_sentiment", "Sentiment", show=True, id="nav_sentiment"),
        Binding("6", "show_config", "Config", show=True, id="nav_config"),
        Binding("7", "show_backtest", "Backtest", show=True, id="nav_backtest"),
        Binding("ctrl+q", "quit", "Quit", show=True, id="nav_quit"),
        Binding("ctrl+c", "quit", "Quit", show=False),
    ]

    # Track running state for clean shutdown
    _running = True
 
    TITLE = "TRADING CLI"
    SUB_TITLE = "Paper Trading Mode"
 
    def __init__(self) -> None:
        super().__init__()
        self.config: dict = {}
        self.db_conn = None
        self.adapter = None
        self.strategy = None
        self.finbert = None
        self.demo_mode: bool = True
        self.market_open: bool = False
        self.watchlist: list[str] = []
        self._prices: dict[str, float] = {}
        self._sentiments: dict[str, float] = {}
        self._signals: dict[str, str] = {}
        self._portfolio_history: list[float] = []
 
    # ── Screens ────────────────────────────────────────────────────────────────
 
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        yield SplashScreen()
        yield OrderedFooter()
 
    # We install all named screens so push_screen(name) works
    CSS = """
    Screen {
        background: $surface;
    }
    #splash-inner {
        align: center middle;
        width: 60;
        height: auto;
        padding: 2 4;
        border: double $primary;
    }
    #splash-title {
        text-align: center;
        margin-bottom: 1;
    }
    #splash-status {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    #account-bar {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    #main-split {
        height: 1fr;
    }
    #left-pane {
        width: 50%;
        border-right: solid $primary-darken-2;
        padding: 0 1;
    }
    #right-pane {
        width: 50%;
        padding: 0 1;
    }
    #signals-label, #positions-label {
        height: 1;
        color: $primary;
        text-style: bold;
    }
    #signal-log {
        height: 1fr;
    }
    #config-scroll {
        width: 100%;
        height: 1fr;
    }
    #config-buttons {
        margin-top: 1;
        height: 3;
        align: center middle;
    }
    #config-buttons Button {
        margin: 0 1;
    }
    #order-grid {
        align: center middle;
        width: 60;
        height: auto;
        border: thick $error;
        padding: 2;
        background: $surface;
    }
    #order-msg {
        margin-bottom: 1;
    }
    #order-buttons {
        height: 3;
    }
    #confirm-grid {
        align: center middle;
        width: 55;
        height: auto;
        border: thick $warning;
        padding: 2;
        background: $surface;
    }
    #confirm-buttons {
        margin-top: 1;
        height: 3;
    }
    #wl-input-row {
        height: 3;
    }
    #wl-help, #sent-help, #trades-help {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }
    #sent-input-row {
        height: 3;
        margin-bottom: 1;
    }
    #sent-progress {
        height: 1;
        margin: 0 1;
    }
    #sent-neg-label, #sent-pos-label {
        height: 1;
        margin: 0 1;
    }
    #sent-summary {
        height: auto;
        max-height: 3;
        padding: 0 1;
    }
    #wl-table, #trades-table, #sent-table, #portfolio-table {
        height: 1fr;
    }
    #portfolio-summary {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    #portfolio-actions {
        height: 3;
        margin-bottom: 1;
    }
    #portfolio-actions Button {
        margin-right: 1;
    }
    #backtest-input-row {
        height: 3;
        margin-bottom: 1;
    }
    #backtest-input-row Input {
        width: 1fr;
    }
    #backtest-input-row Button {
        margin-left: 1;
    }
    #backtest-summary {
        height: auto;
        max-height: 3;
        padding: 0 1;
    }
    #trades-filter-row {
        height: 3;
    }
    #auto-trade-row {
        height: 3;
        margin-top: 1;
        align: left middle;
    }
    #strategy-info {
        height: auto;
        max-height: 3;
        padding: 0 1 0 2;
        color: $text-muted;
        text-style: italic;
    }
    Collapsible {
        width: 100%;
        height: auto;
    }
    """

    def on_mount(self) -> None:
        from trading_cli.screens.dashboard import DashboardScreen
        from trading_cli.screens.watchlist import WatchlistScreen
        from trading_cli.screens.portfolio import PortfolioScreen
        from trading_cli.screens.trades import TradesScreen
        from trading_cli.screens.sentiment import SentimentScreen
        from trading_cli.screens.config_screen import ConfigScreen
        from trading_cli.screens.backtest import BacktestScreen

        self.install_screen(DashboardScreen(), name="dashboard")
        self.install_screen(WatchlistScreen(), name="watchlist")
        self.install_screen(PortfolioScreen(), name="portfolio")
        self.install_screen(TradesScreen(), name="trades")
        self.install_screen(SentimentScreen(), name="sentiment")
        self.install_screen(ConfigScreen(), name="config")
        self.install_screen(BacktestScreen(), name="backtest")

        self._boot()

    @work(thread=True, name="boot")
    def _boot(self) -> None:
        """Boot sequence: load config → FinBERT → Alpaca → DB → start workers."""
        splash = self._get_splash()

        def status(msg: str) -> None:
            if splash:
                self.call_from_thread(splash.set_status, msg)
            logger.info(msg)
 
        # 1. Config
        status("Loading configuration…")
        from trading_cli.config import load_config, get_db_path, is_demo_mode
        self.config = load_config()
 
        # 2. Database
        status("Initialising database…")
        from trading_cli.data.db import init_db
        self.db_conn = init_db(get_db_path())
        from trading_cli.data.db import get_watchlist
        self.watchlist = get_watchlist(self.db_conn)
        if not self.watchlist:
            self.watchlist = list(self.config.get("default_symbols", ["AAPL", "TSLA"]))
            from trading_cli.data.db import add_to_watchlist
            for sym in self.watchlist:
                add_to_watchlist(self.db_conn, sym)
 
        # 3. FinBERT
        status("Loading FinBERT model (this may take ~30s on first run)…")
        from trading_cli.sentiment.finbert import FinBERTAnalyzer
        self.finbert = FinBERTAnalyzer.get_instance()
        success = self.finbert.load(progress_callback=status)
        if not success:
            error_msg = self.finbert.load_error or "Unknown error"
            status(f"FinBERT failed to load: {error_msg}")
 
        # 4. Trading adapter
        status("Connecting to trading platform…")
        from trading_cli.execution.adapter_factory import create_trading_adapter
        self.adapter = create_trading_adapter(self.config)
        self.demo_mode = self.adapter.is_demo_mode

        # 5. Asset search engine (for autocomplete)
        status("Loading asset search index…")
        from trading_cli.data.asset_search import AssetSearchEngine
        self.asset_search = AssetSearchEngine()
        asset_count = self.asset_search.load_assets(self.adapter)
        status(f"Asset search ready: {asset_count} assets indexed")
        # Load embedding model in background (optional, improves search quality)
        self._load_embedding_model_async()

        # 6. Strategy adapter
        status(f"Loading strategy: {self.config.get('strategy_id', 'hybrid')}…")
        from trading_cli.strategy.strategy_factory import create_trading_strategy
        self.strategy = create_trading_strategy(self.config)
        strategy_name = self.strategy.info().name
        status(f"Strategy: {strategy_name}")

        try:
            clock = self.adapter.get_market_clock()
            self.market_open = clock.is_open
        except Exception:
            self.market_open = False
 
        mode_str = "[DEMO MODE]" if self.demo_mode else "[PAPER MODE]"
        status(f"Ready! {mode_str} — loading dashboard…")
        time.sleep(0.5)
 
        # Switch to dashboard
        self.call_from_thread(self._switch_to_dashboard)
 
        # Start background workers
        self.call_from_thread(self._start_workers)
 
    def _get_splash(self) -> SplashScreen | None:
        try:
            return self.query_one(SplashScreen)
        except Exception:
            return None
 
    def _switch_to_dashboard(self) -> None:
        # Push dashboard on top of splash, then dismiss splash
        self.push_screen("dashboard")
        # Close the splash screen
        splash = self._get_splash()
        if splash:
            splash.dismiss()
        if self.demo_mode:
            self.notify("Running in DEMO MODE — add Alpaca keys in Config (6)", timeout=5)
        if self.finbert and not self.finbert.is_loaded:
            error_detail = self.finbert.load_error or "Unknown error"
            self.notify(
                f"FinBERT failed to load: {error_detail}\n"
                "Sentiment will show neutral. Press [r] on Sentiment screen to retry.",
                severity="warning",
                timeout=10,
            )
 
    def _start_workers(self) -> None:
        """Start all background polling workers."""
        self._running = True
        auto_enabled = self.config.get("auto_trading", False)
        logger.info("Starting workers (auto_trading=%s)", auto_enabled)
        self._poll_prices()
        self._poll_positions()
        self._poll_signals()
        if auto_enabled:
            logger.info("Auto-trading enabled — first signal cycle starting")

    @work(thread=True, name="load-embeddings", exclusive=False)
    def _load_embedding_model_async(self) -> None:
        """Load embedding model for semantic asset search (background)."""
        try:
            self.asset_search.load_embedding_model()
            if self.asset_search.has_semantic_search:
                self.call_from_thread(
                    self.notify,
                    "Semantic asset search enabled",
                    severity="information",
                    timeout=3,
                )
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s", exc)

    def _stop_workers(self) -> None:
        """Signal all workers to stop."""
        self._running = False

    def on_unmount(self) -> None:
        """Clean up on app shutdown."""
        self._stop_workers()
        logger.info("TradingApp shutting down...")
        # Ensure we exit with code 0 for clean shutdown
        self.exit(0)

    # ── Background workers ─────────────────────────────────────────────────────

    @work(thread=True, name="poll-prices", exclusive=False)
    def _poll_prices(self) -> None:
        """Continuously fetch latest prices for watchlist symbols."""
        while self._running:
            try:
                interval = self.config.get("poll_interval_prices", 30)
                if self.watchlist and self.adapter:
                    prices = self.adapter.get_latest_quotes_batch(self.watchlist)
                    if prices:
                        self._prices = prices
                        self.call_from_thread(self._on_prices_updated)
            except Exception as exc:
                logger.warning("Price poll error: %s", exc)
            time.sleep(self.config.get("poll_interval_prices", 30))
 
    @work(thread=True, name="poll-positions", exclusive=False)
    def _poll_positions(self) -> None:
        """Sync positions from Alpaca and update dashboard."""
        while self._running:
            try:
                if self.adapter:
                    acct = self.adapter.get_account()
                    positions = self.adapter.get_positions()
                    self._portfolio_history.append(acct.portfolio_value)
                    if len(self._portfolio_history) > 1000:
                        self._portfolio_history = self._portfolio_history[-1000:]
                    self.call_from_thread(self._on_positions_updated, acct, positions)
            except Exception as exc:
                logger.warning("Position poll error: %s", exc)
            time.sleep(self.config.get("poll_interval_positions", 60))

    @work(thread=True, name="poll-signals", exclusive=False)
    def _poll_signals(self) -> None:
        """Generate trading signals and optionally execute auto-trades."""
        time.sleep(5)  # short delay so prices are available first
        logger.info("Signal poll worker started, running first cycle")
        while self._running:
            try:
                self._run_signal_cycle()
            except Exception as exc:
                logger.warning("Signal cycle error: %s", exc)
            time.sleep(self.config.get("poll_interval_signals", 300))
 
    def _run_signal_cycle(self) -> None:
        from trading_cli.data.market import fetch_ohlcv_yfinance
        from trading_cli.data.news import fetch_headlines
        from trading_cli.sentiment.aggregator import aggregate_scores_weighted, aggregate_scores
        from trading_cli.sentiment.news_classifier import classify_headlines, EventType, DEFAULT_WEIGHTS as EVENT_WEIGHTS
        from trading_cli.strategy.risk import check_max_drawdown
        from trading_cli.data.db import save_signal

        auto_enabled = self.config.get("auto_trading", False)
        cycle_time = datetime.now().strftime("%H:%M:%S")
        logger.info("Running signal cycle at %s (auto_trading=%s)", cycle_time, auto_enabled)

        # Build event weight map from config
        event_weights = {
            EventType.EARNINGS: self.config.get("event_weight_earnings", EVENT_WEIGHTS[EventType.EARNINGS]),
            EventType.EXECUTIVE: self.config.get("event_weight_executive", EVENT_WEIGHTS[EventType.EXECUTIVE]),
            EventType.PRODUCT: self.config.get("event_weight_product", EVENT_WEIGHTS[EventType.PRODUCT]),
            EventType.MACRO: self.config.get("event_weight_macro", EVENT_WEIGHTS[EventType.MACRO]),
            EventType.GENERIC: self.config.get("event_weight_generic", EVENT_WEIGHTS[EventType.GENERIC]),
        }

        half_life = self.config.get("sentiment_half_life_hours", 24.0)
        auto_enabled = self.config.get("auto_trading", False)
        cycle_time = datetime.now().strftime("%H:%M:%S")

        # Update dashboard with cycle time
        self.call_from_thread(self._on_cycle_completed, cycle_time, auto_enabled)

        # When auto-trading, scan broader asset universe instead of just watchlist
        scan_universe = auto_enabled and hasattr(self, 'asset_search') and self.asset_search.is_ready
        if scan_universe:
            all_assets = [a["symbol"] for a in self.asset_search._assets]
            # Rotate through universe: scan 50 stocks per cycle, offset by cycle count
            cycle_offset = getattr(self, '_signal_cycle_count', 0) * 50
            self._signal_cycle_count = getattr(self, '_signal_cycle_count', 0) + 1
            symbols = all_assets[cycle_offset % len(all_assets):][:50]
        else:
            symbols = list(self.watchlist)

        for symbol in symbols:
            try:
                # Use shorter window for live signals (only need recent data for breakout detection)
                ohlcv = fetch_ohlcv_yfinance(symbol, days=30)
                if ohlcv.empty:
                    logger.warning(f"No OHLCV data for {symbol}, skipping")
                    continue

                price = self._prices.get(symbol)
                if price is None:
                    from trading_cli.data.market import get_latest_quote_yfinance
                    price = get_latest_quote_yfinance(symbol)
                    if price is None:
                        logger.warning(f"No price data for {symbol}, skipping")
                        continue

                headlines = fetch_headlines(symbol, max_articles=10)

                # Classify headlines by event type
                classifications = classify_headlines(headlines) if headlines else []

                sent_results = []
                timestamps = []
                if self.finbert and self.finbert.is_loaded:
                    sent_results = self.finbert.analyze_with_cache(headlines, self.db_conn)
                elif not self.finbert or not self.finbert.is_loaded:
                    logger.warning(f"FinBERT not loaded for {symbol}, using neutral sentiment")
                    error_detail = self.finbert.load_error if self.finbert else "FinBERT not initialized"
                    self.call_from_thread(
                        self._on_autotrade_error,
                        f"FinBERT not loaded: {error_detail}"
                    )

                # Use weighted aggregation if we have classifications
                if classifications and sent_results:
                    timestamps = [r.get("timestamp", 0) for r in sent_results] if sent_results else None
                    sent_score = aggregate_scores_weighted(
                        sent_results,
                        classifications=classifications,
                        timestamps=timestamps if any(timestamps) else None,
                        event_weights=event_weights,
                        half_life_hours=half_life,
                    )
                else:
                    sent_score = aggregate_scores(sent_results)

                self._sentiments[symbol] = sent_score

                # Use strategy adapter for signal generation
                signal_result = self.strategy.generate_signal(
                    symbol=symbol,
                    ohlcv=ohlcv,
                    sentiment_score=sent_score,
                    prices=self._prices,
                    config=self.config,
                )

                # Build signal dict for backward compatibility with DB and UI
                signal = {
                    "symbol": signal_result.symbol,
                    "action": signal_result.action,
                    "confidence": signal_result.confidence,
                    "hybrid_score": signal_result.score,
                    "technical_score": signal_result.metadata.get("sma_score", 0.0),
                    "sentiment_score": sent_score,
                    "reason": signal_result.reason,
                    "price": price or 0.0,
                }
                self._signals[symbol] = signal_result.action

                save_signal(
                    self.db_conn,
                    symbol=symbol,
                    action=signal["action"],
                    confidence=signal["confidence"],
                    technical_score=signal["technical_score"],
                    sentiment_score=signal["sentiment_score"],
                    reason=signal["reason"],
                )

                self.call_from_thread(self._on_signal_generated, signal)

                # ── Risk management checks before auto-execution ──────────────
                if auto_enabled and signal_result.action != "HOLD":
                    logger.info("Auto-trade %s signal for %s (confidence=%.2f)", signal_result.action, symbol, signal_result.confidence)
                    if check_max_drawdown(self._portfolio_history, self.config.get("max_drawdown", 0.15)):
                        logger.warning("Auto-trade skipped: max drawdown exceeded")
                        self.call_from_thread(
                            self._on_autotrade_blocked,
                            "Auto-trade blocked: Max drawdown exceeded"
                        )
                        continue
                    logger.info("Executing auto-trade: %s %s", signal_result.action, symbol)
                    self.call_from_thread(self._auto_execute, signal)
                elif auto_enabled and signal_result.action == "HOLD":
                    logger.debug(f"HOLD signal for {symbol}, no action taken")
                elif not auto_enabled:
                    logger.debug("Auto-trading disabled, signal %s for %s not executed", signal_result.action, symbol)

            except Exception as exc:
                logger.warning("Signal error for %s: %s", symbol, exc)
                self.call_from_thread(
                    self._on_autotrade_error,
                    f"Error processing {symbol}: {exc}"
                )
 
    # ── UI callbacks (called from thread via call_from_thread) ─────────────────

    def _on_prices_updated(self) -> None:
        try:
            wl_screen = self.get_screen("watchlist")
            if hasattr(wl_screen, "update_data"):
                wl_screen.update_data(self._prices, self._sentiments, self._signals)
        except Exception:
            pass

    def _on_cycle_completed(self, cycle_time: str, auto_enabled: bool) -> None:
        """Called when a signal cycle completes (from worker thread)."""
        try:
            dash = self.get_screen("dashboard")
            if hasattr(dash, "update_autotrade_status"):
                dash.update_autotrade_status(auto_enabled, cycle_time)
        except Exception:
            pass

    def _on_autotrade_error(self, error_msg: str) -> None:
        """Called when auto-trade encounters an error."""
        try:
            dash = self.get_screen("dashboard")
            if hasattr(dash, "update_autotrade_status"):
                dash.update_autotrade_status(error=error_msg)
        except Exception:
            pass

    def _on_autotrade_blocked(self, reason: str) -> None:
        """Called when auto-trade is blocked by risk management."""
        try:
            self.notify(reason, severity="warning", timeout=5)
        except Exception:
            pass
 
    def _on_positions_updated(self, acct, positions: list) -> None:
        try:
            dash = self.get_screen("dashboard")
            if hasattr(dash, "refresh_positions"):
                dash.refresh_positions(positions)
            if hasattr(dash, "refresh_account"):
                dash.refresh_account(acct)
        except Exception:
            pass
 
    def _on_signal_generated(self, signal: dict) -> None:
        try:
            dash = self.get_screen("dashboard")
            if hasattr(dash, "log_signal"):
                dash.log_signal(signal)
        except Exception:
            pass
 
    def _auto_execute(self, signal: dict) -> None:
        """Execute a signal automatically (auto_trading=True) with full risk management."""
        symbol = signal["symbol"]
        action = signal["action"]
        price = signal.get("price", 0.0)

        from trading_cli.strategy.risk import (
            calculate_position_size,
            validate_buy,
            validate_sell,
            check_stop_loss,
        )

        try:
            acct = self.adapter.get_account()
            positions = self.adapter.get_positions()
            positions_dict = {p.symbol: {"qty": p.qty, "avg_entry_price": p.avg_entry_price} for p in positions}

            if action == "BUY":
                ok, reason = validate_buy(
                    symbol, price, 1, acct.cash, positions_dict,
                    max_positions=self.config.get("max_positions", 10),
                )
                if not ok:
                    logger.warning("Auto-buy blocked: %s", reason)
                    self.call_from_thread(
                        self._on_autotrade_blocked,
                        f"Auto-buy {symbol} blocked: {reason}"
                    )
                    return

            elif action == "SELL":
                # Check stop-loss for existing position
                pos = positions_dict.get(symbol)
                if pos:
                    entry_price = pos.get("avg_entry_price", 0)
                    if check_stop_loss(entry_price, price, self.config.get("stop_loss_pct", 0.05)):
                        self.notify(f"Stop-loss triggered for {symbol} @ ${price:.2f}", severity="warning")

                ok, reason = validate_sell(symbol, 1, positions_dict)
                if not ok:
                    logger.warning("Auto-sell blocked: %s", reason)
                    self.call_from_thread(
                        self._on_autotrade_blocked,
                        f"Auto-sell {symbol} blocked: {reason}"
                    )
                    return

            qty = calculate_position_size(
                acct.portfolio_value,
                price or 1.0,
                risk_pct=self.config.get("risk_pct", 0.02),
                max_position_pct=0.10,
            )
            if qty < 1:
                logger.info(f"Auto-trade skipped: calculated qty < 1 for {symbol}")
                return

            result = self.adapter.submit_market_order(symbol, qty, action)
            if result.status not in ("rejected",):
                from trading_cli.data.db import save_trade
                save_trade(
                    self.db_conn, symbol, action,
                    result.filled_price or price, qty,
                    order_id=result.order_id,
                    reason=f"Auto: {signal['reason']}",
                )
                self.notify(
                    f"AUTO {action} {qty} {symbol} @ ${result.filled_price or price:.2f}",
                    timeout=5,
                )
            else:
                logger.warning(f"Auto-trade rejected: {symbol} {action}")
                self.call_from_thread(
                    self._on_autotrade_blocked,
                    f"Order rejected for {symbol} {action}"
                )
        except Exception as exc:
            logger.error("Auto-execute error: %s", exc)
            self.call_from_thread(
                self._on_autotrade_error,
                f"Auto-execute failed: {exc}"
            )
 
    # ── Manual order execution ─────────────────────────────────────────────────
 
    def execute_manual_order(
        self, symbol: str, action: str, qty: int, price: float, reason: str
    ) -> None:
        """Called from screens to submit a manual order with confirmation dialog."""
 
        def on_confirm(confirmed: bool) -> None:
            if not confirmed:
                return
            try:
                result = self.adapter.submit_market_order(symbol, qty, action)
                if result.status not in ("rejected",):
                    from trading_cli.data.db import save_trade
                    save_trade(
                        self.db_conn, symbol, action,
                        result.filled_price or price, qty,
                        order_id=result.order_id,
                        reason=reason,
                    )
                    self.notify(
                        f"{action} {qty} {symbol} @ ${result.filled_price or price:.2f} [{result.status}]"
                    )
                else:
                    self.notify(f"Order rejected for {symbol}", severity="error")
            except Exception as exc:
                self.notify(f"Order failed: {exc}", severity="error")
 
        self.push_screen(OrderConfirmScreen(symbol, action, qty, price, reason), callback=on_confirm)
 
    # ── Watchlist helpers ──────────────────────────────────────────────────────
 
    def add_to_watchlist(self, symbol: str) -> None:
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
            if self.db_conn:
                from trading_cli.data.db import add_to_watchlist
                add_to_watchlist(self.db_conn, symbol)
            self.notify(f"Added {symbol} to watchlist")
 
    def remove_from_watchlist(self, symbol: str) -> None:
        if symbol in self.watchlist:
            self.watchlist.remove(symbol)
            if self.db_conn:
                from trading_cli.data.db import remove_from_watchlist
                remove_from_watchlist(self.db_conn, symbol)
            self.notify(f"Removed {symbol} from watchlist")
 
    # ── Screen actions ─────────────────────────────────────────────────────────

    def action_show_dashboard(self) -> None:
        self.push_screen("dashboard")

    def action_show_watchlist(self) -> None:
        self.push_screen("watchlist")

    def action_show_portfolio(self) -> None:
        self.push_screen("portfolio")

    def action_show_trades(self) -> None:
        self.push_screen("trades")

    def action_show_sentiment(self) -> None:
        self.push_screen("sentiment")

    def action_show_config(self) -> None:
        self.push_screen("config")

    def action_show_backtest(self) -> None:
        self.push_screen("backtest")
