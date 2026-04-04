"""Backtest results screen — displays performance metrics and trade log."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, DataTable, Label, Static, Input, Button, LoadingIndicator
from textual.containers import Vertical, Horizontal, Center
from textual import work
from rich.text import Text

from trading_cli.widgets.ordered_footer import OrderedFooter

if TYPE_CHECKING:
    from trading_cli.strategy.backtest import BacktestResult


class BacktestScreen(Screen):
    """Screen for viewing backtest results."""

    CSS = """
    #backtest-progress {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
    }
    """


class BacktestSummary(Static):
    """Displays key backtest metrics."""

    def __init__(self, result: "BacktestResult | None" = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._result = result

    def set_result(self, result: "BacktestResult") -> None:
        self._result = result
        self.refresh()

    def render(self) -> str:
        if not self._result:
            return "[dim]No backtest data[/dim]"
        r = self._result
        ret_style = "bold green" if r.total_return_pct >= 0 else "bold red"
        dd_style = "bold red" if r.max_drawdown_pct > 10 else "bold yellow"
        sharpe_style = "bold green" if r.sharpe_ratio > 1 else ("bold yellow" if r.sharpe_ratio > 0 else "dim")
        return (
            f"[bold]{r.symbol}[/bold]  "
            f"[{ret_style}]Return: {r.total_return_pct:+.2f}%[/{ret_style}]  "
            f"[{dd_style}]MaxDD: {r.max_drawdown_pct:.2f}%[/{dd_style}]  "
            f"[{sharpe_style}]Sharpe: {r.sharpe_ratio:.2f}[/{sharpe_style}]  "
            f"Win Rate: {r.win_rate:.1f}%  "
            f"Trades: {r.total_trades} ({r.winning_trades}W / {r.losing_trades}L)  "
            f"${r.initial_capital:,.0f} → ${r.final_equity:,.0f}"
        )


class BacktestScreen(Screen):
    """Screen for viewing backtest results."""

    BINDINGS = [
        Binding("r", "run_backtest", "Run", show=True),
    ]

    _last_symbol: str = ""
    _last_result: "BacktestResult | None" = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Input(placeholder="Search by symbol or company name…", id="backtest-symbol-input")
            yield Button("🚀 Run", id="btn-backtest-run", variant="success")
            yield BacktestSummary(id="backtest-summary")
            yield Label("", id="backtest-progress")
            yield LoadingIndicator(id="backtest-loading")
            yield DataTable(id="backtest-table", cursor_type="row")
        yield OrderedFooter()

    def on_mount(self) -> None:
        tbl = self.query_one("#backtest-table", DataTable)
        tbl.add_column("Date", key="date")
        tbl.add_column("Action", key="action")
        tbl.add_column("Price $", key="price")
        tbl.add_column("Qty", key="qty")
        tbl.add_column("P&L $", key="pnl")
        tbl.add_column("Reason", key="reason")

        # Hide loading indicator initially
        try:
            loader = self.query_one("#backtest-loading", LoadingIndicator)
            loader.display = False
        except Exception:
            pass

        # Set progress label initially empty
        try:
            prog = self.query_one("#backtest-progress", Label)
            prog.update("")
        except Exception:
            pass

        # Set up autocomplete on the input
        self._setup_autocomplete()

    def _update_progress(self, text: str) -> None:
        """Update the backtest progress label."""
        try:
            prog = self.query_one("#backtest-progress", Label)
            prog.update(text)
        except Exception:
            pass

    def _setup_autocomplete(self) -> None:
        """Replace plain input with autocomplete-enabled input if asset search is ready."""
        app = self.app
        if not hasattr(app, 'asset_search') or not app.asset_search.is_ready:
            return

        try:
            from trading_cli.widgets.asset_autocomplete import create_asset_autocomplete
            from textual_autocomplete import AutoComplete

            # Find the plain input
            old_input = self.query_one("#backtest-symbol-input", Input)

            # Create autocomplete-enabled input + dropdown
            new_input, autocomplete_widget = create_asset_autocomplete(
                app.asset_search,
                placeholder="Search by symbol or company name… (Tab to complete)",
                id="backtest-symbol-input",
            )

            # Mount new input before old one, then remove old
            self.mount(new_input, before=old_input)
            old_input.remove()
            # Mount dropdown
            self.mount(autocomplete_widget)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to setup autocomplete: %s", exc)

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-backtest-run":
            self.action_run_backtest()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        
        # Extract symbol from autocomplete format "SYMBOL — Company Name"
        if " — " in value:
            symbol = value.split(" — ")[0].strip().upper()
        else:
            symbol = value.upper()
        
        if symbol:
            self._last_symbol = symbol
            self._run_backtest(symbol)

    def action_run_backtest(self) -> None:
        # Try to get symbol from input first
        try:
            inp = self.query_one("#backtest-symbol-input", Input)
            value = inp.value.strip()
            if value:
                # Extract symbol from autocomplete format
                if " — " in value:
                    symbol = value.split(" — ")[0].strip().upper()
                else:
                    symbol = value.upper()
                self._last_symbol = symbol
            elif not self._last_symbol:
                self.app.notify("Enter a symbol first", severity="warning")
                return
        except Exception:
            if not self._last_symbol:
                self.app.notify("Enter a symbol first", severity="warning")
                return

        self._run_backtest(self._last_symbol)

    def _run_backtest(self, symbol: str) -> None:
        """Kick off backtest in a background worker thread."""
        app = self.app
        if not hasattr(app, "adapter") or not hasattr(app, "config"):
            app.notify("App not fully initialized", severity="error")
            return

        # Show loading indicator
        try:
            loader = self.query_one("#backtest-loading", LoadingIndicator)
            loader.display = True
        except Exception:
            pass

        self.app.notify(f"Backtesting {symbol}…", timeout=2)
        self._execute_backtest(symbol)

    @work(thread=True, name="backtest-worker", exclusive=True)
    def _execute_backtest(self, symbol: str) -> None:
        """Run backtest in background thread (non-blocking)."""
        try:
            app = self.app
            from trading_cli.data.news import fetch_headlines_with_timestamps
            from trading_cli.strategy.backtest import BacktestEngine

            # Use adapter for historical data
            adapter = getattr(app, "adapter", None)
            self.app.call_from_thread(self._update_progress, "[dim]Fetching OHLCV data…[/dim]")
            if adapter and not adapter.is_demo_mode:
                ohlcv = adapter.fetch_ohlcv(symbol, days=365)
            else:
                # Fallback to yfinance for demo mode
                from trading_cli.data.market import fetch_ohlcv_yfinance
                ohlcv = fetch_ohlcv_yfinance(symbol, days=365)

            if ohlcv.empty:
                self.app.call_from_thread(self.app.notify, f"No data for {symbol}", severity="warning")
                self.app.call_from_thread(self._hide_loading)
                return

            # Build news_fetcher closure using adapter
            cfg = app.config

            def news_fetcher(sym: str, days_ago: int = 0) -> list[tuple[str, float]]:
                if adapter and hasattr(adapter, 'fetch_news'):
                    headlines = adapter.fetch_news(sym, max_articles=30, days_ago=days_ago)
                    if headlines:
                        return headlines
                # Fallback to Alpaca News via data module
                return fetch_headlines_with_timestamps(
                    sym,
                    days_ago=days_ago,
                    alpaca_key=cfg.get("alpaca_api_key", ""),
                    alpaca_secret=cfg.get("alpaca_api_secret", ""),
                    max_articles=30,
                )

            finbert = getattr(app, "finbert", None)
            has_adapter_news = adapter and hasattr(adapter, 'fetch_news')
            has_api_keys = bool(cfg.get("alpaca_api_key") and cfg.get("alpaca_api_secret"))
            use_sentiment = bool((has_adapter_news or has_api_keys) and finbert and finbert.is_loaded)

            # Use the app's strategy adapter if available
            strategy = getattr(app, "strategy", None)
            strategy_name = strategy.info().name if strategy else "default"
            self.app.call_from_thread(self.app.notify, f"Strategy: {strategy_name}", timeout=2)

            if use_sentiment:
                self.app.call_from_thread(self._update_progress, "[dim]Analyzing sentiment…[/dim]")
            else:
                self.app.call_from_thread(self._update_progress, "[dim]Running technical backtest…[/dim]")

            engine = BacktestEngine(
                config=cfg,
                finbert=finbert if finbert and finbert.is_loaded else None,
                news_fetcher=news_fetcher if (has_adapter_news or has_api_keys) else None,
                use_sentiment=use_sentiment,
                strategy=strategy,
                progress_callback=lambda msg: self.app.call_from_thread(self._update_progress, msg),
            )
            result = engine.run(symbol, ohlcv, initial_capital=100_000.0)
            self.app.call_from_thread(self._display_result, result)
        except Exception as exc:
            self.app.call_from_thread(
                self.app.notify,
                f"Backtest failed: {exc}",
                severity="error",
            )
            import logging
            logging.getLogger(__name__).error("Backtest error: %s", exc, exc_info=True)
            self.app.call_from_thread(self._hide_loading)

    def _hide_loading(self) -> None:
        """Hide the loading indicator."""
        try:
            loader = self.query_one("#backtest-loading", LoadingIndicator)
            loader.display = False
        except Exception:
            pass

    def _display_result(self, result: "BacktestResult") -> None:
        self._last_result = result
        self._hide_loading()
        self._update_progress("")

        summary = self.query_one("#backtest-summary", BacktestSummary)
        summary.set_result(result)

        tbl = self.query_one("#backtest-table", DataTable)
        tbl.clear()
        for trade in result.trades:
            action_style = "bold green" if trade.action == "BUY" else "bold red"
            pnl_val = trade.pnl if trade.pnl is not None else 0
            pnl_str = f"{pnl_val:+,.2f}" if pnl_val != 0 else "—"
            tbl.add_row(
                trade.timestamp[:10],
                Text(trade.action, style=action_style),
                f"{trade.price:.2f}",
                str(trade.qty),
                Text(pnl_str, style="green" if pnl_val > 0 else ("red" if pnl_val < 0 else "dim")),
                trade.reason[:50] if trade.reason else "",
            )
