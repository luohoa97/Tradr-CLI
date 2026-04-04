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
from trading_cli.backtest.engine import BacktestResult


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

    CSS = """
    #backtest-progress {
        height: 1;
        padding: 0 1;
        color: $text-muted;
        text-style: italic;
    }

    #backtest-controls {
        height: auto;
        padding: 0 1;
    }

    #backtest-date-row {
        height: auto;
        layout: horizontal;
    }

    #backtest-date-row Input {
        width: 1fr;
    }

    #btn-backtest-run {
        width: 100%;
    }

    #backtest-summary {
        height: auto;
        padding: 0 1;
        color: $text;
    }

    #backtest-table {
        width: 100%;
    }
    """

    BINDINGS = [
        Binding("r", "run_backtest", "Run", show=True),
    ]

    _last_symbol: str = ""
    _last_result: "BacktestResult | None" = None
    _all_results: list["BacktestResult"] = []
    _pending_symbols: list[str] = []
    _start_date: str | None = None
    _end_date: str | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="backtest-controls"):
            with Horizontal(id="backtest-date-row"):
                yield Input(placeholder="Start date (YYYY-MM-DD)", id="backtest-start-date")
                yield Input(placeholder="End date (YYYY-MM-DD)", id="backtest-end-date")
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

    def _update_progress(self, text: str) -> None:
        """Update the backtest progress label."""
        try:
            prog = self.query_one("#backtest-progress", Label)
            prog.update(text)
        except Exception:
            pass

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-backtest-run":
            self.action_run_backtest()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id in ("backtest-start-date", "backtest-end-date"):
            self.action_run_backtest()

    def action_run_backtest(self) -> None:
        # Parse date range
        start_date = end_date = None
        try:
            start_input = self.query_one("#backtest-start-date", Input)
            end_input = self.query_one("#backtest-end-date", Input)
            if start_input.value.strip():
                start_date = start_input.value.strip()
            if end_input.value.strip():
                end_date = end_input.value.strip()
        except Exception:
            pass

        app = self.app
        if not hasattr(app, "config"):
            self.app.notify("App not fully initialized", severity="error")
            return

        symbols = app.config.get("default_symbols", ["AAPL", "TSLA", "NVDA"])
        if not symbols:
            self.app.notify("No symbols configured", severity="warning")
            return

        # Reset accumulated results
        self._all_results = []
        self._pending_symbols = list(symbols)
        self._start_date = start_date
        self._end_date = end_date

        label = f"{start_date or 'start'} → {end_date or 'now'}"
        self.app.notify(f"Backtesting {len(symbols)} symbols ({label})", timeout=2)

        # Show loading
        try:
            loader = self.query_one("#backtest-loading", LoadingIndicator)
            loader.display = True
        except Exception:
            pass

        # Clear table
        tbl = self.query_one("#backtest-table", DataTable)
        tbl.clear()

        # Update summary to show "Running…"
        summary = self.query_one("#backtest-summary", BacktestSummary)
        summary._result = None
        summary.refresh()

        # Kick off first symbol
        self._next_backtest()

    def _next_backtest(self) -> None:
        """Run the next pending symbol, or display final results."""
        if not self._pending_symbols:
            self._display_all_results()
            return

        symbol = self._pending_symbols.pop(0)
        self._update_progress(f"[dim]Backtesting {symbol}… ({len(self._pending_symbols)} remaining)[/dim]")
        self._execute_backtest(symbol, self._start_date, self._end_date)

    @work(thread=True, name="backtest-worker", exclusive=True)
    def _execute_backtest(self, symbol: str, start_date: str | None = None, end_date: str | None = None) -> None:
        """Run backtest in background thread (non-blocking)."""
        try:
            app = self.app
            from trading_cli.data.news import fetch_headlines_with_timestamps
            from trading_cli.backtest.engine import BacktestEngine

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
                debug=True,
            )
            result = engine.run(symbol, ohlcv, start_date=start_date, end_date=end_date, initial_capital=100_000.0)
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
        """Store result, then chain to next symbol or show combined view."""
        self._all_results.append(result)
        # Chain to next symbol
        self.app.call_from_thread(self._next_backtest)

    def _display_all_results(self) -> None:
        """Display combined backtest results for all symbols."""
        self._hide_loading()
        self._update_progress("")

        if not self._all_results:
            self.app.notify("No results", severity="warning")
            return

        # Aggregate metrics
        total_trades = sum(r.total_trades for r in self._all_results)
        total_wins = sum(r.winning_trades for r in self._all_results)
        total_losses = sum(r.losing_trades for r in self._all_results)
        total_initial = sum(r.initial_capital for r in self._all_results)
        total_final = sum(r.final_equity for r in self._all_results)
        total_return_pct = ((total_final - total_initial) / total_initial * 100) if total_initial else 0
        max_dd_pct = max(r.max_drawdown_pct for r in self._all_results)
        sharpe = sum(r.sharpe_ratio for r in self._all_results) / len(self._all_results) if self._all_results else 0
        win_rate = (total_wins / total_trades * 100) if total_trades else 0

        # Build combined symbol list
        symbols_str = ", ".join(r.symbol for r in self._all_results)

        # Create a synthetic combined result for the summary widget
        combined = BacktestResult(
            symbol=symbols_str,
            start_date=min(r.start_date for r in self._all_results),
            end_date=max(r.end_date for r in self._all_results),
            initial_capital=total_initial,
            final_equity=total_final,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=total_wins,
            losing_trades=total_losses,
            trades=[t for r in self._all_results for t in r.trades],
        )
        self._last_result = combined

        summary = self.query_one("#backtest-summary", BacktestSummary)
        summary.set_result(combined)

        tbl = self.query_one("#backtest-table", DataTable)
        tbl.clear()
        for trade in combined.trades:
            action_style = "bold green" if trade.action == "BUY" else "bold red"
            pnl_val = trade.pnl if trade.pnl is not None else 0
            pnl_str = f"{pnl_val:+,.2f}" if pnl_val != 0 else "—"
            tbl.add_row(
                f"[dim]{trade.symbol}[/dim] {trade.timestamp[:10]}",
                Text(trade.action, style=action_style),
                f"{trade.price:.2f}",
                str(trade.qty),
                Text(pnl_str, style="green" if pnl_val > 0 else ("red" if pnl_val < 0 else "dim")),
                trade.reason[:50] if trade.reason else "",
            )
