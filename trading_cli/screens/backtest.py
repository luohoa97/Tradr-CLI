"""Backtest results screen — displays performance metrics and trade log."""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, DataTable, Label, Static, Input, Button
from textual.containers import Vertical, Horizontal
from rich.text import Text

from trading_cli.widgets.ordered_footer import OrderedFooter

if TYPE_CHECKING:
    from trading_cli.strategy.backtest import BacktestResult


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
        Binding("r", "run_backtest", "Run", show=False),
    ]

    _last_symbol: str = ""
    _last_result: "BacktestResult | None" = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="backtest-input-row"):
                yield Label("Symbol:")
                yield Input(placeholder="e.g. AAPL", id="backtest-symbol-input")
                yield Button("🚀 Run", id="btn-backtest-run", variant="success")
            yield BacktestSummary(id="backtest-summary")
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

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-backtest-run":
            self.action_run_backtest()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        symbol = event.value.strip().upper()
        if symbol:
            self._last_symbol = symbol
            self._run_backtest(symbol)

    def action_run_backtest(self) -> None:
        # Try to get symbol from input first
        try:
            inp = self.query_one("#backtest-symbol-input", Input)
            symbol = inp.value.strip().upper()
            if symbol:
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
        app = self.app
        if not hasattr(app, "adapter") or not hasattr(app, "config"):
            app.notify("App not fully initialized", severity="error")
            return

        self.app.notify(f"Backtesting {symbol}…", timeout=2)

        from trading_cli.data.news import fetch_headlines_with_timestamps
        from trading_cli.strategy.backtest import BacktestEngine

        try:
            # Use adapter for historical data
            adapter = getattr(app, "adapter", None)
            if adapter and not adapter.is_demo_mode:
                ohlcv = adapter.fetch_ohlcv(symbol, days=365)
            else:
                # Fallback to yfinance for demo mode
                from trading_cli.data.market import fetch_ohlcv_yfinance
                ohlcv = fetch_ohlcv_yfinance(symbol, days=365)

            if ohlcv.empty:
                self.app.notify(f"No data for {symbol}", severity="warning")
                return

            # Build news_fetcher closure using adapter
            cfg = app.config
            adapter = getattr(app, "adapter", None)

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

            # Use the app's strategy adapter if available
            strategy = getattr(app, "strategy", None)
            strategy_name = strategy.info().name if strategy else "default"
            self.app.notify(f"Strategy: {strategy_name}", timeout=2)

            engine = BacktestEngine(
                config=cfg,
                finbert=finbert if finbert and finbert.is_loaded else None,
                news_fetcher=news_fetcher if (has_adapter_news or has_api_keys) else None,
                use_sentiment=bool((has_adapter_news or has_api_keys) and finbert and finbert.is_loaded),
                strategy=strategy,
            )
            result = engine.run(symbol, ohlcv, initial_capital=100_000.0)
            self._display_result(result)
        except Exception as exc:
            self.app.notify(f"Backtest failed: {exc}", severity="error")
            import logging
            logging.getLogger(__name__).error("Backtest error: %s", exc, exc_info=True)

    def _display_result(self, result: "BacktestResult") -> None:
        self._last_result = result

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
