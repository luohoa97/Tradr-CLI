"""Backtest results screen — displays performance metrics and trade log."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, DataTable, Label, Static
from textual.containers import Vertical, Horizontal
from rich.text import Text

from trading_cli.widgets.ordered_footer import OrderedFooter


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
    _last_result: BacktestResult | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Label("[dim]Enter a symbol and press Enter to backtest · [r] re-run[/dim]", id="backtest-help")
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

    def on_input_submitted(self, event) -> None:
        symbol = event.value.strip().upper()
        if symbol:
            self._last_symbol = symbol
            self._run_backtest(symbol)

    def action_run_backtest(self) -> None:
        if self._last_symbol:
            self._run_backtest(self._last_symbol)

    def _run_backtest(self, symbol: str) -> None:
        app = self.app
        if not hasattr(app, "client") or not hasattr(app, "config"):
            app.notify("App not fully initialized", severity="error")
            return

        self.app.notify(f"Backtesting {symbol}…", timeout=2)

        from trading_cli.data.market import fetch_ohlcv_alpaca, fetch_ohlcv_yfinance
        from trading_cli.data.news import fetch_headlines_with_timestamps
        from trading_cli.strategy.backtest import BacktestEngine

        try:
            # Try Alpaca historical data first, fall back to yfinance
            client = getattr(app, "client", None)
            use_alpaca_data = client and not client.demo_mode

            if use_alpaca_data:
                ohlcv = fetch_ohlcv_alpaca(client, symbol, days=365)
            else:
                ohlcv = fetch_ohlcv_yfinance(symbol, days=365)

            if ohlcv.empty:
                self.app.notify(f"No data for {symbol}", severity="warning")
                return

            # Build news_fetcher closure with API keys from config
            cfg = app.config
            alpaca_key = cfg.get("alpaca_api_key", "")
            alpaca_secret = cfg.get("alpaca_api_secret", "")

            def news_fetcher(sym: str, days_ago: int = 0) -> list[tuple[str, float]]:
                return fetch_headlines_with_timestamps(
                    sym,
                    days_ago=days_ago,
                    alpaca_key=alpaca_key,
                    alpaca_secret=alpaca_secret,
                    max_articles=30,
                )

            finbert = getattr(app, "finbert", None)
            has_api_keys = bool(alpaca_key and alpaca_secret)

            engine = BacktestEngine(
                config=cfg,
                finbert=finbert if finbert and finbert.is_loaded else None,
                news_fetcher=news_fetcher if has_api_keys else None,
                use_sentiment=bool(has_api_keys and finbert and finbert.is_loaded),
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
            pnl_str = Text(f"{trade.pnl:+,.2f}" if trade.pnl != 0 else "—",
                           style="green" if trade.pnl > 0 else ("red" if trade.pnl < 0 else "dim"))
            tbl.add_row(
                trade.timestamp[:10],
                Text(trade.action, style=action_style),
                f"{trade.price:.2f}",
                str(trade.qty),
                pnl_str,
                trade.reason[:50],
            )
