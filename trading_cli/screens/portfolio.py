"""Portfolio screen — detailed positions with close-position action."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, DataTable, Label, Static, Button
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from rich.text import Text

from trading_cli.widgets.positions_table import PositionsTable
from trading_cli.widgets.ordered_footer import OrderedFooter


class PortfolioSummary(Static):
    equity: reactive[float] = reactive(0.0)
    cash: reactive[float] = reactive(0.0)
    total_pl: reactive[float] = reactive(0.0)

    def render(self) -> Text:
        t = Text()
        t.append("Portfolio Value: ", style="bold")
        t.append(f"${self.equity:,.2f}  ", style="bold cyan")
        t.append("Cash: ", style="bold")
        t.append(f"${self.cash:,.2f}  ", style="cyan")
        pl_style = "bold green" if self.total_pl >= 0 else "bold red"
        t.append("Total P&L: ", style="bold")
        t.append(f"${self.total_pl:+,.2f}", style=pl_style)
        return t


class PortfolioScreen(Screen):
    """Screen ID 3 — full position details from Alpaca."""

    BINDINGS = [
        Binding("x", "close_position", "Close position", show=False),
        Binding("r", "refresh_data", "Refresh", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield PortfolioSummary(id="portfolio-summary")
            with Horizontal(id="portfolio-actions"):
                yield Button("🔄 Refresh", id="btn-refresh", variant="primary")
                yield Button("❌ Close Selected", id="btn-close", variant="error")
            yield PositionsTable(id="portfolio-table")
        yield OrderedFooter()

    def on_mount(self) -> None:
        self.action_refresh_data()

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-refresh":
            self.action_refresh_data()
        elif event.button.id == "btn-close":
            self.action_close_position()

    def action_refresh_data(self) -> None:
        app = self.app
        if not hasattr(app, "client"):
            return
        try:
            acct = app.client.get_account()
            summary = self.query_one("#portfolio-summary", PortfolioSummary)
            summary.equity = acct.equity
            summary.cash = acct.cash

            positions = app.client.get_positions()
            total_pl = sum(p.unrealized_pl for p in positions)
            summary.total_pl = total_pl

            tbl = self.query_one("#portfolio-table", PositionsTable)
            tbl.refresh_positions(positions)
        except Exception as exc:
            self.app.notify(f"Refresh failed: {exc}", severity="error")

    def action_close_position(self) -> None:
        tbl = self.query_one("#portfolio-table", PositionsTable)
        if len(tbl.rows) == 0:
            self.app.notify("No positions to close", severity="warning")
            return
        if tbl.cursor_row is None:
            self.app.notify("Select a position first", severity="warning")
            return
        row = tbl.get_row_at(tbl.cursor_row)
        if not row:
            return
        symbol = str(row[0])
        self.app.push_screen(
            ConfirmCloseScreen(symbol),
            callback=self._on_close_confirmed,
        )

    def _on_close_confirmed(self, confirmed: bool) -> None:
        if not confirmed:
            return
        if not hasattr(self, "_pending_close"):
            return
        symbol = self._pending_close
        try:
            result = self.app.client.close_position(symbol)
            if result:
                from trading_cli.data.db import save_trade
                save_trade(
                    self.app.db_conn, symbol, "SELL",
                    result.filled_price or 0.0,
                    result.qty,
                    order_id=result.order_id,
                    reason="Manual close from Portfolio screen",
                )
                self.app.notify(f"Closed {symbol}: {result.status}")
        except Exception as exc:
            self.app.notify(f"Close failed: {exc}", severity="error")
        self.action_refresh_data()


class ConfirmCloseScreen(Screen):
    """Modal confirmation dialog for closing a position."""

    def __init__(self, symbol: str) -> None:
        super().__init__()
        self._symbol = symbol

    def compose(self) -> ComposeResult:
        from textual.containers import Grid

        with Grid(id="confirm-grid"):
            yield Label(
                f"[bold red]Close position in {self._symbol}?[/bold red]\n"
                "This will submit a market SELL order.",
                id="confirm-msg",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes, close", id="btn-yes", variant="error")
                yield Button("Cancel", id="btn-no", variant="default")

    def on_button_pressed(self, event) -> None:
        self.dismiss(event.button.id == "btn-yes")
