"""Dashboard screen — main view with positions, signals and account summary."""
 
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, Static, Label, Rule
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.reactive import reactive
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich import box

from trading_cli.widgets.positions_table import PositionsTable
from trading_cli.widgets.signal_log import SignalLog
from trading_cli.widgets.ordered_footer import OrderedFooter
 
 
class AccountBar(Static):
    cash: reactive[float] = reactive(0.0)
    equity: reactive[float] = reactive(0.0)
    demo: reactive[bool] = reactive(False)
    market_open: reactive[bool] = reactive(False)
 
    def render(self) -> Text:
        t = Text()
        mode = "[DEMO] " if self.demo else ""
        t.append(mode, style="bold yellow")
        t.append(f"Cash: ${self.cash:,.2f}  ", style="bold cyan")
        t.append(f"Equity: ${self.equity:,.2f}  ", style="bold white")
        status_style = "bold green" if self.market_open else "bold red"
        status_text = "● OPEN" if self.market_open else "● CLOSED"
        t.append(status_text, style=status_style)
        return t
 
 
class DashboardScreen(Screen):
    """Screen ID 1 — main dashboard."""
 
    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=False),
    ]
 
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield AccountBar(id="account-bar")
            yield Rule()
            with Horizontal(id="main-split"):
                with Vertical(id="left-pane"):
                    yield Label("[bold]RECENT SIGNALS[/bold]", id="signals-label")
                    yield SignalLog(id="signal-log", max_lines=50, markup=True)
                with Vertical(id="right-pane"):
                    yield Label("[bold]POSITIONS[/bold]", id="positions-label")
                    yield PositionsTable(id="positions-table")
        yield OrderedFooter()
 
    def on_mount(self) -> None:
        self._refresh_from_app()
 
    def action_refresh(self) -> None:
        self._refresh_from_app()
 
    def _refresh_from_app(self) -> None:
        app = self.app
        if not hasattr(app, "client"):
            return
        try:
            acct = app.client.get_account()
            bar = self.query_one("#account-bar", AccountBar)
            bar.cash = acct.cash
            bar.equity = acct.equity
            bar.demo = app.demo_mode
            bar.market_open = app.market_open
 
            positions = app.client.get_positions()
            self.query_one("#positions-table", PositionsTable).refresh_positions(positions)
        except Exception:
            pass
 
    # Called by app worker when new data arrives
    def refresh_positions(self, positions: list) -> None:
        try:
            self.query_one("#positions-table", PositionsTable).refresh_positions(positions)
        except Exception:
            pass
 
    def refresh_account(self, acct) -> None:
        try:
            bar = self.query_one("#account-bar", AccountBar)
            bar.cash = acct.cash
            bar.equity = acct.equity
            bar.demo = self.app.demo_mode
            bar.market_open = self.app.market_open
        except Exception:
            pass
 
    def log_signal(self, signal: dict) -> None:
        try:
            self.query_one("#signal-log", SignalLog).log_signal(signal)
        except Exception:
            pass
