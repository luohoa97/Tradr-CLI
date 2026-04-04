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
 
 
class AutoTradeStatus(Static):
    """Shows auto-trade status and last cycle time."""
    enabled: reactive[bool] = reactive(False)
    last_cycle: reactive[str] = reactive("--")
    last_error: reactive[str] = reactive("")

    def render(self) -> Text:
        status = "[AUTO] ON" if self.enabled else "[AUTO] OFF"
        style = "bold green" if self.enabled else "bold yellow"
        t = Text(status, style=style)
        t.append(f"  Last: {self.last_cycle}", style="dim")
        if self.last_error:
            t.append(f"  Error: {self.last_error}", style="bold red")
        return t


class DashboardScreen(Screen):
    """Screen ID 1 — main dashboard."""

    BINDINGS = [
        Binding("r", "refresh", "Refresh", show=False),
        Binding("t", "toggle_autotrade", "Toggle Auto", show=True),
    ]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield AccountBar(id="account-bar")
            yield Rule()
            yield AutoTradeStatus(id="autotrade-status")
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
        if not hasattr(app, "adapter"):
            return
        try:
            acct = app.adapter.get_account()
            bar = self.query_one("#account-bar", AccountBar)
            bar.cash = acct.cash
            bar.equity = acct.equity
            bar.demo = app.demo_mode
            bar.market_open = app.market_open

            positions = app.adapter.get_positions()
            self.query_one("#positions-table", PositionsTable).refresh_positions(positions)
            
            # Initialize auto-trade status
            auto_enabled = app.config.get("auto_trading", False)
            self.update_autotrade_status(auto_enabled)
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

    def update_autotrade_status(self, enabled: bool, last_cycle: str = "", error: str = "") -> None:
        """Update the auto-trade status indicator."""
        try:
            status = self.query_one("#autotrade-status", AutoTradeStatus)
            status.enabled = enabled
            if last_cycle:
                status.last_cycle = last_cycle
            if error:
                status.last_error = error
        except Exception:
            pass

    def action_toggle_autotrade(self) -> None:
        """Toggle auto-trading on/off from dashboard."""
        app = self.app
        if not hasattr(app, "config"):
            return
        
        current = app.config.get("auto_trading", False)
        new_value = not current
        app.config["auto_trading"] = new_value
        
        # Persist to disk
        from trading_cli.config import save_config
        save_config(app.config)
        
        # Update status indicator
        self.update_autotrade_status(new_value)
        
        # Notify user
        status = "enabled" if new_value else "disabled"
        app.notify(f"Auto-trading {status}", severity="information" if new_value else "warning")
