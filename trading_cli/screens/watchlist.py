"""Watchlist screen — add/remove symbols, live prices and signals."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, DataTable, Input, Label, Static
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from rich.text import Text

from trading_cli.widgets.ordered_footer import OrderedFooter


class WatchlistScreen(Screen):
    """Screen ID 2 — symbol watchlist with live prices and signals."""

    BINDINGS = [
        Binding("a", "focus_add", "Add", show=True),
        Binding("d", "delete_selected", "Delete", show=True),
        Binding("r", "refresh", "Refresh", show=True),
    ]

    _prices: dict[str, float] = {}
    _sentiments: dict[str, float] = {}
    _signals: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            # Primary search input (like sentiment screen)
            app = self.app
            if hasattr(app, 'asset_search') and app.asset_search.is_ready:
                from trading_cli.widgets.asset_autocomplete import create_asset_autocomplete
                input_widget, autocomplete_widget = create_asset_autocomplete(
                    app.asset_search,
                    placeholder="Search by symbol or company name… (Tab to complete)",
                    id="wl-input",
                )
                yield input_widget
                yield autocomplete_widget
            else:
                yield Input(placeholder="Search by symbol or company name…", id="wl-input")
            
            yield DataTable(id="wl-table", cursor_type="row")
        yield OrderedFooter()
 
    def on_mount(self) -> None:
        tbl = self.query_one("#wl-table", DataTable)
        tbl.add_column("Symbol", key="symbol")
        tbl.add_column("Price $", key="price")
        tbl.add_column("Sentiment", key="sentiment")
        tbl.add_column("Signal", key="signal")
        self._populate_table()
 
    def _populate_table(self) -> None:
        tbl = self.query_one("#wl-table", DataTable)
        tbl.clear()
        app = self.app
        watchlist = getattr(app, "watchlist", [])
        for sym in watchlist:
            price = self._prices.get(sym, 0.0)
            sent = self._sentiments.get(sym, 0.0)
            sig = self._signals.get(sym, "HOLD")
 
            price_str = f"${price:.2f}" if price else "—"
            sent_str = Text(f"{sent:+.3f}", style="green" if sent > 0 else ("red" if sent < 0 else "dim"))
            sig_style = {"BUY": "bold green", "SELL": "bold red", "HOLD": "yellow"}.get(sig, "white")
            sig_str = Text(sig, style=sig_style)
 
            tbl.add_row(sym, price_str, sent_str, sig_str, key=sym)
 
    def update_data(
        self,
        prices: dict[str, float],
        sentiments: dict[str, float],
        signals: dict[str, str],
    ) -> None:
        self._prices = prices
        self._sentiments = sentiments
        self._signals = signals
        self._populate_table()
 
    def action_focus_add(self) -> None:
        self.query_one("#wl-input", Input).focus()
 
    def action_delete_selected(self) -> None:
        tbl = self.query_one("#wl-table", DataTable)
        if tbl.cursor_row is not None:
            row_key = tbl.get_row_at(tbl.cursor_row)
            if row_key:
                symbol = str(row_key[0])
                app = self.app
                if hasattr(app, "remove_from_watchlist"):
                    app.remove_from_watchlist(symbol)
                self._populate_table()
 
    def action_refresh(self) -> None:
        self._populate_table()
 
    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        
        # Extract symbol from autocomplete format "SYMBOL — Company Name"
        # If it contains " — ", take the first part as the symbol
        if " — " in value:
            symbol = value.split(" — ")[0].strip().upper()
        else:
            symbol = value.upper()
        
        if symbol:
            app = self.app
            if hasattr(app, "add_to_watchlist"):
                app.add_to_watchlist(symbol)
            event.input.value = ""
            self._populate_table()
