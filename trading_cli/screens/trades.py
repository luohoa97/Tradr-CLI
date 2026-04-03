"""Trade history screen — scrollable log with filter and CSV export."""
 
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, DataTable, Input, Label, Static
from textual.containers import Vertical, Horizontal
from rich.text import Text

from trading_cli.widgets.ordered_footer import OrderedFooter
 
 
class TradesScreen(Screen):
    """Screen ID 4 — all executed trades with filter and export."""
 
    BINDINGS = [
        Binding("e", "export_csv", "Export CSV", show=False),
        Binding("r", "refresh_data", "Refresh", show=False),
        Binding("f", "focus_filter", "Filter", show=False),
    ]
 
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="trades-filter-row"):
                yield Label("Filter: ", id="filter-label")
                yield Input(placeholder="symbol or action…", id="trades-filter")
            yield Label(
                "[dim][e] export CSV · [r] refresh · [f] filter[/dim]",
                id="trades-help",
            )
            yield DataTable(id="trades-table", cursor_type="row")
        yield OrderedFooter()
 
    def on_mount(self) -> None:
        tbl = self.query_one("#trades-table", DataTable)
        tbl.add_column("Time", key="time")
        tbl.add_column("Symbol", key="symbol")
        tbl.add_column("Action", key="action")
        tbl.add_column("Price $", key="price")
        tbl.add_column("Qty", key="qty")
        tbl.add_column("P&L $", key="pnl")
        tbl.add_column("Order ID", key="order_id")
        tbl.add_column("Reason", key="reason")
        self.action_refresh_data()
 
    def action_refresh_data(self, filter_text: str = "") -> None:
        from trading_cli.data.db import get_trade_history
 
        app = self.app
        if not hasattr(app, "db_conn"):
            return
        trades = get_trade_history(app.db_conn, limit=200)
        tbl = self.query_one("#trades-table", DataTable)
        tbl.clear()
        ft = filter_text.upper()
        for trade in trades:
            if ft and ft not in trade["symbol"] and ft not in trade["action"]:
                continue
            ts = trade["timestamp"][:19].replace("T", " ")
            action = trade["action"]
            action_style = {"BUY": "bold green", "SELL": "bold red"}.get(action, "yellow")
            pnl = trade.get("pnl") or 0.0
            pnl_str = Text(f"{pnl:+.2f}" if pnl != 0 else "—",
                           style="green" if pnl > 0 else ("red" if pnl < 0 else "dim"))
            tbl.add_row(
                ts,
                trade["symbol"],
                Text(action, style=action_style),
                f"{trade['price']:.2f}",
                str(trade["quantity"]),
                pnl_str,
                trade.get("order_id") or "—",
                (trade.get("reason") or "")[:40],
            )
 
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_refresh_data(event.value.strip())
 
    def action_export_csv(self) -> None:
        from trading_cli.data.db import get_trade_history
 
        app = self.app
        if not hasattr(app, "db_conn"):
            return
        trades = get_trade_history(app.db_conn, limit=10000)
        export_dir = Path.home() / "Downloads"
        export_dir.mkdir(exist_ok=True)
        fname = export_dir / f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(fname, "w", newline="") as f:
            if not trades:
                f.write("No trades\n")
            else:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)
        app.notify(f"Exported to {fname}")
 
    def action_focus_filter(self) -> None:
        self.query_one("#trades-filter", Input).focus()
