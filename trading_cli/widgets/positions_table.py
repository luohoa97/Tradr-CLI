"""Reusable positions DataTable widget."""
 
from __future__ import annotations
 
from textual.widgets import DataTable
from rich.text import Text
 
 
class PositionsTable(DataTable):
    """DataTable pre-configured for displaying portfolio positions with P&L colouring."""
 
    COLUMNS = ("Symbol", "Qty", "Entry $", "Current $", "P&L $", "P&L %", "Value $")
 
    def on_mount(self) -> None:
        self.cursor_type = "row"
        for col in self.COLUMNS:
            self.add_column(col, key=col)
 
    def refresh_positions(self, positions: list) -> None:
        """Re-populate table from a list of Position objects."""
        self.clear()
        for p in positions:
            pl = getattr(p, "unrealized_pl", 0.0)
            plpc = getattr(p, "unrealized_plpc", 0.0) * 100
 
            pl_str = Text(f"{pl:+.2f}", style="bold green" if pl >= 0 else "bold red")
            plpc_str = Text(f"{plpc:+.2f}%", style="bold green" if plpc >= 0 else "bold red")
 
            self.add_row(
                p.symbol,
                str(int(p.qty)),
                f"{p.avg_entry_price:.2f}",
                f"{p.current_price:.2f}",
                pl_str,
                plpc_str,
                f"{p.market_value:,.2f}",
                key=p.symbol,
            )
