"""Scrolling signal/trade feed widget."""
 
from __future__ import annotations
 
from datetime import datetime
 
from textual.widgets import RichLog
from rich.text import Text
 
 
class SignalLog(RichLog):
    """Auto-scrolling log widget for trading signals and notifications."""
 
    def log_signal(self, signal: dict) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        action = signal.get("action", "HOLD")
        symbol = signal.get("symbol", "???")
        price = signal.get("price", 0.0)
        reason = signal.get("reason", "")
        conf = signal.get("confidence", 0.0)
 
        action_style = {
            "BUY": "bold green",
            "SELL": "bold red",
            "HOLD": "yellow",
        }.get(action, "white")
 
        line = Text()
        line.append(f"{ts} ", style="dim")
        line.append(f"{symbol:<6}", style="bold white")
        line.append(f" {action:<4}", style=action_style)
        if price:
            line.append(f" ${price:<8.2f}", style="cyan")
        line.append(f"  ({reason}  conf={conf:.2f})", style="dim")
        self.write(line)
 
    def log_order(self, order_result) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        action = order_result.action.upper()
        style = "bold green" if action == "BUY" else "bold red"
        fp = order_result.filled_price
        price_str = f"@ ${fp:.2f}" if fp else ""
        msg = Text()
        msg.append(f"{ts} ", style="dim")
        msg.append("ORDER ", style="bold")
        msg.append(f"{action} {order_result.qty} {order_result.symbol} {price_str}", style=style)
        msg.append(f"  [{order_result.status}]", style="dim")
        self.write(msg)
 
    def log_error(self, message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        line = Text()
        line.append(f"{ts} ", style="dim")
        line.append("ERROR ", style="bold red")
        line.append(message, style="red")
        self.write(line)
 
    def log_info(self, message: str) -> None:
        ts = datetime.utcnow().strftime("%H:%M:%S")
        line = Text()
        line.append(f"{ts} ", style="dim")
        line.append(message, style="dim white")
        self.write(line)
