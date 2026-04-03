"""Visual sentiment gauge widget — renders a [-1, +1] bar."""
 
from __future__ import annotations
 
from textual.widget import Widget
from textual.reactive import reactive
from rich.text import Text
from rich.console import RenderableType
 
 
class SentimentGauge(Widget):
    """
    Renders a horizontal sentiment gauge like:
 
        AAPL  [negative ◄═══════════●══════╍╍╍╍ positive]  +0.42
    """
 
    score: reactive[float] = reactive(0.0)
    symbol: reactive[str] = reactive("")
    width_chars: int = 30
 
    def __init__(self, symbol: str = "", score: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.symbol = symbol
        self.score = score
 
    def render(self) -> RenderableType:
        return self._build_gauge(self.symbol, self.score)
 
    def update_score(self, symbol: str, score: float) -> None:
        self.symbol = symbol
        self.score = score
 
    def _build_gauge(self, symbol: str, score: float) -> Text:
        w = self.width_chars
        mid = w // 2
        clamped = max(-1.0, min(1.0, score))
        pos = int(mid + clamped * (mid - 1))
        pos = max(0, min(w - 1, pos))
 
        # Build bar segments
        bar = ["─"] * w
        bar[mid] = "┼"
        bar[pos] = "●"
        bar_str = "".join(bar)
 
        # Colour: negative portion red, positive green, neutral white
        t = Text()
        if symbol:
            t.append(f"{symbol:<6} ", style="bold white")
        t.append("[", style="dim")
        t.append("neg ", style="dim red")
 
        for i, ch in enumerate(bar_str):
            if ch == "●":
                style = "bold green" if clamped >= 0 else "bold red"
            elif i < mid:
                style = "red" if i < pos else "dim"
            elif i > mid:
                style = "green" if i <= pos else "dim"
            else:
                style = "yellow"
            t.append(ch, style=style)
 
        t.append(" pos", style="dim green")
        t.append("]", style="dim")
 
        score_style = "bold green" if score > 0.05 else ("bold red" if score < -0.05 else "yellow")
        t.append(f"  {score:+.3f}", style=score_style)
        return t
