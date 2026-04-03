"""Sentiment analysis screen — interactive FinBERT analysis per symbol."""
 
from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, Input, Label, DataTable, Static
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from rich.text import Text

from trading_cli.widgets.sentiment_gauge import SentimentGauge
from trading_cli.sentiment.aggregator import get_sentiment_summary, score_to_bar
from trading_cli.widgets.ordered_footer import OrderedFooter
 
 
class SentimentScreen(Screen):
    """Screen ID 5 — on-demand FinBERT sentiment analysis."""
 
    BINDINGS = [
        Binding("r", "refresh_symbol", "Refresh", show=False),
    ]
 
    _current_symbol: str = ""
 
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            with Horizontal(id="sent-input-row"):
                yield Label("Symbol: ", id="sent-label")
                yield Input(placeholder="e.g. TSLA", id="sent-input")
            yield Label(
                "[dim]Type a symbol and press Enter · [r] refresh · [ESC] back[/dim]",
                id="sent-help",
            )
            yield SentimentGauge(id="sent-gauge")
            yield Label("", id="sent-summary")
            yield DataTable(id="sent-table", cursor_type="row")
        yield OrderedFooter()
 
    def on_mount(self) -> None:
        tbl = self.query_one("#sent-table", DataTable)
        tbl.add_column("Headline", key="headline")
        tbl.add_column("Label", key="label")
        tbl.add_column("Score", key="score")
        self.query_one("#sent-input", Input).focus()
 
    def on_input_submitted(self, event: Input.Submitted) -> None:
        symbol = event.value.strip().upper()
        if symbol:
            self._current_symbol = symbol
            self._run_analysis(symbol)
 
    def action_refresh_symbol(self) -> None:
        if self._current_symbol:
            self._run_analysis(self._current_symbol)
 
    def _run_analysis(self, symbol: str) -> None:
        app = self.app
        config = getattr(app, "config", {})
        db_conn = getattr(app, "db_conn", None)
        analyzer = getattr(app, "finbert", None)
 
        self.app.notify(f"Fetching news for {symbol}…", timeout=2)
 
        from trading_cli.data.news import fetch_headlines
        headlines = fetch_headlines(symbol, max_articles=20)
 
        if not headlines:
            self.app.notify(f"No headlines found for {symbol}", severity="warning")
            return
 
        results = []
        if analyzer and analyzer.is_loaded:
            if db_conn:
                results = analyzer.analyze_with_cache(headlines, db_conn)
            else:
                results = analyzer.analyze_batch(headlines)
        else:
            # Fallback: neutral
            results = [{"label": "neutral", "score": 0.5}] * len(headlines)
            self.app.notify("FinBERT not loaded — showing neutral", severity="warning")
 
        self._display_results(symbol, headlines, results)
 
    def _display_results(self, symbol: str, headlines: list[str], results: list[dict]) -> None:
        summary = get_sentiment_summary(results)
 
        gauge = self.query_one("#sent-gauge", SentimentGauge)
        gauge.update_score(symbol, summary["score"])
 
        lbl = self.query_one("#sent-summary", Label)
        score = summary["score"]
        bar = score_to_bar(score, width=24)
        dominant = summary["dominant"].upper()
        dom_style = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "yellow"}.get(dominant, "white")
        lbl.update(
            f"[bold]{symbol}[/bold]  "
            f"[{dom_style}]{bar}[/{dom_style}]  "
            f"Score: [bold]{score:+.3f}[/bold]  "
            f"+{summary['positive_count']} −{summary['negative_count']} ={summary['neutral_count']}"
        )
 
        tbl = self.query_one("#sent-table", DataTable)
        tbl.clear()
        for headline, result in zip(headlines, results):
            label = result.get("label", "neutral")
            score_val = result.get("score", 0.5)
            label_style = {"positive": "green", "negative": "red", "neutral": "yellow"}.get(label, "white")
            tbl.add_row(
                headline[:80],
                Text(label.upper(), style=f"bold {label_style}"),
                Text(f"{score_val:.3f}", style=label_style),
            )
