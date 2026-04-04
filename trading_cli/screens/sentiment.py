"""Sentiment analysis screen — interactive FinBERT analysis per symbol."""

from __future__ import annotations

import threading

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, Input, Label, DataTable, Static
from textual.containers import Vertical
from textual.reactive import reactive
from textual import work
from rich.text import Text

from trading_cli.sentiment.aggregator import get_sentiment_summary
from trading_cli.widgets.ordered_footer import OrderedFooter


class SentimentScoreDisplay(Static):
    """Displays sentiment score with a simple label."""

    score: reactive[float] = reactive(0.0)
    symbol: reactive[str] = reactive("")
    positive_count: reactive[int] = reactive(0)
    negative_count: reactive[int] = reactive(0)
    neutral_count: reactive[int] = reactive(0)
    dominant: reactive[str] = reactive("NEUTRAL")

    def render(self) -> str:
        if not self.symbol:
            return ""
        dom_style = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "yellow"}.get(self.dominant, "white")
        return (
            f"[bold]{self.symbol}[/bold] — "
            f"[{dom_style}]{self.dominant}[/{dom_style}] "
            f"(score: [bold]{self.score:+.3f}[/bold], "
            f"+{self.positive_count} / −{self.negative_count} / ={self.neutral_count})"
        )


class SentimentScreen(Screen):
    """Screen ID 5 — on-demand FinBERT sentiment analysis."""

    BINDINGS = [
        Binding("r", "refresh_symbol", "Refresh", show=False),
    ]

    _current_symbol: str = ""
    _analysis_task: str = ""  # Track the latest symbol being analyzed

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            # Create asset autocomplete input
            app = self.app
            if hasattr(app, 'asset_search') and app.asset_search.is_ready:
                from trading_cli.widgets.asset_autocomplete import create_asset_autocomplete
                input_widget, autocomplete_widget = create_asset_autocomplete(
                    app.asset_search,
                    placeholder="Search by symbol or company name… (Tab to complete)",
                    id="sent-input",
                )
                yield input_widget
                yield autocomplete_widget
            else:
                yield Input(placeholder="Search by symbol or company name…", id="sent-input")
            
            yield Label("", id="sent-loading-status")
            yield SentimentScoreDisplay(id="sent-summary")
            yield DataTable(id="sent-table", cursor_type="row")
        yield OrderedFooter()

    def on_mount(self) -> None:
        tbl = self.query_one("#sent-table", DataTable)
        tbl.add_column("Headline", key="headline")
        tbl.add_column("Label", key="label")
        tbl.add_column("Score", key="score")
        self.query_one("#sent-input", Input).focus()
        self._clear_loading_status()

    # ------------------------------------------------------------------
    # Loading status helpers
    # ------------------------------------------------------------------

    def _set_loading_status(self, text: str) -> None:
        """Update the status label text."""
        def _update():
            try:
                self.query_one("#sent-loading-status", Label).update(f"[dim]{text}[/dim]")
            except Exception:
                pass
        
        # Only use call_from_thread if we're in a background thread
        if threading.get_ident() != self.app._thread_id:
            self.app.call_from_thread(_update)
        else:
            _update()

    def _clear_loading_status(self) -> None:
        """Clear the status label."""
        def _update():
            try:
                self.query_one("#sent-loading-status", Label).update("")
            except Exception:
                pass
        
        # Only use call_from_thread if we're in a background thread
        if threading.get_ident() != self.app._thread_id:
            self.app.call_from_thread(_update)
        else:
            _update()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        value = event.value.strip()
        if not value:
            return
        
        # Extract symbol from autocomplete format "SYMBOL — Company Name"
        if " — " in value:
            symbol = value.split(" — ")[0].strip().upper()
        else:
            symbol = value.upper()
        
        if symbol:
            self._current_symbol = symbol
            self._run_analysis(symbol)

    def action_refresh_symbol(self) -> None:
        if self._current_symbol:
            self._run_analysis(self._current_symbol)

    # ------------------------------------------------------------------
    # Analysis (background thread)
    # ------------------------------------------------------------------

    def _run_analysis(self, symbol: str) -> None:
        """Kick off background analysis."""
        # Update the task tracker to the latest symbol (cancels previous tasks)
        self._analysis_task = symbol

        # Clear the table to show we're working on a new request
        tbl = self.query_one("#sent-table", DataTable)
        tbl.clear()

        # Reset summary display
        lbl = self.query_one("#sent-summary", SentimentScoreDisplay)
        lbl.symbol = ""
        lbl.score = 0.0

        self._do_analysis(symbol)

    @work(thread=True, exclusive=False, description="Analyzing sentiment")
    def _do_analysis(self, symbol: str) -> None:
        """Analyze sentiment for a symbol (non-blocking, allows cancellation)."""
        analyzer = getattr(self.app, "finbert", None)
        db_conn = getattr(self.app, "db_conn", None)

        # Check if this task has been superseded by a newer request
        def is_cancelled() -> bool:
            return self._analysis_task != symbol

        # Attempt to reload FinBERT if not loaded
        if analyzer and not analyzer.is_loaded:
            self._set_loading_status("Loading FinBERT model…")
            success = analyzer.reload(
                progress_callback=lambda msg: self._set_loading_status(msg),
            )
            if not success:
                error_msg = analyzer.load_error or "Unknown error"
                self.app.call_from_thread(
                    self.app.notify,
                    f"FinBERT failed to load: {error_msg}",
                    severity="error",
                )
                self._set_loading_status(f"Failed: {error_msg}")
                return

        # Check cancellation after model loading
        if is_cancelled():
            return

        self._set_loading_status(f"Fetching headlines for {symbol}…")

        from trading_cli.data.news import fetch_headlines
        headlines = fetch_headlines(symbol, max_articles=20)

        # Check cancellation after network call
        if is_cancelled():
            return

        if not headlines:
            self.app.call_from_thread(
                self.app.notify, f"No headlines found for {symbol}", severity="warning",
            )
            self._clear_loading_status()
            return

        self._set_loading_status("Running sentiment analysis…")

        results = []
        if analyzer and analyzer.is_loaded:
            if db_conn:
                results = analyzer.analyze_with_cache(headlines, db_conn)
            else:
                results = analyzer.analyze_batch(headlines)
        else:
            results = [{"label": "neutral", "score": 0.5}] * len(headlines)

        # Check cancellation after heavy computation
        if is_cancelled():
            return

        self._clear_loading_status()

        # Only update UI if this is still the latest task
        if not is_cancelled():
            # Dispatch UI update back to main thread
            self.app.call_from_thread(self._display_results, symbol, headlines, results)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_results(self, symbol: str, headlines: list[str], results: list[dict]) -> None:
        summary = get_sentiment_summary(results)

        # Update summary
        lbl = self.query_one("#sent-summary", SentimentScoreDisplay)
        lbl.symbol = symbol
        lbl.score = summary["score"]
        lbl.positive_count = summary["positive_count"]
        lbl.negative_count = summary["negative_count"]
        lbl.neutral_count = summary["neutral_count"]
        lbl.dominant = summary["dominant"].upper()

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
