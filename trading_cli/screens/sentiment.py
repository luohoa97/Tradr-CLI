"""Sentiment analysis screen — interactive FinBERT analysis per symbol."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Header, Input, Label, DataTable, Static, ProgressBar, LoadingIndicator
from textual.containers import Vertical, Horizontal
from textual.reactive import reactive
from textual import work
from rich.text import Text

from trading_cli.sentiment.aggregator import get_sentiment_summary, score_to_bar
from trading_cli.widgets.ordered_footer import OrderedFooter


class SentimentScoreDisplay(Static):
    """Displays sentiment score with a visual bar."""

    score: reactive[float] = reactive(0.0)
    symbol: reactive[str] = reactive("")
    positive_count: reactive[int] = reactive(0)
    negative_count: reactive[int] = reactive(0)
    neutral_count: reactive[int] = reactive(0)
    dominant: reactive[str] = reactive("NEUTRAL")

    def render(self) -> str:
        if not self.symbol:
            return "[dim]No sentiment data[/dim]"
        bar = score_to_bar(self.score, width=24)
        dom_style = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "yellow"}.get(self.dominant, "white")
        return (
            f"[bold]{self.symbol}[/bold]  "
            f"[{dom_style}]{bar}[/{dom_style}]  "
            f"Score: [bold]{self.score:+.3f}[/bold]  "
            f"+{self.positive_count} −{self.negative_count} ={self.neutral_count}"
        )


class SentimentProgressBar(ProgressBar):
    """Progress bar that maps 0-100 to -1 to +1 sentiment score."""

    _current_score: float = 0.0

    @property
    def score(self) -> float:
        return self._current_score

    @score.setter
    def score(self, value: float) -> None:
        self._current_score = value
        # Map -1..+1 to 0..100
        self.total = 100
        self.progress = int((value + 1.0) / 2.0 * 100)


class SentimentScreen(Screen):
    """Screen ID 5 — on-demand FinBERT sentiment analysis."""

    BINDINGS = [
        Binding("r", "refresh_symbol", "Refresh", show=False),
    ]

    _current_symbol: str = ""
    _loading: reactive[bool] = reactive(False)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical():
            yield Input(placeholder="Enter a symbol and press Enter…", id="sent-input")
            yield Label("", id="sent-help")
            with Horizontal(id="sent-loading-row", classes="sent-loading"):
                yield LoadingIndicator(id="sent-loading-spinner")
                yield Label("", id="sent-loading-status")
            yield Label("[red]← Negative[/red]", id="sent-neg-label")
            yield SentimentProgressBar(id="sent-progress")
            yield Label("[green]Positive →[/green]", id="sent-pos-label")
            yield SentimentScoreDisplay(id="sent-summary")
            yield DataTable(id="sent-table", cursor_type="row")
        yield OrderedFooter()

    def on_mount(self) -> None:
        tbl = self.query_one("#sent-table", DataTable)
        tbl.add_column("Headline", key="headline")
        tbl.add_column("Label", key="label")
        tbl.add_column("Score", key="score")
        self.query_one("#sent-input", Input).focus()

        # Hide loading row initially
        self._hide_loading()

    # ------------------------------------------------------------------
    # Loading visibility helpers
    # ------------------------------------------------------------------

    def _show_loading(self) -> None:
        """Show the loading indicator + status label."""
        try:
            row = self.query_one("#sent-loading-row")
            row.display = True
            spinner = self.query_one("#sent-loading-spinner")
            spinner.display = True
        except Exception:
            pass

    def _hide_loading(self) -> None:
        """Hide the loading indicator + status label."""
        try:
            row = self.query_one("#sent-loading-row")
            row.display = False
            spinner = self.query_one("#sent-loading-spinner")
            spinner.display = False
        except Exception:
            pass

    def _set_loading_status(self, text: str) -> None:
        """Update the status label text (called from any thread)."""
        def _update():
            try:
                self.query_one("#sent-loading-status", Label).update(text)
            except Exception:
                pass
        self.app.call_from_thread(_update)

    def _watch__loading(self, loading: bool) -> None:
        """Toggle loading UI when _loading changes."""
        if loading:
            self._show_loading()
        else:
            self._hide_loading()

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        symbol = event.value.strip().upper()
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
        self._loading = True
        self._do_analysis(symbol)

    @work(thread=True, exclusive=True, description="Analyzing sentiment")
    def _do_analysis(self, symbol: str) -> None:
        analyzer = getattr(self.app, "finbert", None)
        db_conn = getattr(self.app, "db_conn", None)

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
                self.app.call_from_thread(
                    self.app.notify, "Showing neutral sentiment", severity="warning",
                )
                self._loading = False
                return

        self._set_loading_status(f"Fetching headlines for {symbol}…")

        from trading_cli.data.news import fetch_headlines
        headlines = fetch_headlines(symbol, max_articles=20)

        if not headlines:
            self.app.call_from_thread(
                self.app.notify, f"No headlines found for {symbol}", severity="warning",
            )
            self._set_loading_status("")
            self._loading = False
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

        self._set_loading_status("")
        self._loading = False

        # Dispatch UI update back to main thread
        self.app.call_from_thread(self._display_results, symbol, headlines, results)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _display_results(self, symbol: str, headlines: list[str], results: list[dict]) -> None:
        summary = get_sentiment_summary(results)

        # Update progress bar
        bar = self.query_one("#sent-progress", SentimentProgressBar)
        bar.score = summary["score"]

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
