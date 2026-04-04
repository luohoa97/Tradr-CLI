"""Asset autocomplete widget with symbol and company name search."""

from __future__ import annotations

import logging
import time
import threading
from typing import TYPE_CHECKING

from textual.widgets import Input
from textual_autocomplete import AutoComplete, DropdownItem
from textual_autocomplete._autocomplete import TargetState

if TYPE_CHECKING:
    from trading_cli.data.asset_search import AssetSearchEngine

logger = logging.getLogger(__name__)


def create_asset_autocomplete(
    search_engine: AssetSearchEngine,
    *,
    placeholder: str = "Search symbol or company name...",
    id: str | None = None,  # noqa: A002
) -> tuple[Input, AutoComplete]:
    """Create an Input widget with autocomplete for asset search.

    Args:
        search_engine: The asset search engine instance.
        placeholder: Placeholder text for the input.
        id: Widget ID.

    Returns:
        Tuple of (Input widget, AutoComplete widget).
        Yield both in your compose() method.

    Example:
        input_widget, autocomplete_widget = create_asset_autocomplete(engine)
        yield input_widget
        yield autocomplete_widget
    """
    input_widget = Input(placeholder=placeholder, id=id)

    # Cache results to avoid repeated searches
    _cache: dict[str, list[DropdownItem]] = {}
    _cache_lock = threading.Lock()
    _last_query = ""
    _last_time = 0.0

    def get_suggestions(state: TargetState) -> list[DropdownItem]:
        nonlocal _last_query, _last_time

        query = state.text.strip()
        if not query:
            return []

        # Debounce: skip if same query within 300ms
        now = time.monotonic()
        if query == _last_query and (now - _last_time) < 0.3:
            return []
        _last_query = query
        _last_time = now

        # Check cache first
        with _cache_lock:
            if query in _cache:
                return _cache[query]

        try:
            results = search_engine.search(query, max_results=10)
            if not results:
                return []

            suggestions = []
            for result in results:
                symbol = result["symbol"]
                name = result.get("name", "")
                # Display format: "AAPL — Apple Inc."
                display_text = f"{symbol} — {name}" if name else symbol
                suggestions.append(DropdownItem(main=display_text))

            # Cache the results
            with _cache_lock:
                _cache[query] = suggestions
                # Limit cache size
                if len(_cache) > 1000:
                    # Remove oldest 500 entries
                    keys_to_remove = list(_cache.keys())[:500]
                    for k in keys_to_remove:
                        del _cache[k]

            return suggestions
        except Exception as exc:
            logger.warning("Asset search failed: %s", exc)
            return []

    autocomplete = AutoComplete(input_widget, candidates=get_suggestions)
    return input_widget, autocomplete
