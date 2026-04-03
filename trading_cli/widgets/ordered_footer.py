"""Custom Footer widget that preserves navigation binding order (1-6 first, quit last).

Always renders bindings — never disappears on screen transitions or resize.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import groupby

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Footer
from textual.widgets._footer import FooterKey
from textual.binding import Binding


class OrderedFooter(Footer):
    """Footer that shows navigation bindings (1-6) first, then quit, then other bindings."""

    NAV_ORDER = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
    QUIT_KEY = "ctrl+q"

    def compose(self) -> ComposeResult:
        try:
            active_bindings = self.screen.active_bindings
        except Exception:
            return

        bindings = [
            (binding, enabled, tooltip)
            for (_, binding, enabled, tooltip) in active_bindings.values()
            if binding.show
        ]

        if not bindings:
            return

        # Sort: nav keys (1-6) first in order, then ctrl+q, then everything else
        def sort_key(item):
            binding, enabled, tooltip = item
            key = binding.key
            if key in self.NAV_ORDER:
                return (0, self.NAV_ORDER[key], key)
            elif key == self.QUIT_KEY:
                return (2, 0, key)
            else:
                return (1, 0, key)

        bindings.sort(key=sort_key)

        action_to_bindings: defaultdict[str, list[tuple[Binding, bool, str]]] = defaultdict(list)
        for binding, enabled, tooltip in bindings:
            action_to_bindings[binding.action].append((binding, enabled, tooltip))

        self.styles.grid_size_columns = len(action_to_bindings)
        for group, multi_bindings_iterable in groupby(
            action_to_bindings.values(),
            lambda multi_bindings_: multi_bindings_[0][0].group,
        ):
            multi_bindings = list(multi_bindings_iterable)
            for multi_binding in multi_bindings:
                binding, enabled, tooltip = multi_binding[0]
                yield FooterKey(
                    binding.key,
                    self.app.get_key_display(binding),
                    binding.description,
                    binding.action,
                    disabled=not enabled,
                    tooltip=tooltip,
                ).data_bind(compact=Footer.compact)

        if self.show_command_palette and self.app.ENABLE_COMMAND_PALETTE:
            try:
                _node, binding, enabled, tooltip = active_bindings[
                    self.app.COMMAND_PALETTE_BINDING
                ]
            except KeyError:
                pass
            else:
                yield FooterKey(
                    binding.key,
                    self.app.get_key_display(binding),
                    binding.description,
                    binding.action,
                    classes="-command-palette",
                    disabled=not enabled,
                    tooltip=binding.tooltip or binding.description,
                )

    def on_mount(self) -> None:
        """Force a recompose after mount to catch app-level bindings."""
        self.call_later(self.recompose)

    def on_screen_resume(self, event: Screen.ScreenResume) -> None:
        """Force footer refresh when screen becomes active."""
        self.recompose()

    def on_resize(self) -> None:
        """Force footer refresh on terminal resize (maximize/minimize fix)."""
        self.recompose()
