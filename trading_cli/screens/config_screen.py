"""Config screen — edit API keys and strategy parameters."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import (
    Header, Input, Label, Switch, Button, Static, Select,
    OptionList, Collapsible,
)
from textual.containers import Vertical, Horizontal, ScrollableContainer
from textual.reactive import reactive

from trading_cli.config import save_config
from trading_cli.widgets.ordered_footer import OrderedFooter


class ConfigRow(Horizontal):
    """Label + Input/Widget row."""

    DEFAULT_CSS = """
    ConfigRow {
        width: 100%;
        height: auto;
        padding: 0 1;
        margin: 0 0 0 0;
        layout: horizontal;
    }
    ConfigRow Label {
        width: 28;
        min-width: 28;
        content-align: right middle;
        padding-right: 1;
    }
    ConfigRow Input, ConfigRow Select {
        width: 1fr;
    }
    """

    def __init__(
        self,
        label: str,
        key: str,
        value: str = "",
        password: bool = False,
        options: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(id=f"row-{key}")
        self._label = label
        self._key = key
        self._value = value
        self._password = password
        self._options = options

    def compose(self) -> ComposeResult:
        yield Label(f"{self._label}:")
        if self._options:
            yield Select(
                options=self._options,
                value=self._value,
                id=f"input-{self._key}",
                allow_blank=False,
            )
        else:
            yield Input(
                value=self._value,
                password=self._password,
                id=f"input-{self._key}",
            )
 
 
class ConfigScreen(Screen):
    """Screen ID 6 — settings editor."""

    BINDINGS = [
        Binding("ctrl+s", "save", "Save", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def compose(self) -> ComposeResult:
        app = self.app
        cfg = getattr(app, "config", {})

        # Build strategy options from registry
        from trading_cli.strategy.adapters.registry import list_strategies
        strategy_id = cfg.get("strategy_id", "hybrid")
        try:
            strategy_options = list_strategies()
        except Exception:
            strategy_options = ["hybrid"]
        strategy_select_options = [(opt.title(), opt) for opt in strategy_options]

        # Build exchange provider options from adapter registry
        from trading_cli.execution.adapters.registry import list_adapters
        current_provider = cfg.get("adapter_id", "yfinance")
        try:
            adapter_ids = list_adapters()
        except Exception:
            adapter_ids = ["yfinance", "alpaca", "binance", "kraken"]
        provider_display = {
            "alpaca": "Alpaca (Stocks/ETFs)",
            "yfinance": "Yahoo Finance (Demo)",
            "binance": "Binance (Crypto)",
            "kraken": "Kraken (Crypto)",
        }
        provider_select_options = [
            (provider_display.get(aid, aid.title()), aid) for aid in adapter_ids
        ]

        # Build sentiment model options
        current_sentiment = cfg.get("sentiment_model", "finbert")
        sentiment_select_options = [
            ("FinBERT", "finbert"),
        ]

        yield Header(show_clock=True)
        with ScrollableContainer(id="config-scroll"):
            yield Label("[bold]Configuration[/bold]  [dim](Ctrl+S to save, ESC to cancel)[/dim]")

            with Collapsible(title="🏦 Exchange Provider", id="collapsible-provider"):
                yield ConfigRow(
                    "Exchange",
                    "adapter_id",
                    current_provider,
                    options=provider_select_options,
                )

            with Collapsible(title="🔑 Alpaca API", id="collapsible-api"):
                yield ConfigRow("API Key", "alpaca_api_key", cfg.get("alpaca_api_key", ""), password=True)
                yield ConfigRow("API Secret", "alpaca_api_secret", cfg.get("alpaca_api_secret", ""), password=True)

            with Collapsible(title="📊 Risk Parameters", id="collapsible-risk"):
                yield ConfigRow("Risk % per trade", "risk_pct", str(cfg.get("risk_pct", 0.02)))
                yield ConfigRow("Max drawdown %", "max_drawdown", str(cfg.get("max_drawdown", 0.15)))
                yield ConfigRow("Stop-loss %", "stop_loss_pct", str(cfg.get("stop_loss_pct", 0.05)))
                yield ConfigRow("Max positions", "max_positions", str(cfg.get("max_positions", 10)))

            with Collapsible(title="🎯 Signal Thresholds", id="collapsible-thresholds"):
                yield ConfigRow("Buy threshold (0–1)", "signal_buy_threshold", str(cfg.get("signal_buy_threshold", 0.5)))
                yield ConfigRow("Sell threshold (-1–0)", "signal_sell_threshold", str(cfg.get("signal_sell_threshold", -0.3)))

            with Collapsible(title="🧠 Strategy", id="collapsible-strategy"):
                yield ConfigRow(
                    "Active strategy",
                    "strategy_id",
                    strategy_id,
                    options=strategy_select_options,
                )
                yield Static("", id="strategy-info")
                yield ConfigRow(
                    "Sentiment model",
                    "sentiment_model",
                    current_sentiment,
                    options=sentiment_select_options,
                )

            with Collapsible(title="⚖️ Strategy Weights", id="collapsible-weights"):
                yield ConfigRow("Technical weight", "tech_weight", str(cfg.get("tech_weight", 0.6)))
                yield ConfigRow("Sentiment weight", "sent_weight", str(cfg.get("sent_weight", 0.4)))

            with Collapsible(title="📈 Technical Indicator Weights", id="collapsible-tech-weights"):
                yield ConfigRow("SMA weight", "weight_sma", str(cfg.get("weight_sma", 0.25)))
                yield ConfigRow("RSI weight", "weight_rsi", str(cfg.get("weight_rsi", 0.25)))
                yield ConfigRow("Bollinger weight", "weight_bb", str(cfg.get("weight_bb", 0.20)))
                yield ConfigRow("EMA weight", "weight_ema", str(cfg.get("weight_ema", 0.15)))
                yield ConfigRow("Volume weight", "weight_volume", str(cfg.get("weight_volume", 0.15)))

            with Collapsible(title="⚙️ Indicator Parameters", id="collapsible-params"):
                yield ConfigRow("SMA short period", "sma_short", str(cfg.get("sma_short", 20)))
                yield ConfigRow("SMA long period", "sma_long", str(cfg.get("sma_long", 50)))
                yield ConfigRow("RSI period", "rsi_period", str(cfg.get("rsi_period", 14)))
                yield ConfigRow("Bollinger window", "bb_window", str(cfg.get("bb_window", 20)))
                yield ConfigRow("Bollinger std dev", "bb_std", str(cfg.get("bb_std", 2.0)))
                yield ConfigRow("EMA fast", "ema_fast", str(cfg.get("ema_fast", 12)))
                yield ConfigRow("EMA slow", "ema_slow", str(cfg.get("ema_slow", 26)))
                yield ConfigRow("Volume SMA window", "volume_window", str(cfg.get("volume_window", 20)))

            with Collapsible(title="📰 Sentiment Event Weights", id="collapsible-event-weights"):
                yield ConfigRow("Earnings weight", "event_weight_earnings", str(cfg.get("event_weight_earnings", 1.5)))
                yield ConfigRow("Executive weight", "event_weight_executive", str(cfg.get("event_weight_executive", 1.3)))
                yield ConfigRow("Product weight", "event_weight_product", str(cfg.get("event_weight_product", 1.2)))
                yield ConfigRow("Macro weight", "event_weight_macro", str(cfg.get("event_weight_macro", 1.4)))
                yield ConfigRow("Generic weight", "event_weight_generic", str(cfg.get("event_weight_generic", 0.8)))
                yield ConfigRow("Sentiment half-life (hrs)", "sentiment_half_life_hours", str(cfg.get("sentiment_half_life_hours", 24.0)))

            with Collapsible(title="⏱️ Poll Intervals (seconds)", id="collapsible-poll"):
                yield ConfigRow("Price poll", "poll_interval_prices", str(cfg.get("poll_interval_prices", 30)))
                yield ConfigRow("News poll", "poll_interval_news", str(cfg.get("poll_interval_news", 900)))
                yield ConfigRow("Signal poll", "poll_interval_signals", str(cfg.get("poll_interval_signals", 300)))
                yield ConfigRow("Positions poll", "poll_interval_positions", str(cfg.get("poll_interval_positions", 60)))

            with Collapsible(title="🤖 Auto-Trading", id="collapsible-auto"):
                with Horizontal(id="auto-trade-row"):
                    yield Label("Enable auto-trading:")
                    yield Switch(
                        value=cfg.get("auto_trading", False),
                        id="switch-auto-trading",
                    )

            with Horizontal(id="config-buttons"):
                yield Button("💾 Save", id="btn-save", variant="success")
                yield Button("💾🔄 Save & Restart", id="btn-restart", variant="warning")
                yield Button("❌ Cancel", id="btn-cancel", variant="default")

        yield OrderedFooter()

    def on_button_pressed(self, event) -> None:
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-restart":
            self.action_save_restart()
        elif event.button.id == "btn-cancel":
            self.app.pop_screen()

    def _read_config(self) -> dict:
        """Read all config values from the form."""
        app = self.app
        cfg = dict(getattr(app, "config", {}))

        str_keys = [
            "alpaca_api_key", "alpaca_api_secret",
        ]
        float_keys = [
            "risk_pct", "max_drawdown", "stop_loss_pct",
            "signal_buy_threshold", "signal_sell_threshold",
            "tech_weight", "sent_weight",
            "weight_sma", "weight_rsi", "weight_bb", "weight_ema", "weight_volume",
            "bb_std",
            "event_weight_earnings", "event_weight_executive", "event_weight_product",
            "event_weight_macro", "event_weight_generic",
            "sentiment_half_life_hours",
        ]
        int_keys = [
            "max_positions", "poll_interval_prices",
            "poll_interval_news", "poll_interval_signals", "poll_interval_positions",
            "sma_short", "sma_long", "rsi_period",
            "bb_window", "ema_fast", "ema_slow", "volume_window",
        ]

        for key in str_keys:
            try:
                widget = self.query_one(f"#input-{key}", Input)
                cfg[key] = widget.value.strip()
            except Exception:
                pass
        for key in float_keys:
            try:
                widget = self.query_one(f"#input-{key}", Input)
                cfg[key] = float(widget.value.strip())
            except Exception:
                pass
        for key in int_keys:
            try:
                widget = self.query_one(f"#input-{key}", Input)
                cfg[key] = int(widget.value.strip())
            except Exception:
                pass

        # Strategy selector (Select widget)
        try:
            sel = self.query_one("#input-strategy_id", Select)
            cfg["strategy_id"] = str(sel.value)
        except Exception:
            pass

        # Exchange provider (Select widget)
        try:
            sel = self.query_one("#input-adapter_id", Select)
            cfg["adapter_id"] = str(sel.value)
        except Exception:
            pass

        # Sentiment model (Select widget)
        try:
            sel = self.query_one("#input-sentiment_model", Select)
            cfg["sentiment_model"] = str(sel.value)
        except Exception:
            pass

        try:
            sw = self.query_one("#switch-auto-trading", Switch)
            cfg["auto_trading"] = sw.value
        except Exception:
            pass

        return cfg

    def action_save(self) -> None:
        app = self.app
        cfg = self._read_config()

        save_config(cfg)
        app.config = cfg
        app.notify("Configuration saved ✓")
        app.pop_screen()

    def action_save_restart(self) -> None:
        app = self.app
        cfg = self._read_config()

        save_config(cfg)
        app.config = cfg
        app.notify("Restarting with new config…")

        import sys
        import os
        # Use os.execv to replace the current process
        python = sys.executable
        script = sys.argv[0]
        os.execv(python, [python, script])

    def on_select_changed(self, event: Select.Changed) -> None:
        """Update info display when selection changes."""
        if event.select.id == "input-strategy_id":
            self._update_strategy_info(str(event.value))

    def _update_strategy_info(self, strategy_id: str) -> None:
        """Display strategy description."""
        try:
            from trading_cli.strategy.adapters.registry import get_strategy
            strategy_cls = get_strategy(strategy_id)
            if strategy_cls:
                info = strategy_cls.__new__(strategy_cls).info()
                info_widget = self.query_one("#strategy-info", Static)
                info_widget.update(
                    f"[dim]{info.description}[/dim]"
                )
        except Exception:
            pass

    def on_mount(self) -> None:
        """Initialize strategy info display."""
        cfg = getattr(self.app, "config", {})
        strategy_id = cfg.get("strategy_id", "hybrid")
        self._update_strategy_info(strategy_id)

    def action_cancel(self) -> None:
        """Handle ESC to cancel without saving."""
        self.app.pop_screen()
