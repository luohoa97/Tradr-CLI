"""Backtesting framework — simulates trades using historical OHLCV + sentiment."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from trading_cli.sentiment.aggregator import aggregate_scores_weighted
from trading_cli.sentiment.news_classifier import classify_headlines, EventType
from trading_cli.strategy.signals import generate_signal, technical_score
from trading_cli.strategy.risk import calculate_position_size, check_stop_loss, check_max_drawdown

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    timestamp: str
    symbol: str
    action: str  # BUY or SELL
    price: float
    qty: int
    reason: str
    pnl: float = 0.0


@dataclass
class BacktestResult:
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    def summary_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "period": f"{self.start_date} to {self.end_date}",
            "initial_capital": f"${self.initial_capital:,.2f}",
            "final_equity": f"${self.final_equity:,.2f}",
            "total_return": f"{self.total_return_pct:+.2f}%",
            "max_drawdown": f"{self.max_drawdown_pct:.2f}%",
            "sharpe_ratio": f"{self.sharpe_ratio:.2f}",
            "win_rate": f"{self.win_rate:.1f}%",
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
        }


class BacktestEngine:
    """Runs historical simulation using the same signal pipeline as live trading."""

    def __init__(
        self,
        config: dict,
        finbert=None,
        news_fetcher=None,
        use_sentiment: bool = True,
        strategy=None,
        progress_callback=None,
        debug: bool = False,
    ):
        """
        Args:
            config: Trading configuration dict.
            finbert: FinBERTAnalyzer instance (or None to skip sentiment).
            news_fetcher: Callable(symbol, days_ago) -> list[tuple[str, float]]
                         Returns list of (headline, unix_timestamp) tuples.
            use_sentiment: If False, skip all sentiment scoring regardless of
                          whether finbert/news_fetcher are provided.
            strategy: StrategyAdapter instance. If None, falls back to legacy
                      hardcoded technical + sentiment pipeline.
            progress_callback: Optional callable(str) to report progress.
            debug: If True, log every bar's signal details at INFO level.
        """
        self.config = config
        self.finbert = finbert
        self.news_fetcher = news_fetcher
        self.use_sentiment = use_sentiment
        self.strategy = strategy
        self.progress_callback = progress_callback
        self.debug = debug
        # Force INFO level on this logger when debug is enabled
        if debug:
            logger.setLevel(logging.INFO)

    def run(
        self,
        symbol: str,
        ohlcv: pd.DataFrame,
        start_date: str | None = None,
        end_date: str | None = None,
        initial_capital: float = 100_000.0,
    ) -> BacktestResult:
        """
        Run backtest on historical OHLCV data.

        Simulates daily signal generation and order execution at next day's open.
        """
        df = ohlcv.copy()
        # Handle both column-based and index-based dates
        if "Date" in df.columns or "date" in df.columns:
            date_col = "Date" if "Date" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)

        # Apply date range filter on the index
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df.index <= pd.Timestamp(end_date)]

        # Reset index to get date back as a column for downstream code
        df = df.reset_index()
        df = df.rename(columns={"index": "date"})

        # Normalize column names to lowercase for consistent access
        # yfinance can return MultiIndex columns (tuples), so flatten them first
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adj_close"})

        logger.info("Backtest %s: %d bars, columns: %s", symbol, len(df), list(df.columns))

        if len(df) < 60:
            logger.warning("Backtest %s: not enough data (%d bars, need 60+)", symbol, len(df))
            date_col = "date" if "date" in df.columns else None
            start_str = str(df.iloc[0][date_col])[:10] if date_col and len(df) > 0 else "N/A"
            end_str = str(df.iloc[-1][date_col])[:10] if date_col and len(df) > 0 else "N/A"
            return BacktestResult(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                initial_capital=initial_capital,
                final_equity=initial_capital,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
            )

        cash = initial_capital
        position_qty = 0
        position_avg_price = 0.0
        equity_curve = [initial_capital]
        trades: list[BacktestTrade] = []
        equity_values = [initial_capital]

        # Normalize column names to lowercase for consistent access
        # yfinance can return MultiIndex columns (tuples), so flatten them first
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.columns = [c.lower() for c in df.columns]
        if "adj close" in df.columns:
            df = df.rename(columns={"adj close": "adj_close"})

        logger.info("Backtest %s: %d bars, columns: %s", symbol, len(df), list(df.columns))

        if len(df) < 60:
            logger.warning("Backtest %s: not enough data (%d bars, need 60+)", symbol, len(df))

        # Config params
        buy_threshold = self.config.get("signal_buy_threshold", 0.5)
        sell_threshold = self.config.get("signal_sell_threshold", -0.3)
        sma_short = self.config.get("sma_short", 20)
        sma_long = self.config.get("sma_long", 50)
        rsi_period = self.config.get("rsi_period", 14)
        bb_window = self.config.get("bb_window", 20)
        bb_std = self.config.get("bb_std", 2.0)
        ema_fast = self.config.get("ema_fast", 12)
        ema_slow = self.config.get("ema_slow", 26)
        vol_window = self.config.get("volume_window", 20)
        tech_weight = self.config.get("tech_weight", 0.6)
        sent_weight = self.config.get("sent_weight", 0.4)
        risk_pct = self.config.get("risk_pct", 0.02)
        max_dd = self.config.get("max_drawdown", 0.15)
        stop_loss_pct = self.config.get("stop_loss_pct", 0.05)

        tech_weights = {
            "sma": self.config.get("weight_sma", 0.25),
            "rsi": self.config.get("weight_rsi", 0.25),
            "bb": self.config.get("weight_bb", 0.20),
            "ema": self.config.get("weight_ema", 0.15),
            "volume": self.config.get("weight_volume", 0.15),
        }

        # ── Pre-fetch and cache all sentiment scores ──────────────────────
        lookback = max(sma_long, ema_slow, bb_window, vol_window) + 30
        logger.info("Backtest %s: lookback=%d, total_bars=%d", symbol, lookback, len(df) - lookback)
        sent_scores = {}
        if self.use_sentiment and self.finbert and self.news_fetcher:
            total_days = len(df) - lookback
            try:
                # Fetch all news once (batch)
                if self.progress_callback:
                    self.progress_callback("Fetching historical news…")
                all_news = self.news_fetcher(symbol, days_ago=len(df))
                if all_news:
                    headlines = [item[0] for item in all_news]
                    timestamps = [item[1] for item in all_news]
                    classifications = classify_headlines(headlines)
                    # Analyze all headlines at once
                    if self.progress_callback:
                        self.progress_callback("Analyzing sentiment (batch)…")
                    results = self.finbert.analyze_batch(headlines)
                    # Single aggregated score for the whole period
                    cached_score = aggregate_scores_weighted(
                        results, classifications, timestamps=timestamps
                    )
                    # Apply same score to all bars (since we fetched once)
                    for i in range(lookback, len(df)):
                        sent_scores[i] = cached_score
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Sentiment pre-fetch failed: %s", exc)
                sent_scores = {}

        # ── Walk forward through data ─────────────────────────────────────
        total_bars = len(df) - lookback
        if self.progress_callback:
            self.progress_callback("Running simulation…")
        for idx, i in enumerate(range(lookback, len(df))):
            if self.progress_callback and idx % 20 == 0:
                pct = int(idx / total_bars * 100) if total_bars else 0
                self.progress_callback(f"Running simulation… {pct}%")

            historical_ohlcv = df.iloc[:i]
            current_bar = df.iloc[i]
            current_price = float(current_bar["close"])
            current_date = str(current_bar.get("date", ""))

            # Use pre-cached sentiment score
            sent_score = sent_scores.get(i, 0.0)

            # Max drawdown check
            if check_max_drawdown(equity_values, max_dd):
                break  # Stop backtest if drawdown exceeded

            # Build mock position object for strategy adapter
            class _MockPosition:
                def __init__(self, symbol, qty, avg_price):
                    self.symbol = symbol
                    self.qty = qty
                    self.avg_entry_price = avg_price

            backtest_positions = [_MockPosition(symbol, position_qty, position_avg_price)] if position_qty > 0 else []

            # Generate signal — use strategy adapter if available, else legacy
            if self.strategy is not None:
                # Use strategy adapter
                signal_result = self.strategy.generate_signal(
                    symbol=symbol,
                    ohlcv=historical_ohlcv,
                    sentiment_score=sent_score,
                    positions=backtest_positions,
                    config=self.config,
                )
                action = signal_result.action
                score = signal_result.score
                reason = signal_result.reason
                buy_threshold = self.config.get("signal_buy_threshold", 0.5)
                sell_threshold = self.config.get("signal_sell_threshold", -0.3)
                if self.debug:
                    logger.info(
                        "Bar %d | %s | price=%.2f | score=%.3f | action=%s | reason=%s",
                        idx, current_date, current_price, score, action, reason,
                    )
            else:
                # Legacy hardcoded technical + sentiment
                tech = technical_score(
                    historical_ohlcv, sma_short, sma_long, rsi_period,
                    bb_window, bb_std, ema_fast, ema_slow, vol_window,
                    tech_weights,
                )
                # Normalize hybrid score: if sentiment is absent (0.0),
                # use tech alone so buy/sell thresholds remain reachable
                if sent_score == 0.0:
                    hybrid = tech
                else:
                    hybrid = tech_weight * tech + sent_weight * sent_score
                score = hybrid
                if hybrid >= buy_threshold:
                    action = "BUY"
                elif hybrid <= sell_threshold:
                    action = "SELL"
                else:
                    action = "HOLD"
                reason = f"hybrid={hybrid:.3f} tech={tech:.3f}"
                if self.debug:
                    logger.info(
                        "Bar %d | %s | price=%.2f | tech=%.3f | sent=%.3f | hybrid=%.3f | action=%s",
                        idx, current_date, current_price, tech, sent_score, hybrid, action,
                    )

            if action == "BUY" and position_qty == 0:
                qty = calculate_position_size(
                    cash + position_qty * position_avg_price,
                    current_price,
                    risk_pct=risk_pct,
                    max_position_pct=0.10,
                )
                if qty > 0 and cash >= qty * current_price:
                    cost = qty * current_price
                    cash -= cost
                    total_shares = position_qty + qty
                    position_avg_price = (
                        (position_avg_price * position_qty + current_price * qty) / total_shares
                    )
                    position_qty = total_shares

                    trades.append(BacktestTrade(
                        timestamp=current_date,
                        symbol=symbol,
                        action="BUY",
                        price=current_price,
                        qty=qty,
                        reason=reason,
                    ))
                    if self.debug:
                        logger.info(
                            "  >>> BUY %d @ %.2f (cost=%.2f, cash=%.2f, pos=%d)",
                            qty, current_price, cost, cash, position_qty,
                        )
                elif self.debug:
                    logger.info(
                        "  >>> BUY blocked: qty=%d, cash=%.2f, need=%.2f",
                        qty, cash, qty * current_price,
                    )

            elif action == "SELL" and position_qty > 0:
                sell_reason = reason
                if check_stop_loss(position_avg_price, current_price, stop_loss_pct):
                    sell_reason = f"stop-loss ({reason})"

                proceeds = position_qty * current_price
                pnl = (current_price - position_avg_price) * position_qty
                cash += proceeds

                trades.append(BacktestTrade(
                    timestamp=current_date,
                    symbol=symbol,
                    action="SELL",
                    price=current_price,
                    qty=position_qty,
                    reason=sell_reason,
                    pnl=pnl,
                ))

                if self.debug:
                    logger.info(
                        "  >>> SELL %d @ %.2f (pnl=%.2f, proceeds=%.2f, cash=%.2f)",
                        position_qty, current_price, pnl, proceeds, cash,
                    )

                position_qty = 0
                position_avg_price = 0.0

            # Track equity
            equity = cash + position_qty * current_price
            equity_curve.append(equity)
            equity_values.append(equity)

        # Close any remaining position at last price
        if position_qty > 0 and len(df) > 0:
            last_price = float(df.iloc[-1]["close"])
            last_date = str(df.iloc[-1]["date"])[:10]
            pnl = (last_price - position_avg_price) * position_qty
            cash += position_qty * last_price
            trades.append(BacktestTrade(
                timestamp=last_date,
                symbol=symbol,
                action="SELL",
                price=last_price,
                qty=position_qty,
                reason="end of backtest",
                pnl=pnl,
            ))
            position_qty = 0

        final_equity = cash
        total_return = ((final_equity - initial_capital) / initial_capital) * 100
        logger.info("Backtest %s: %d trades, return=%.2f%%", symbol, len(trades), total_return)

        # Compute metrics
        peak = equity_values[0]
        max_dd_actual = 0.0
        for val in equity_values:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd_actual = max(max_dd_actual, dd)

        # Win rate
        sell_trades = [t for t in trades if t.action == "SELL"]
        winning = sum(1 for t in sell_trades if t.pnl > 0)
        losing = sum(1 for t in sell_trades if t.pnl < 0)
        win_rate = (winning / len(sell_trades) * 100) if sell_trades else 0.0

        # Sharpe ratio (daily returns)
        if len(equity_values) > 1:
            returns = np.diff(equity_values) / equity_values[:-1]
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0.0
        else:
            sharpe = 0.0

        return BacktestResult(
            symbol=symbol,
            start_date=str(df.iloc[0]["date"])[:10] if len(df) > 0 else "N/A",
            end_date=str(df.iloc[-1]["date"])[:10] if len(df) > 0 else "N/A",
            initial_capital=initial_capital,
            final_equity=final_equity,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd_actual * 100,
            sharpe_ratio=sharpe,
            win_rate=win_rate,
            total_trades=len(trades),
            winning_trades=winning,
            losing_trades=losing,
            trades=trades,
            equity_curve=equity_curve,
        )
