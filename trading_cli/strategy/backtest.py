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
    ):
        """
        Args:
            config: Trading configuration dict.
            finbert: FinBERTAnalyzer instance (or None to skip sentiment).
            news_fetcher: Callable(symbol, days_ago) -> list[tuple[str, float]]
                         Returns list of (headline, unix_timestamp) tuples.
            use_sentiment: If False, skip all sentiment scoring regardless of
                          whether finbert/news_fetcher are provided.
        """
        self.config = config
        self.finbert = finbert
        self.news_fetcher = news_fetcher
        self.use_sentiment = use_sentiment

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
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            if start_date:
                df = df[df["Date"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["Date"] <= pd.Timestamp(end_date)]

        if len(df) < 60:
            return BacktestResult(
                symbol=symbol,
                start_date=str(df.index[0]) if len(df) > 0 else "N/A",
                end_date=str(df.index[-1]) if len(df) > 0 else "N/A",
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

        # Walk forward through data
        lookback = max(sma_long, ema_slow, bb_window, vol_window) + 30
        for i in range(lookback, len(df)):
            historical_ohlcv = df.iloc[:i]
            current_bar = df.iloc[i]
            current_price = float(current_bar.get("Close", current_bar.get("close", 0)))
            current_date = str(current_bar.get("Date", df.index[i]))

            # Technical score
            tech = technical_score(
                historical_ohlcv, sma_short, sma_long, rsi_period,
                bb_window, bb_std, ema_fast, ema_slow, vol_window,
                tech_weights,
            )

            # Sentiment score — fetches historical news for the current walk-forward date
            sent_score = 0.0
            if self.use_sentiment and self.finbert and self.news_fetcher:
                try:
                    news_items = self.news_fetcher(symbol, days_ago=(len(df) - i))
                    if news_items:
                        headlines = [item[0] for item in news_items]
                        timestamps = [item[1] for item in news_items]
                        classifications = classify_headlines(headlines)
                        # Use cached sentiment if available (via FinBERT's analyze_with_cache)
                        results = self.finbert.analyze_batch(headlines)
                        sent_score = aggregate_scores_weighted(
                            results, classifications, timestamps=timestamps
                        )
                except Exception:
                    pass

            hybrid = tech_weight * tech + sent_weight * sent_score

            # Max drawdown check
            if check_max_drawdown(equity_values, max_dd):
                break  # Stop backtest if drawdown exceeded

            # Generate signal
            if hybrid >= buy_threshold and position_qty == 0:
                # BUY signal
                qty = calculate_position_size(
                    cash + position_qty * position_avg_price,
                    current_price,
                    risk_pct=risk_pct,
                    max_position_pct=0.10,
                )
                if qty > 0 and cash >= qty * current_price:
                    cost = qty * current_price
                    cash -= cost
                    # Update position average price
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
                        reason=f"hybrid={hybrid:.3f} tech={tech:.3f}",
                    ))

            elif hybrid <= sell_threshold and position_qty > 0:
                # SELL signal or stop-loss
                sell_reason = f"hybrid={hybrid:.3f}"
                if check_stop_loss(position_avg_price, current_price, stop_loss_pct):
                    sell_reason = "stop-loss"

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

                position_qty = 0
                position_avg_price = 0.0

            # Track equity
            equity = cash + position_qty * current_price
            equity_curve.append(equity)
            equity_values.append(equity)

        # Close any remaining position at last price
        if position_qty > 0 and len(df) > 0:
            last_price = float(df.iloc[-1].get("Close", df.iloc[-1].get("close", 0)))
            last_date = str(df.index[-1])
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
            start_date=str(df.index[0]) if len(df) > 0 else "N/A",
            end_date=str(df.index[-1]) if len(df) > 0 else "N/A",
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
