"""SQLite database layer — schema, queries, and connection management."""
 
from __future__ import annotations
 
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any
 
 
def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
 
 
def init_db(db_path: Path) -> sqlite3.Connection:
    """Create all tables and return an open connection."""
    conn = get_connection(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp   TEXT    NOT NULL,
            symbol      TEXT    NOT NULL,
            action      TEXT    NOT NULL,
            price       REAL    NOT NULL,
            quantity    INTEGER NOT NULL,
            order_id    TEXT,
            reason      TEXT,
            pnl         REAL,
            portfolio_value REAL
        );
 
        CREATE TABLE IF NOT EXISTS signals (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            action          TEXT NOT NULL,
            confidence      REAL,
            technical_score REAL,
            sentiment_score REAL,
            reason          TEXT,
            executed        INTEGER DEFAULT 0
        );
 
        CREATE TABLE IF NOT EXISTS watchlist (
            symbol      TEXT PRIMARY KEY,
            added_at    TEXT NOT NULL
        );
 
        CREATE TABLE IF NOT EXISTS sentiment_cache (
            headline_hash   TEXT PRIMARY KEY,
            headline        TEXT NOT NULL,
            label           TEXT NOT NULL,
            score           REAL NOT NULL,
            cached_at       TEXT NOT NULL
        );
 
        CREATE TABLE IF NOT EXISTS price_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            open        REAL,
            high        REAL,
            low         REAL,
            close       REAL,
            volume      INTEGER,
            UNIQUE(symbol, timestamp)
        );
 
        CREATE TABLE IF NOT EXISTS config (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn
 
 
# ── Trades ─────────────────────────────────────────────────────────────────────
 
def save_trade(
    conn: sqlite3.Connection,
    symbol: str,
    action: str,
    price: float,
    quantity: int,
    order_id: str | None = None,
    reason: str | None = None,
    pnl: float | None = None,
    portfolio_value: float | None = None,
) -> int:
    ts = datetime.utcnow().isoformat()
    cur = conn.execute(
        """INSERT INTO trades
           (timestamp, symbol, action, price, quantity, order_id, reason, pnl, portfolio_value)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, symbol, action, price, quantity, order_id, reason, pnl, portfolio_value),
    )
    conn.commit()
    return cur.lastrowid
 
 
def get_trade_history(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    action: str | None = None,
    limit: int = 100,
) -> list[dict]:
    q = "SELECT * FROM trades"
    params: list[Any] = []
    clauses = []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol.upper())
    if action:
        clauses.append("action = ?")
        params.append(action.upper())
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)
    return [dict(r) for r in conn.execute(q, params).fetchall()]
 
 
# ── Signals ────────────────────────────────────────────────────────────────────
 
def save_signal(
    conn: sqlite3.Connection,
    symbol: str,
    action: str,
    confidence: float,
    technical_score: float,
    sentiment_score: float,
    reason: str,
    executed: bool = False,
) -> int:
    ts = datetime.utcnow().isoformat()
    cur = conn.execute(
        """INSERT INTO signals
           (timestamp, symbol, action, confidence, technical_score, sentiment_score, reason, executed)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (ts, symbol, action, confidence, technical_score, sentiment_score, reason, int(executed)),
    )
    conn.commit()
    return cur.lastrowid
 
 
def get_recent_signals(conn: sqlite3.Connection, limit: int = 20) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM signals ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    ]
 
 
# ── Watchlist ──────────────────────────────────────────────────────────────────
 
def get_watchlist(conn: sqlite3.Connection) -> list[str]:
    return [r["symbol"] for r in conn.execute("SELECT symbol FROM watchlist ORDER BY symbol").fetchall()]
 
 
def add_to_watchlist(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
        (symbol.upper(), datetime.utcnow().isoformat()),
    )
    conn.commit()
 
 
def remove_from_watchlist(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol.upper(),))
    conn.commit()
 
 
# ── Sentiment cache ────────────────────────────────────────────────────────────
 
def headline_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()
 
 
def get_cached_sentiment(conn: sqlite3.Connection, text: str) -> dict | None:
    h = headline_hash(text)
    row = conn.execute(
        "SELECT label, score FROM sentiment_cache WHERE headline_hash = ?", (h,)
    ).fetchone()
    return dict(row) if row else None
 
 
def cache_sentiment(conn: sqlite3.Connection, text: str, label: str, score: float) -> None:
    h = headline_hash(text)
    conn.execute(
        """INSERT OR REPLACE INTO sentiment_cache
           (headline_hash, headline, label, score, cached_at)
           VALUES (?, ?, ?, ?, ?)""",
        (h, text[:500], label, score, datetime.utcnow().isoformat()),
    )
    conn.commit()
 
 
# ── Price history ──────────────────────────────────────────────────────────────
 
def upsert_price_bar(
    conn: sqlite3.Connection,
    symbol: str,
    timestamp: str,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: int,
) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO price_history
           (symbol, timestamp, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (symbol, timestamp, open_, high, low, close, volume),
    )
    conn.commit()
 
 
def get_price_history(
    conn: sqlite3.Connection, symbol: str, limit: int = 200
) -> list[dict]:
    return [
        dict(r)
        for r in conn.execute(
            "SELECT * FROM price_history WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
            (symbol.upper(), limit),
        ).fetchall()
    ]
