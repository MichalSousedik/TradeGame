"""
Live state persistence.

LiveState  — opened by the loop process (read-write).
LiveReader — opened by the dashboard process (read-only).
Both operate on data/live.db, separate from the OHLCV cache.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

_CREATE_EQUITY = """
CREATE TABLE IF NOT EXISTS live_equity (
    ts_ms      BIGINT  NOT NULL,
    agent_name VARCHAR NOT NULL,
    equity     DOUBLE  NOT NULL,
    position   DOUBLE  NOT NULL,
    cash       DOUBLE  NOT NULL,
    price      DOUBLE  NOT NULL
)
"""

_CREATE_META = """
CREATE TABLE IF NOT EXISTS live_meta (
    key   VARCHAR NOT NULL,
    value VARCHAR NOT NULL
)
"""


class LiveState:
    """Write-side state store used by the live loop process."""

    def __init__(self, data_dir: Path) -> None:
        import duckdb
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(data_dir / "live.db"))
        self._conn.execute(_CREATE_EQUITY)
        self._conn.execute(_CREATE_META)

    def record(
        self,
        ts: pd.Timestamp,
        price: float,
        snapshot: dict[str, dict],
    ) -> None:
        """Persist one equity snapshot per agent for the just-closed bar."""
        ts_ms = int(ts.timestamp() * 1000)
        rows = [
            (ts_ms, name, d["equity"], d["position"], d["cash"], price)
            for name, d in snapshot.items()
        ]
        df = pd.DataFrame(
            rows, columns=["ts_ms", "agent_name", "equity", "position", "cash", "price"]
        )
        self._conn.register("_eq_tmp", df)
        self._conn.execute("INSERT INTO live_equity SELECT * FROM _eq_tmp")
        self._conn.unregister("_eq_tmp")

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute("DELETE FROM live_meta WHERE key=?", [key])
        self._conn.execute("INSERT INTO live_meta VALUES (?, ?)", [key, value])

    def close(self) -> None:
        self._conn.close()


class LiveReader:
    """Read-only view of live.db used by the dashboard."""

    def __init__(self, data_dir: Path) -> None:
        import duckdb
        db_path = Path(data_dir) / "live.db"
        self._conn: Optional[object] = None
        if db_path.exists():
            try:
                self._conn = duckdb.connect(str(db_path), read_only=True)
            except Exception:
                pass  # loop has exclusive lock — show "not running" gracefully

    @property
    def is_available(self) -> bool:
        return self._conn is not None

    def read_equity_history(self, last_n: int = 500) -> pd.DataFrame:
        """Return equity history for all agents (last_n bars each)."""
        if not self.is_available:
            return pd.DataFrame()
        df = self._conn.execute(  # type: ignore[union-attr]
            """
            SELECT ts_ms, agent_name, equity, price
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY agent_name ORDER BY ts_ms DESC
                       ) AS rn
                FROM live_equity
            )
            WHERE rn <= ?
            ORDER BY ts_ms
            """,
            [last_n],
        ).df()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.drop(columns=["ts_ms"])

    def read_latest_snapshot(self) -> pd.DataFrame:
        """Return the most recent equity row for every agent."""
        if not self.is_available:
            return pd.DataFrame()
        df = self._conn.execute(  # type: ignore[union-attr]
            """
            SELECT ts_ms, agent_name, equity, position, cash, price
            FROM live_equity
            WHERE ts_ms = (SELECT MAX(ts_ms) FROM live_equity)
            ORDER BY agent_name
            """
        ).df()
        if df.empty:
            return df
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.drop(columns=["ts_ms"])

    def get_meta(self, key: str) -> Optional[str]:
        if not self.is_available:
            return None
        row = self._conn.execute(  # type: ignore[union-attr]
            "SELECT value FROM live_meta WHERE key=?", [key]
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
