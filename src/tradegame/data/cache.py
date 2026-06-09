from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS ohlcv (
    ts_ms  BIGINT  NOT NULL,
    symbol VARCHAR NOT NULL,
    tf     VARCHAR NOT NULL,
    open   DOUBLE  NOT NULL,
    high   DOUBLE  NOT NULL,
    low    DOUBLE  NOT NULL,
    close  DOUBLE  NOT NULL,
    volume DOUBLE  NOT NULL
)
"""


class DataCache:
    """DuckDB-backed local cache for OHLCV data with incremental fetching.

    Use as a context manager so the file lock is released promptly:
        with DataCache(data_dir) as cache:
            df = cache.get_or_fetch(...)
    """

    def __init__(self, data_dir: Path) -> None:
        import duckdb

        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(str(data_dir / "cache.db"))
        self._conn.execute(_CREATE_SQL)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "DataCache":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def put(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return

        min_ms = _to_ms(df.index.min())
        max_ms = _to_ms(df.index.max())
        self._conn.execute(
            "DELETE FROM ohlcv WHERE symbol=? AND tf=? AND ts_ms>=? AND ts_ms<=?",
            [symbol, timeframe, min_ms, max_ms],
        )

        insert = df.reset_index()
        insert.columns = ["ts", "open", "high", "low", "close", "volume"]
        insert["ts_ms"] = insert["ts"].apply(_to_ms)
        insert["symbol"] = symbol
        insert["tf"] = timeframe
        insert = insert[["ts_ms", "symbol", "tf", "open", "high", "low", "close", "volume"]]

        self._conn.register("_put_tmp", insert)
        self._conn.execute("INSERT INTO ohlcv SELECT * FROM _put_tmp")
        self._conn.unregister("_put_tmp")

    def get(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[pd.DataFrame]:
        start_ms = _to_ms(_ensure_utc(start))
        end_ms = _to_ms(_ensure_utc(end))

        df = self._conn.execute(
            "SELECT ts_ms, open, high, low, close, volume FROM ohlcv "
            "WHERE symbol=? AND tf=? AND ts_ms>=? AND ts_ms<? ORDER BY ts_ms",
            [symbol, timeframe, start_ms, end_ms],
        ).df()

        if df.empty:
            return None

        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.drop(columns=["ts_ms"]).set_index("ts")
        return df

    def get_or_fetch(self, source, symbol: str, timeframe: str,
                     start: datetime, end: datetime) -> pd.DataFrame:
        """Return cached data if available for the full range; otherwise fetch and cache."""
        cached = self.get(symbol, timeframe, start, end)
        if cached is not None and not cached.empty:
            return cached

        df = source.history(symbol, timeframe, start, end)
        if not df.empty:
            self.put(symbol, timeframe, df)
        return df if not (df is None) else pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"]
        )


def _to_ms(ts) -> int:
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, pd.Timestamp):
        return int(ts.timestamp() * 1000)
    if isinstance(ts, datetime):
        dt = _ensure_utc(ts)
        return int(dt.timestamp() * 1000)
    raise TypeError(f"Cannot convert {type(ts)} to epoch ms")


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
