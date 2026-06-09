from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterator

import pandas as pd

from tradegame.data.source import Bar


class CryptoSource:
    """Fetches OHLCV data from a CCXT-compatible exchange (default: Binance)."""

    def __init__(self, exchange_id: str = "binance") -> None:
        import ccxt
        self._exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    def history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        since_ms = _to_ms(start)
        end_ms = _to_ms(end)
        raw: list = []

        while since_ms < end_ms:
            bars = self._exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=1000)
            if not bars:
                break
            within = [b for b in bars if b[0] < end_ms]
            raw.extend(within)
            if len(bars) < 1000 or bars[-1][0] >= end_ms:
                break
            since_ms = bars[-1][0] + 1

        if not raw:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        return df

    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """Yields closed bars via polling (WebSocket upgrade in Phase 1)."""
        raise NotImplementedError("Live streaming implemented in Phase 1 (live/loop.py)")


def _to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)
