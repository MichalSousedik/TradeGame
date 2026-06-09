from __future__ import annotations

from datetime import datetime
from typing import Iterator, Protocol

import pandas as pd


class Bar:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts: datetime, open: float, high: float,
                 low: float, close: float, volume: float) -> None:
        self.ts = ts
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class DataSource(Protocol):
    """Canonical interface for market data providers (crypto, stocks, …)."""

    def history(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with UTC DatetimeIndex, columns [open,high,low,close,volume]."""
        ...

    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        """Yield closed bars in real time (used by live/loop.py)."""
        ...
