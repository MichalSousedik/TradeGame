"""Alpaca data source — Phase 4 stub."""
from __future__ import annotations

from datetime import datetime
from typing import Iterator

import pandas as pd

from tradegame.data.source import Bar


class AlpacaSource:
    """US-stock intraday data via Alpaca free IEX feed + paper-trading account."""

    def __init__(self, api_key: str = "", api_secret: str = "") -> None:
        self._api_key = api_key
        self._api_secret = api_secret

    def history(self, symbol: str, timeframe: str,
                start: datetime, end: datetime) -> pd.DataFrame:
        raise NotImplementedError("Alpaca source implemented in Phase 4")

    def stream(self, symbol: str, timeframe: str) -> Iterator[Bar]:
        raise NotImplementedError("Alpaca source implemented in Phase 4")
