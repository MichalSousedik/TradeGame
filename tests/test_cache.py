import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradegame.data.cache import DataCache


def _make_ohlcv(n: int = 100, start: str = "2023-01-01") -> pd.DataFrame:
    close = 30_000.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(n),
        },
        index=pd.date_range(start, periods=n, freq="1h", tz="UTC"),
    )


def test_write_read_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DataCache(Path(tmp))
        df = _make_ohlcv(100)
        cache.put("BTC/USDT", "1h", df)

        result = cache.get(
            "BTC/USDT", "1h",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 8, tzinfo=timezone.utc),
        )
        assert result is not None
        assert not result.empty
        assert "close" in result.columns
        assert result.index.tz is not None    # timezone preserved


def test_read_empty_returns_none():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DataCache(Path(tmp))
        result = cache.get(
            "BTC/USDT", "1h",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 8, tzinfo=timezone.utc),
        )
        assert result is None


def test_incremental_write_no_overlap():
    with tempfile.TemporaryDirectory() as tmp:
        cache = DataCache(Path(tmp))
        df1 = _make_ohlcv(50, "2023-01-01")
        df2 = _make_ohlcv(50, "2023-03-01")
        cache.put("BTC/USDT", "1h", df1)
        cache.put("BTC/USDT", "1h", df2)

        result = cache.get(
            "BTC/USDT", "1h",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 4, 1, tzinfo=timezone.utc),
        )
        assert result is not None
        assert len(result) == 100


def test_overwrite_same_range():
    """Writing the same range twice must not create duplicate rows."""
    with tempfile.TemporaryDirectory() as tmp:
        cache = DataCache(Path(tmp))
        df = _make_ohlcv(50)
        cache.put("BTC/USDT", "1h", df)
        cache.put("BTC/USDT", "1h", df)    # second write should overwrite

        result = cache.get(
            "BTC/USDT", "1h",
            datetime(2023, 1, 1, tzinfo=timezone.utc),
            datetime(2023, 1, 10, tzinfo=timezone.utc),
        )
        assert result is not None
        assert not result.index.duplicated().any()
