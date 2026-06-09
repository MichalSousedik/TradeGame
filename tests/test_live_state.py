"""Tests for live equity state persistence (LiveState writer + LiveReader)."""
import tempfile
from datetime import timezone
from pathlib import Path

import pandas as pd
import pytest

from tradegame.live.state import LiveState, LiveReader


def _snapshot(equity: float, position: float = 0.5, cash: float = 50_000.0) -> dict:
    return {"equity": equity, "position": position, "cash": cash}


def _ts(offset_hours: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2024-01-01T00:00:00", tz="UTC") + pd.Timedelta(hours=offset_hours)


# ── LiveState (writer) ────────────────────────────────────────────────────────

def test_record_and_read_back():
    with tempfile.TemporaryDirectory() as tmp:
        state = LiveState(Path(tmp))
        ts = _ts(0)
        state.record(ts, 42_000.0, {
            "Agent A": _snapshot(100_000.0),
            "Agent B": _snapshot(98_000.0),
        })
        state.close()

        reader = LiveReader(Path(tmp))
        assert reader.is_available
        snap = reader.read_latest_snapshot()
        assert len(snap) == 2
        assert set(snap["agent_name"]) == {"Agent A", "Agent B"}
        reader.close()


def test_meta_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        state = LiveState(Path(tmp))
        state.set_meta("symbol", "BTC/USDT")
        state.set_meta("status", "running")
        state.close()

        reader = LiveReader(Path(tmp))
        assert reader.get_meta("symbol") == "BTC/USDT"
        assert reader.get_meta("status") == "running"
        assert reader.get_meta("nonexistent") is None
        reader.close()


def test_latest_snapshot_returns_most_recent_bar():
    with tempfile.TemporaryDirectory() as tmp:
        state = LiveState(Path(tmp))
        for h in range(5):
            state.record(_ts(h), 40_000.0 + h * 100, {"Bot": _snapshot(100_000.0 + h * 50)})
        state.close()

        reader = LiveReader(Path(tmp))
        snap = reader.read_latest_snapshot()
        assert len(snap) == 1
        assert snap["equity"].iloc[0] == pytest.approx(100_000.0 + 4 * 50)
        reader.close()


def test_equity_history_last_n():
    with tempfile.TemporaryDirectory() as tmp:
        state = LiveState(Path(tmp))
        for h in range(20):
            state.record(_ts(h), 40_000.0, {"Bot": _snapshot(100_000.0 + h)})
        state.close()

        reader = LiveReader(Path(tmp))
        hist = reader.read_equity_history(last_n=5)
        # 5 rows for "Bot"
        assert len(hist) == 5
        reader.close()


def test_reader_not_available_when_db_missing():
    with tempfile.TemporaryDirectory() as tmp:
        reader = LiveReader(Path(tmp))          # no live.db exists yet
        assert not reader.is_available
        assert reader.read_latest_snapshot().empty
        assert reader.read_equity_history().empty
        assert reader.get_meta("anything") is None
