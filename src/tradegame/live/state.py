"""
Live state persistence — plain JSON files, no DuckDB locking.

Three files in data/:
  live_meta.json    — key/value metadata (status, symbol, last price, …)
  live_latest.json  — most recent equity snapshot per agent (overwritten each bar)
  live_equity.jsonl — full equity history, one JSON line appended per closed bar

No connection management, no file locks, safe to read from a separate process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd


def _atomic_write(path: Path, data) -> None:
    """Write JSON atomically via a temp file so the dashboard never reads a partial write."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, str(path))


class LiveState:
    """Write-side state store — used exclusively by the live loop process."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / "live_meta.json"
        self._latest_path = self._dir / "live_latest.json"
        self._equity_path = self._dir / "live_equity.jsonl"
        if not self._meta_path.exists():
            _atomic_write(self._meta_path, {})

    def record(self, ts: pd.Timestamp, price: float, snapshot: dict) -> None:
        """Persist one equity snapshot per agent for the just-closed bar."""
        ts_ms = int(ts.timestamp() * 1000)

        # Append each agent row to history
        with open(self._equity_path, "a") as f:
            for name, d in snapshot.items():
                f.write(json.dumps({
                    "ts_ms": ts_ms, "agent_name": name, "price": price,
                    "equity": d["equity"], "position": d["position"], "cash": d["cash"],
                }) + "\n")

        # Overwrite latest snapshot (O(1) for the dashboard's current-equity table)
        latest = [
            {"ts_ms": ts_ms, "agent_name": name, "price": price, **d}
            for name, d in snapshot.items()
        ]
        _atomic_write(self._latest_path, latest)

    def set_meta(self, key: str, value: str) -> None:
        meta = self._read_meta()
        meta[key] = value
        _atomic_write(self._meta_path, meta)

    def close(self) -> None:
        pass  # nothing to close

    def _read_meta(self) -> dict:
        try:
            return json.loads(self._meta_path.read_text())
        except Exception:
            return {}


class LiveReader:
    """Read-only view of live state — used by the Streamlit dashboard."""

    def __init__(self, data_dir: Path) -> None:
        self._dir = Path(data_dir)
        self._meta_path = self._dir / "live_meta.json"
        self._latest_path = self._dir / "live_latest.json"
        self._equity_path = self._dir / "live_equity.jsonl"

    @property
    def is_available(self) -> bool:
        return self._meta_path.exists()

    def get_meta(self, key: str) -> Optional[str]:
        if not self._meta_path.exists():
            return None
        try:
            return json.loads(self._meta_path.read_text()).get(key)
        except Exception:
            return None

    def read_latest_snapshot(self) -> pd.DataFrame:
        """Current equity for every agent — fast, reads a single small JSON file."""
        if not self._latest_path.exists():
            return pd.DataFrame()
        try:
            rows = json.loads(self._latest_path.read_text())
            df = pd.DataFrame(rows)
            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            return df.drop(columns=["ts_ms"])
        except Exception:
            return pd.DataFrame()

    def read_equity_history(self, last_n: int = 500) -> pd.DataFrame:
        """Equity curves — reads the JSONL history file."""
        if not self._equity_path.exists():
            return pd.DataFrame()
        rows = []
        try:
            with open(self._equity_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
        except Exception:
            return pd.DataFrame()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        # Keep last_n bars per agent
        parts = [
            agent_df.tail(last_n)
            for _, agent_df in df.groupby("agent_name", sort=False)
        ]
        df = pd.concat(parts) if parts else pd.DataFrame()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
            df = df.drop(columns=["ts_ms"]).sort_values("ts")
        return df

    def close(self) -> None:
        pass
