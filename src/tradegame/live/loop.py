"""
Phase 1 — Live WebSocket loop via ccxt.pro.

Architecture
────────────
• A REST warmup call loads the last 300 bars from cache (or Binance) for
  indicator history.
• ccxt.pro watch_ohlcv() streams kline updates via WebSocket.
• We detect a *closed* bar when the forming-bar timestamp advances — the
  previous bar is now confirmed closed.
• On every closed bar: append to the rolling window, recompute features,
  step all agents, rebalance portfolios, persist equity snapshot.

Run via:  python -m tradegame.live.runner
       or tradegame-live  (after pip install -e .)
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Sequence

import pandas as pd

from tradegame.live.state import LiveState

log = logging.getLogger(__name__)


class LiveLoop:
    """Drives all agents on a live WebSocket feed, persisting equity to LiveState."""

    _WINDOW_CAP = 500   # rolling bar buffer kept in memory

    def __init__(
        self,
        agents: Sequence,
        portfolios: Sequence,
        state: LiveState,
        symbol: str,
        timeframe: str,
        warmup_df: pd.DataFrame,
    ) -> None:
        self._agents = list(agents)
        self._portfolios = list(portfolios)
        self._state = state
        self._symbol = symbol
        self._timeframe = timeframe
        self._df = warmup_df.copy()    # rolling OHLCV window

    # ── public ────────────────────────────────────────────────────────────────

    def start(self, exchange_id: str = "binance") -> None:
        """Blocking entry point — runs until KeyboardInterrupt."""
        asyncio.run(self._run(exchange_id))

    # ── async internals ───────────────────────────────────────────────────────

    async def _run(self, exchange_id: str) -> None:
        try:
            import ccxt.pro as ccxtpro
        except ImportError as exc:
            raise ImportError(
                "ccxt.pro not available. Install with: pip install 'ccxt[pro]'"
            ) from exc

        exchange = getattr(ccxtpro, exchange_id)({"enableRateLimit": True})
        last_forming_ts: int | None = None

        log.info("WebSocket connected — %s %s", self._symbol, self._timeframe)
        print(f"[live] Watching {self._symbol} {self._timeframe}  (Ctrl-C to stop)")

        try:
            while True:
                bars = await exchange.watch_ohlcv(
                    self._symbol, self._timeframe, limit=300
                )
                if not bars or len(bars) < 2:
                    continue

                forming_ts = bars[-1][0]   # ms timestamp of the currently-forming bar

                if last_forming_ts is None:
                    last_forming_ts = forming_ts
                    continue

                if forming_ts > last_forming_ts:
                    # New bar just opened — the bar at last_forming_ts is closed.
                    # ccxt.pro keeps closed bars in the list; bars[-2] is the one
                    # that just closed (its timestamp == last_forming_ts).
                    closed = next(
                        (b for b in reversed(bars) if b[0] == last_forming_ts),
                        bars[-2],
                    )
                    self._on_closed_bar(closed)
                    last_forming_ts = forming_ts

        except asyncio.CancelledError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            print("[live] Shutting down.")
            await exchange.close()
            self._state.set_meta("status", "stopped")
            self._state.close()

    def _on_closed_bar(self, bar: list) -> None:
        from tradegame.features.indicators import compute_features

        ts_ms, open_, high, low, close, volume = bar
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        price = float(close)

        # Append to rolling window
        new_row = pd.DataFrame(
            {"open": [open_], "high": [high], "low": [low],
             "close": [close], "volume": [volume]},
            index=pd.DatetimeIndex([ts]),
        )
        self._df = pd.concat([self._df, new_row]).iloc[-self._WINDOW_CAP:]

        features = compute_features(self._df, timeframe=self._timeframe)
        snapshot: dict[str, dict] = {}

        for agent, portfolio in zip(self._agents, self._portfolios):
            action = agent.observe(features)
            portfolio.rebalance(price, action.target_weight, ts)
            snapshot[agent.name] = {
                "equity": portfolio.equity(price),
                "position": portfolio.position,
                "cash": portfolio.cash,
            }

        self._state.record(ts, price, snapshot)
        self._state.set_meta("last_bar_ts", str(ts_ms))
        self._state.set_meta("last_price", f"{price:.8f}")

        equities = {n: f"${d['equity']:,.0f}" for n, d in snapshot.items()}
        print(f"[live] {ts.strftime('%Y-%m-%d %H:%M UTC')} | "
              f"price={price:,.2f} | {equities}")
