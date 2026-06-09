"""
Phase 1 — Live WebSocket loop.

Polls Binance for closed bars and feeds them to all agents in real time.
This is a stub; the full async implementation arrives in Phase 1.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from typing import Sequence, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tradegame.agents.base import Agent
    from tradegame.engine.portfolio import Portfolio


def run_paper(
    agents: Sequence["Agent"],
    portfolios: Sequence["Portfolio"],
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    exchange_id: str = "binance",
    on_bar=None,
) -> None:
    """
    Poll for closed hourly bars and step all agents.

    on_bar(ts, price, results): optional callback for dashboard updates.

    Phase 1 will replace this polling loop with a true WebSocket connection.
    """
    import ccxt

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    from tradegame.features.indicators import compute_features

    print(f"[live] Starting paper loop — {symbol} {timeframe}")
    bar_buffer: list[list] = []

    while True:
        try:
            # Fetch last 300 bars for indicator warmup
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=300)
            if not raw:
                time.sleep(10)
                continue

            # Only act on a newly closed bar
            latest_ts_ms = raw[-1][0]
            df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts")

            features = compute_features(df)
            price = float(df.iloc[-1]["close"])
            ts = df.index[-1]

            results = {}
            for agent, portfolio in zip(agents, portfolios):
                action = agent.observe(features.iloc[:-1])   # exclude the live (open) bar
                portfolio.rebalance(price, action.target_weight, ts)
                results[agent.name] = portfolio.equity(price)

            if on_bar:
                on_bar(ts, price, results)

            # Sleep until the next bar closes
            tf_seconds = exchange.parse_timeframe(timeframe)
            next_close = ts + timedelta(seconds=tf_seconds)
            sleep_secs = (next_close - datetime.now(tz=timezone.utc)).total_seconds()
            time.sleep(max(10, sleep_secs + 2))   # +2s buffer for exchange lag

        except KeyboardInterrupt:
            print("[live] Stopped.")
            break
        except Exception as exc:
            print(f"[live] Error: {exc} — retrying in 30s")
            time.sleep(30)
