"""
Entry point for the live paper-trading loop.

Usage
─────
  python -m tradegame.live.runner
  python -m tradegame.live.runner --symbol ETH/USDT --timeframe 1h
  tradegame-live                          # after pip install -e .

The loop fetches the last 300 bars for indicator warmup, then switches to
a Binance WebSocket feed.  Equity snapshots are written to data/live.db
so the Streamlit dashboard can display them in real time.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Allow running as `python -m tradegame.live.runner` from the project root
_src = Path(__file__).resolve().parents[3]
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TradeGame live paper-trading loop")
    p.add_argument("--symbol", default=None, help="e.g. BTC/USDT")
    p.add_argument("--timeframe", default=None, help="e.g. 1h, 4h, 1d")
    p.add_argument("--exchange", default=None, help="ccxt exchange id (default: binance)")
    p.add_argument("--capital", type=float, default=None, help="Initial capital per agent")
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to config.yaml (default: config.yaml in cwd)"
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from tradegame.config import load_settings
    from tradegame.data.cache import DataCache
    from tradegame.data.crypto import CryptoSource
    from tradegame.agents.rule_based import ALL_AGENTS
    from tradegame.engine.portfolio import Portfolio
    from tradegame.live.loop import LiveLoop
    from tradegame.live.state import LiveState

    settings = load_settings(args.config)

    symbol = args.symbol or settings.symbol
    timeframe = args.timeframe or settings.timeframe
    exchange_id = args.exchange or settings.exchange
    capital = args.capital or settings.backtest.initial_capital

    print(f"[runner] symbol={symbol}  timeframe={timeframe}  "
          f"exchange={exchange_id}  capital=${capital:,.0f}")

    # ── warmup: fetch last 300 bars from cache or REST ──────────────────────
    data_dir = Path(args.config).parent / settings.data_dir
    print(f"[runner] Fetching warmup history from {exchange_id}…")

    source = CryptoSource(exchange_id)
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=30)     # 30 days = 720 h-bars, enough for EMA-200

    # Use context manager so the file lock is released before the WebSocket loop
    # starts — otherwise the dashboard can't open cache.db simultaneously.
    with DataCache(data_dir) as cache:
        warmup_df = cache.get_or_fetch(source, symbol, timeframe, start, end)

    if warmup_df is None or warmup_df.empty:
        print("[runner] ERROR: failed to fetch warmup data. Check internet connection.")
        sys.exit(1)

    print(f"[runner] Warmup: {len(warmup_df)} bars loaded "
          f"({warmup_df.index[0]} → {warmup_df.index[-1]})")

    # ── agents & portfolios ──────────────────────────────────────────────────
    agents = [cls() for cls in ALL_AGENTS]
    portfolios = [
        Portfolio(capital, settings.backtest.fee_bps, settings.backtest.slippage_bps)
        for _ in agents
    ]

    # ── live state ───────────────────────────────────────────────────────────
    state = LiveState(data_dir)
    state.set_meta("symbol", symbol)
    state.set_meta("timeframe", timeframe)
    state.set_meta("exchange", exchange_id)
    state.set_meta("started_at", datetime.now(tz=timezone.utc).isoformat())
    state.set_meta("status", "running")

    # ── start loop ───────────────────────────────────────────────────────────
    loop = LiveLoop(agents, portfolios, state, symbol, timeframe, warmup_df)
    loop.start(exchange_id)


if __name__ == "__main__":
    main()
