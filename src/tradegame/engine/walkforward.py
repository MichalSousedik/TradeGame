"""
Walk-forward validation — the only honest way to evaluate a fitting agent.

Each fold: train agent on train_slice → evaluate on the following test_slice.
The combined out-of-sample equity is what the leaderboard shows.
"""
from __future__ import annotations

from typing import Callable, Iterator, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tradegame.agents.base import Agent
    from tradegame.config import Settings


def splits(
    n_bars: int, train_bars: int, test_bars: int
) -> Iterator[tuple[slice, slice]]:
    """Yield (train_slice, test_slice) non-overlapping walk-forward windows."""
    start = 0
    while start + train_bars + test_bars <= n_bars:
        yield (
            slice(start, start + train_bars),
            slice(start + train_bars, start + train_bars + test_bars),
        )
        start += test_bars  # anchored=False: window rolls forward by test_bars


def run(
    agent_factory: Callable[[], "Agent"],
    ohlcv: pd.DataFrame,
    settings: "Settings",
    initial_capital: float = 100_000.0,
    fee_bps: int = 10,
    slippage_bps: int = 5,
) -> tuple[pd.Series, dict]:
    """
    Run walk-forward validation.

    agent_factory is called once per fold to get a fresh agent.
    Returns (combined OOS equity Series, aggregate metrics dict).
    """
    from tradegame.engine.backtest import run as _run_backtest
    from tradegame.engine import metrics as _metrics

    train_bars = settings.walkforward.train_bars
    test_bars = settings.walkforward.test_bars

    oos_equity_parts: list[pd.Series] = []

    for train_sl, test_sl in splits(len(ohlcv), train_bars, test_bars):
        agent = agent_factory()
        train_df = ohlcv.iloc[train_sl]
        test_df = ohlcv.iloc[test_sl]
        agent.fit(train_df)
        hist, _, _ = _run_backtest(agent, test_df, initial_capital, fee_bps, slippage_bps)
        if not hist.empty:
            oos_equity_parts.append(hist["equity"])

    if not oos_equity_parts:
        return pd.Series(dtype=float), {}

    combined = pd.concat(oos_equity_parts)
    return combined, _metrics.summary(combined)
