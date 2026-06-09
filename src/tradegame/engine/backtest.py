from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tradegame.engine.portfolio import Portfolio
from tradegame.engine import metrics as _metrics

if TYPE_CHECKING:
    from tradegame.agents.base import Agent


def run(
    agent: "Agent",
    ohlcv: pd.DataFrame,
    initial_capital: float = 100_000.0,
    fee_bps: int = 10,
    slippage_bps: int = 5,
    window_size: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Simulate agent on historical OHLCV data.

    For each bar i the agent sees features computed on rows 0..i-1 (causal window),
    then executes at bar i's close price.

    Returns: (equity_history, trades, metrics_dict)
    """
    from tradegame.features.indicators import compute_features

    features = compute_features(ohlcv)
    portfolio = Portfolio(initial_capital, fee_bps, slippage_bps)

    for i in range(1, len(features)):
        window_start = max(0, i - window_size)
        window = features.iloc[window_start:i]      # rows up to (not incl.) bar i
        price = float(ohlcv.iloc[i]["close"])
        ts = ohlcv.index[i]
        action = agent.observe(window)
        portfolio.rebalance(price, action.target_weight, ts)

    hist = portfolio.history()
    trades = portfolio.trades()
    m = _metrics.summary(hist["equity"]) if not hist.empty else {}
    return hist, trades, m
