import numpy as np
import pandas as pd
import pytest

from tradegame.agents.rule_based import ALL_AGENTS, BuyAndHold
from tradegame.engine.backtest import run


def _rising_market(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    log_returns = rng.standard_normal(n) * 0.005 + 0.001   # slight upward drift
    close = 30_000.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.994,
            "close": close,
            "volume": np.ones(n) * 100.0,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_buy_and_hold_profits_in_rising_market():
    hist, trades, m = run(BuyAndHold(), _rising_market(), fee_bps=0, slippage_bps=0)
    assert not hist.empty
    assert hist["equity"].iloc[-1] > 100_000.0
    assert "Total Return" in m


def test_equity_curve_is_continuous():
    hist, _, _ = run(BuyAndHold(), _rising_market())
    assert not hist.empty
    assert not hist["equity"].isna().any()
    assert (hist["equity"] > 0).all()


def test_all_agents_run_without_error():
    ohlcv = _rising_market()
    for AgentClass in ALL_AGENTS:
        agent = AgentClass()
        hist, trades, m = run(agent, ohlcv)
        assert not hist.empty, f"{agent.name} returned empty history"


def test_fees_lower_return_vs_no_fees():
    ohlcv = _rising_market()
    hist_free, _, _ = run(BuyAndHold(), ohlcv, fee_bps=0, slippage_bps=0)
    hist_fee, _, _ = run(BuyAndHold(), ohlcv, fee_bps=10, slippage_bps=5)
    assert hist_free["equity"].iloc[-1] > hist_fee["equity"].iloc[-1]


def test_metrics_keys_present():
    hist, _, m = run(BuyAndHold(), _rising_market())
    for key in ("Total Return", "CAGR", "Sharpe", "Sortino", "Max Drawdown"):
        assert key in m, f"Missing metric key: {key}"
