import pandas as pd
import pytest

from tradegame.engine.portfolio import Portfolio


TS = pd.Timestamp("2023-01-01", tz="UTC")


def test_flat_position_unchanged_equity():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    p.rebalance(price=10.0, target_weight=0.0, ts=TS)
    assert p.equity(10.0) == pytest.approx(100_000.0)
    assert p.equity(99_999.0) == pytest.approx(100_000.0)   # flat = price-insensitive


def test_full_long_doubles_on_2x_price():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    p.rebalance(price=10.0, target_weight=1.0, ts=TS)
    assert p.equity(10.0) == pytest.approx(100_000.0, rel=1e-4)
    assert p.equity(20.0) == pytest.approx(200_000.0, rel=1e-4)


def test_fees_reduce_equity():
    p = Portfolio(100_000, fee_bps=100, slippage_bps=0)   # 1% fee
    p.rebalance(price=10.0, target_weight=1.0, ts=TS)
    assert p.equity(10.0) < 100_000.0


def test_slippage_reduces_equity():
    p_no_slip = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    p_slip = Portfolio(100_000, fee_bps=0, slippage_bps=50)   # 0.5% slippage
    p_no_slip.rebalance(10.0, 1.0, TS)
    p_slip.rebalance(10.0, 1.0, TS)
    assert p_slip.equity(10.0) < p_no_slip.equity(10.0)


def test_history_recorded_every_bar():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    for i in range(5):
        p.rebalance(price=float(10 + i), target_weight=0.5, ts=TS + pd.Timedelta(hours=i))
    hist = p.history()
    assert len(hist) == 5


def test_dust_trades_skipped():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    p.rebalance(10.0, 0.5, TS)                    # initial buy
    p.rebalance(10.0, 0.5001, TS)                 # < 0.1% delta → skip
    trades = p.trades()
    assert len(trades) == 1                        # only the initial buy


def test_round_trip_break_even_no_costs():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    ts1 = TS
    ts2 = TS + pd.Timedelta(hours=1)
    p.rebalance(price=10.0, target_weight=1.0, ts=ts1)
    p.rebalance(price=10.0, target_weight=0.0, ts=ts2)   # sell at same price
    assert p.equity(10.0) == pytest.approx(100_000.0, rel=1e-6)


def test_short_profits_when_price_falls():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    p.rebalance(price=10.0, target_weight=-1.0, ts=TS)   # go fully short
    # price drops 20 % → short should gain ~20 %
    assert p.equity(8.0) == pytest.approx(120_000.0, rel=1e-4)


def test_short_loses_when_price_rises():
    p = Portfolio(100_000, fee_bps=0, slippage_bps=0)
    p.rebalance(price=10.0, target_weight=-1.0, ts=TS)
    # price rises 20 % → short should lose ~20 %
    assert p.equity(12.0) == pytest.approx(80_000.0, rel=1e-4)


def test_action_clamps_to_minus_one():
    from tradegame.agents.base import Action
    assert Action(-2.0).target_weight == pytest.approx(-1.0)
    assert Action(-0.5).target_weight == pytest.approx(-0.5)
    assert Action(2.0).target_weight  == pytest.approx(1.0)
