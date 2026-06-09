"""
Critical test: indicators must be causal (no look-ahead bias).
Computing features on N bars vs N+k bars must give identical values for rows 0..N-1.
"""
import numpy as np
import pandas as pd
import pytest

from tradegame.features.indicators import compute_features

INDICATOR_COLS = ["ema_fast", "ema_slow", "rsi", "macd", "macd_signal",
                  "bb_upper", "bb_mid", "bb_lower", "atr", "roc"]


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 30_000.0 + np.cumsum(rng.standard_normal(n) * 200)
    # volume is constant so that _make_ohlcv(300) and _make_ohlcv(400) are
    # identical in rows 0..299 — required by the no-lookahead test setup check
    return pd.DataFrame(
        {
            "open": close * 0.9990,
            "high": close * 1.0015,
            "low": close * 0.9985,
            "close": close,
            "volume": np.ones(n) * 100.0,
        },
        index=pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC"),
    )


def test_no_lookahead_by_extension():
    """
    Adding future bars must not change any past indicator value.
    This is the single most important backtest correctness test.
    """
    partial = _make_ohlcv(300)
    extended = _make_ohlcv(400)          # 100 extra future bars
    assert (partial == extended.iloc[:300]).all().all(), "OHLCV mismatch — test setup error"

    feat_partial = compute_features(partial)
    feat_extended = compute_features(extended)

    check_row = 299   # last row of partial
    for col in INDICATOR_COLS:
        if col not in feat_partial.columns:
            continue
        v_partial = feat_partial[col].iloc[check_row]
        v_extended = feat_extended[col].iloc[check_row]
        if pd.isna(v_partial) and pd.isna(v_extended):
            continue
        assert abs(v_partial - v_extended) < 1e-8, (
            f"Look-ahead bias detected in '{col}': "
            f"partial={v_partial:.6f}, extended={v_extended:.6f}"
        )


def test_indicators_are_finite_after_warmup():
    """All indicators should produce finite values after sufficient warmup (200 bars)."""
    df = _make_ohlcv(300)
    feat = compute_features(df)
    tail = feat.iloc[210:]   # well past the EMA-200 warmup period
    for col in INDICATOR_COLS:
        if col not in feat.columns:
            continue
        assert tail[col].notna().all(), f"NaN in '{col}' after warmup"
