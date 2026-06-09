"""
Technical indicators — implemented from scratch (no pandas-ta dependency).
All indicators are causal: value at row i uses only rows 0..i.
compute_features() is the single public entry point.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ── primitives ────────────────────────────────────────────────────────────────

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(com=length - 1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length - 1, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(
    series: pd.Series, length: int = 20, std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, mid, lower)."""
    mid = sma(series, length)
    sigma = series.rolling(length).std()
    return mid + std * sigma, mid, mid - std * sigma


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(com=length - 1, min_periods=length).mean()


def roc(series: pd.Series, length: int = 10) -> pd.Series:
    return (series / series.shift(length) - 1.0) * 100.0


# ── public entry point ────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all indicator columns to a copy of df.
    Input : OHLCV DataFrame, UTC DatetimeIndex.
    Output: same rows + columns [ema_fast, ema_slow, ema_200, rsi,
                                  macd, macd_signal, macd_hist,
                                  bb_upper, bb_mid, bb_lower, atr, roc].
    """
    out = df.copy()
    c = df["close"]
    out["ema_fast"] = ema(c, 12)
    out["ema_slow"] = ema(c, 26)
    out["ema_200"] = ema(c, 200)
    out["rsi"] = rsi(c, 14)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(c)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(c)
    out["atr"] = atr(df["high"], df["low"], c)
    out["roc"] = roc(c, 10)
    return out
