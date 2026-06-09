"""
Technical indicators — implemented from scratch (no pandas-ta dependency).
All indicators are causal: value at row i uses only rows 0..i.

compute_features(df, timeframe) is the single public entry point.
Indicator periods scale with timeframe so the same agent logic works
correctly on 1h, 4h, and 1d bars.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ── indicator period presets per timeframe ────────────────────────────────────

def _params(timeframe: str) -> dict:
    """
    Return indicator periods suited to the given bar timeframe.

    Rule of thumb: periods represent *calendar days* worth of data.
      EMA fast  ~  2 days
      EMA slow  ~  8 days
      EMA trend ~ 21 days
      RSI       ~  1 day
      MACD      ~  2 / 4.3 / 1.5 days
      Bollinger ~ 20 days
      ATR       ~ 14 days
      ROC       ~ 10 days
    """
    if timeframe in ("1h", "2h"):
        return dict(
            ema_fast=48,  ema_slow=200,  ema_trend=500,
            rsi=24,
            macd_fast=48, macd_slow=104, macd_sig=36,
            bb=480, atr=336, roc=240,
        )
    if timeframe in ("4h", "6h", "8h", "12h"):
        return dict(
            ema_fast=12,  ema_slow=50,   ema_trend=200,
            rsi=14,
            macd_fast=12, macd_slow=26,  macd_sig=9,
            bb=120, atr=84,  roc=60,
        )
    # 1d, 3d, 1w, or unknown — original textbook values
    return dict(
        ema_fast=12,  ema_slow=26,   ema_trend=200,
        rsi=14,
        macd_fast=12, macd_slow=26,  macd_sig=9,
        bb=20,  atr=14,  roc=10,
    )


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

def compute_features(df: pd.DataFrame, timeframe: str = "1d") -> pd.DataFrame:
    """
    Append indicator columns to a copy of df.

    Periods are scaled to the timeframe so that EMA fast/slow represent
    roughly the same calendar-time horizon regardless of bar size.

    Input : OHLCV DataFrame, UTC DatetimeIndex.
    Output: same rows + [ema_fast, ema_slow, ema_200, rsi,
                          macd, macd_signal, macd_hist,
                          bb_upper, bb_mid, bb_lower, atr, roc].
    """
    p = _params(timeframe)
    out = df.copy()
    c = df["close"]
    out["ema_fast"] = ema(c, p["ema_fast"])
    out["ema_slow"] = ema(c, p["ema_slow"])
    out["ema_200"]  = ema(c, p["ema_trend"])
    out["rsi"]      = rsi(c, p["rsi"])
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(
        c, p["macd_fast"], p["macd_slow"], p["macd_sig"]
    )
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(c, p["bb"])
    out["atr"] = atr(df["high"], df["low"], c, p["atr"])
    out["roc"] = roc(c, p["roc"])
    return out
