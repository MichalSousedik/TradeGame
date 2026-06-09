from __future__ import annotations

import pandas as pd

from tradegame.agents.base import Action, Agent


class BuyAndHold(Agent):
    """Always fully long. The benchmark every other agent must beat."""

    name = "Buy & Hold"

    def observe(self, window: pd.DataFrame) -> Action:
        return Action(1.0)


class EMACross(Agent):
    """
    Long (+1) when fast EMA is above slow EMA (uptrend).
    Short (-1) when fast EMA is below slow EMA (downtrend).
    """

    name = "EMA Cross"

    def observe(self, window: pd.DataFrame) -> Action:
        row = window.iloc[-1]
        ema_fast = row.get("ema_fast", float("nan"))
        ema_slow = row.get("ema_slow", float("nan"))
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return Action(0.0)
        return Action(1.0 if ema_fast > ema_slow else -1.0)


class RSIMeanReversion(Agent):
    """
    Long (+1) when RSI signals oversold (< 30).
    Short (-1) when RSI signals overbought (> 70).
    Flat (0) in the neutral zone — waits for an extreme.
    """

    name = "RSI Mean Reversion"

    def __init__(self, oversold: float = 30.0, overbought: float = 70.0) -> None:
        self._oversold = oversold
        self._overbought = overbought
        self._weight = 0.0

    def observe(self, window: pd.DataFrame) -> Action:
        rsi_val = window["rsi"].iloc[-1] if "rsi" in window.columns else float("nan")
        if pd.isna(rsi_val):
            return Action(self._weight)
        if rsi_val < self._oversold:
            self._weight = 1.0
        elif rsi_val > self._overbought:
            self._weight = -1.0
        return Action(self._weight)

    def reset(self) -> None:
        self._weight = 0.0


class BollingerReversion(Agent):
    """
    Long (+1) when price dips below lower Bollinger Band (oversold).
    Short (-1) when price rises above upper band (overbought).
    Flat (0) while price is inside the bands.
    """

    name = "Bollinger Reversion"

    def __init__(self) -> None:
        self._weight = 0.0

    def observe(self, window: pd.DataFrame) -> Action:
        row = window.iloc[-1]
        price = row["close"]
        lower = row.get("bb_lower", float("nan"))
        upper = row.get("bb_upper", float("nan"))
        if pd.isna(lower) or pd.isna(upper):
            return Action(self._weight)
        if price < lower:
            self._weight = 1.0
        elif price > upper:
            self._weight = -1.0
        else:
            self._weight = 0.0
        return Action(self._weight)

    def reset(self) -> None:
        self._weight = 0.0


class MACDTrend(Agent):
    """
    Long (+1) on MACD-line bullish crossover above signal.
    Short (-1) on bearish crossover below signal.
    Holds until the next crossover.
    """

    name = "MACD Trend"

    def __init__(self) -> None:
        self._weight = 0.0

    def observe(self, window: pd.DataFrame) -> Action:
        if len(window) < 2 or "macd" not in window.columns:
            return Action(self._weight)
        macd_now  = window["macd"].iloc[-1]
        sig_now   = window["macd_signal"].iloc[-1]
        macd_prev = window["macd"].iloc[-2]
        sig_prev  = window["macd_signal"].iloc[-2]
        if pd.isna(macd_now) or pd.isna(sig_now):
            return Action(self._weight)
        if macd_prev <= sig_prev and macd_now > sig_now:   # bullish cross
            self._weight = 1.0
        elif macd_prev >= sig_prev and macd_now < sig_now: # bearish cross
            self._weight = -1.0
        return Action(self._weight)

    def reset(self) -> None:
        self._weight = 0.0


ALL_AGENTS = [BuyAndHold, EMACross, RSIMeanReversion, BollingerReversion, MACDTrend]
