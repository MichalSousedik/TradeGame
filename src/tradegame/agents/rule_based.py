from __future__ import annotations

import pandas as pd

from tradegame.agents.base import Action, Agent


class BuyAndHold(Agent):
    """Always fully invested. The benchmark every other agent must beat."""

    name = "Buy & Hold"

    def observe(self, window: pd.DataFrame) -> Action:
        return Action(1.0)


class EMACross(Agent):
    """Long when EMA(12) > EMA(26), flat otherwise."""

    name = "EMA Cross (12/26)"

    def observe(self, window: pd.DataFrame) -> Action:
        row = window.iloc[-1]
        ema_fast = row.get("ema_fast", float("nan"))
        ema_slow = row.get("ema_slow", float("nan"))
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return Action(0.0)
        return Action(1.0 if ema_fast > ema_slow else 0.0)


class RSIMeanReversion(Agent):
    """
    Buy when RSI dips below 30 (oversold); exit when RSI rises above 70 (overbought).
    Holds position in between.
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
            self._weight = 0.0
        return Action(self._weight)

    def reset(self) -> None:
        self._weight = 0.0


class BollingerReversion(Agent):
    """
    Buy when price crosses below lower Bollinger Band; exit above upper band.
    Holds in between.
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
            self._weight = 0.0
        return Action(self._weight)

    def reset(self) -> None:
        self._weight = 0.0


class MACDTrend(Agent):
    """Long when MACD line crosses above signal; flat when it crosses below."""

    name = "MACD Trend"

    def __init__(self) -> None:
        self._weight = 0.0

    def observe(self, window: pd.DataFrame) -> Action:
        if len(window) < 2 or "macd" not in window.columns:
            return Action(self._weight)
        macd_now = window["macd"].iloc[-1]
        sig_now = window["macd_signal"].iloc[-1]
        macd_prev = window["macd"].iloc[-2]
        sig_prev = window["macd_signal"].iloc[-2]
        if pd.isna(macd_now) or pd.isna(sig_now):
            return Action(self._weight)
        if macd_prev <= sig_prev and macd_now > sig_now:
            self._weight = 1.0
        elif macd_prev >= sig_prev and macd_now < sig_now:
            self._weight = 0.0
        return Action(self._weight)

    def reset(self) -> None:
        self._weight = 0.0


ALL_AGENTS = [BuyAndHold, EMACross, RSIMeanReversion, BollingerReversion, MACDTrend]
