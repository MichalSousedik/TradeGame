from __future__ import annotations

import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return equity.iloc[-1] / equity.iloc[0] - 1.0


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    seconds = (equity.index[-1] - equity.index[0]).total_seconds()
    years = seconds / (365.25 * 24 * 3600)
    if years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def sharpe(equity: pd.Series, periods_per_year: int = 8760) -> float:
    rets = equity.pct_change().dropna()
    if rets.std() == 0 or len(rets) < 2:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(periods_per_year))


def sortino(equity: pd.Series, periods_per_year: int = 8760) -> float:
    rets = equity.pct_change().dropna()
    downside = rets[rets < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(rets.mean() / downside.std() * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    dd = (equity - roll_max) / roll_max
    return float(dd.min())


def summary(equity: pd.Series, periods_per_year: int = 8760) -> dict:
    if equity.empty:
        return {}
    return {
        "Total Return": f"{total_return(equity):+.1%}",
        "CAGR":         f"{cagr(equity):+.1%}",
        "Sharpe":       f"{sharpe(equity, periods_per_year):.2f}",
        "Sortino":      f"{sortino(equity, periods_per_year):.2f}",
        "Max Drawdown": f"{max_drawdown(equity):.1%}",
    }
