from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class _Trade:
    ts: pd.Timestamp
    delta_units: float
    exec_price: float
    fee: float


class Portfolio:
    """
    Single-asset paper-trading portfolio.

    rebalance() is called on every bar; it either trades or holds.
    Fees are charged on the trade notional value.
    Slippage worsens the execution price in the direction of the trade.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        fee_bps: int = 10,
        slippage_bps: int = 5,
    ) -> None:
        self.initial_capital = initial_capital
        self.cash = float(initial_capital)
        self.position = 0.0           # units of base asset held
        self.fee_bps = fee_bps
        self.slippage_bps = slippage_bps
        self._history: list[dict] = []
        self._trades: list[_Trade] = []

    # ── public API ────────────────────────────────────────────────────────────

    def rebalance(self, price: float, target_weight: float, ts: pd.Timestamp) -> None:
        """Move to target_weight (0–1 fraction of equity) at current price."""
        eq = self.equity(price)
        target_units = (eq * target_weight) / price
        delta_units = target_units - self.position

        if abs(delta_units * price) < eq * 0.001:   # skip dust trades (< 0.1% of equity)
            self._record(ts, price)
            return

        # Slippage: buying costs more, selling earns less
        if delta_units > 0:
            exec_price = price * (1.0 + self.slippage_bps / 10_000.0)
        else:
            exec_price = price * (1.0 - self.slippage_bps / 10_000.0)

        fee = abs(delta_units) * exec_price * self.fee_bps / 10_000.0

        self.position += delta_units
        self.cash -= delta_units * exec_price   # positive when buying (reduces cash)
        self.cash -= fee                        # fee always reduces cash

        self._trades.append(_Trade(ts, delta_units, exec_price, fee))
        self._record(ts, price)

    def equity(self, price: float) -> float:
        return self.cash + self.position * price

    def history(self) -> pd.DataFrame:
        if not self._history:
            return pd.DataFrame()
        return pd.DataFrame(self._history).set_index("ts")

    def trades(self) -> pd.DataFrame:
        if not self._trades:
            return pd.DataFrame()
        return pd.DataFrame([
            {"ts": t.ts, "delta_units": t.delta_units,
             "exec_price": t.exec_price, "fee": t.fee}
            for t in self._trades
        ])

    def reset(self) -> None:
        self.cash = float(self.initial_capital)
        self.position = 0.0
        self._history.clear()
        self._trades.clear()

    # ── private ───────────────────────────────────────────────────────────────

    def _record(self, ts: pd.Timestamp, price: float) -> None:
        self._history.append(
            {"ts": ts, "equity": self.equity(price),
             "cash": self.cash, "position": self.position, "price": price}
        )
