from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Action:
    """
    Target portfolio weight in [-1, 1].
      +1.0  fully long  (all equity invested in the asset)
       0.0  flat        (all cash, no exposure)
      -1.0  fully short (equity-sized short position)
    """
    target_weight: float

    def __post_init__(self) -> None:
        self.target_weight = max(-1.0, min(1.0, self.target_weight))


class Agent(ABC):
    """Base class for all trading agents."""

    name: str = "Agent"

    @abstractmethod
    def observe(self, window: pd.DataFrame) -> Action:
        """
        Given a window of OHLCV + feature rows (up to and including the last
        *closed* bar), return the desired portfolio weight for the next bar.
        The window is always causal — no future data.
        """

    def fit(self, train: pd.DataFrame) -> None:
        """Optional: train on historical data (no-op for rule-based agents)."""

    def reset(self) -> None:
        """Reset any internal state (called between walk-forward folds)."""
