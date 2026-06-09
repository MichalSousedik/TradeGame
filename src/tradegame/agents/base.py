from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass
class Action:
    """
    Target portfolio weight in [0, 1].
    0.0 = fully flat (all cash), 1.0 = fully invested long.
    Phase 0 is long-only. Short extension (negative weights) in Phase 1+.
    """
    target_weight: float

    def __post_init__(self) -> None:
        self.target_weight = max(0.0, min(1.0, self.target_weight))


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
