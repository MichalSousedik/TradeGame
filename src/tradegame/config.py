from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BacktestConfig:
    start: str = "2023-01-01"
    end: str = "2025-01-01"
    initial_capital: float = 100_000.0
    fee_bps: int = 10
    slippage_bps: int = 5


@dataclass
class WalkforwardConfig:
    train_bars: int = 2016  # 12 weeks at 1h
    test_bars: int = 336    # 2 weeks at 1h


@dataclass
class Settings:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    exchange: str = "binance"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walkforward: WalkforwardConfig = field(default_factory=WalkforwardConfig)


def load_settings(path: str | Path = "config.yaml") -> Settings:
    p = Path(path)
    if not p.exists():
        return Settings()

    raw = yaml.safe_load(p.read_text())
    d = raw.get("default", {})
    bt_raw = raw.get("backtest", {})
    wf_raw = raw.get("walkforward", {})

    bt_fields = {f for f in BacktestConfig.__dataclass_fields__}
    wf_fields = {f for f in WalkforwardConfig.__dataclass_fields__}

    return Settings(
        symbol=d.get("symbol", "BTC/USDT"),
        timeframe=d.get("timeframe", "1h"),
        exchange=d.get("exchange", "binance"),
        data_dir=Path(raw.get("data_dir", "data")),
        backtest=BacktestConfig(**{k: v for k, v in bt_raw.items() if k in bt_fields}),
        walkforward=WalkforwardConfig(**{k: v for k, v in wf_raw.items() if k in wf_fields}),
    )
