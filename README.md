# TradeGame

AI-powered paper-trading arena. A population of competing agents trade live crypto data.
The game tracks who makes the most money — and evolves smarter strategies over time.

## Quickstart

```bash
pip install -e ".[dev]"
make run          # opens the Streamlit dashboard
make test         # run test suite
```

## Architecture

```
data/       DuckDB cache (OHLCV history, gitignored)
src/tradegame/
  config.py           settings loader
  data/               CCXT/Binance fetcher + DuckDB cache
  features/           technical indicators (causal, no look-ahead)
  agents/             Agent interface + rule-based baselines
  engine/             portfolio sim, backtest runner, metrics, walk-forward
  live/               WebSocket live loop (Phase 1)
  dashboard/          Streamlit app
tests/
```

## Phases

| Phase | Status | What |
|-------|--------|------|
| 0 — Foundation | ✅ | Rule-based agents, backtest engine, Streamlit dashboard |
| 1 — Live crypto | 🔜 | Binance WebSocket → paper trading |
| 2 — ML agent | 🔜 | XGBoost walk-forward |
| 3 — GA agent | 🔜 | DEAP genetic programming, "breeding" UI |
| 4 — US stocks | 🔜 | Alpaca paper trading |
