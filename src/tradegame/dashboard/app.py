"""TradeGame — Streamlit dashboard."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="TradeGame",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── resolve project root so imports work regardless of launch directory ───────
import sys
_src = Path(__file__).resolve().parents[3]   # …/TradeGame/src
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tradegame.config import load_settings
from tradegame.data.cache import DataCache
from tradegame.data.crypto import CryptoSource
from tradegame.agents.rule_based import ALL_AGENTS
from tradegame.engine.backtest import run as run_backtest
from tradegame.engine import metrics as _metrics

# ── settings ──────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
settings = load_settings(_CFG_PATH if _CFG_PATH.exists() else "config.yaml")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ TradeGame")
    st.markdown("---")

    symbol = st.selectbox("Symbol", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
    timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d"])

    default_end = datetime.utcnow().date()
    default_start = (datetime.utcnow() - timedelta(days=365)).date()
    start_date = st.date_input("From", value=default_start)
    end_date = st.date_input("To", value=default_end)

    agent_names = [a.name for a in [cls() for cls in ALL_AGENTS]]
    selected = st.multiselect("Agents", agent_names, default=agent_names)

    run_btn = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(f"Capital: ${settings.backtest.initial_capital:,.0f}  "
               f"| Fee: {settings.backtest.fee_bps/100:.2f}%  "
               f"| Slippage: {settings.backtest.slippage_bps/100:.2f}%")

# ── main area ─────────────────────────────────────────────────────────────────
st.title("📈 TradeGame — AI Trading Arena")
st.caption("Baseline rule-based agents competing against each other and Buy & Hold.")

if not run_btn and "results" not in st.session_state:
    st.info("👈 Configure a simulation in the sidebar and click **Run Simulation**.")
    st.stop()

if run_btn:
    start_dt = datetime(start_date.year, start_date.month, start_date.day)
    end_dt = datetime(end_date.year, end_date.month, end_date.day)

    data_dir = _CFG_PATH.parent / settings.data_dir if _CFG_PATH.exists() else settings.data_dir

    with st.spinner("Fetching market data (cached after first run)…"):
        try:
            cache = DataCache(data_dir)
            source = CryptoSource(settings.exchange)
            ohlcv = cache.get_or_fetch(source, symbol, timeframe, start_dt, end_dt)
        except Exception as exc:
            st.error(f"Data fetch failed: {exc}")
            st.stop()

    if ohlcv is None or ohlcv.empty:
        st.error("No data returned. Check your internet connection.")
        st.stop()

    agents = [cls() for cls in ALL_AGENTS if cls().name in selected]
    results: dict[str, dict] = {}
    progress = st.progress(0.0, "Running backtests…")

    for idx, agent in enumerate(agents):
        hist, trades, m = run_backtest(
            agent, ohlcv,
            initial_capital=settings.backtest.initial_capital,
            fee_bps=settings.backtest.fee_bps,
            slippage_bps=settings.backtest.slippage_bps,
        )
        results[agent.name] = {
            "equity": hist["equity"] if not hist.empty else pd.Series(dtype=float),
            "trades": trades,
            "metrics": m,
        }
        progress.progress((idx + 1) / len(agents))

    progress.empty()
    st.session_state["results"] = results
    st.session_state["ohlcv"] = ohlcv

results = st.session_state["results"]
ohlcv = st.session_state.get("ohlcv", pd.DataFrame())

# ── leaderboard ───────────────────────────────────────────────────────────────
st.subheader("🏆 Leaderboard")
board_rows = []
for name, r in results.items():
    row = {"Agent": name, **r["metrics"]}
    board_rows.append(row)

if board_rows:
    board_df = pd.DataFrame(board_rows).set_index("Agent")
    st.dataframe(board_df, use_container_width=True)

# ── equity curves ─────────────────────────────────────────────────────────────
st.subheader("📊 Equity Curves")
fig_eq = go.Figure()
for name, r in results.items():
    eq = r["equity"]
    if not eq.empty:
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq,
            name=name, mode="lines",
            hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
        ))

fig_eq.update_layout(
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    height=420,
    legend=dict(orientation="h", y=-0.15),
    margin=dict(l=0, r=0, t=10, b=10),
)
fig_eq.update_yaxes(tickformat="$,.0f")
st.plotly_chart(fig_eq, use_container_width=True)

# ── drawdown ──────────────────────────────────────────────────────────────────
st.subheader("📉 Drawdown")
fig_dd = go.Figure()
for name, r in results.items():
    eq = r["equity"]
    if not eq.empty:
        dd = (eq - eq.cummax()) / eq.cummax()
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd,
            name=name, mode="lines",
            fill="tozeroy",
            hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
        ))

fig_dd.update_layout(
    yaxis_tickformat=".0%",
    height=260,
    hovermode="x unified",
    showlegend=False,
    margin=dict(l=0, r=0, t=10, b=10),
)
st.plotly_chart(fig_dd, use_container_width=True)

# ── price chart with trade markers ───────────────────────────────────────────
if not ohlcv.empty:
    st.subheader("🕯️ Price + Trades")
    agent_for_trades = st.selectbox("Show trades for agent", list(results.keys()))
    trades_df = results[agent_for_trades]["trades"]

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=ohlcv.index, y=ohlcv["close"],
        name="Price", mode="lines",
        line=dict(color="#888", width=1),
    ))

    if trades_df is not None and not trades_df.empty:
        buys = trades_df[trades_df["delta_units"] > 0]
        sells = trades_df[trades_df["delta_units"] < 0]
        if not buys.empty:
            fig_p.add_trace(go.Scatter(
                x=buys["ts"], y=buys["exec_price"],
                mode="markers", name="Buy",
                marker=dict(symbol="triangle-up", color="green", size=10),
            ))
        if not sells.empty:
            fig_p.add_trace(go.Scatter(
                x=sells["ts"], y=sells["exec_price"],
                mode="markers", name="Sell",
                marker=dict(symbol="triangle-down", color="red", size=10),
            ))

    fig_p.update_layout(
        height=350,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_p, use_container_width=True)

st.markdown("---")
st.caption("Phase 0 — rule-based agents. Phase 1: live feed. Phase 2: ML. Phase 3: Genetic Programming.")
