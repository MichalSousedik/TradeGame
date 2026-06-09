"""TradeGame — Streamlit dashboard (Phase 0 + Phase 1 live tab)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow launch from any directory when not installed
_src = Path(__file__).resolve().parents[2]   # .../TradeGame/src
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from tradegame.config import load_settings
from tradegame.data.cache import DataCache
from tradegame.data.crypto import CryptoSource
from tradegame.agents.rule_based import ALL_AGENTS
from tradegame.engine.backtest import run as run_backtest

st.set_page_config(
    page_title="TradeGame",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── settings ──────────────────────────────────────────────────────────────────
_CFG_PATH = Path(__file__).resolve().parents[3] / "config.yaml"
settings = load_settings(_CFG_PATH if _CFG_PATH.exists() else "config.yaml")
_DATA_DIR = (
    _CFG_PATH.parent / settings.data_dir
    if _CFG_PATH.exists()
    else settings.data_dir
)

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

    agent_names = [cls().name for cls in ALL_AGENTS]
    selected = st.multiselect("Agents", agent_names, default=agent_names)

    run_btn = st.button("▶  Run Backtest", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption(
        f"Capital: ${settings.backtest.initial_capital:,.0f}  "
        f"| Fee: {settings.backtest.fee_bps / 100:.2f}%  "
        f"| Slippage: {settings.backtest.slippage_bps / 100:.2f}%"
    )

# ── page tabs ─────────────────────────────────────────────────────────────────
tab_bt, tab_live = st.tabs(["📊 Backtest", "📡 Live"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_bt:
    st.header("AI Trading Arena — Backtest")
    st.caption("Rule-based agents vs Buy & Hold on historical crypto data.")

    if not run_btn and "bt_results" not in st.session_state:
        st.info("👈 Set a date range and click **Run Backtest** in the sidebar.")
        st.stop()

    if run_btn:
        start_dt = datetime(start_date.year, start_date.month, start_date.day)
        end_dt = datetime(end_date.year, end_date.month, end_date.day)

        with st.spinner("Fetching market data (cached after first run)…"):
            try:
                with DataCache(_DATA_DIR) as cache:
                    ohlcv = cache.get_or_fetch(
                        CryptoSource(settings.exchange), symbol, timeframe, start_dt, end_dt
                    )
            except Exception as exc:
                st.error(f"Data fetch failed: {exc}")
                st.stop()

        if ohlcv is None or ohlcv.empty:
            st.error("No data returned. Check your internet connection.")
            st.stop()

        agents = [cls() for cls in ALL_AGENTS if cls().name in selected]
        results: dict[str, dict] = {}
        prog = st.progress(0.0, "Running backtests…")

        for idx, agent in enumerate(agents):
            hist, trades, m = run_backtest(
                agent, ohlcv,
                initial_capital=settings.backtest.initial_capital,
                fee_bps=settings.backtest.fee_bps,
                slippage_bps=settings.backtest.slippage_bps,
                timeframe=timeframe,
            )
            results[agent.name] = {
                "equity": hist["equity"] if not hist.empty else pd.Series(dtype=float),
                "trades": trades,
                "metrics": m,
            }
            prog.progress((idx + 1) / len(agents))

        prog.empty()
        st.session_state["bt_results"] = results
        st.session_state["bt_ohlcv"] = ohlcv

    results = st.session_state["bt_results"]
    ohlcv_bt = st.session_state.get("bt_ohlcv", pd.DataFrame())

    # Leaderboard
    st.subheader("🏆 Leaderboard")
    board = [{"Agent": n, **r["metrics"]} for n, r in results.items()]
    if board:
        st.dataframe(pd.DataFrame(board).set_index("Agent"), use_container_width=True)

    # Equity curves
    st.subheader("📊 Equity Curves")
    fig_eq = go.Figure()
    for name, r in results.items():
        eq = r["equity"]
        if not eq.empty:
            fig_eq.add_trace(go.Scatter(
                x=eq.index, y=eq, name=name, mode="lines",
                hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
            ))
    fig_eq.update_layout(
        xaxis_title="Date", yaxis_title="Portfolio Value ($)",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", y=-0.18),
        margin=dict(l=0, r=0, t=10, b=10),
    )
    fig_eq.update_yaxes(tickformat="$,.0f")
    st.plotly_chart(fig_eq, use_container_width=True)

    # Drawdown
    st.subheader("📉 Drawdown")
    fig_dd = go.Figure()
    for name, r in results.items():
        eq = r["equity"]
        if not eq.empty:
            dd = (eq - eq.cummax()) / eq.cummax()
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd, name=name, mode="lines", fill="tozeroy",
                hovertemplate="%{y:.1%}<extra>%{fullData.name}</extra>",
            ))
    fig_dd.update_layout(
        yaxis_tickformat=".0%", height=260, hovermode="x unified",
        showlegend=False, margin=dict(l=0, r=0, t=10, b=10),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Price chart with trade markers
    if not ohlcv_bt.empty:
        st.subheader("🕯️ Price + Trades")
        agent_pick = st.selectbox("Show trades for", list(results.keys()))
        trades_df = results[agent_pick]["trades"]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=ohlcv_bt.index, y=ohlcv_bt["close"],
            name="Price", mode="lines", line=dict(color="#888", width=1),
        ))
        if trades_df is not None and not trades_df.empty:
            buys = trades_df[trades_df["delta_units"] > 0]
            sells = trades_df[trades_df["delta_units"] < 0]
            if not buys.empty:
                fig_p.add_trace(go.Scatter(
                    x=buys["ts"], y=buys["exec_price"], mode="markers",
                    name="Buy", marker=dict(symbol="triangle-up", color="green", size=10),
                ))
            if not sells.empty:
                fig_p.add_trace(go.Scatter(
                    x=sells["ts"], y=sells["exec_price"], mode="markers",
                    name="Sell", marker=dict(symbol="triangle-down", color="red", size=10),
                ))
        fig_p.update_layout(
            height=350, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=0, r=0, t=10, b=10),
        )
        st.plotly_chart(fig_p, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — LIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab_live:
    st.header("Live Paper Trading")

    from tradegame.live.state import LiveReader

    # Auto-refresh every 30 seconds while on this tab
    refresh_secs = st.sidebar.slider(
        "Live auto-refresh (s)", 15, 120, 30,
        help="How often the Live tab re-fetches from live.db"
    )
    st.markdown(
        f'<meta http-equiv="refresh" content="{refresh_secs}">',
        unsafe_allow_html=True,
    )

    reader = LiveReader(_DATA_DIR)

    if not reader.is_available:
        st.info(
            "**Live loop is not running.**\n\n"
            "Start it in a separate terminal:\n"
            "```\n"
            "tradegame-live\n"
            "# or: python -m tradegame.live.runner\n"
            "```\n"
            "The dashboard will update automatically once the first bar closes."
        )
    else:
        # ── status bar ────────────────────────────────────────────────────────
        live_symbol = reader.get_meta("symbol") or "–"
        live_tf = reader.get_meta("timeframe") or "–"
        live_status = reader.get_meta("status") or "unknown"
        last_price_str = reader.get_meta("last_price")
        last_bar_ts_ms = reader.get_meta("last_bar_ts")

        last_price = float(last_price_str) if last_price_str else None
        last_bar_ts = (
            pd.Timestamp(int(last_bar_ts_ms), unit="ms", tz="UTC")
            if last_bar_ts_ms else None
        )

        status_color = "🟢" if live_status == "running" else "🔴"
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", f"{status_color} {live_status.capitalize()}")
        col2.metric("Symbol / TF", f"{live_symbol} {live_tf}")
        col3.metric(
            "Last Price",
            f"${last_price:,.2f}" if last_price else "–"
        )
        col4.metric(
            "Last Bar",
            last_bar_ts.strftime("%Y-%m-%d %H:%M UTC") if last_bar_ts else "–"
        )

        st.markdown("---")

        # ── latest snapshot ───────────────────────────────────────────────────
        latest = reader.read_latest_snapshot()
        if not latest.empty:
            st.subheader("💰 Current Equity")
            display = latest[["agent_name", "equity", "position", "cash"]].copy()
            display["equity"] = display["equity"].map("${:,.2f}".format)
            display["cash"] = display["cash"].map("${:,.2f}".format)
            display["position"] = display["position"].map("{:.6f}".format)
            display.columns = ["Agent", "Equity", "Position (base)", "Cash"]
            st.dataframe(display.set_index("Agent"), use_container_width=True)
        else:
            st.warning("Waiting for first closed bar…")

        # ── equity curves ─────────────────────────────────────────────────────
        history = reader.read_equity_history(last_n=500)
        if not history.empty:
            st.subheader("📈 Live Equity Curves")
            fig_live = go.Figure()
            for agent_name in history["agent_name"].unique():
                agent_df = history[history["agent_name"] == agent_name].sort_values("ts")
                fig_live.add_trace(go.Scatter(
                    x=agent_df["ts"], y=agent_df["equity"],
                    name=agent_name, mode="lines",
                    hovertemplate="%{y:$,.0f}<extra>%{fullData.name}</extra>",
                ))
            fig_live.update_layout(
                xaxis_title="Time (UTC)", yaxis_title="Portfolio Value ($)",
                hovermode="x unified", height=400,
                legend=dict(orientation="h", y=-0.2),
                margin=dict(l=0, r=0, t=10, b=10),
            )
            fig_live.update_yaxes(tickformat="$,.0f")
            st.plotly_chart(fig_live, use_container_width=True)

            # Price chart
            price_df = (
                history[["ts", "price"]]
                .drop_duplicates("ts")
                .sort_values("ts")
            )
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=price_df["ts"], y=price_df["price"],
                name="Price", mode="lines",
                line=dict(color="#f0a500", width=1.5),
                hovertemplate="%{y:$,.2f}<extra>Price</extra>",
            ))
            fig_price.update_layout(
                xaxis_title="Time (UTC)", yaxis_title="Price ($)",
                height=250, hovermode="x unified",
                margin=dict(l=0, r=0, t=10, b=10),
            )
            st.plotly_chart(fig_price, use_container_width=True)

        st.caption(f"Auto-refreshing every {refresh_secs}s  |  data from `data/live.db`")
        reader.close()

st.markdown("---")
st.caption("Phase 0 — rule-based agents  |  Phase 1 — live WebSocket feed  "
           "|  Phase 2: ML  |  Phase 3: Genetic Programming")
