"""
dashboard.py — InvestEng Streamlit Dashboard
=============================================
Interactive investment intelligence dashboard.
Connects to the FastAPI backend (api.py) for all data and computation.

Sections
--------
  1. Sidebar         — universe selector, strategy picker, date range
  2. Portfolio        — allocation chart, weights table, metrics cards
  3. Efficient Frontier — interactive risk/return curve
  4. Backtesting      — equity curve, drawdown, rolling Sharpe, trade log
  5. Risk Scorecard   — per-ticker risk heatmap and comparison
  6. Correlation      — interactive heatmap

Run with:
    streamlit run dashboard.py
    (Requires api.py running on localhost:8000)
"""

import time
import requests
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title = "InvestEng",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------------

COLORS = {
    "primary"   : "#2563EB",
    "success"   : "#16A34A",
    "danger"    : "#DC2626",
    "warning"   : "#D97706",
    "neutral"   : "#6B7280",
    "bg_card"   : "#F8FAFC",
    "portfolio" : px.colors.qualitative.Plotly,
}

STRATEGY_LABELS = {
    "max_sharpe"   : "📈 Max Sharpe",
    "min_variance" : "🛡️ Min Variance",
    "mean_variance": "⚖️ Mean-Variance",
    "risk_parity"  : "🔵 Risk Parity",
    "black_litterman": "🧠 Black-Litterman",
}

REBALANCE_LABELS = {
    "monthly"    : "Monthly",
    "quarterly"  : "Quarterly",
    "semi-annual": "Semi-Annual",
    "annual"     : "Annual",
}

ASSET_UNIVERSE = {
    "us_stocks": ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","BRK-B","UNH"],
    "etfs"     : ["SPY","QQQ","VTI","BND","GLD","IWM","VEA","VWO"],
    "crypto"   : ["BTC-USD","ETH-USD","BNB-USD","SOL-USD","XRP-USD"],
    "indices"  : ["^GSPC","^IXIC","^DJI","^NSEI","^BSESN","^FTSE","^N225"],
}


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------

def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to InvestEng API. Start the backend: `uvicorn api:app --reload`")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to InvestEng API. Start the backend: `uvicorn api:app --reload`")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------

def pct(v, decimals=2):
    if v is None:
        return "—"
    return f"{v * 100:+.{decimals}f}%"

def pct_abs(v, decimals=2):
    if v is None:
        return "—"
    return f"{v * 100:.{decimals}f}%"

def flt(v, decimals=3):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"

def usd(v):
    if v is None:
        return "—"
    return f"${v:,.2f}"

def colour_metric(v, good_positive=True):
    if v is None:
        return "normal"
    return "normal" if (v >= 0) == good_positive else "inverse"


# ---------------------------------------------------------------------------
# Metric Card Row
# ---------------------------------------------------------------------------

def metric_row(metrics: dict):
    """Render a row of 5 metric cards."""
    cols = st.columns(5)
    items = [
        ("CAGR",          pct(metrics.get("cagr")),          None),
        ("Sharpe Ratio",  flt(metrics.get("sharpe")),         None),
        ("Max Drawdown",  pct(metrics.get("max_drawdown")),   None),
        ("Volatility",    pct_abs(metrics.get("annual_vol")), None),
        ("Alpha",         pct(metrics.get("alpha")),           None),
    ]
    for col, (label, value, delta) in zip(cols, items):
        col.metric(label, value, delta)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> dict:
    """Render sidebar controls and return selected parameters."""
    st.sidebar.image("https://via.placeholder.com/200x50?text=InvestEng", width=200)
    st.sidebar.title("⚙️ Configuration")

    # ── Asset Universe ─────────────────────────────────────────────────────
    st.sidebar.markdown("### 🌐 Asset Universe")
    selected_classes = st.sidebar.multiselect(
        "Asset Classes",
        options=list(ASSET_UNIVERSE.keys()),
        default=["us_stocks", "etfs"],
        format_func=lambda x: x.replace("_", " ").title(),
    )

    all_tickers = []
    for cls in selected_classes:
        all_tickers.extend(ASSET_UNIVERSE.get(cls, []))

    selected_tickers = st.sidebar.multiselect(
        "Select Tickers",
        options=all_tickers,
        default=all_tickers[:8] if len(all_tickers) >= 8 else all_tickers,
    )

    # ── Strategy ───────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🧠 Strategy")
    strategy = st.sidebar.selectbox(
        "Optimisation Strategy",
        options=list(STRATEGY_LABELS.keys()),
        format_func=lambda x: STRATEGY_LABELS[x],
    )

    use_risk_profile = st.sidebar.toggle("Use Risk Profile Instead", value=False)
    risk_score = None
    if use_risk_profile:
        risk_score = st.sidebar.slider(
            "Risk Score (0=Conservative, 100=Aggressive)",
            min_value=0, max_value=100, value=50, step=5,
        )
        profile_labels = {10: "Conservative", 30: "Moderate", 50: "Balanced",
                          70: "Growth", 90: "Aggressive"}
        nearest = min(profile_labels.keys(), key=lambda x: abs(x - risk_score))
        st.sidebar.caption(f"📊 Profile: **{profile_labels[nearest]}**")

    # ── Constraints ────────────────────────────────────────────────────────
    st.sidebar.markdown("### 📏 Constraints")
    min_wt = st.sidebar.slider("Min Weight per Asset", 0.0, 0.15, 0.01, 0.01,
                                format="%.2f")
    max_wt = st.sidebar.slider("Max Weight per Asset", 0.10, 1.0, 0.40, 0.05,
                                format="%.2f")

    # ── Backtest Settings ──────────────────────────────────────────────────
    st.sidebar.markdown("### 📅 Backtest")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Start", value=date(2020, 1, 1))
    end_date   = col2.date_input("End",   value=date.today())

    rebalance   = st.sidebar.selectbox(
        "Rebalance Frequency",
        options=list(REBALANCE_LABELS.keys()),
        format_func=lambda x: REBALANCE_LABELS[x],
        index=1,
    )
    train_years = st.sidebar.slider("Train Window (years)", 1, 5, 1)

    st.sidebar.markdown("### 💰 Capital & Costs")
    initial_capital  = st.sidebar.number_input(
        "Initial Capital (USD)", min_value=1000, max_value=10_000_000,
        value=100_000, step=10_000,
    )
    tx_cost = st.sidebar.slider(
        "Transaction Cost", 0.0, 0.02, 0.001, 0.0005, format="%.4f"
    )

    return {
        "tickers"         : selected_tickers,
        "strategy"        : strategy,
        "use_risk_profile": use_risk_profile,
        "risk_score"      : risk_score,
        "min_weight"      : min_wt,
        "max_weight"      : max_wt,
        "start_date"      : str(start_date),
        "end_date"        : str(end_date),
        "rebalance"       : rebalance,
        "train_years"     : train_years,
        "initial_capital" : initial_capital,
        "transaction_cost": tx_cost,
    }


# ---------------------------------------------------------------------------
# Tab 1 — Portfolio Allocation
# ---------------------------------------------------------------------------

def render_portfolio_tab(params: dict):
    st.header("📊 Portfolio Allocation")

    if len(params["tickers"]) < 2:
        st.warning("Select at least 2 tickers in the sidebar to optimise.")
        return

    constraints = {
        "min_weight": params["min_weight"],
        "max_weight": params["max_weight"],
        "long_only" : True,
        "leverage"  : 1.0,
    }

    with st.spinner("Optimising portfolio..."):
        if params["use_risk_profile"] and params["risk_score"] is not None:
            data = api_post("/portfolio/optimise-profile", {
                "tickers"   : params["tickers"],
                "risk_score": params["risk_score"],
            })
        else:
            data = api_post("/portfolio/optimise", {
                "tickers"    : params["tickers"],
                "strategy"   : params["strategy"],
                "constraints": constraints,
            })

    if data is None:
        return

    # ── Metrics Cards ────────────────────────────────────────────────────
    m = data.get("metrics", {})
    st.subheader("Portfolio Metrics")
    cols = st.columns(4)
    cols[0].metric("Expected Return",  pct(m.get("expected_return")))
    cols[1].metric("Volatility",       pct_abs(m.get("volatility")))
    cols[2].metric("Sharpe Ratio",     flt(m.get("sharpe_ratio")))
    cols[3].metric("Diversification",  flt(m.get("diversification")))

    # ── Explanation ──────────────────────────────────────────────────────
    st.info(f"💡 {data.get('explanation', '')}")

    # ── Allocation Charts ────────────────────────────────────────────────
    alloc = pd.DataFrame(data.get("allocation", []))
    if alloc.empty:
        st.warning("No allocation data returned.")
        return

    alloc = alloc.sort_values("weight", ascending=False)
    alloc["weight_pct"]            = alloc["weight"] * 100
    alloc["risk_contribution_pct"] = alloc["risk_contribution"] * 100
    alloc["expected_return_pct"]   = alloc["expected_return"] * 100

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = px.pie(
            alloc[alloc["weight"] > 0.005],
            values="weight_pct",
            names="ticker",
            title="Weight Allocation",
            hole=0.42,
            color_discrete_sequence=COLORS["portfolio"],
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(showlegend=False, margin=dict(t=40, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_rc = px.bar(
            alloc[alloc["weight"] > 0.005],
            x="ticker",
            y="risk_contribution_pct",
            color="ticker",
            title="Risk Contribution (%)",
            labels={"risk_contribution_pct": "Risk %", "ticker": ""},
            color_discrete_sequence=COLORS["portfolio"],
        )
        fig_rc.update_layout(showlegend=False, margin=dict(t=40, b=0))
        st.plotly_chart(fig_rc, use_container_width=True)

    # ── Allocation Table ─────────────────────────────────────────────────
    st.subheader("Allocation Detail")
    display = alloc[alloc["weight"] > 0.001][
        ["ticker", "weight_pct", "risk_contribution_pct", "expected_return_pct"]
    ].rename(columns={
        "weight_pct"            : "Weight (%)",
        "risk_contribution_pct" : "Risk Contrib (%)",
        "expected_return_pct"   : "Exp. Return (%)",
    })
    st.dataframe(
        display.style.format({
            "Weight (%)": "{:.2f}",
            "Risk Contrib (%)": "{:.2f}",
            "Exp. Return (%)": "{:.2f}",
        }).background_gradient(subset=["Weight (%)"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Tab 2 — Efficient Frontier
# ---------------------------------------------------------------------------

def render_frontier_tab(params: dict):
    st.header("🎯 Efficient Frontier")

    if len(params["tickers"]) < 2:
        st.warning("Select at least 2 tickers.")
        return

    n_points = st.slider("Resolution (frontier points)", 20, 80, 40, 5)

    with st.spinner("Computing efficient frontier..."):
        data = api_post("/portfolio/frontier", {
            "tickers"    : params["tickers"],
            "n_points"   : n_points,
            "constraints": {
                "min_weight": params["min_weight"],
                "max_weight": params["max_weight"],
                "long_only" : True,
                "leverage"  : 1.0,
            },
        })

    if data is None:
        return

    frontier = pd.DataFrame(data.get("frontier", []))
    if frontier.empty:
        st.warning("Could not compute efficient frontier with current settings.")
        return

    max_sharpe_pt = data.get("max_sharpe_point", {})

    fig = go.Figure()

    # Frontier curve
    fig.add_trace(go.Scatter(
        x=frontier["volatility"] * 100,
        y=frontier["portfolio_return"] * 100,
        mode="lines+markers",
        name="Efficient Frontier",
        line=dict(color=COLORS["primary"], width=2.5),
        marker=dict(size=5, color=frontier["sharpe"],
                    colorscale="Blues", showscale=True,
                    colorbar=dict(title="Sharpe")),
        hovertemplate=(
            "<b>Vol:</b> %{x:.2f}%<br>"
            "<b>Return:</b> %{y:.2f}%<br>"
            "<extra></extra>"
        ),
    ))

    # Max Sharpe point
    if max_sharpe_pt:
        fig.add_trace(go.Scatter(
            x=[max_sharpe_pt.get("volatility", 0) * 100],
            y=[max_sharpe_pt.get("portfolio_return", 0) * 100],
            mode="markers",
            name="Max Sharpe",
            marker=dict(size=14, color=COLORS["success"],
                        symbol="star", line=dict(width=2, color="white")),
        ))

    fig.update_layout(
        title       = "Efficient Frontier — Risk vs Return Trade-off",
        xaxis_title = "Annual Volatility (%)",
        yaxis_title = "Expected Annual Return (%)",
        hovermode   = "closest",
        height      = 480,
        legend      = dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Strategy comparison table
    st.subheader("Strategy Comparison")
    with st.spinner("Comparing strategies..."):
        comp_data = api_post("/portfolio/compare", {
            "tickers"    : params["tickers"],
            "strategy"   : "max_sharpe",
            "constraints": {"min_weight": params["min_weight"],
                            "max_weight": params["max_weight"],
                            "long_only": True, "leverage": 1.0},
        })

    if comp_data:
        comp_df = pd.DataFrame(comp_data.get("comparison", []))
        if not comp_df.empty and "strategy" in comp_df.columns:
            comp_df = comp_df.set_index("strategy")
            for col in ["annual_return", "volatility", "max_weight"]:
                if col in comp_df.columns:
                    comp_df[col] = comp_df[col].apply(
                        lambda v: f"{v*100:.2f}%" if v is not None else "—"
                    )
            st.dataframe(comp_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3 — Backtesting
# ---------------------------------------------------------------------------

def render_backtest_tab(params: dict):
    st.header("🔁 Walk-Forward Backtest")

    if len(params["tickers"]) < 2:
        st.warning("Select at least 2 tickers.")
        return

    run_btn = st.button("▶ Run Backtest", type="primary", use_container_width=True)
    if not run_btn:
        st.info("Configure parameters in the sidebar, then click **Run Backtest**.")
        return

    payload = {
        "tickers"          : params["tickers"],
        "strategy"         : params["strategy"],
        "start_date"       : params["start_date"],
        "end_date"         : params["end_date"],
        "rebalance"        : params["rebalance"],
        "train_years"      : params["train_years"],
        "initial_capital"  : params["initial_capital"],
        "transaction_cost" : params["transaction_cost"],
        "benchmark"        : "^GSPC",
        "constraints": {
            "min_weight": params["min_weight"],
            "max_weight": params["max_weight"],
            "long_only" : True,
            "leverage"  : 1.0,
        },
    }

    with st.spinner("Running walk-forward backtest (this may take a minute)..."):
        data = api_post("/backtest/run", payload)

    if data is None:
        return

    metrics = data.get("metrics", {})
    eq_data = pd.DataFrame(data.get("equity_curve", []))

    # ── Key Metric Cards ─────────────────────────────────────────────────
    st.subheader("Performance Summary")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("CAGR",         pct(metrics.get("cagr")))
    c2.metric("Sharpe",       flt(metrics.get("sharpe")))
    c3.metric("Sortino",      flt(metrics.get("sortino")))
    c4.metric("Max Drawdown", pct(metrics.get("max_drawdown")))
    c5.metric("Alpha",        pct(metrics.get("alpha")))
    c6.metric("Final Value",  usd(metrics.get("final_value")))

    if eq_data.empty:
        st.warning("No equity curve data returned.")
        return

    eq_data["date"] = pd.to_datetime(eq_data["date"])

    # ── Equity Curve + Drawdown (2-panel) ────────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.04,
        subplot_titles=("Portfolio vs Benchmark", "Drawdown"),
    )

    fig.add_trace(go.Scatter(
        x=eq_data["date"], y=eq_data["portfolio_value"],
        name="Portfolio", line=dict(color=COLORS["primary"], width=2.2),
        fill="tonexty" if False else None,
    ), row=1, col=1)

    if "benchmark_value" in eq_data.columns:
        fig.add_trace(go.Scatter(
            x=eq_data["date"], y=eq_data["benchmark_value"],
            name="Benchmark (S&P 500)",
            line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
        ), row=1, col=1)

    if "drawdown" in eq_data.columns:
        fig.add_trace(go.Scatter(
            x=eq_data["date"],
            y=eq_data["drawdown"] * 100,
            name="Drawdown (%)",
            fill="tozeroy",
            line=dict(color=COLORS["danger"], width=1),
            fillcolor="rgba(220,38,38,0.15)",
        ), row=2, col=1)

    fig.update_layout(
        height=550, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=50, b=20),
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)",        row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # ── Rolling Sharpe ───────────────────────────────────────────────────
    if "daily_return" in eq_data.columns:
        eq_data["daily_return"] = pd.to_numeric(eq_data["daily_return"], errors="coerce")
        rolling_sharpe = (
            eq_data.set_index("date")["daily_return"]
            .rolling(63)
            .apply(lambda r: r.mean() / r.std() * (252 ** 0.5) if r.std() > 0 else 0)
        )
        fig_sharpe = go.Figure(go.Scatter(
            x=rolling_sharpe.index, y=rolling_sharpe.values,
            fill="tozeroy", name="Rolling Sharpe (63d)",
            line=dict(color=COLORS["primary"]),
            fillcolor="rgba(37,99,235,0.10)",
        ))
        fig_sharpe.add_hline(y=1.0, line_dash="dash",
                              line_color=COLORS["success"], annotation_text="Sharpe=1")
        fig_sharpe.add_hline(y=0.0, line_dash="dash", line_color=COLORS["neutral"])
        fig_sharpe.update_layout(title="Rolling 63-Day Sharpe Ratio", height=280,
                                  margin=dict(t=40, b=20))
        st.plotly_chart(fig_sharpe, use_container_width=True)

    # ── Full Metrics Table ───────────────────────────────────────────────
    with st.expander("📋 Full Performance Report"):
        st.text(data.get("report_text", "No report generated."))

    # ── Rebalance Log ────────────────────────────────────────────────────
    reb_log = data.get("rebalance_log", [])
    if reb_log:
        with st.expander(f"🔄 Rebalance Log ({len(reb_log)} events)"):
            reb_df = pd.DataFrame([{
                "Date"            : e["date"],
                "Turnover"        : pct_abs(e.get("turnover")),
                "Tx Cost"         : pct_abs(e.get("transaction_cost")),
                "# Tickers"       : len(e.get("tickers_used", [])),
            } for e in reb_log])
            st.dataframe(reb_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 4 — Risk Scorecard
# ---------------------------------------------------------------------------

def render_risk_tab(params: dict):
    st.header("🧮 Risk Scorecard")

    asset_type_filter = st.selectbox(
        "Filter by Asset Class",
        options=["All", "us_stocks", "etfs", "crypto", "indices"],
        format_func=lambda x: "All Asset Classes" if x == "All" else x.replace("_", " ").title(),
    )

    with st.spinner("Loading risk summary..."):
        data = api_post("/features/risk-summary", {
            "asset_type": None if asset_type_filter == "All" else asset_type_filter
        })

    if data is None:
        return

    df = pd.DataFrame(data.get("risk_summary", []))
    if df.empty:
        st.warning("No risk data found. Run the feature engineering pipeline first via /features/build.")
        return

    # Filter to selected tickers only if sidebar selection active
    if params["tickers"]:
        df_filtered = df[df["ticker"].isin(params["tickers"])]
        if not df_filtered.empty:
            df = df_filtered

    # Format percentages
    pct_cols = ["annual_return", "avg_vol_20d", "max_drawdown", "avg_var_95", "avg_cvar_95"]
    for col in pct_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Sharpe Ranking Bar ───────────────────────────────────────────────
    df_sorted = df.sort_values("avg_sharpe", ascending=False).head(20)
    fig_sharpe = px.bar(
        df_sorted,
        x="ticker", y="avg_sharpe",
        color="avg_sharpe",
        color_continuous_scale=["red", "yellow", "green"],
        title="Sharpe Ratio Ranking",
        labels={"avg_sharpe": "Sharpe Ratio", "ticker": ""},
    )
    fig_sharpe.update_layout(height=350, showlegend=False, margin=dict(t=40, b=10))
    st.plotly_chart(fig_sharpe, use_container_width=True)

    # ── Risk/Return Scatter ──────────────────────────────────────────────
    col_scatter, col_dd = st.columns(2)

    with col_scatter:
        fig_rr = px.scatter(
            df,
            x="avg_vol_20d", y="annual_return",
            text="ticker", color="avg_sharpe",
            color_continuous_scale="RdYlGn",
            title="Risk vs Return",
            labels={"avg_vol_20d": "Volatility", "annual_return": "Ann. Return"},
            size_max=14,
        )
        fig_rr.update_traces(textposition="top center", marker_size=10)
        fig_rr.update_layout(height=380, coloraxis_colorbar_title="Sharpe")
        st.plotly_chart(fig_rr, use_container_width=True)

    with col_dd:
        fig_dd = px.bar(
            df.sort_values("max_drawdown").head(20),
            x="ticker", y="max_drawdown",
            color="max_drawdown",
            color_continuous_scale="Reds_r",
            title="Max Drawdown by Ticker",
            labels={"max_drawdown": "Max Drawdown", "ticker": ""},
        )
        fig_dd.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_dd, use_container_width=True)

    # ── Full Scorecard Table ─────────────────────────────────────────────
    st.subheader("Full Risk Scorecard")
    disp = df[["ticker", "asset_type", "annual_return", "avg_vol_20d",
               "avg_sharpe", "avg_sortino", "max_drawdown", "avg_var_95",
               "avg_beta"]].copy()

    for c in ["annual_return", "avg_vol_20d", "max_drawdown", "avg_var_95"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v*100:.2f}%" if pd.notna(v) else "—")

    for c in ["avg_sharpe", "avg_sortino", "avg_beta"]:
        if c in disp.columns:
            disp[c] = disp[c].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")

    disp.columns = ["Ticker", "Class", "Ann.Return", "Volatility",
                    "Sharpe", "Sortino", "Max DD", "VaR 95", "Beta"]
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Tab 5 — Correlation Heatmap
# ---------------------------------------------------------------------------

def render_correlation_tab(params: dict):
    st.header("🔗 Correlation Analysis")

    col1, col2 = st.columns([2, 1])
    with col1:
        window = st.slider("Rolling Window (trading days)", 60, 504, 252, 21)
    with col2:
        method = st.selectbox("Method", ["pearson", "spearman", "kendall"])

    tickers = params["tickers"] if params["tickers"] else None

    with st.spinner("Computing correlation matrix..."):
        data = api_post("/features/correlation", {
            "tickers": tickers,
            "window" : window,
            "method" : method,
        })

    if data is None:
        return

    labels = data.get("tickers", [])
    matrix = data.get("matrix", [])

    if not matrix:
        st.warning("No correlation data. Run feature engineering first.")
        return

    corr_df = pd.DataFrame(matrix, index=labels, columns=labels)

    # ── Heatmap ──────────────────────────────────────────────────────────
    fig_heat = px.imshow(
        corr_df,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"{method.title()} Correlation Matrix ({window}d rolling)",
        text_auto=".2f",
        aspect="auto",
    )
    fig_heat.update_layout(height=550, margin=dict(t=50, b=10))
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Top / Bottom Correlations ─────────────────────────────────────────
    col_top, col_bot = st.columns(2)
    pairs = []
    for i, t1 in enumerate(labels):
        for j, t2 in enumerate(labels):
            if j > i:
                pairs.append((t1, t2, corr_df.loc[t1, t2]))

    pairs_df = pd.DataFrame(pairs, columns=["Asset 1", "Asset 2", "Correlation"])
    pairs_df = pairs_df.dropna().sort_values("Correlation")

    with col_top:
        st.markdown("**Lowest Correlations** (best diversifiers)")
        st.dataframe(
            pairs_df.head(8).assign(
                Correlation=lambda d: d["Correlation"].apply(lambda v: f"{v:.3f}")
            ),
            use_container_width=True, hide_index=True,
        )

    with col_bot:
        st.markdown("**Highest Correlations** (most similar)")
        st.dataframe(
            pairs_df.tail(8).sort_values("Correlation", ascending=False).assign(
                Correlation=lambda d: d["Correlation"].apply(lambda v: f"{v:.3f}")
            ),
            use_container_width=True, hide_index=True,
        )


# ---------------------------------------------------------------------------
# Tab 6 — Backtest History
# ---------------------------------------------------------------------------

def render_history_tab():
    st.header("🗂️ Backtest History")
    data = api_get("/backtest/history")
    if data is None:
        return

    df = pd.DataFrame(data.get("history", []))
    if df.empty:
        st.info("No backtest runs recorded yet.")
        return

    # Metric trend chart
    if "sharpe" in df.columns and "run_timestamp" in df.columns:
        df["run_timestamp"] = pd.to_datetime(df["run_timestamp"])
        df_sorted = df.sort_values("run_timestamp")

        fig = px.line(
            df_sorted, x="run_timestamp", y=["sharpe", "cagr"],
            title="Sharpe & CAGR Across Backtest Runs",
            markers=True,
        )
        fig.update_layout(height=300, legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)

    # History table
    display_cols = [c for c in [
        "run_id", "strategy", "start_date", "end_date", "rebalance",
        "cagr", "sharpe", "max_drawdown", "final_value", "run_timestamp",
    ] if c in df.columns]

    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main():
    st.title("📈 InvestEng — Intelligent Investment Engine")
    st.caption("Data-driven portfolio intelligence: Ingest → Engineer → Optimise → Backtest")

    # ── API Health Check ─────────────────────────────────────────────────
    health = api_get("/health")
    if health:
        comp = health.get("components", {})
        ready = all(comp.values())
        if ready:
            st.success("✅ Engine online — all components initialised.")
        else:
            not_ready = [k for k, v in comp.items() if not v]
            st.warning(f"⚠️ Some components not ready: {not_ready}")
    else:
        st.error("API offline. Run `uvicorn api:app --reload --port 8000`")

    # ── Sidebar ───────────────────────────────────────────────────────────
    params = render_sidebar()

    # ── Pipeline Actions ──────────────────────────────────────────────────
    with st.expander("⚡ Pipeline Actions", expanded=False):
        a1, a2 = st.columns(2)
        if a1.button("🔄 Ingest Market Data", use_container_width=True):
            result = api_post("/ingest", {"asset_types": ["us_stocks","etfs","crypto","indices"]})
            if result:
                st.success(f"Ingestion started | Job ID: {result.get('job_id')}")

        if a2.button("⚙️ Build Features", use_container_width=True):
            result = api_post("/features/build", {"asset_types": ["us_stocks","etfs","crypto","indices"]})
            if result:
                st.success(f"Feature engineering started | Job ID: {result.get('job_id')}")

    # ── Main Tabs ─────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Portfolio",
        "🎯 Efficient Frontier",
        "🔁 Backtest",
        "🧮 Risk Scorecard",
        "🔗 Correlation",
        "🗂️ History",
    ])

    with tabs[0]:
        render_portfolio_tab(params)
    with tabs[1]:
        render_frontier_tab(params)
    with tabs[2]:
        render_backtest_tab(params)
    with tabs[3]:
        render_risk_tab(params)
    with tabs[4]:
        render_correlation_tab(params)
    with tabs[5]:
        render_history_tab()


if __name__ == "__main__":
    main()
