import sqlite3
import time
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Drift Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design System ───────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0d1117",
    "surface":     "#161b22",
    "surface2":    "#1c2230",
    "border":      "#2d3348",
    "text":        "#e6edf3",
    "muted":       "#8b949e",
    "blue":        "#58a6ff",
    "green":       "#3fb950",
    "yellow":      "#d29922",
    "red":         "#f85149",
    "purple":      "#bc8cff",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["muted"], size=12),
    margin=dict(l=0, r=0, t=24, b=0),
    xaxis=dict(showgrid=False, zeroline=False, color=COLORS["muted"]),
    yaxis=dict(showgrid=True, gridcolor=COLORS["border"], zeroline=False, color=COLORS["muted"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
    hoverlabel=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["border"], font_color=COLORS["text"]),
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1117; }

    /* ── Sidebar ── */
    div[data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #2d3348;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2230 100%);
        border: 1px solid #2d3348;
        border-radius: 14px;
        padding: 22px 20px 18px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #58a6ff44; }
    .metric-label {
        color: #8b949e; font-size: 11px; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px;
    }
    .metric-value { color: #e6edf3; font-size: 30px; font-weight: 700; line-height: 1; }
    .metric-sub   { color: #58a6ff; font-size: 12px; margin-top: 6px; }

    /* ── Risk Colors ── */
    .risk-healthy       { color: #3fb950; }
    .risk-drift         { color: #d29922; }
    .risk-high          { color: #f85149; }

    /* ── Badges ── */
    .badge {
        display: inline-block; padding: 3px 10px;
        border-radius: 20px; font-size: 11px; font-weight: 600; letter-spacing: 0.03em;
    }
    .badge-healthy { background: rgba(63,185,80,0.12);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
    .badge-drift   { background: rgba(210,153,34,0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
    .badge-high    { background: rgba(248,81,73,0.12);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }

    /* ── Section Headers ── */
    .section-title {
        color: #e6edf3; font-size: 14px; font-weight: 600;
        letter-spacing: 0.04em; text-transform: uppercase;
        margin: 28px 0 14px 0; padding-bottom: 10px;
        border-bottom: 1px solid #2d3348;
        display: flex; align-items: center; gap: 8px;
    }

    /* ── Alert Feed ── */
    .alert-item {
        background: rgba(248,81,73,0.06);
        border: 1px solid rgba(248,81,73,0.25);
        border-left: 3px solid #f85149;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .alert-item-drift {
        background: rgba(210,153,34,0.06);
        border: 1px solid rgba(210,153,34,0.25);
        border-left: 3px solid #d29922;
        border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;
    }
    .alert-id   { color: #e6edf3; font-weight: 600; font-size: 13px; }
    .alert-meta { color: #8b949e; font-size: 12px; margin-top: 4px; }

    /* ── Logo ── */
    .logo-block {
        padding: 20px 4px 16px;
        border-bottom: 1px solid #2d3348;
        margin-bottom: 20px;
    }
    .logo-name {
        font-size: 17px; font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .logo-sub  { color: #8b949e; font-size: 11px; margin-top: 4px; }
    .status-dot-live   { display:inline-block; width:7px; height:7px; border-radius:50%; background:#3fb950; margin-right:5px; animation: pulse 2s infinite; }
    .status-dot-paused { display:inline-block; width:7px; height:7px; border-radius:50%; background:#8b949e; margin-right:5px; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

    /* ── Divider ── */
    hr { border-color: #2d3348 !important; }

    /* ── Dataframe ── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Footer ── */
    .footer { text-align:center; color:#30363d; font-size:11px; margin-top:48px; padding-top:16px; border-top:1px solid #1c2230; }
</style>
""", unsafe_allow_html=True)

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")


@st.cache_data(ttl=10)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query("SELECT * FROM executions ORDER BY created_at DESC", conn)
    return df


def risk_badge(risk: str) -> str:
    cls = {
        "healthy": "badge-healthy",
        "drift_detected": "badge-drift",
        "high_risk": "badge-high",
    }.get(risk, "badge-drift")
    label = risk.replace("_", " ").title()
    return f'<span class="badge {cls}">{label}</span>'


def metric_card(col, label: str, value: str, sub: str = ""):
    col.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        </div>""",
        unsafe_allow_html=True,
    )


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    auto_refresh = st.toggle("⚡ Auto-Refresh (10s)", value=False, key="auto_refresh")
    status_dot = '<span class="status-dot-live"></span>' if auto_refresh else '<span class="status-dot-paused"></span>'
    status_text = "Live" if auto_refresh else "Paused"
    st.markdown(
        f"""<div class="logo-block">
            <div class="logo-name">🧠 Drift Detector</div>
            <div class="logo-sub">{status_dot}{status_text} · Agentic AI Observability</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Filters")
    min_score = st.slider("Min Drift Score", 0, 100, 0)
    severity_filter = st.multiselect(
        "Severity", ["low", "medium", "high"], default=["low", "medium", "high"]
    )
    decision_filter = st.multiselect(
        "Decision", ["auto_resolve", "escalate"], default=["auto_resolve", "escalate"]
    )

    st.markdown("---")
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    # DB Stats
    if os.path.exists(DB_PATH):
        db_size_kb = os.path.getsize(DB_PATH) // 1024
        st.markdown(
            f"<div style='color:#8b949e;font-size:12px;'>"
            f"📦 DB: <b style='color:#e6edf3'>{db_size_kb} KB</b></div>",
            unsafe_allow_html=True,
        )

# ─── Auto-Refresh Logic ───────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()

# ─── Load & Filter ────────────────────────────────────────────────────────────
df_raw = load_data()

if df_raw.empty:
    st.title("🧠 Agentic Drift Detector")
    st.warning(
        "No telemetry data found. Run `python run.py --simulate-batch 50` to build a baseline first."
    )
    st.stop()

df = df_raw.copy()
df = df[df["drift_score"] >= min_score]
if severity_filter:
    df = df[df["severity"].isin(severity_filter)]
if decision_filter:
    df = df[df["decision"].isin(decision_filter)]

total = len(df)
if total == 0:
    st.warning("No records match the current filters.")
    st.stop()

# ─── Page Header ─────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(
        "<h1 style='color:#e6edf3;font-size:26px;margin-bottom:4px;font-weight:700;'>"
        "🧠 Agentic Drift Detector</h1>"
        "<p style='color:#8b949e;font-size:14px;margin-top:0;'>"
        "Real-time behavioral observability for autonomous AI agent workflows.</p>",
        unsafe_allow_html=True,
    )
with col_h2:
    st.markdown(
        f"<div style='text-align:right;padding-top:12px;'>"
        f"<span style='color:#8b949e;font-size:13px;'>Showing <b style='color:#e6edf3'>{total}</b> "
        f"of <b style='color:#e6edf3'>{len(df_raw)}</b> executions</span></div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin:12px 0 24px;'>", unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
avg_drift    = df["drift_score"].mean()
avg_latency  = df["execution_time_ms"].mean() / 1000
avg_retries  = df["retry_count"].mean()
esc_rate     = (df["decision"] == "escalate").mean() * 100
heal_count   = (
    df["path_taken"].str.contains("intervention", na=False).sum()
    if "path_taken" in df.columns else 0
)
high_risk_ct = (df["risk_level"] == "high_risk").sum() if "risk_level" in df.columns else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)
metric_card(c1, "Total Runs",      f"{total:,}")
metric_card(c2, "Avg Drift Score", f"{avg_drift:.1f}",  sub="0 = perfect")
metric_card(c3, "Avg Latency",     f"{avg_latency:.2f}s")
metric_card(c4, "Escalation Rate", f"{esc_rate:.1f}%")
metric_card(c5, "Healing Events",  f"{heal_count:,}",   sub="intervention node")
metric_card(c6, "High-Risk Runs",  f"{high_risk_ct:,}", sub="score ≥ 60")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1 ────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        "<div class='section-title'>📈 Drift Score Over Time</div>",
        unsafe_allow_html=True,
    )
    chart_df = df[["drift_score"]].reset_index(drop=True)
    chart_df.index.name = "Run #"

    # Rolling average
    chart_df["rolling_avg"] = chart_df["drift_score"].rolling(10, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=chart_df["drift_score"], name="Drift Score",
        line=dict(color=COLORS["blue"], width=1.5),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.06)",
        hovertemplate="Run %{x}<br>Score: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        y=chart_df["rolling_avg"], name="10-Run Avg",
        line=dict(color=COLORS["purple"], width=2, dash="dot"),
        hovertemplate="Avg: %{y:.1f}<extra></extra>",
    ))
    # Threshold band
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(248,81,73,0.05)",
                  line_width=0, annotation_text="High Risk",
                  annotation_position="top right",
                  annotation_font_color=COLORS["red"],
                  annotation_font_size=11)
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(210,153,34,0.04)",
                  line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_right:
    st.markdown(
        "<div class='section-title'>⏱️ Latency Over Time</div>",
        unsafe_allow_html=True,
    )
    lat_df = df[["execution_time_ms"]].reset_index(drop=True)
    lat_df["rolling_avg"] = lat_df["execution_time_ms"].rolling(10, min_periods=1).mean()
    lat_df.index.name = "Run #"

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        y=lat_df["execution_time_ms"], name="Latency (ms)",
        line=dict(color=COLORS["green"], width=1.5),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.06)",
        hovertemplate="Run %{x}<br>Latency: %{y} ms<extra></extra>",
    ))
    fig2.add_trace(go.Scatter(
        y=lat_df["rolling_avg"], name="10-Run Avg",
        line=dict(color=COLORS["yellow"], width=2, dash="dot"),
        hovertemplate="Avg: %{y:.0f} ms<extra></extra>",
    ))
    fig2.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ─── Charts Row 2 ────────────────────────────────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown(
        "<div class='section-title'>📊 Severity Distribution</div>",
        unsafe_allow_html=True,
    )
    sev_counts = df["severity"].value_counts().reset_index()
    sev_counts.columns = ["Severity", "Count"]
    sev_color_map = {"low": COLORS["green"], "medium": COLORS["yellow"], "high": COLORS["red"]}
    fig3 = px.bar(
        sev_counts, x="Count", y="Severity", orientation="h",
        color="Severity", color_discrete_map=sev_color_map,
        text="Count",
    )
    fig3.update_traces(
        textposition="outside",
        textfont_color=COLORS["text"],
        marker_line_width=0,
    )
    fig3.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False,
                       bargap=0.35)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with col_r2:
    st.markdown(
        "<div class='section-title'>🎯 Decision Distribution</div>",
        unsafe_allow_html=True,
    )
    dec_counts = df["decision"].value_counts().reset_index()
    dec_counts.columns = ["Decision", "Count"]
    dec_color_map = {
        "auto_resolve": COLORS["green"],
        "escalate": COLORS["red"],
    }
    fig4 = px.bar(
        dec_counts, x="Count", y="Decision", orientation="h",
        color="Decision", color_discrete_map=dec_color_map,
        text="Count",
    )
    fig4.update_traces(
        textposition="outside",
        textfont_color=COLORS["text"],
        marker_line_width=0,
    )
    fig4.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False,
                       bargap=0.35)
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ─── Risk Level Breakdown ─────────────────────────────────────────────────────
if "risk_level" in df.columns:
    st.markdown(
        "<div class='section-title'>🔰 Risk Level Breakdown</div>",
        unsafe_allow_html=True,
    )
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_counts["Percentage"] = (risk_counts["Count"] / total * 100).round(1).astype(str) + "%"

    r_cols = st.columns(len(risk_counts))
    color_cls = {
        "healthy": ("risk-healthy", "3fb950"),
        "drift_detected": ("risk-drift", "d29922"),
        "high_risk": ("risk-high", "f85149"),
    }
    for i, row in risk_counts.iterrows():
        cls, hex_c = color_cls.get(row["Risk Level"], ("", "58a6ff"))
        r_cols[i].markdown(
            f"<div class='metric-card' style='border-color:#{hex_c}22;'>"
            f"<div class='metric-label'>{row['Risk Level'].replace('_', ' ').title()}</div>"
            f"<div class='metric-value {cls}'>{row['Count']}</div>"
            f"<div class='metric-sub'>{row['Percentage']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ─── Alert Feed ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>🚨 Recent Alerts</div>", unsafe_allow_html=True)

alert_df = df_raw[df_raw["risk_level"].isin(["high_risk", "drift_detected"])].head(5) \
    if "risk_level" in df_raw.columns else pd.DataFrame()

if alert_df.empty:
    st.markdown(
        "<div style='color:#3fb950;padding:12px;background:rgba(63,185,80,0.06);"
        "border:1px solid rgba(63,185,80,0.2);border-radius:8px;font-size:13px;'>"
        "✅ No active alerts — all executions are healthy.</div>",
        unsafe_allow_html=True,
    )
else:
    for _, row in alert_df.iterrows():
        css_cls = "alert-item" if row["risk_level"] == "high_risk" else "alert-item-drift"
        icon = "🔴" if row["risk_level"] == "high_risk" else "🟡"
        ts = row.get("created_at", "—")
        st.markdown(
            f"<div class='{css_cls}'>"
            f"<div class='alert-id'>{icon} Incident <code>{row['incident_id']}</code> "
            f"— Drift Score <b>{row['drift_score']}</b> · {row['risk_level'].replace('_',' ').title()}</div>"
            f"<div class='alert-meta'>"
            f"Severity: {row.get('severity','—')} &nbsp;·&nbsp; "
            f"Decision: {row.get('decision','—')} &nbsp;·&nbsp; "
            f"Retries: {row.get('retry_count', 0)} &nbsp;·&nbsp; "
            f"<span style='color:#3d444d'>{ts}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

# ─── Recent Executions Table ──────────────────────────────────────────────────
st.markdown("<div class='section-title'>📋 Recent Executions</div>", unsafe_allow_html=True)

display_cols = [
    c for c in [
        "incident_id", "severity", "decision", "confidence",
        "retry_count", "drift_score", "risk_level", "execution_time_ms", "created_at",
    ]
    if c in df.columns
]

st.dataframe(
    df[display_cols].head(50),
    use_container_width=True,
    hide_index=True,
    column_config={
        "drift_score":       st.column_config.ProgressColumn(
            "Drift Score", min_value=0, max_value=100, format="%d"
        ),
        "confidence":        st.column_config.NumberColumn("Confidence", format="%.2f"),
        "execution_time_ms": st.column_config.NumberColumn("Latency (ms)", format="%d ms"),
        "risk_level":        st.column_config.TextColumn("Risk Level"),
    },
)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Agentic Drift Detector &nbsp;·&nbsp; Built with LangGraph &amp; Streamlit</div>",
    unsafe_allow_html=True,
)
