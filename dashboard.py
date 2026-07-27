import json
import os
import sqlite3
import time
from dotenv import load_dotenv

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Drift Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend Detection ─────────────────────────────────────────────────────────
load_dotenv()
USE_POSTGRES = bool(os.getenv("DATABASE_URL"))
DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")

# ── Design System ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":      "#0d1117", "surface": "#161b22", "surface2": "#1c2230",
    "border":  "#2d3348", "text":    "#e6edf3",  "muted":   "#8b949e",
    "blue":    "#58a6ff", "green":   "#3fb950",  "yellow":  "#d29922",
    "red":     "#f85149", "purple":  "#bc8cff",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=COLORS["muted"], size=12),
    margin=dict(l=0, r=0, t=24, b=0),
    xaxis=dict(showgrid=False, zeroline=False, color=COLORS["muted"]),
    yaxis=dict(showgrid=True, gridcolor=COLORS["border"], zeroline=False, color=COLORS["muted"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["muted"])),
    hoverlabel=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["border"],
                    font_color=COLORS["text"]),
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #0d1117; }
    div[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #2d3348; }

    .metric-card {
        background: linear-gradient(135deg, #161b22 0%, #1c2230 100%);
        border: 1px solid #2d3348; border-radius: 14px;
        padding: 22px 20px 18px; text-align: center; transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #58a6ff44; }
    .metric-label { color: #8b949e; font-size: 11px; font-weight: 600;
        letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 8px; }
    .metric-value { color: #e6edf3; font-size: 30px; font-weight: 700; line-height: 1; }
    .metric-sub   { color: #58a6ff; font-size: 12px; margin-top: 6px; }

    .risk-healthy { color: #3fb950; }
    .risk-drift   { color: #d29922; }
    .risk-high    { color: #f85149; }

    .badge { display: inline-block; padding: 3px 10px; border-radius: 20px;
             font-size: 11px; font-weight: 600; letter-spacing: 0.03em; }
    .badge-healthy { background: rgba(63,185,80,0.12);  color: #3fb950; border: 1px solid rgba(63,185,80,0.3); }
    .badge-drift   { background: rgba(210,153,34,0.12); color: #d29922; border: 1px solid rgba(210,153,34,0.3); }
    .badge-high    { background: rgba(248,81,73,0.12);  color: #f85149; border: 1px solid rgba(248,81,73,0.3); }

    .section-title { color: #e6edf3; font-size: 14px; font-weight: 600;
        letter-spacing: 0.04em; text-transform: uppercase;
        margin: 28px 0 14px; padding-bottom: 10px; border-bottom: 1px solid #2d3348; }

    .alert-item      { background: rgba(248,81,73,0.06);  border: 1px solid rgba(248,81,73,0.25);
                       border-left: 3px solid #f85149; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
    .alert-item-drift { background: rgba(210,153,34,0.06); border: 1px solid rgba(210,153,34,0.25);
                        border-left: 3px solid #d29922; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; }
    .alert-id   { color: #e6edf3; font-weight: 600; font-size: 13px; }
    .alert-meta { color: #8b949e; font-size: 12px; margin-top: 4px; }

    .logo-block { padding: 20px 4px 16px; border-bottom: 1px solid #2d3348; margin-bottom: 20px; }
    .logo-name  { font-size: 17px; font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #bc8cff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .logo-sub   { color: #8b949e; font-size: 11px; margin-top: 4px; }
    .status-dot-live   { display:inline-block; width:7px; height:7px; border-radius:50%;
                         background:#3fb950; margin-right:5px; animation: pulse 2s infinite; }
    .status-dot-paused { display:inline-block; width:7px; height:7px; border-radius:50%;
                         background:#8b949e; margin-right:5px; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

    .tenant-badge { display:inline-block; background:rgba(88,166,255,0.12);
        border:1px solid rgba(88,166,255,0.3); border-radius:6px;
        padding:3px 10px; font-size:11px; color:#58a6ff; font-weight:600;
        letter-spacing:0.05em; text-transform:uppercase; }
    .mode-badge-pg  { background:rgba(63,185,80,0.1); border:1px solid rgba(63,185,80,0.3);
        color:#3fb950; border-radius:6px; padding:2px 8px; font-size:10px; font-weight:600; }
    .mode-badge-sq  { background:rgba(210,153,34,0.1); border:1px solid rgba(210,153,34,0.3);
        color:#d29922; border-radius:6px; padding:2px 8px; font-size:10px; font-weight:600; }

    hr { border-color: #2d3348 !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    .footer { text-align:center; color:#30363d; font-size:11px; margin-top:48px;
              padding-top:16px; border-top:1px solid #1c2230; }
</style>
""", unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def get_tenants() -> list[str]:
    """Fetch available tenant IDs (PostgreSQL mode only)."""
    if not USE_POSTGRES:
        return ["default"]
    try:
        import psycopg2
        with psycopg2.connect(os.getenv("DATABASE_URL")) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id FROM tenants ORDER BY created_at")
                tenants = [row[0] for row in cur.fetchall()]
        return tenants or ["default"]
    except Exception as exc:
        st.warning(f"Could not fetch tenants: {exc}")
        return ["default"]


@st.cache_data(ttl=10)
def load_data(tenant_id: str = "default") -> pd.DataFrame:
    """Load executions for the given tenant from the active backend."""
    if USE_POSTGRES:
        return _load_postgres(tenant_id)
    return _load_sqlite()


def _load_postgres(tenant_id: str) -> pd.DataFrame:
    try:
        import psycopg2
        with psycopg2.connect(os.getenv("DATABASE_URL")) as conn:
            df = pd.read_sql(
                """
                SELECT incident_id, severity, decision, confidence, step_count,
                       retry_count, path_taken, execution_time_ms,
                       drift_score, risk_level, created_at, ml_explanation,
                       sdoh_risk_label, sdoh_risk_score, sdoh_shap_factors
                FROM executions
                WHERE tenant_id = %s
                ORDER BY created_at DESC
                LIMIT 500
                """,
                conn,
                params=(tenant_id,),
            )
        # Normalise path_taken (JSONB → JSON string for .str.contains())
        if "path_taken" in df.columns:
            df["path_taken"] = df["path_taken"].apply(
                lambda x: json.dumps(x) if isinstance(x, list) else (x or "[]")
            )
        return df
    except Exception as exc:
        st.error(f"PostgreSQL connection failed: {exc}")
        return pd.DataFrame()


def _load_sqlite() -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query("SELECT * FROM executions ORDER BY created_at DESC", conn)


@st.cache_data(ttl=60)
def get_queue_depth() -> int:
    """Return Redis queue depth (0 if Redis not configured)."""
    try:
        from telemetry.queue import queue_depth
        return queue_depth()
    except Exception:
        return 0


def metric_card(col, label: str, value: str, sub: str = ""):
    col.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        </div>""",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    auto_refresh = st.toggle("⚡ Auto-Refresh (10s)", value=False, key="auto_refresh")
    status_dot = '<span class="status-dot-live"></span>' if auto_refresh else '<span class="status-dot-paused"></span>'
    mode_badge = (
        '<span class="mode-badge-pg">PostgreSQL</span>'
        if USE_POSTGRES else
        '<span class="mode-badge-sq">SQLite</span>'
    )
    st.markdown(
        f"""<div class="logo-block">
            <div class="logo-name">🧠 Drift Detector</div>
            <div class="logo-sub">{status_dot}{'Live' if auto_refresh else 'Paused'} &nbsp;·&nbsp; {mode_badge}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Workflow selector ─────────────────────────────────────────────────────
    st.markdown("### 🔬 Workflow")
    workflow_mode = st.radio(
        "Active Workflow",
        options=["Incident Triage", "Clinical Coding"],
        index=0,
        label_visibility="collapsed",
        horizontal=True,
    )
    IS_CLINICAL = workflow_mode == "Clinical Coding"

    # ── Clinical file upload ──────────────────────────────────────────────────
    uploaded_note: str | None = None
    if IS_CLINICAL:
        st.markdown("### 📄 Upload Clinical Note")
        uploaded_file = st.file_uploader(
            "Upload a .txt clinical note to code",
            type=["txt"],
            help="Plain-text clinical note. The agent will extract and map ICD-10 codes.",
        )
        if uploaded_file is not None:
            uploaded_note = uploaded_file.read().decode("utf-8", errors="replace")
            st.success(f"Loaded: **{uploaded_file.name}** ({len(uploaded_note)} chars)")
            if st.button("▶ Run Clinical Coding", use_container_width=True, type="primary"):
                with st.spinner("Running clinical coding agent…"):
                    import subprocess
                    import sys
                    result = subprocess.run(
                        [sys.executable, "-m", "clinical.run_clinical",
                         "--note", uploaded_note, "--no-alerts"],
                        capture_output=True, text=True,
                    )
                st.toast("✅ Coding complete! Refresh to see results.")
                st.cache_data.clear()
                st.rerun()

    # ── Tenant selector (PostgreSQL mode only) ────────────────────────────────
    if USE_POSTGRES:
        st.markdown("### 🏢 Tenant")
        tenants = get_tenants()
        selected_tenant = st.selectbox("Active Tenant", tenants, index=0, label_visibility="collapsed")
    else:
        selected_tenant = "default"

    st.markdown("### ⚙️ Filters")
    min_score = st.slider("Min Drift Score", 0, 100, 0)
    
    if IS_CLINICAL:
        severity_filter = []
        decision_filter = st.multiselect(
            "Decision", 
            ["complete", "requires_clinical_review"], 
            default=["complete", "requires_clinical_review"]
        )
    else:
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
    # Infrastructure stats
    if USE_POSTGRES:
        depth = get_queue_depth()
        st.markdown(
            f"<div style='color:#8b949e;font-size:12px;'>"
            f"⚡ Queue depth: <b style='color:#e6edf3'>{depth}</b></div>",
            unsafe_allow_html=True,
        )
    elif os.path.exists(DB_PATH):
        db_size_kb = os.path.getsize(DB_PATH) // 1024
        st.markdown(
            f"<div style='color:#8b949e;font-size:12px;'>"
            f"📦 DB: <b style='color:#e6edf3'>{db_size_kb} KB</b></div>",
            unsafe_allow_html=True,
        )

# ── Auto-Refresh ──────────────────────────────────────────────────────────────
if auto_refresh:
    time.sleep(10)
    st.cache_data.clear()
    st.rerun()

# ── Load & Filter ─────────────────────────────────────────────────────────────
df_raw = load_data(selected_tenant)

# Filter by workflow type if clinical mode is selected
if not df_raw.empty and "workflow_type" in df_raw.columns:
    workflow_type_filter = "clinical_coding" if IS_CLINICAL else "incident_triage"
    df_raw = df_raw[df_raw["workflow_type"] == workflow_type_filter]

if df_raw.empty:
    st.title("🧠 Agentic Drift Detector")
    if IS_CLINICAL:
        backend_hint = (
            "No clinical coding data found. Run:\n"
            "```bash\npython -m clinical.run_clinical --simulate-batch 20\n```"
            "\nor upload a clinical note using the sidebar."
        )
    elif USE_POSTGRES:
        backend_hint = f"No data found for tenant **{selected_tenant}** in PostgreSQL."
    else:
        backend_hint = "No telemetry data found. Run `python run.py --simulate-batch 50`."
    st.warning(backend_hint)
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

# ── Page Header ───────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    tenant_badge = f'<span class="tenant-badge">{selected_tenant}</span>' if USE_POSTGRES else ""
    workflow_label = "⚕️ Clinical Coding" if IS_CLINICAL else "🧠 Incident Triage"
    st.markdown(
        f"<h1 style='color:#e6edf3;font-size:26px;margin-bottom:4px;font-weight:700;'>"
        f"{workflow_label} Drift Detector &nbsp;{tenant_badge}</h1>"
        "<p style='color:#8b949e;font-size:14px;margin-top:0;'>"
        + ("ICD-10 clinical coding observability — behavioral safety for medical AI."
           if IS_CLINICAL else
           "Real-time behavioral observability for autonomous AI agent workflows.")
        + "</p>",
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

# ── KPI Cards ─────────────────────────────────────────────────────────────────
avg_drift   = df["drift_score"].mean()
avg_latency = df["execution_time_ms"].mean() / 1000
heal_count  = (
    df["path_taken"].str.contains("intervention", na=False).sum()
    if "path_taken" in df.columns else 0
)
high_risk_ct = (df["risk_level"] == "high_risk").sum() if "risk_level" in df.columns else 0

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
metric_card(c1, "Total Runs",      f"{total:,}")
metric_card(c2, "Avg Drift Score", f"{avg_drift:.1f}",  sub="0 = perfect")
metric_card(c3, "Avg Latency",     f"{avg_latency:.2f}s")

if IS_CLINICAL:
    # Clinical-specific KPIs
    avg_conf = df["overall_confidence"].mean() if "overall_confidence" in df.columns else 0
    review_ct = (df["decision"] == "requires_clinical_review").sum() if "decision" in df.columns else 0
    avg_privacy = df["privacy_leak_risk"].mean() if "privacy_leak_risk" in df.columns else 0
    
    metric_card(c4, "Avg Confidence",  f"{avg_conf:.2f}",   sub="coding accuracy")
    metric_card(c5, "Human Review",    f"{review_ct:,}",    sub="clinical_intervention")
    metric_card(c6, "De-ID Leak Risk", f"{avg_privacy:.2f}",sub="0 = safe")
else:
    esc_rate = (df["decision"] == "escalate").mean() * 100
    metric_card(c4, "Escalation Rate", f"{esc_rate:.1f}%")
    metric_card(c5, "Healing Events",  f"{heal_count:,}",   sub="intervention node")
    # c6 is intentionally left blank for Incident Triage to keep alignment
    metric_card(c6, "-", "-")

metric_card(c7, "High-Risk Runs",  f"{high_risk_ct:,}", sub="score ≥ 60")
st.markdown("<br>", unsafe_allow_html=True)

# ── Charts Row 1 ──────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("<div class='section-title'>📈 Drift Score Over Time</div>", unsafe_allow_html=True)
    cdf = df[["drift_score"]].reset_index(drop=True)
    cdf["rolling_avg"] = cdf["drift_score"].rolling(10, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cdf["drift_score"], name="Drift Score",
        line=dict(color=COLORS["blue"], width=1.5), fill="tozeroy",
        fillcolor="rgba(88,166,255,0.06)", hovertemplate="Run %{x}<br>Score: %{y}<extra></extra>"))
    fig.add_trace(go.Scatter(y=cdf["rolling_avg"], name="10-Run Avg",
        line=dict(color=COLORS["purple"], width=2, dash="dot"),
        hovertemplate="Avg: %{y:.1f}<extra></extra>"))
    fig.add_hrect(y0=60, y1=100, fillcolor="rgba(248,81,73,0.05)", line_width=0,
        annotation_text="High Risk", annotation_position="top right",
        annotation_font_color=COLORS["red"], annotation_font_size=11)
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(210,153,34,0.04)", line_width=0)
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with col_right:
    st.markdown("<div class='section-title'>⏱️ Latency Over Time</div>", unsafe_allow_html=True)
    ldf = df[["execution_time_ms"]].reset_index(drop=True)
    ldf["rolling_avg"] = ldf["execution_time_ms"].rolling(10, min_periods=1).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=ldf["execution_time_ms"], name="Latency (ms)",
        line=dict(color=COLORS["green"], width=1.5), fill="tozeroy",
        fillcolor="rgba(63,185,80,0.06)", hovertemplate="Run %{x}<br>%{y} ms<extra></extra>"))
    fig2.add_trace(go.Scatter(y=ldf["rolling_avg"], name="10-Run Avg",
        line=dict(color=COLORS["yellow"], width=2, dash="dot"),
        hovertemplate="Avg: %{y:.0f} ms<extra></extra>"))
    fig2.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=True)
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

# ── Charts Row 2 ──────────────────────────────────────────────────────────────
col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("<div class='section-title'>📊 Severity Distribution</div>", unsafe_allow_html=True)
    sev = df["severity"].value_counts().reset_index()
    sev.columns = ["Severity", "Count"]
    fig3 = px.bar(sev, x="Count", y="Severity", orientation="h", color="Severity",
        color_discrete_map={"low": COLORS["green"], "medium": COLORS["yellow"], "high": COLORS["red"]},
        text="Count")
    fig3.update_traces(textposition="outside", textfont_color=COLORS["text"], marker_line_width=0)
    fig3.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False, bargap=0.35)
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

with col_r2:
    st.markdown("<div class='section-title'>🎯 Decision Distribution</div>", unsafe_allow_html=True)
    dec = df["decision"].value_counts().reset_index()
    dec.columns = ["Decision", "Count"]
    fig4 = px.bar(dec, x="Count", y="Decision", orientation="h", color="Decision",
        color_discrete_map={"auto_resolve": COLORS["green"], "escalate": COLORS["red"]},
        text="Count")
    fig4.update_traces(textposition="outside", textfont_color=COLORS["text"], marker_line_width=0)
    fig4.update_layout(**PLOTLY_LAYOUT, height=200, showlegend=False, bargap=0.35)
    st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

# ── Risk Level Breakdown ──────────────────────────────────────────────────────
if "risk_level" in df.columns:
    st.markdown("<div class='section-title'>🔰 Risk Level Breakdown</div>", unsafe_allow_html=True)
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_counts["Percentage"] = (risk_counts["Count"] / total * 100).round(1).astype(str) + "%"
    r_cols = st.columns(len(risk_counts))
    color_cls = {"healthy": ("risk-healthy", "3fb950"),
                 "drift_detected": ("risk-drift", "d29922"), "high_risk": ("risk-high", "f85149")}
    for i, row in risk_counts.iterrows():
        cls, hex_c = color_cls.get(row["Risk Level"], ("", "58a6ff"))
        r_cols[i].markdown(
            f"<div class='metric-card' style='border-color:#{hex_c}22;'>"
            f"<div class='metric-label'>{row['Risk Level'].replace('_',' ').title()}</div>"
            f"<div class='metric-value {cls}'>{row['Count']}</div>"
            f"<div class='metric-sub'>{row['Percentage']}</div></div>",
            unsafe_allow_html=True)

# ── Alert Feed & Explainable AI ────────────────────────────────────────────────
st.markdown("<div class='section-title'>🚨 Recent Alerts & Explainable AI</div>", unsafe_allow_html=True)
alert_df = (df_raw[df_raw["risk_level"].isin(["high_risk", "drift_detected", "high", "critical"])].head(5)
            if "risk_level" in df_raw.columns else pd.DataFrame())
if alert_df.empty:
    st.markdown(
        "<div style='color:#3fb950;padding:12px;background:rgba(63,185,80,0.06);"
        "border:1px solid rgba(63,185,80,0.2);border-radius:8px;font-size:13px;'>"
        "✅ No active alerts — all executions are healthy.</div>", unsafe_allow_html=True)
else:
    for _, row in alert_df.iterrows():
        css_cls = "alert-item" if row["risk_level"] in ["high_risk", "high", "critical"] else "alert-item-drift"
        icon = "🔴" if row["risk_level"] in ["high_risk", "high", "critical"] else "🟡"
        
        explanation_html = ""
        explanation = row.get("ml_explanation")
        if pd.notna(explanation) and explanation:
            explanation_html = f"<div style='margin-top:8px;padding:8px;background:rgba(88,166,255,0.1);border-left:3px solid #58a6ff;border-radius:4px;color:#e6edf3;font-size:12px;'><b>🧠 SHAP Root Cause Analysis:</b> {explanation}</div>"
            
        st.markdown(
            f"<div class='{css_cls}'>"
            f"<div class='alert-id'>{icon} Incident <code>{row['incident_id']}</code> "
            f"— Drift Score <b>{row['drift_score']}</b> · {row['risk_level'].replace('_',' ').title()}</div>"
            f"<div class='alert-meta'>"
            f"Severity: {row.get('severity','—')} &nbsp;·&nbsp; "
            f"Decision: {row.get('decision','—')} &nbsp;·&nbsp; "
            f"Retries: {row.get('retry_count',0)} &nbsp;·&nbsp; "
            f"<span style='color:#3d444d'>{row.get('created_at','—')}</span>"
            f"</div>"
            f"{explanation_html}"
            f"</div>", unsafe_allow_html=True)

# ── Recent Executions Table ───────────────────────────────────────────────────
st.markdown("<div class='section-title'>📋 Recent Executions</div>", unsafe_allow_html=True)
display_cols = [c for c in [
    "incident_id", "severity", "decision", "confidence",
    "retry_count", "drift_score", "risk_level", "execution_time_ms", "created_at",
] if c in df.columns]
st.dataframe(
    df[display_cols].head(50), use_container_width=True, hide_index=True,
    column_config={
        "drift_score":       st.column_config.ProgressColumn("Drift Score", min_value=0, max_value=100, format="%d"),
        "confidence":        st.column_config.NumberColumn("Confidence", format="%.2f"),
        "execution_time_ms": st.column_config.NumberColumn("Latency (ms)", format="%d ms"),
        "risk_level":        st.column_config.TextColumn("Risk Level"),
    },
)

# ── Population Insights (Clinical Coding mode — Miimansa RWE pipeline) ────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🧬 Population Insights — Real World Evidence</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Aggregated statistics across all coded clinical records. "
        "This is the foundation of Real World Evidence (RWE) generation — "
        "mirroring Miimansa's 'Learning from Every Patient' pipeline."
    )

    pop_col1, pop_col2 = st.columns(2)

    with pop_col1:
        # Claims-ready rate gauge
        if "decision" in df.columns:
            complete_ct    = (df["decision"] == "complete").sum()
            total_clinical = len(df)
            claims_rate    = (complete_ct / total_clinical * 100) if total_clinical else 0

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=claims_rate,
                title={"text": "Claims-Ready Rate (%)", "font": {"color": "#e6edf3", "size": 14}},
                number={"suffix": "%", "font": {"color": "#58a6ff", "size": 28}},
                gauge={
                    "axis":  {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar":   {"color": "#3fb950"},
                    "steps": [
                        {"range": [0,  50], "color": "rgba(248,81,73,0.2)"},
                        {"range": [50, 80], "color": "rgba(210,153,34,0.2)"},
                        {"range": [80,100], "color": "rgba(63,185,80,0.2)"},
                    ],
                    "threshold": {"line": {"color": "#58a6ff", "width": 2}, "value": 85},
                    "bgcolor": "#1c2230", "bordercolor": "#2d3348",
                },
            ))
            fig_gauge.update_layout(
                paper_bgcolor="#161b22", font_color="#e6edf3",
                height=220, margin=dict(t=30, b=10, l=20, r=20),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Coding confidence trend over time
        if "overall_confidence" in df.columns and "created_at" in df.columns:
            df_trend = df[["created_at", "overall_confidence"]].dropna()
            if not df_trend.empty:
                df_trend["created_at"] = pd.to_datetime(df_trend["created_at"])
                df_trend = df_trend.sort_values("created_at")
                df_trend["rolling_conf"] = df_trend["overall_confidence"].rolling(5, min_periods=1).mean()
                fig_trend = px.line(
                    df_trend, x="created_at", y="rolling_conf",
                    title="Coding Confidence Over Time (5-run rolling avg)",
                    labels={"created_at": "", "rolling_conf": "Confidence"},
                    color_discrete_sequence=["#58a6ff"],
                )
                fig_trend.update_layout(
                    paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                    font_color="#e6edf3", title_font_size=13,
                    height=220, margin=dict(t=40, b=20, l=20, r=20),
                    yaxis=dict(range=[0, 1], gridcolor="#2d3348"),
                    xaxis=dict(gridcolor="#2d3348"),
                )
                fig_trend.add_hline(y=0.85, line_dash="dot", line_color="#3fb950",
                                    annotation_text="Claims-ready threshold")
                st.plotly_chart(fig_trend, use_container_width=True)

    with pop_col2:
        # Risk level distribution bar chart
        if "risk_level" in df.columns and "drift_score" in df.columns:
            risk_counts = df["risk_level"].value_counts().reset_index()
            risk_counts.columns = ["Risk Level", "Count"]
            color_map = {
                "healthy":        "#3fb950",
                "low_drift":      "#58a6ff",
                "moderate_drift": "#d29922",
                "high_risk":      "#f85149",
            }
            fig_risk = px.bar(
                risk_counts, x="Risk Level", y="Count",
                title="Risk Level Distribution",
                color="Risk Level", color_discrete_map=color_map,
            )
            fig_risk.update_layout(
                paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
                font_color="#e6edf3", title_font_size=13, showlegend=False,
                height=220, margin=dict(t=40, b=20, l=20, r=20),
                yaxis=dict(gridcolor="#2d3348"), xaxis=dict(gridcolor="#2d3348"),
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        # Population summary card
        if "overall_confidence" in df.columns:
            avg_conf = df["overall_confidence"].dropna().mean()
            std_conf = df["overall_confidence"].dropna().std()
            n_runs   = len(df)
            n_review = (df["decision"] == "requires_clinical_review").sum() if "decision" in df.columns else 0
            st.markdown(
                f"""
                <div style="background:#1c2230;border:1px solid #2d3348;border-radius:8px;
                            padding:16px;font-size:13px;color:#e6edf3;margin-top:8px;">
                  <div style="font-weight:700;color:#58a6ff;margin-bottom:8px;">📊 Population Summary</div>
                  <table style="width:100%;border-collapse:collapse;">
                    <tr><td style="color:#8b949e;padding:3px 0;">Total coded records</td>
                        <td style="text-align:right;font-weight:600;">{n_runs:,}</td></tr>
                    <tr><td style="color:#8b949e;padding:3px 0;">Avg coding confidence</td>
                        <td style="text-align:right;font-weight:600;">{avg_conf:.3f} ± {std_conf:.3f}</td></tr>
                    <tr><td style="color:#8b949e;padding:3px 0;">Sent to human review</td>
                        <td style="text-align:right;font-weight:600;color:#f85149;">{n_review:,}</td></tr>
                    <tr><td style="color:#8b949e;padding:3px 0;">Human review rate</td>
                        <td style="text-align:right;font-weight:600;">{(n_review/n_runs*100):.1f}%</td></tr>
                  </table>
                  <div style="margin-top:12px;font-size:11px;color:#3d444d;
                              border-top:1px solid #2d3348;padding-top:8px;">
                    ⚕️ Adaptive baseline updates with each run — the more records coded,
                    the tighter the drift detection becomes.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── SDOH Risk Trajectories Section ───────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div class='section-title'>🧬 SDOH Longitudinal Risk Trajectories</div>",
    unsafe_allow_html=True,
)
st.caption(
    "Social Determinants of Health (SDOH) risk trajectory model — "
    "tracks patient risk across visits using lifestyle, demographic, and environmental signals."
)

if "sdoh_risk_label" in df.columns and "sdoh_risk_score" in df.columns:
    sdoh_df = df.dropna(subset=["sdoh_risk_label"]).copy()
    
    if not sdoh_df.empty:
        sdoh_col1, sdoh_col2 = st.columns([1, 2])
        
        with sdoh_col1:
            st.markdown(
                "<div class='section-title'>🌍 Population Risk Distribution</div>",
                unsafe_allow_html=True,
            )
            label_counts = sdoh_df["sdoh_risk_label"].value_counts().reset_index()
            label_counts.columns = ["Risk Label", "Count"]
            color_map = {
                "low": COLORS["green"], "moderate": COLORS["yellow"],
                "high": COLORS["red"], "critical": "#ff0000",
            }
            fig_pie = px.pie(
                label_counts, names="Risk Label", values="Count",
                color="Risk Label", color_discrete_map=color_map,
                title="Risk Label Distribution"
            )
            fig_pie.update_layout(
                paper_bgcolor="#161b22", font_color="#e6edf3",
                title_font_size=13, height=320,
                margin=dict(t=40, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
            
        with sdoh_col2:
            st.markdown(
                "<div class='section-title'>⚠️ Explanatory AI (SHAP Drivers)</div>",
                unsafe_allow_html=True,
            )
            
            # Let the user pick an incident to drill down
            incidents_with_shap = sdoh_df[sdoh_df["sdoh_shap_factors"].notna()]["incident_id"].tolist()
            if incidents_with_shap:
                selected_incident = st.selectbox("Select Record to Explain", incidents_with_shap)
                
                record = sdoh_df[sdoh_df["incident_id"] == selected_incident].iloc[0]
                
                st.markdown(
                    f"<div style='color:#e6edf3;font-size:13px;margin-bottom:8px;'>"
                    f"Risk Score: <b style='color:#58a6ff'>{record['sdoh_risk_score']:.3f}</b> &nbsp;·&nbsp; "
                    f"Label: <b style='color:{color_map.get(record['sdoh_risk_label'], '#ffffff')}'>{record['sdoh_risk_label'].upper()}</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
                try:
                    import json
                    shap_data = record["sdoh_shap_factors"]
                    if isinstance(shap_data, str):
                        shap_data = json.loads(shap_data)
                        
                    if shap_data and isinstance(shap_data, list):
                        factor_df = pd.DataFrame(shap_data).sort_values("shap_value", ascending=True)
                        fig_factors = px.bar(
                            factor_df, x="shap_value", y="feature", orientation="h",
                            color="shap_value",
                            color_continuous_scale=[[0, COLORS["green"]], [0.5, COLORS["yellow"]], [1, COLORS["red"]]],
                        )
                        fig_factors.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False,
                                                  coloraxis_showscale=False)
                        st.plotly_chart(fig_factors, use_container_width=True, config={"displayModeBar": False})
                    else:
                        st.info("No SHAP data available for this record.")
                except Exception as e:
                    st.error(f"Error parsing SHAP data: {e}")
            else:
                st.info("No records with SHAP explanations found.")

# ── Human-in-the-Loop Clinical Review Portal ─────────────────────────────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>⚕️ Human-in-the-Loop Clinical Review Portal</div>",
        unsafe_allow_html=True,
    )
    st.caption("Interactive clinical override & audit interface for records marked `requires_clinical_review`.")

    try:
        from telemetry.store import get_pending_reviews, get_review_history, save_human_intervention, update_execution_human_status
        pending_list = get_pending_reviews(limit=50)
        history_list = get_review_history(limit=50)

        review_tab1, review_tab2 = st.tabs([f"📥 Pending Queue ({len(pending_list)})", f"📜 Review Audit History ({len(history_list)})"])

        with review_tab1:
            if pending_list:
                rec_ids = [p["incident_id"] for p in pending_list]
                selected_rec_id = st.selectbox("Select Flagged Record to Review", rec_ids)
                selected_p = next(p for p in pending_list if p["incident_id"] == selected_rec_id)

                col_rev1, col_rev2 = st.columns([1, 1])

                with col_rev1:
                    st.markdown("#### Record Summary")
                    st.json({
                        "Record ID": selected_p["incident_id"],
                        "Decision": selected_p["decision"],
                        "Retry Count": selected_p.get("retry_count", 0),
                        "Confidence": selected_p.get("overall_confidence", 0.0),
                        "SDOH Risk Label": selected_p.get("sdoh_risk_label", "N/A"),
                        "Created At": selected_p.get("created_at", "N/A")
                    })

                with col_rev2:
                    st.markdown("#### Clinician Override Action")
                    action_choice = st.radio("Review Action", ["approved", "edited", "rejected"], horizontal=True)
                    clinician_name = st.text_input("Clinician Identifier", value="Dr. Alex Taylor")
                    review_notes = st.text_area("Clinical Notes & Rationale", placeholder="Provide clinical justification for code approval or override...")
                    custom_codes_input = st.text_input("Final ICD-10 Codes (comma-separated if editing)", value="E11.9, I10")

                    if st.button("💾 Submit Clinical Review", type="primary", use_container_width=True):
                        # Parse custom codes
                        parsed_codes = [{"code": c.strip(), "description": "Clinician Override"} for c in custom_codes_input.split(",") if c.strip()]
                        
                        new_status = "approved_by_clinician" if action_choice in ("approved", "edited") else "rejected_by_clinician"
                        
                        save_human_intervention(
                            incident_id=selected_rec_id,
                            action=action_choice,
                            reviewed_by=clinician_name,
                            notes=review_notes,
                            original_codes=[],
                            final_codes=parsed_codes
                        )
                        update_execution_human_status(
                            record_id=selected_rec_id,
                            new_status=new_status,
                            human_action=action_choice,
                            notes=review_notes,
                            reviewed_by=clinician_name,
                            final_codes=parsed_codes
                        )
                        st.toast(f"✅ Record '{selected_rec_id}' updated to {new_status}!")
                        st.cache_data.clear()
                        st.rerun()
            else:
                st.success("🎉 No pending records requiring clinical review. All extraction models operating within bounds.")

        with review_tab2:
            if history_list:
                hist_df = pd.DataFrame(history_list)
                st.dataframe(
                    hist_df[["incident_id", "action", "reviewed_by", "notes", "created_at"]],
                    use_container_width=True
                )
            else:
                st.info("No past human interventions recorded yet.")

    except Exception as exc:
        st.error(f"Error loading Clinical Review Portal: {exc}")

# ── PRISM v2 Live Clinical Trial Matching Explorer ──────────────────────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🎯 PRISM v2 Live Clinical Trial Matching Engine</div>",
        unsafe_allow_html=True,
    )
    st.caption("Queries live recruiting studies on ClinicalTrials.gov API v2 and matches patient biomarkers and staging criteria.")

    col_t1, col_t2 = st.columns([3, 1])
    with col_t1:
        search_condition = st.text_input("Search Condition / Histology", value="Non-Small Cell Lung Cancer")
    with col_t2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_btn = st.button("🔎 Fetch Live Trials", type="primary", use_container_width=True)

    try:
        from clinical.tools.clinical_trials_api import search_recruiting_trials
        trials_list = search_recruiting_trials(condition=search_condition, limit=4)

        if trials_list:
            t_cols = st.columns(len(trials_list))
            for idx, trial in enumerate(trials_list):
                with t_cols[idx]:
                    st.markdown(
                        f"""
                        <div style="background:#161b22;border:1px solid #2d3348;border-radius:10px;padding:14px;height:100%;">
                            <span class="badge badge-healthy">{trial.get('phase', 'PHASE2')}</span>
                            <div style="font-weight:700;color:#58a6ff;font-size:14px;margin-top:6px;">{trial['nct_id']}</div>
                            <div style="font-size:12px;color:#e6edf3;font-weight:600;margin:6px 0;line-height:1.3;">{trial['brief_title'][:65]}...</div>
                            <div style="font-size:11px;color:#8b949e;">Sponsor: <b>{trial.get('sponsor', 'Academic')}</b></div>
                            <div style="font-size:11px;color:#3fb950;margin-top:8px;font-weight:600;">Eligible Range: 18+ Yrs · {trial.get('gender', 'ALL')}</div>
                            <a href="{trial['url']}" target="_blank" style="display:inline-block;margin-top:10px;font-size:12px;color:#bc8cff;text-decoration:none;font-weight:600;">🔗 View Study on ClinicalTrials.gov →</a>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        else:
            st.info("No recruiting trials found for this condition.")
    except Exception as exc:
        st.error(f"Error fetching live clinical trials: {exc}")

# ── Pharmacovigilance & Drug Safety Scanner ────────────────────────────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>💊 Pharmacovigilance & Drug Safety Scanner</div>",
        unsafe_allow_html=True,
    )
    st.caption("Scans medication regimens for NLM RxNav high-severity drug interactions and extracts unstructured Adverse Drug Reaction (ADR) signals.")

    col_p1, col_p2 = st.columns([2, 1])

    with col_p1:
        med_input = st.text_input("Prescribed Medications (comma-separated)", value="Warfarin, Aspirin, Keytruda")
        adr_note_input = st.text_area("Patient Symptoms / Clinical Note (for ADR signal extraction)", value="Patient reports epistaxis and severe skin rash after starting Keytruda 10 days ago.")

    with col_p2:
        st.markdown("<br>", unsafe_allow_html=True)
        scan_btn = st.button("🛡️ Run Safety Scan", type="primary", use_container_width=True)

    try:
        from clinical.tools.pharmacovigilance_api import check_drug_interactions
        from clinical.steps.pharmacovigilance_step import _extract_adverse_reactions, _compute_safety_risk

        med_list = [m.strip() for m in med_input.split(",") if m.strip()]
        interactions = check_drug_interactions(med_list) if len(med_list) >= 2 else []
        adrs = _extract_adverse_reactions(adr_note_input, med_list)
        overall_risk = _compute_safety_risk(interactions, adrs)

        risk_color_map = {"low": "#3fb950", "moderate": "#d29922", "high": "#f85149", "critical": "#ff0000"}

        st.markdown(
            f"<div style='font-size:14px;color:#e6edf3;margin-bottom:12px;'>"
            f"Overall Safety Risk: <b style='color:{risk_color_map.get(overall_risk, '#ffffff')}'>{overall_risk.upper()}</b>"
            f"</div>",
            unsafe_allow_html=True
        )

        col_out1, col_out2 = st.columns(2)

        with col_out1:
            st.markdown("##### ⚠️ Drug-Drug Interactions")
            if interactions:
                for inter in interactions:
                    st.markdown(
                        f"""
                        <div class="alert-item">
                            <div class="alert-id">⚡ {inter['pair']}</div>
                            <div class="alert-meta">{inter['description']}</div>
                            <div style="font-size:10px;color:#8b949e;margin-top:4px;">Source: {inter['source']}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.success("No severe drug-drug interactions detected.")

        with col_out2:
            st.markdown("##### 🩺 Extracted ADR Signals")
            if adrs:
                for adr in adrs:
                    st.markdown(
                        f"""
                        <div class="alert-item-drift">
                            <div class="alert-id">🚩 {adr['category']}: {adr['symptom']}</div>
                            <div class="alert-meta">Suspected Drug: <b>{adr['suspected_drug']}</b> · Evidence: <i>"{adr['evidence_span']}"</i></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info("No adverse drug reaction signals detected in note text.")

    except Exception as exc:
        st.error(f"Error running pharmacovigilance scan: {exc}")

# ── CMS Financial RAF & RADV Audit Scorecard ────────────────────────────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>📊 CMS Financial RAF & RADV Audit Scorecard</div>",
        unsafe_allow_html=True,
    )
    st.caption("Calculates patient Risk Adjustment Factor (RAF) scores and audits Medicare RADV financial clawback exposure ($ USD) for missing MEAT documentation.")

    try:
        from clinical.tools.raf_audit_calculator import calculate_raf_audit_metrics

        sample_codes_for_audit = [
            {"code": "E11.40", "description": "Type 2 diabetes with diabetic neuropathy", "hcc_category": "HCC 18", "raf_weight": 0.368, "meat_met": True},
            {"code": "I50.9", "description": "Heart failure, unspecified", "hcc_category": "HCC 85", "raf_weight": 0.323, "meat_met": True},
            {"code": "J44.9", "description": "Chronic obstructive pulmonary disease, unspecified", "hcc_category": "HCC 111", "raf_weight": 0.335, "meat_met": False},
        ]

        raf_metrics = calculate_raf_audit_metrics(sample_codes_for_audit, {"age": 74, "gender": "M"})

        raf_col1, raf_col2, raf_col3, raf_col4 = st.columns(4)
        with raf_col1:
            metric_card(raf_col1, "Total Patient RAF", f"{raf_metrics['total_raf_score']:.3f}", sub="Base Demo + Disease")
        with raf_col2:
            metric_card(raf_col2, "Verified Compliant RAF", f"{raf_metrics['verified_raf_score']:.3f}", sub="MEAT Proof Present")
        with raf_col3:
            metric_card(raf_col3, "Unverified RAF Risk", f"{raf_metrics['unverified_raf_score']:.3f}", sub="Missing MEAT Proof")
        with raf_col4:
            metric_card(raf_col4, "RADV Audit Exposure", f"${raf_metrics['radv_financial_exposure_usd']:,.2f}", sub="Annual Clawback Risk")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 📋 RADV Code Compliance Audit Detail")
        audit_df = pd.DataFrame(raf_metrics["code_audit_details"])
        st.dataframe(audit_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Error loading CMS RAF & RADV Audit Scorecard: {exc}")

# ── SYMPHONY v2 Longitudinal Disease Timeline & RECIST 1.1 Tracker ────────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>📜 SYMPHONY v2 Longitudinal Patient Timeline & RECIST 1.1 Tracker</div>",
        unsafe_allow_html=True,
    )
    st.caption("Synthesizes multi-visit patient notes into chronological timelines and tracks serial radiology target lesion diameters per RECIST 1.1 standards.")

    try:
        from clinical.tools.symphony_engine import synthesize_patient_timeline

        sample_visit_history = [
            {"date": "2025-10-10", "doc_type": "Baseline CT Scan", "summary": "Right upper lobe lung lesion measuring 45mm. EGFR exon 19 deletion detected.", "target_lesion_mm": 45.0, "new_lesions": False},
            {"date": "2026-02-14", "doc_type": "Follow-up CT #1", "summary": "Partial response observed after Osimertinib therapy. Primary lesion reduced to 30mm (-33.3%).", "target_lesion_mm": 30.0, "new_lesions": False},
            {"date": "2026-07-01", "doc_type": "Follow-up CT #2", "summary": "Continued response. Primary target lesion further decreased to 22mm (-51.1% from baseline). No new metastases.", "target_lesion_mm": 22.0, "new_lesions": False},
        ]

        sym_res = synthesize_patient_timeline(sample_visit_history)

        sym_col1, sym_col2 = st.columns([1, 2])

        with sym_col1:
            st.markdown(
                f"""
                <div style="background:#161b22;border:1px solid #2d3348;border-radius:10px;padding:16px;margin-bottom:12px;">
                    <div style="font-size:12px;color:#8b949e;text-transform:uppercase;font-weight:600;">RECIST 1.1 Overall Response</div>
                    <div style="font-size:28px;font-weight:800;color:#3fb950;margin:6px 0;">{sym_res['recist_overall_response']} (PARTIAL RESPONSE)</div>
                    <div style="font-size:13px;color:#e6edf3;">Target Lesion Change: <b style="color:#3fb950;">{sym_res['recist_delta_pct']:+.1f}%</b></div>
                    <div style="font-size:11px;color:#8b949e;margin-top:8px;">{sym_res['pre_chart_summary']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with sym_col2:
            st.markdown("##### 📈 Serial Target Lesion Trajectory (mm)")
            measures_df = pd.DataFrame(sym_res["serial_measurements"])
            fig_lesion = px.line(
                measures_df,
                x="date",
                y="target_lesion_mm",
                markers=True,
                labels={"target_lesion_mm": "Lesion Sum (mm)", "date": "Scan Date"},
                title="Target Lesion Diameter Over Time (RECIST 1.1)"
            )
            fig_lesion.update_traces(line_color="#3fb950", marker_size=10)
            fig_lesion.update_layout(
                template="plotly_dark",
                height=220,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_lesion, use_container_width=True)

        st.markdown("##### 🗓️ Chronological Patient Visit Timeline")
        timeline_cols = st.columns(len(sym_res["chronological_timeline"]))
        for idx, visit_item in enumerate(sym_res["chronological_timeline"]):
            with timeline_cols[idx]:
                st.markdown(
                    f"""
                    <div style="background:#161b22;border:1px solid #2d3348;border-radius:10px;padding:12px;height:100%;">
                        <div style="font-weight:700;color:#58a6ff;font-size:13px;">📅 {visit_item['date']}</div>
                        <div style="font-size:11px;color:#bc8cff;font-weight:600;">{visit_item['doc_type']}</div>
                        <div style="font-size:12px;color:#e6edf3;margin-top:6px;line-height:1.3;">{visit_item['summary']}</div>
                        <div style="font-size:11px;color:#3fb950;margin-top:8px;font-weight:600;">Lesion: {visit_item['target_lesion_mm']} mm</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    except Exception as exc:
        st.error(f"Error loading SYMPHONY v2 Longitudinal Engine: {exc}")

# ── SMART-on-FHIR R4 Adapter & Synthetic Patient Generator ────────────────────
if IS_CLINICAL:
    st.markdown("---")
    st.markdown(
        "<div class='section-title'>🔥 SMART-on-FHIR R4 Adapter &amp; Synthetic Patient Generator</div>",
        unsafe_allow_html=True,
    )
    st.caption("Exports extracted clinical state objects to HL7 FHIR R4 JSON Bundles (Patient, Condition, MedicationStatement, Observation) and generates synthetic patient charts.")

    fhir_col1, fhir_col2 = st.columns([1, 1])

    with fhir_col1:
        st.markdown("##### 🎲 Synthetic Patient Chart Generator")
        synth_condition = st.text_input("Oncology Condition", value="Non-Small Cell Lung Cancer", key="synth_cond")
        if st.button("✨ Seed Synthetic Patient Record", type="primary", use_container_width=True):
            try:
                from clinical.tools.fhir_adapter import generate_synthetic_patient_chart
                synth_data = generate_synthetic_patient_chart(synth_condition)
                st.session_state["active_synth_chart"] = synth_data
                st.success(f"Generated patient {synth_data['patient_id']} ({synth_data['demographics']['age']}yo {synth_data['demographics']['gender']})")
            except Exception as exc:
                st.error(f"Error seeding synthetic patient chart: {exc}")

        if "active_synth_chart" in st.session_state:
            active_p = st.session_state["active_synth_chart"]
            st.text_area("Generated Synthetic Note Text", value=active_p["raw_note"], height=160)

    with fhir_col2:
        st.markdown("##### 📦 HL7 FHIR R4 Bundle Exporter")
        try:
            from clinical.tools.fhir_adapter import export_clinical_state_to_fhir
            sample_state_for_fhir = {
                "record_id": "FHIR-DEMO-001",
                "demographics": {"age": 68, "gender": "M"},
                "icd10_codes": [
                    {"code": "C34.11", "description": "Malignant neoplasm of upper lobe, right bronchus or lung", "hcc_category": "HCC 12", "meat_met": True},
                    {"code": "E11.40", "description": "Type 2 diabetes with diabetic neuropathy", "hcc_category": "HCC 18", "meat_met": True}
                ],
                "extracted_medications": [
                    {"drug_name": "Osimertinib", "rxcui": "1730058"},
                    {"drug_name": "Warfarin", "rxcui": "11289"}
                ],
                "sdoh_risk_label": "moderate",
                "total_raf_score": 0.852
            }
            demo_fhir_bundle = export_clinical_state_to_fhir(sample_state_for_fhir)
            st.json(demo_fhir_bundle, expanded=False)
            st.download_button(
                label="📥 Download HL7 FHIR R4 Bundle (JSON)",
                data=json.dumps(demo_fhir_bundle, indent=2),
                file_name="patient_fhir_r4_bundle.json",
                mime="application/json",
                use_container_width=True
            )
        except Exception as exc:
            st.error(f"Error exporting FHIR Bundle: {exc}")

# ── Footer ────────────────────────────────────────────────────────────────────
backend_label = "PostgreSQL + TimescaleDB" if USE_POSTGRES else "SQLite"
st.markdown(
    f"<div class='footer'>Agentic Drift Detector &nbsp;·&nbsp; "
    f"Built with LangGraph &amp; Streamlit &nbsp;·&nbsp; {backend_label}</div>",
    unsafe_allow_html=True,
)






