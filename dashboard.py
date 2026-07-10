import json
import os
import sqlite3
import time

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
from dotenv import load_dotenv
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
                    import subprocess, sys
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

# ── Footer ────────────────────────────────────────────────────────────────────
backend_label = "PostgreSQL + TimescaleDB" if USE_POSTGRES else "SQLite"
st.markdown(
    f"<div class='footer'>Agentic Drift Detector &nbsp;·&nbsp; "
    f"Built with LangGraph &amp; Streamlit &nbsp;·&nbsp; {backend_label}</div>",
    unsafe_allow_html=True,
)
