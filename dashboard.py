import sqlite3
import streamlit as st
import pandas as pd
import os

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic Drift Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background-color: #0d1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #232938 100%);
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-label { color: #8b949e; font-size: 13px; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 6px; }
    .metric-value { color: #e6edf3; font-size: 32px; font-weight: 700; line-height: 1; }
    .metric-sub   { color: #58a6ff; font-size: 13px; margin-top: 4px; }

    .risk-healthy       { color: #3fb950; }
    .risk-drift         { color: #d29922; }
    .risk-high          { color: #f85149; }

    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-healthy { background: rgba(63,185,80,0.15); color: #3fb950; }
    .badge-drift   { background: rgba(210,153,34,0.15); color: #d29922; }
    .badge-high    { background: rgba(248,81,73,0.15);  color: #f85149; }

    .section-title {
        color: #e6edf3;
        font-size: 18px;
        font-weight: 600;
        margin: 28px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #2d3348;
    }

    .stDataFrame { border-radius: 8px; }
    div[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #2d3348; }
</style>
""", unsafe_allow_html=True)

# ─── Database ────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "telemetry.db")

@st.cache_data(ttl=10)
def load_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM executions ORDER BY created_at DESC",
        conn
    )
    conn.close()
    return df

def risk_badge(risk: str) -> str:
    cls = {"healthy": "badge-healthy", "drift_detected": "badge-drift", "high_risk": "badge-high"}.get(risk, "badge-drift")
    return f'<span class="badge {cls}">{risk}</span>'

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Agentic Drift Detector")
    st.markdown("Real-time telemetry dashboard for monitoring autonomous AI agent behavior.")
    st.markdown("---")
    st.markdown("### Filters")
    min_score = st.slider("Min Drift Score", 0, 100, 0)
    severity_filter = st.multiselect("Severity", ["low", "medium", "high"], default=["low", "medium", "high"])
    decision_filter = st.multiselect("Decision", ["auto_resolve", "escalate"], default=["auto_resolve", "escalate"])
    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

# ─── Load & Filter ───────────────────────────────────────────────────────────
df_raw = load_data()

if df_raw.empty:
    st.title("🧠 Agentic Drift Detector")
    st.warning("No telemetry data found. Run `python run.py --simulate-batch 50` to build a baseline first.")
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

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#e6edf3; font-size:28px; margin-bottom:4px;'>🧠 Agentic Drift Detector</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:#8b949e;'>Showing <b>{total}</b> of <b>{len(df_raw)}</b> total executions</p>", unsafe_allow_html=True)

# ─── KPI Cards ───────────────────────────────────────────────────────────────
avg_drift    = df["drift_score"].mean()
avg_latency  = df["execution_time_ms"].mean() / 1000
avg_retries  = df["retry_count"].mean()
esc_rate     = (df["decision"] == "escalate").mean() * 100
heal_count   = (df["path_taken"].str.contains("intervention", na=False)).sum() if "path_taken" in df.columns else 0
high_risk_ct = (df["risk_level"] == "high_risk").sum() if "risk_level" in df.columns else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)

def metric_card(col, label, value, sub=""):
    col.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
        </div>
    """, unsafe_allow_html=True)

metric_card(c1, "Total Runs",       f"{total:,}")
metric_card(c2, "Avg Drift Score",  f"{avg_drift:.1f}", sub="0 = perfect")
metric_card(c3, "Avg Latency",      f"{avg_latency:.2f}s")
metric_card(c4, "Escalation Rate",  f"{esc_rate:.1f}%")
metric_card(c5, "Healing Events",   f"{heal_count:,}",  sub="intervention node")
metric_card(c6, "High-Risk Runs",   f"{high_risk_ct:,}", sub="score >= 60")

st.markdown("---")

# ─── Charts ──────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("<div class='section-title'>Drift Score Over Time</div>", unsafe_allow_html=True)
    chart_df = df[["drift_score"]].reset_index(drop=True)
    chart_df.index.name = "Run #"
    st.line_chart(chart_df, color="#58a6ff", height=240)

with col_right:
    st.markdown("<div class='section-title'>Latency Over Time (ms)</div>", unsafe_allow_html=True)
    lat_df = df[["execution_time_ms"]].reset_index(drop=True)
    lat_df.index.name = "Run #"
    st.line_chart(lat_df, color="#3fb950", height=240)

col_l2, col_r2 = st.columns(2)

with col_l2:
    st.markdown("<div class='section-title'>Severity Distribution</div>", unsafe_allow_html=True)
    sev_counts = df["severity"].value_counts()
    st.bar_chart(sev_counts, color="#d29922", height=220)

with col_r2:
    st.markdown("<div class='section-title'>Decision Distribution</div>", unsafe_allow_html=True)
    dec_counts = df["decision"].value_counts()
    st.bar_chart(dec_counts, color="#bc8cff", height=220)

# ─── Risk Level Breakdown ────────────────────────────────────────────────────
if "risk_level" in df.columns:
    st.markdown("<div class='section-title'>Risk Level Breakdown</div>", unsafe_allow_html=True)
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    risk_counts["Percentage"] = (risk_counts["Count"] / total * 100).round(1).astype(str) + "%"

    r_cols = st.columns(len(risk_counts))
    for i, row in risk_counts.iterrows():
        cls = {"healthy": "risk-healthy", "drift_detected": "risk-drift", "high_risk": "risk-high"}.get(row["Risk Level"], "")
        r_cols[i].markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-label'>{row['Risk Level'].replace('_', ' ').title()}</div>"
            f"<div class='metric-value {cls}'>{row['Count']}</div>"
            f"<div class='metric-sub'>{row['Percentage']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

# ─── Recent Executions Table ─────────────────────────────────────────────────
st.markdown("<div class='section-title'>Recent Executions</div>", unsafe_allow_html=True)

display_cols = [c for c in ["incident_id", "severity", "decision", "confidence", "retry_count", "drift_score", "risk_level", "execution_time_ms", "created_at"] if c in df.columns]
st.dataframe(
    df[display_cols].head(50),
    use_container_width=True,
    hide_index=True,
    column_config={
        "drift_score":       st.column_config.ProgressColumn("Drift Score", min_value=0, max_value=100),
        "confidence":        st.column_config.NumberColumn("Confidence", format="%.2f"),
        "execution_time_ms": st.column_config.NumberColumn("Latency (ms)", format="%d ms"),
    }
)

st.markdown("<p style='text-align:center; color:#30363d; font-size:12px; margin-top:32px;'>Agentic Drift Detector — Built with LangGraph & Streamlit</p>", unsafe_allow_html=True)
