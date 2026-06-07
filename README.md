# Agentic Drift Detector

Detect **behavioral drift and semantic bias** in autonomous, agentic AI workflows by analyzing execution telemetry — even when the system does not explicitly fail.

---

## 🧠 Why This Project Exists

Modern agentic AI systems rarely fail loudly.
Instead, they **silently drift**:

* More steps than before
* Excessive retries
* Escalation bias
* Classification bias
* Rising latency and cost

These issues don't throw errors — they **erode reliability over time**.

**Agentic Drift Detector** is a production-grade reference implementation that shows how to:

* Orchestrate agentic workflows using **LangGraph**
* Instrument and persist execution behavior in **SQLite**
* Detect semantic drift using population-level baselines
* **Heal** the agent automatically when a drift loop is detected
* Visualize everything in a live **Streamlit dashboard**
* Alert teams via **Slack and Discord webhooks**

---

## 🎯 Core Idea

> *Treat agent execution as a behavioral system, not just a prompt pipeline.*

Instead of validating outputs, this project monitors:

* Execution paths
* Retry patterns
* Step ordering
* Decision instability
* Semantic bias (escalation & classification)

This allows early detection of **autonomy degradation** — and automatic correction.

---

## 🏗️ Architecture Overview

```
Incident Trigger
      ↓
LangGraph State Machine Engine
      ↓
LLM-Powered Agent Nodes (Triage → Investigation → Decision → Notification)
      ↓                             ↑
      └──── Drift Loop? ────→ Intervention (Agentic Healing)
      ↓
Execution Telemetry (SQLite)
      ↓
Drift Detection Engine
      ↓
Webhook Alerting (Slack / Discord) + Streamlit Dashboard
```

Each layer has a **single responsibility**, making the system observable, extensible, and self-healing.

---

## 🤖 Agentic Workflow

The incident triage workflow consists of autonomous nodes managed by LangGraph:

1. **Triage Node**
   * Uses **ChatOpenAI (gpt-3.5-turbo)** to classify incident severity
   * Falls back to weighted simulation if no API key is configured

2. **Investigation Node**
   * Gathers contextual evidence from the incident state

3. **Decision Node**
   * Uses **ChatOpenAI** to decide: `escalate` or `auto_resolve`
   * Returns a confidence score between 0.0 and 1.0

4. **Intervention Node** *(Agentic Healing)*
   * Triggered automatically if retry_count ≥ 2 (persistent drift loop detected)
   * Forces the agent back to a stable, deterministic decision
   * Prevents infinite loops without crashing the workflow

5. **Notification Node**
   * Communicates outcomes
   * Triggers webhook alerts on drift events

---

## 📡 Telemetry & Observability

Each execution emits telemetry including:

* Step name, execution order, retry count, path taken
* Execution latency (ms)
* Drift score and risk level

Telemetry is stored in a **SQLite database** enabling:

* Population-level baseline calculations
* Categorical rate tracking (Escalation Rate, High-Severity Rate)
* Live visualization in the Streamlit dashboard

---

## 🚨 Drift & Bias Detection

Drift is defined as **deviation from historically stable behavior**, not explicit failure.

The drift engine detects:

| Signal | Trigger |
|---|---|
| **Retry Drift** | Retry count exceeds historical average |
| **Step Count Drift** | More steps than the historical norm |
| **Latency Drift** | Execution time > 1.5× the historical average |
| **Escalation Bias** | Agent escalates a low-severity incident against historical norms |
| **Classification Bias** | Agent over-classifies incidents as high-severity |

---

## 🩹 Agentic Healing

When the system detects a **persistent retry loop** (retry_count ≥ 2 with low confidence), the LangGraph conditional edge routes to the `intervention` node which:

1. Logs the healing event
2. Forcibly resets confidence to a stable value
3. Sets the decision to `escalate` (safest action under uncertainty)
4. Passes control cleanly to the notification node

This transforms the detector from **passive observer** to **active guardian**.

---

## 🛠️ Tech Stack

* **Python** — core language
* **LangGraph** — deterministic state machine orchestration
* **LangChain + OpenAI** — real LLM-powered triage and decision nodes (optional, with simulation fallback)
* **SQLite** — population-level telemetry persistence
* **Streamlit** — live telemetry visualization dashboard
* **Slack / Discord Webhooks** — real-time drift alerts
* **pytest** — automated test suite (8 tests)

---

## 🚀 Running the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment (optional)
Create a `.env` file:
```env
# Optional — enables real LLM reasoning
OPENAI_API_KEY=sk-...

# Optional — enables real-time Slack alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

# Optional — enables real-time Discord alerts
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

> If no API key is provided, the workflow runs in **simulation mode** automatically.

### 3. Build a healthy baseline
```bash
python run.py --simulate-batch 60
```

### 4. Run a single execution (or trigger biased simulation)
```bash
# Normal run
python run.py

# Trigger classification + escalation bias
python run.py --bias
```

### 5. Launch the Streamlit dashboard
```bash
streamlit run dashboard.py
```

### 6. Run the test suite
```bash
python -m pytest
```

---

## 🧪 Test Suite

| Test | Description |
|---|---|
| `test_escalation_bias_anomalous` | Verifies +25 penalty for anomalous low-severity escalation |
| `test_escalation_bias_normal` | Verifies zero penalty for healthy auto-resolution |
| `test_classification_bias_anomalous` | Verifies +20 penalty for anomalous high-severity classification |
| `test_latency_drift_high` | Verifies max latency penalty for severe degradation |
| `test_step_count_drift` | Verifies step count penalty calculation |
| `test_workflow_healthy_auto_resolve` | Integration: healthy path → auto_resolve |
| `test_workflow_low_confidence_single_retry` | Integration: one retry, no intervention |
| `test_workflow_intervention_on_drift_loop` | Integration: drift loop triggers healing node |

---

## 🔮 Roadmap

* Multi-incident workflow support
* LangSmith trace integration
* ML-based anomaly detection (replacing rule-based signals)
* Cloud deployment (Cloud Run)

---

## 👤 Author

Built as a systems-level exploration of **agentic AI reliability, observability, autonomy drift, and self-healing**.

This project emphasizes **engineering judgment over demos**.

---

## 📜 License

MIT
