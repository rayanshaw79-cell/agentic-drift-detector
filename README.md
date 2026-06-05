# Agentic Drift Detector

Detect **behavioral drift** in autonomous, agentic AI workflows by analyzing execution telemetry — even when the system does not explicitly fail.

---

## 🧠 Why This Project Exists

Modern agentic AI systems rarely fail loudly.
Instead, they **silently drift**:

* More steps than before
* Excessive retries
* Tool overuse
* Escalation bias
* Rising latency and cost

These issues don’t throw errors — they **erode reliability over time**.

**Agentic Drift Detector** is a reference implementation that shows how to:

* Instrument agentic workflows
* Capture execution behavior
* Detect drift before it becomes an incident

---

## 🎯 Core Idea

> *Treat agent execution as a behavioral system, not just a prompt pipeline.*

Instead of validating outputs, this project monitors:

* Execution paths
* Retry patterns
* Step ordering
* Decision instability

This allows early detection of **autonomy degradation**.

---

## 🏗️ Architecture Overview

```
Incident Trigger
      ↓
LangGraph State Machine Engine
      ↓
Agent Nodes (Triage → Investigation → Decision → Notification)
      ↓
Execution Telemetry (SQLite & LangSmith)
      ↓
Drift Detection Engine
```

Each layer has a **single responsibility**, making the system observable and extensible.

---

## 🤖 Agentic Workflow

The incident triage workflow consists of four autonomous steps:

1. **Triage Step**

   * Classifies incident severity
   * Introduces classification drift signals

2. **Investigation Step**

   * Gathers contextual evidence
   * Enables depth and repetition drift detection

3. **Decision Step**

   * Determines auto-resolution vs escalation
   * Allows a single controlled retry
   * Primary source of behavioral drift

4. **Notification Step**

   * Communicates outcomes
   * Detects duplicate or premature alerts

The workflow supports **branching and retry**, which are critical for realistic drift scenarios.

---

## 🧾 Shared State Contract

All agents operate on a shared `IncidentState` contract:

* Incident identity
* Agent outputs
* Execution metadata
* Performance indicators

This contract is the **single source of truth** for:

* Telemetry
* Drift analysis
* Replay and debugging

---

## 📡 Telemetry & Observability

Each step emits execution telemetry, including:

* Step name
* Execution order
* Retry count
* Path taken
* Execution latency (ms)

Telemetry is stored in a **SQLite database**, making it possible to:

* Query historical population baselines
* Track categorical rates (e.g., Escalation Rate)
* Perform sliding-window analysis

---

## 🚨 What Is Drift?

Drift is defined as **deviation from historically stable behavior**, not explicit failure.

Examples:

* Decision retries increase over time
* Escalation rate spikes for low-severity incidents
* Execution paths become longer
* Notifications fire before decisions stabilize

These patterns indicate **loss of autonomy quality**.

---

## 🧪 Drift Simulation & Alerting

This repository includes **intentional drift simulation** to demonstrate how autonomy can degrade *without failures* — and how the system detects it early.

### 🔁 Scenario: Retry Explosion

In this simulation, the decision agent is configured to produce **low confidence scores**, causing:

* Repeated decision retries
* Increased step count
* Higher execution cost and latency

Despite this degradation:

* The workflow completes successfully
* No exceptions are raised
* The system remains "operational"

This mirrors **real-world AI failures**, where systems don’t crash — they quietly get worse.

### 🚨 Drift Detection Outcome

The drift engine detects behavioral changes using:

* Retry count deviation
* Execution path inflation
* Decision loop detection
* Latency degradation
* Semantic Bias (Escalation & Classification Bias)

Example output:

```
[DRIFT ANALYSIS]
{'drift_score': 45, 'risk_level': 'drift_detected', 'baseline_used': {'avg_steps': 4.2, 'avg_retries': 0.2, 'avg_latency': 2137.9, 'escalation_rate': 0.57, 'high_severity_rate': 0.11, 'low_severity_escalation_rate': 0.12}}
```

### 🧠 Why This Matters

Most AI monitoring focuses on **outputs**.
This system focuses on **behavior**.

By detecting drift early, teams can:

* Intervene before incidents escalate
* Reduce operational cost
* Maintain trust in autonomous systems

This approach reflects how **production AI reliability teams** think about safety and observability.

---

---

## 🛠️ Tech Stack

* Python
* **LangGraph** for deterministic state machine orchestration
* **SQLite** for population-level telemetry baselines
* **LangSmith** for UI visualization
* Rules-first drift detection (ML-ready later)

---

## 🚀 Running the Project

First, generate a healthy statistical baseline:
```bash
python run.py --simulate-batch 50
```

Then, trigger a single biased execution to watch the drift detector flag it:
```bash
python run.py --bias
```

This will:
* Execute the LangGraph workflow
* Emit telemetry to SQLite (and LangSmith if configured via `.env`)
* Calculate the drift score against the aggregate baseline
* Print the final execution state and drift analysis

---

## 🔮 Roadmap

Planned enhancements:

* Drift scoring & risk classification
* Slack / alert integrations
* Baseline learning
* Visualization dashboard
* Multi-workflow support

---

## 👤 Author

Built as a systems-level exploration of **agentic AI reliability, observability, and autonomy drift**.

This project emphasizes **engineering judgment over demos**.

---

## 📜 License

MIT
