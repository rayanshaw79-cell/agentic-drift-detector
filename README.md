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
Antigravity Workflow Engine
      ↓
Agent Steps (Triage → Investigation → Decision → Notification)
      ↓
Execution Telemetry (JSONL)
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
* Timestamp

Telemetry is stored as **append-only JSONL**, making it:

* Simple
* Transparent
* Easy to analyze

Telemetry never makes decisions — it only observes.

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

## 🧪 Example Drift Scenario

After a prompt update:

* Average steps increase from 4 → 7
* Decision confidence drops
* Escalation rate rises from 30% → 65%

The system still "works" — but autonomy has degraded.

This project detects that change **before humans feel the pain**.

---

## 🛠️ Tech Stack

* Python
* Antigravity-style workflow orchestration
* JSON-based telemetry
* Rules-first drift detection (ML-ready later)

No heavy dependencies. No magic frameworks.

---

## 🚀 Running the Project

```bash
python run.py
```

This will:

* Execute the agentic workflow
* Emit telemetry
* Print final execution state

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
