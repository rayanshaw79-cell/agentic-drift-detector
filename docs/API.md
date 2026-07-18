# API Documentation

The Agentic Drift Detector exposes a high-performance REST API powered by **FastAPI**. It serves as the primary entry point for triggering LLM workflows (LangGraph) and retrieving telemetry.

## Base URL
When running via Docker Compose: `http://localhost:8000`

Interactive Swagger Documentation is available at: `/docs`

---

## 1. Incident Triage Endpoint

**`POST /api/v1/triage`**

Triggers the IT Incident Triage LangGraph workflow. Evaluates the incident severity, retrieves historical telemetry, and makes an agentic decision (`escalate` vs `auto_resolve`).

### Request Body

```json
{
  "incident_text": "Production database latency spiking over 500ms.",
  "simulate_bias": false
}
```

- `simulate_bias` (boolean): If `true`, intentionally forces the agent to make a high-severity classification and low-confidence decision to test the Agentic Healing circuit breakers.

### Response

```json
{
  "status": "success",
  "result": {
    "current_step": "notification",
    "step_count": 3,
    "path_taken": ["triage", "investigation", "decision", "notification"],
    "confidence": 0.85,
    "decision": "escalate",
    "execution_time_ms": 1205
  },
  "execution_time_ms": 1215
}
```

---

## 2. Clinical Data Extraction Endpoint

**`POST /api/v1/clinical/extract`**

Triggers the advanced Medical Real-World Evidence (RWE) pipeline. Runs Bayesian NER, Disambiguation, MEAT Validation, and SDOH Risk Prediction.

### Request Body

```json
{
  "note_text": "Patient is a 65 y/o male presenting with uncontrolled Type 2 Diabetes Mellitus with peripheral neuropathy. Currently experiencing housing instability."
}
```

### Response

```json
{
  "status": "success",
  "patient_id": "sim_84f9",
  "coding_status": "complete",
  "icd10_codes": [
    {
      "code": "E11.40",
      "description": "Type 2 diabetes mellitus with diabetic neuropathy, unspecified",
      "hcc_category": "HCC 18",
      "raf_weight": 0.318,
      "meat_verified": true,
      "meat_evidence": "uncontrolled Type 2 Diabetes Mellitus with peripheral neuropathy"
    }
  ],
  "sdoh_risk_score": 0.82,
  "sdoh_risk_label": "high",
  "execution_time_ms": 2450
}
```

---

## 3. Health & Readiness Probes

**`GET /health`**

Standard Kubernetes-compatible liveness and readiness probe.

### Response
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "postgres_connected": true,
  "redis_connected": true
}
```

---

## 4. Webhook System (Outbound)

If `SLACK_WEBHOOK_URL` or `DISCORD_WEBHOOK_URL` is set in the environment, the background worker will emit real-time JSON payloads to your channels whenever a **Drift Anomaly** is detected (e.g., latency spikes, retry loops).

*Note: Webhook processing is handled asynchronously via Redis task queues to prevent blocking the main FastAPI thread.*
