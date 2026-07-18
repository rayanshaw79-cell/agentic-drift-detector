# Data Model & Observability Schema

The system relies on a robust relational data model implemented in **TimescaleDB / PostgreSQL**. The primary goals are:
1. Long-term persistence of LLM behavioral telemetry for drift detection.
2. Clinical patient records mapped to ICD-10 and HCC codes.
3. Persistent storage of SDOH (Social Determinants of Health) Machine Learning models and SHAP explainability factors.

## 1. LLM Telemetry Schema

Every time a LangGraph node executes, it emits telemetry. Over thousands of runs, this data forms the "baseline" used to detect Agentic Drift.

### `telemetry_logs`
Stores granular execution details of every workflow step.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL (PK) | Unique log ID |
| `timestamp` | TIMESTAMPTZ | When the execution finished |
| `incident_id` | VARCHAR | UUID grouping steps of the same workflow |
| `step_name` | VARCHAR | Name of the LangGraph node (e.g., `triage`, `intervention`) |
| `execution_time_ms` | INTEGER | Time taken to execute the node |
| `retry_count` | INTEGER | Number of times the node retried its action |
| `path_taken` | JSONB | Array of steps taken leading to this node |
| `confidence` | FLOAT | The agent's stated confidence score (0.0 to 1.0) |
| `decision` | VARCHAR | The action taken (e.g., `escalate`, `auto_resolve`) |
| `workflow_type` | VARCHAR | Identifier separating `IT_INCIDENT` vs `CLINICAL` |

---

## 2. Clinical Pipeline Schema

For the clinical extraction workflow, we store the resulting, audit-proof medical codes.

### `patient_records`
Stores the final processed state of the clinical document.

| Column | Type | Description |
|---|---|---|
| `patient_id` | VARCHAR (PK) | Unique Patient ID |
| `created_at` | TIMESTAMPTZ | Extraction timestamp |
| `coding_status` | VARCHAR | Overall extraction status (`complete`, `requires_clinical_review`) |
| `sdoh_risk_score` | FLOAT | ML-predicted Risk Score (0.0 to 1.0) |
| `sdoh_risk_label` | VARCHAR | Categorical threshold (`low`, `moderate`, `high`, `critical`) |
| `sdoh_shap_factors` | JSONB | Top 3 features driving the risk prediction |

### `extracted_conditions`
Maps the extracted entities to the patient and stores CMS financial auditing proofs (MEAT).

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL (PK) | Unique condition record |
| `patient_id` | VARCHAR (FK) | Reference to `patient_records` |
| `icd10_code` | VARCHAR | ICD-10 diagnostic code (e.g., `E11.40`) |
| `description` | TEXT | Description of the code |
| `hcc_category` | VARCHAR | CMS Hierarchical Condition Category (e.g., `HCC 18`) |
| `raf_weight` | FLOAT | Risk Adjustment Factor financial weight |
| `meat_verified` | BOOLEAN | Did the Agentic Healing circuit verify clinical action? |
| `meat_evidence` | TEXT | Extracted snippet proving M, E, A, or T |

---

## 3. Drift Machine Learning Model State

To calculate if an agent's latency or retry patterns are anomalous, the system calculates running baselines (mean, standard deviation).

### `drift_ml_models`
Stores serialized instances of the background anomaly detection algorithms (e.g., Isolation Forests).

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL (PK) | Unique ID |
| `model_name` | VARCHAR | Name of the model (e.g., `latency_isolation_forest`) |
| `trained_at` | TIMESTAMPTZ | When the model was last refit |
| `model_binary` | BYTEA | Serialized (joblib) Scikit-Learn model |
| `accuracy_score` | FLOAT | Evaluation metric from the last training pass |
