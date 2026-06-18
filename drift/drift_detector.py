from telemetry.store import get_historical_metrics

# -------------------------
# 1. DRIFT SIGNAL FUNCTIONS
# -------------------------
# Signals are grouped into two categories:
#   a) Generic workflow signals (used by incident_triage)
#   b) Clinical-specific signals (used by clinical_coding)

def step_count_drift(state, baseline):
    max_steps = max(5, baseline["avg_steps"] + 1)
    actual_steps = state.get("step_count", 0)

    if actual_steps <= max_steps:
        return 0

    overflow = actual_steps - max_steps
    return min(30, overflow * 10)


def retry_drift(state, baseline):
    # We dynamically allow retries up to the baseline average + a buffer
    max_retries = max(1, baseline["avg_retries"])
    retries = state.get("retry_count", 0)

    if retries <= max_retries:
        return 0

    return min(25, (retries - max_retries) * 15)


def path_drift(state):
    expected = ["triage", "investigation", "decision"]
    actual = state.get("path_taken", [])

    if actual[:len(expected)] != expected:
        return 25

    return 0


def decision_loop_drift(state):
    decision_count = state.get("path_taken", []).count("decision")
    allowed = 2

    if decision_count <= allowed:
        return 0

    return min(20, (decision_count - allowed) * 10)


def latency_drift(state, baseline):
    # E.g. anything over 1.5x baseline latency is considered drift
    expected_latency = max(500, baseline["avg_latency"] * 1.5)
    actual = state.get("execution_time_ms", 0)

    if actual <= expected_latency:
        return 0

    overflow = (actual - expected_latency) / 1000.0  # seconds over
    return min(20, int(overflow * 5))

def escalation_bias(state, baseline):
    decision = state.get("decision")
    severity = state.get("severity")
    
    if decision != "escalate" or severity != "low":
        return 0
        
    # If the historical low_severity_escalation_rate is small (e.g. 5%),
    # but the agent escalated this low-severity issue, we flag it.
    historical_rate = baseline.get("low_severity_escalation_rate", 0.05)
    
    if historical_rate < 0.2:
        # Strong deviation from baseline: agent is panicking
        return 25
    elif historical_rate < 0.5:
        return 10
    
    return 0

def classification_bias(state, baseline):
    severity = state.get("severity")
    
    if severity != "high":
        return 0
        
    # If the historical high_severity_rate is low, but the agent assigned high severity
    historical_rate = baseline.get("high_severity_rate", 0.2)
    
    if historical_rate < 0.15:
        # Anomalous high severity classification
        return 20
    elif historical_rate < 0.3:
        return 10
        
    return 0


# ── Clinical-Specific Drift Signals ──────────────────────────────────────────

def coding_confidence_drift(state, baseline):
    """
    Fires when the agent's overall_confidence drops significantly below the
    historical average — a strong hallucination or ambiguity signal.
    """
    confidence = state.get("overall_confidence")
    if confidence is None:
        return 0  # Not a clinical run

    avg_confidence = baseline.get("avg_coding_confidence", 0.75)
    if confidence >= avg_confidence * 0.85:
        return 0  # Within 15% of baseline — healthy
    elif confidence >= avg_confidence * 0.65:
        return 20  # Moderate drop
    else:
        return 35  # Severe confidence collapse


def unresolved_entity_drift(state, baseline):
    """
    Fires when the proportion of unresolved medical terms (no ICD-10 match)
    exceeds the historical baseline. Signals ambiguous notes or model drift.
    """
    codes = state.get("icd10_codes") or []
    if not codes:
        return 0

    unresolved_count = sum(1 for c in codes if c.get("code") == "UNRESOLVED")
    unresolved_rate = unresolved_count / len(codes)

    historical_rate = baseline.get("avg_unresolved_rate", 0.05)
    if unresolved_rate <= historical_rate * 1.5:
        return 0
    elif unresolved_rate <= 0.3:
        return 15  # Some unresolvable terms
    else:
        return 30  # High unresolved rate — note quality or model drift


def clinical_api_retry_drift(state, baseline):
    """
    Fires when NLM API retries (reflected in retry_count) exceed the baseline.
    High retries indicate fragile ontology resolution.
    """
    retries = state.get("retry_count", 0)
    avg_retries = baseline.get("avg_retries", 0.0)
    max_allowed = max(1, avg_retries + 0.5)

    if retries <= max_allowed:
        return 0
    return min(20, int((retries - max_allowed) * 10))


# -------------------------
# 2. DRIFT SCORE AGGREGATOR
# -------------------------

def calculate_drift_score(state, baseline, workflow_type: str = "incident_triage"):
    score = 0

    # Generic signals — applied to all workflows
    score += step_count_drift(state, baseline)
    score += retry_drift(state, baseline)
    score += path_drift(state)
    score += decision_loop_drift(state)
    score += latency_drift(state, baseline)

    if workflow_type == "clinical_coding":
        # Clinical-specific signals replace incident-triage semantic signals
        score += coding_confidence_drift(state, baseline)
        score += unresolved_entity_drift(state, baseline)
        score += clinical_api_retry_drift(state, baseline)
    else:
        # Incident-triage semantic signals
        score += escalation_bias(state, baseline)
        score += classification_bias(state, baseline)

    return score

# -------------------------
# 3. RISK CLASSIFICATION
# -------------------------

def classify_risk(score):
    if score < 30:
        return "healthy"
    elif score < 60:
        return "drift_detected"
    else:
        return "high_risk"

# -------------------------
# 4. PUBLIC ENTRY POINT
# -------------------------

def analyze_workflow(state, workflow_type: str = "incident_triage"):
    """
    Analyze a completed workflow state and return a drift analysis dict.

    Args:
        state:         The final LangGraph state dict.
        workflow_type: "incident_triage" (default) or "clinical_coding".
                       Controls which semantic drift signals are applied.
    """
    baseline = get_historical_metrics()
    score = calculate_drift_score(state, baseline, workflow_type=workflow_type)
    risk = classify_risk(score)

    return {
        "drift_score":    score,
        "risk_level":     risk,
        "workflow_type":  workflow_type,
        "baseline_used":  baseline,
    }
