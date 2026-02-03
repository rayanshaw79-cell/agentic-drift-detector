
# -------------------------
# 1. BASELINE DEFINITION
# -------------------------

BASELINE = {
    "max_steps": 5,
    "max_retries": 1,
    "expected_path": [
        "triage",
        "investigation",
        "decision",
        "notification"
    ],
    "max_decision_repeats": 2
}

# -------------------------
# 2. DRIFT SIGNAL FUNCTIONS
# -------------------------

def step_count_drift(state):
    max_steps = BASELINE["max_steps"]
    actual_steps = state["step_count"]

    if actual_steps <= max_steps:
        return 0

    overflow = actual_steps - max_steps
    return min(30, overflow * 10)


def retry_drift(state):
    max_retries = BASELINE["max_retries"]
    retries = state["retry_count"]

    if retries <= max_retries:
        return 0

    return min(25, (retries - max_retries) * 15)


def path_drift(state):
    expected = BASELINE["expected_path"]
    actual = state["path_taken"]

    if actual[:len(expected)] != expected:
        return 25

    return 0


def decision_loop_drift(state):
    decision_count = state["path_taken"].count("decision")
    allowed = BASELINE["max_decision_repeats"]

    if decision_count <= allowed:
        return 0

    return min(20, (decision_count - allowed) * 10)

# -------------------------
# 3. DRIFT SCORE AGGREGATOR
# -------------------------

def calculate_drift_score(state):
    score = 0

    score += step_count_drift(state)
    score += retry_drift(state)
    score += path_drift(state)
    score += decision_loop_drift(state)

    return score

# -------------------------
# 4. RISK CLASSIFICATION
# -------------------------

def classify_risk(score):
    if score < 30:
        return "healthy"
    elif score < 60:
        return "drift_detected"
    else:
        return "high_risk"

# -------------------------
# 5. PUBLIC ENTRY POINT
# -------------------------

def analyze_workflow(state):
    score = calculate_drift_score(state)
    risk = classify_risk(score)

    return {
        "drift_score": score,
        "risk_level": risk
    }


