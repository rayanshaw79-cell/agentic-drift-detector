import os
import joblib
import numpy as np
import logging
from sklearn.ensemble import IsolationForest

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "model.joblib")

class DriftModel:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                log.info("Loaded ML drift model from %s", MODEL_PATH)
            except Exception as e:
                log.error("Failed to load ML drift model: %s", e)

    def _encode_state(self, state: dict) -> np.ndarray:
        severity_map = {"low": 1, "medium": 2, "high": 3}
        decision_map = {"auto_resolve": 1, "escalate": 2}

        # Handle missing or clinical state formats gracefully
        step_count = state.get("step_count", 0)
        retry_count = state.get("retry_count", 0)
        execution_time_ms = state.get("execution_time_ms", 0)
        severity = severity_map.get(state.get("severity", "low"), 1)
        decision = decision_map.get(state.get("decision", "auto_resolve"), 1)

        return np.array([[step_count, retry_count, execution_time_ms, severity, decision]])

    def train(self, historical_data: list[dict]):
        if not historical_data:
            log.warning("No historical data provided for training.")
            return

        X = []
        for state in historical_data:
            X.append(self._encode_state(state)[0])
        
        X = np.array(X)
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X)
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        log.info("Trained and saved ML drift model to %s", MODEL_PATH)

    def predict(self, state: dict) -> tuple[int, str]:
        """
        Returns (drift_score, risk_level) based on ML model.
        If model is not loaded, returns (0, "healthy") - acting as a no-op 
        so fallback logic can handle it.
        """
        if self.model is None:
            return 0, "healthy"

        X = self._encode_state(state)
        # IsolationForest returns 1 for inliers, -1 for outliers
        prediction = self.model.predict(X)[0]
        # decision_function gives scores, lower (negative) is more anomalous. 
        score = self.model.decision_function(X)[0]
        
        # Normalize score into a 0-100 drift score
        if prediction == -1:
            # Boost the base score for outliers so they cross the 60 threshold easier
            drift_score = min(100, int(abs(score) * 400) + 60)
            if drift_score >= 60:
                return drift_score, "high_risk"
            return drift_score, "drift_detected"
        
        # It's an inlier
        return max(0, int((0.5 - score) * 10)), "healthy"
