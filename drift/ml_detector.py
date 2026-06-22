import os
import joblib
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
import shap

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "model.joblib")
VEC_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "vectorizer.joblib")

class DriftModel:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VEC_PATH)
                log.info("Loaded ML drift model and vectorizer from %s", MODEL_PATH)
            except Exception as e:
                log.error("Failed to load ML drift model: %s", e)

    def _extract_numeric_features(self, state: dict) -> list:
        severity_map = {"low": 1, "medium": 2, "high": 3}
        decision_map = {"auto_resolve": 1, "escalate": 2, "requires_clinical_review": 3, "manual_review": 3}

        step_count = state.get("step_count", 0)
        retry_count = state.get("retry_count", 0)
        execution_time_ms = state.get("execution_time_ms", 0)
        severity = severity_map.get(state.get("severity", "low"), 1)
        
        raw_decision = state.get("decision") or state.get("coding_status", "auto_resolve")
        decision = decision_map.get(raw_decision, 1)

        return [step_count, retry_count, execution_time_ms, severity, decision]

    def _extract_text_features(self, state: dict) -> str:
        # We represent path_taken as a space-separated string of steps
        path_taken = state.get("path_taken", [])
        return " ".join(path_taken) if path_taken else "unknown"

    def _encode_state(self, state: dict) -> np.ndarray:
        num_feats = self._extract_numeric_features(state)
        text_feat = self._extract_text_features(state)
        
        # Transform text feature
        if self.vectorizer is not None:
            tfidf_feats = self.vectorizer.transform([text_feat]).toarray()[0]
        else:
            # Fallback if no vectorizer is loaded but we are predicting
            tfidf_feats = np.zeros(0)
            
        return np.concatenate((num_feats, tfidf_feats)).reshape(1, -1)

    def train(self, historical_data: list[dict]):
        if not historical_data:
            log.warning("No historical data provided for training.")
            return

        num_X = []
        text_X = []
        for state in historical_data:
            num_X.append(self._extract_numeric_features(state))
            text_X.append(self._extract_text_features(state))
        
        self.vectorizer = TfidfVectorizer(max_features=10)
        tfidf_X = self.vectorizer.fit_transform(text_X).toarray()
        
        X = np.hstack((np.array(num_X), tfidf_X))
        
        self.model = IsolationForest(contamination=0.05, random_state=42)
        self.model.fit(X)
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(self.vectorizer, VEC_PATH)
        log.info("Trained and saved ML drift model and vectorizer to config/")

    def _generate_explanation(self, X: np.ndarray) -> str:
        try:
            # Initialize explainer on demand
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)[0]
            
            # Reconstruct feature names
            base_feats = ["step_count", "retry_count", "execution_time_ms", "severity", "decision"]
            if self.vectorizer is not None:
                text_feats = [f"path_{w}" for w in self.vectorizer.get_feature_names_out()]
            else:
                text_feats = []
            
            feat_names = base_feats + text_feats
            
            # Sort features by absolute SHAP value
            contributions = list(zip(feat_names, shap_values))
            # Sort descending by absolute value
            contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Top 3 features that contributed to the anomaly
            top_factors = [f"{name} ({val:.2f})" for name, val in contributions[:3] if abs(val) > 0.01]
            if not top_factors:
                top_factors = [f"{name} ({val:.2f})" for name, val in contributions[:3]]
                
            return f"Anomaly driven by: {', '.join(top_factors)}"
        except Exception as e:
            log.error("Failed to generate SHAP explanation: %s", e)
            return "Anomaly detected (explanation unavailable)"

    def predict(self, state: dict) -> tuple[int, str, str | None]:
        """
        Returns (drift_score, risk_level, ml_explanation) based on ML model.
        If model is not loaded, returns (0, "healthy", None).
        """
        if self.model is None or self.vectorizer is None:
            return 0, "healthy", None

        X = self._encode_state(state)
        # IsolationForest returns 1 for inliers, -1 for outliers
        prediction = self.model.predict(X)[0]
        # decision_function gives scores, lower (negative) is more anomalous. 
        score = self.model.decision_function(X)[0]
        
        # Normalize score into a 0-100 drift score
        if prediction == -1:
            drift_score = min(100, int(abs(score) * 400) + 60)
            explanation = self._generate_explanation(X)
            if drift_score >= 60:
                return drift_score, "high_risk", explanation
            return drift_score, "drift_detected", explanation
        
        # It's an inlier
        return max(0, int((0.5 - score) * 10)), "healthy", None
