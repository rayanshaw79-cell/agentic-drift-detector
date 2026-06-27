"""
clinical/sdoh/risk_model.py — SDOH Longitudinal Risk Gradient Boosting Model.

Loads or trains a GradientBoostingClassifier that predicts a patient's
SDOH risk label ("low" | "moderate" | "high" | "critical") from 12
tabular features extracted per clinical visit.

The model is persisted to clinical/sdoh/risk_model.joblib after training.
"""

import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent / "risk_model.joblib"

FEATURES = [
    "age",
    "visit_number",
    "hcc_score",
    "env_aqi",
    "env_poverty_rate",
    "food_risk_score",
    "smoking_flag",
    "alcohol_flag",
    "exercise_score",
    "icd10_code_count",
    "chain_stage",
    "sdoh_risk_score",   # cumulative input signal; target is the label
]

LABEL_ORDER = ["low", "moderate", "high", "critical"]


def _build_feature_vector(visit: dict) -> np.ndarray:
    """Extract a 1-D feature vector from a visit record dict."""
    return np.array([
        visit.get("age",              45),
        visit.get("visit_number",      1),
        visit.get("hcc_score",       0.0),
        visit.get("env_aqi",          80),
        visit.get("env_poverty_rate", 0.15),
        visit.get("food_risk_score",  0.0),
        int(visit.get("smoking_flag",   0)),
        int(visit.get("alcohol_flag",   0)),
        visit.get("exercise_score",   0.5),
        visit.get("icd10_code_count",   0),
        visit.get("chain_stage",        0),
        visit.get("sdoh_risk_score",  0.0),
    ], dtype=float)


def train(df: pd.DataFrame) -> object:
    """
    Train a GradientBoostingClassifier on *df* and return the fitted model.
    The model is also persisted to MODEL_PATH.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    X = df[FEATURES].fillna(0).values
    y = df["sdoh_risk_label"].values

    le = LabelEncoder()
    le.fit(LABEL_ORDER)
    y_enc = le.transform(y)

    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42,
    )
    clf.fit(X, y_enc)

    # Cross-validated F1 (macro)
    scores = cross_val_score(clf, X, y_enc, cv=5, scoring="f1_macro")
    log.info("GBM CV F1 (macro): %.3f +/- %.3f", scores.mean(), scores.std())
    print(f"  [OK] Cross-validated F1 (macro): {scores.mean():.3f} +/- {scores.std():.3f}")

    bundle = {"model": clf, "label_encoder": le, "features": FEATURES}
    joblib.dump(bundle, MODEL_PATH)
    log.info("Risk model saved to %s", MODEL_PATH)
    return bundle


def load() -> dict:
    """
    Load the persisted model bundle from disk.
    Raises FileNotFoundError if the model has not been trained yet.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Risk model not found at {MODEL_PATH}. "
            "Run `python -m clinical.sdoh.train_risk_model` first."
        )
    return joblib.load(MODEL_PATH)


def predict_proba(bundle: dict, visit: dict) -> tuple[str, float, list[dict]]:
    """
    Predict the risk label and probability for a single visit record.

    Returns:
        (label, probability, shap_factors)
        shap_factors: list of {"feature": str, "contribution": float}
    """
    import shap

    clf = bundle["model"]
    le  = bundle["label_encoder"]

    x = _build_feature_vector(visit).reshape(1, -1)
    proba   = clf.predict_proba(x)[0]
    pred_idx = int(np.argmax(proba))
    label    = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    # SHAP explanation for the predicted class
    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x)          # shape: (n_classes, 1, n_features)
        class_shap  = shap_values[pred_idx][0]
        shap_factors = sorted(
            [{"feature": FEATURES[i], "contribution": round(float(class_shap[i]), 4)}
             for i in range(len(FEATURES))],
            key=lambda d: abs(d["contribution"]),
            reverse=True,
        )[:6]  # top 6 drivers
    except Exception as exc:
        log.warning("SHAP computation failed: %s", exc)
        shap_factors = []

    return label, confidence, shap_factors
