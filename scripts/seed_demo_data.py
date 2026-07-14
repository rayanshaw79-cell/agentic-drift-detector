import random
import uuid
import sys
import os

# Add parent dir to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from telemetry.store import save_execution_state, init_db

def generate_mock_state(i):
    record_id = f"demo-pt-{i}-{uuid.uuid4().hex[:6]}"
    
    # SDOH Profiles (label, base_score, shap_factors)
    sdoh_profiles = [
        ("low", 0.1, [{"feature": "food_risk_score", "contribution": 0.05}, {"feature": "env_poverty_rate", "contribution": -0.02}]),
        ("moderate", 0.45, [{"feature": "env_aqi", "contribution": 0.15}, {"feature": "food_risk_score", "contribution": 0.1}]),
        ("high", 0.75, [{"feature": "env_poverty_rate", "contribution": 0.3}, {"feature": "food_risk_score", "contribution": 0.25}, {"feature": "smoking_flag", "contribution": 0.1}]),
        ("critical", 0.92, [{"feature": "env_poverty_rate", "contribution": 0.4}, {"feature": "smoking_flag", "contribution": 0.25}, {"feature": "env_aqi", "contribution": 0.2}]),
    ]
    
    profile = random.choices(sdoh_profiles, weights=[40, 30, 20, 10])[0]
    
    # ICD10 Profiles
    icd10_codes = [
        {"code": "I10", "description": "Essential (primary) hypertension", "confidence": 0.95, "meat_met": True, "raf_weight": 0.3},
        {"code": "E11.9", "description": "Type 2 diabetes mellitus without complications", "confidence": 0.90, "meat_met": True, "raf_weight": 0.1}
    ] if random.random() > 0.5 else [{"code": "J45.909", "description": "Unspecified asthma", "confidence": 0.85, "meat_met": True, "raf_weight": 0.2}]
    
    # 5% chance of unresolved code
    if random.random() < 0.05:
        icd10_codes.append({"code": "UNRESOLVED", "description": "Unknown condition", "confidence": 0.0, "meat_met": False, "raf_weight": 0.0})
        coding_status = "requires_clinical_review"
    else:
        coding_status = "complete"

    state = {
        "record_id": record_id,
        "coding_status": coding_status,
        "overall_confidence": random.uniform(0.7, 0.99),
        "step_count": random.randint(4, 7),
        "retry_count": 0 if coding_status == "complete" else 1,
        "path_taken": ["deid", "compliance_checker", "ner", "ontology_lookup", "disambiguation", "meat_validation", "validation", "sdoh_integration", "clinical_output"],
        "execution_time_ms": random.randint(300, 1500),
        "icd10_codes": icd10_codes,
        "sdoh_risk_label": profile[0],
        "sdoh_risk_score": profile[1] + random.uniform(-0.05, 0.05),
        "sdoh_shap_factors": profile[2]
    }
    
    analysis = {
        "workflow_type": "clinical_coding",
        "risk_level": "healthy" if coding_status == "complete" else "drift",
    }
    
    return state, analysis

def main():
    init_db()
    count = 50
    if len(sys.argv) > 1 and sys.argv[1] == "--count":
        count = int(sys.argv[2])

    print(f"Generating {count} demo records...")
    for i in range(count):
        state, analysis = generate_mock_state(i)
        save_execution_state(state, analysis=analysis, tenant_id="demo")
    
    print("Done! The database is seeded. You can now run `streamlit run dashboard.py`.")

if __name__ == "__main__":
    main()
