"""
clinical/sdoh/patient_simulator.py — Synthetic Patient Dataset Generator.

Generates 500 realistic synthetic patient records across 12 months of visits,
encoding known clinical correlations between SDOH factors and disease progression.

Usage:
    from clinical.sdoh.patient_simulator import generate_dataset
    df = generate_dataset(n_patients=500, n_months=12)

Or run directly to save a CSV:
    python -m clinical.sdoh.patient_simulator
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42

# ── ICD-10 Progression Chains ─────────────────────────────────────────────────
# Patients in high-risk SDOH environments progress through these chains faster.
PROGRESSION_CHAINS = {
    "diabetes": [
        ["R73.09"],                      # Prediabetes
        ["E11.9"],                       # T2DM uncomplicated
        ["E11.65"],                      # T2DM with hyperglycaemia
        ["E11.40", "E11.9"],             # T2DM with diabetic neuropathy
        ["E11.40", "E11.9", "N18.3"],    # T2DM + CKD stage 3
    ],
    "hypertension": [
        ["R03.0"],                       # Elevated blood pressure
        ["I10"],                         # Essential hypertension
        ["I10", "I50.9"],               # Hypertension + Heart failure
        ["I10", "I50.9", "N18.2"],      # + CKD stage 2
    ],
    "copd": [
        ["J44.1"],                       # COPD with exacerbation
        ["J44.1", "J96.00"],            # + Respiratory failure
        ["J44.1", "J96.00", "I27.20"],  # + Pulmonary hypertension
    ],
    "obesity": [
        ["E66.09"],                      # Morbid obesity
        ["E66.09", "G47.33"],           # + Sleep apnoea
        ["E66.09", "G47.33", "M47.816"], # + Spondylosis
    ],
}

RACE_OPTIONS     = ["White", "Black", "Hispanic", "Asian", "Other"]
GENDER_OPTIONS   = ["M", "F"]
ZIP_PROFILES = {
    # zip: (poverty_rate, aqi, food_risk_base)
    "73301": (0.32, 145, 0.75),   # High poverty, poor air, food desert
    "10001": (0.18, 95,  0.40),   # Moderate
    "94102": (0.22, 65,  0.35),   # Lower poverty, good air
    "60601": (0.14, 80,  0.20),   # Low poverty, good air
    "30301": (0.28, 110, 0.60),   # Moderate-high poverty
}


def _sdoh_risk_multiplier(poverty_rate, aqi, food_risk, smoking, alcohol) -> float:
    """Compute an overall SDOH risk multiplier (1.0 = baseline)."""
    score = 1.0
    score += poverty_rate * 1.5       # High poverty accelerates disease
    score += (aqi / 200.0) * 0.8     # Poor air quality
    score += food_risk * 0.7          # Food insecurity
    score += smoking * 0.9            # Smoking
    score += alcohol * 0.4            # Alcohol
    return score


def _assign_label(risk_score: float) -> str:
    if risk_score < 0.30:
        return "low"
    if risk_score < 0.55:
        return "moderate"
    if risk_score < 0.75:
        return "high"
    return "critical"


def generate_dataset(n_patients: int = 500, n_months: int = 12) -> pd.DataFrame:
    """Generate a synthetic longitudinal patient dataset."""
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    records = []
    base_date = datetime(2024, 1, 1)

    for p_idx in range(n_patients):
        patient_id   = f"PT-{p_idx + 1:04d}"
        age          = rng.randint(35, 78)
        gender       = rng.choice(GENDER_OPTIONS)
        race         = rng.choice(RACE_OPTIONS)
        zip_code     = rng.choice(list(ZIP_PROFILES.keys()))
        poverty_rate, base_aqi, food_risk_base = ZIP_PROFILES[zip_code]

        smoking   = rng.random() < (0.30 + poverty_rate * 0.5)
        alcohol   = rng.random() < (0.20 + poverty_rate * 0.3)
        exercise  = round(rng.uniform(0.0, 1.0) * (1.0 - poverty_rate * 0.5), 2)

        chain_name = rng.choice(list(PROGRESSION_CHAINS.keys()))
        chain      = PROGRESSION_CHAINS[chain_name]

        # Compute how fast this patient progresses (SDOH multiplier)
        multiplier = _sdoh_risk_multiplier(
            poverty_rate, base_aqi, food_risk_base, int(smoking), int(alcohol)
        )
        # Higher multiplier = faster chain progression
        visits_per_stage = max(1, int(n_months / (len(chain) * multiplier * 0.5)))

        visit_date = base_date
        stage      = 0

        for month in range(n_months):
            # Progress to next stage if enough time has passed
            stage = min(int(month / visits_per_stage), len(chain) - 1)
            icd10 = chain[stage]

            aqi          = base_aqi + np.random.normal(0, 10)
            food_risk    = min(1.0, food_risk_base + np.random.normal(0, 0.05))
            hcc_score    = round(0.2 + stage * 0.25 * multiplier + np.random.normal(0, 0.05), 3)

            # Normalised SDOH risk score for ML training (0.0 – 1.0)
            raw_risk = (
                poverty_rate * 0.20 +
                (aqi / 200.0) * 0.15 +
                food_risk * 0.15 +
                int(smoking) * 0.15 +
                int(alcohol) * 0.10 +
                (1.0 - exercise) * 0.05 +
                (stage / len(chain)) * 0.20
            )
            sdoh_risk_score = round(min(1.0, raw_risk + np.random.normal(0, 0.02)), 3)

            records.append({
                "patient_id":       patient_id,
                "visit_number":     month + 1,
                "visit_date":       visit_date.strftime("%Y-%m-%d"),
                "zip_code":         zip_code,
                "age":              age,
                "gender":           gender,
                "race":             race,
                "smoking_flag":     int(smoking),
                "alcohol_flag":     int(alcohol),
                "exercise_score":   exercise,
                "food_risk_score":  round(food_risk, 3),
                "env_aqi":          round(aqi, 1),
                "env_poverty_rate": round(poverty_rate, 3),
                "hcc_score":        hcc_score,
                "icd10_codes":      "|".join(icd10),
                "icd10_code_count": len(icd10),
                "chain_stage":      stage,
                "sdoh_risk_score":  sdoh_risk_score,
                "sdoh_risk_label":  _assign_label(sdoh_risk_score),
            })

            # Next visit: 25–35 days later
            visit_date += timedelta(days=rng.randint(25, 35))

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "clinical/sdoh/synthetic_patients.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df):,} visit records for {df['patient_id'].nunique()} patients → {out_path}")
    print(df["sdoh_risk_label"].value_counts())
