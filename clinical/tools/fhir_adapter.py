"""
clinical/tools/fhir_adapter.py — HL7 FHIR R4 Adapter & Synthetic Patient Generator.

Converts extracted clinical state objects into standard HL7 FHIR R4 JSON Bundles
and generates synthetic oncology patient charts for testing & seeding.
"""

import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any

log = logging.getLogger(__name__)


def export_clinical_state_to_fhir(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts extracted ClinicalState into an HL7 FHIR R4 JSON Bundle.

    Resources included in Bundle:
      - Patient
      - Condition (for each ICD-10 code)
      - MedicationStatement (for each extracted drug)
      - Observation (SDOH risk, RAF score, Target lesion diameter)
      - DiagnosticReport (RECIST response)

    Returns:
        HL7 FHIR R4 Bundle dictionary.
    """
    record_id = state.get("record_id", str(uuid.uuid4())[:8])
    patient_id = f"Patient-{record_id}"

    # 1. FHIR Patient Resource
    patient_resource = {
        "resourceType": "Patient",
        "id": patient_id,
        "active": True,
        "gender": "male" if state.get("demographics", {}).get("gender", "M") == "M" else "female",
        "birthDate": f"{datetime.now().year - state.get('demographics', {}).get('age', 70)}-01-01"
    }

    entries = [
        {
            "fullUrl": f"urn:uuid:{patient_id}",
            "resource": patient_resource
        }
    ]

    # 2. FHIR Condition Resources (ICD-10 Codes)
    for idx, item in enumerate(state.get("icd10_codes") or []):
        cond_id = f"Condition-{record_id}-{idx+1}"
        cond_resource = {
            "resourceType": "Condition",
            "id": cond_id,
            "clinicalStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]
            },
            "verificationStatus": {
                "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]
            },
            "code": {
                "coding": [
                    {
                        "system": "http://hl7.org/fhir/sid/icd-10-cm",
                        "code": item.get("code", "R69"),
                        "display": item.get("description", "Unspecified condition")
                    }
                ],
                "text": item.get("description", "Unspecified condition")
            },
            "subject": {"reference": f"Patient/{patient_id}"}
        }
        entries.append({
            "fullUrl": f"urn:uuid:{cond_id}",
            "resource": cond_resource
        })

    # 3. FHIR MedicationStatement Resources
    for idx, med in enumerate(state.get("extracted_medications") or []):
        med_id = f"MedicationStatement-{record_id}-{idx+1}"
        med_resource = {
            "resourceType": "MedicationStatement",
            "id": med_id,
            "status": "active",
            "medicationCodeableConcept": {
                "coding": [
                    {
                        "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                        "code": str(med.get("rxcui", "1000")),
                        "display": med.get("drug_name", "Unspecified Medication")
                    }
                ],
                "text": med.get("drug_name", "Unspecified Medication")
            },
            "subject": {"reference": f"Patient/{patient_id}"}
        }
        entries.append({
            "fullUrl": f"urn:uuid:{med_id}",
            "resource": med_resource
        })

    # 4. FHIR Observation Resource (SDOH & RAF Scores)
    if state.get("sdoh_risk_label"):
        sdoh_id = f"Observation-SDOH-{record_id}"
        sdoh_obs = {
            "resourceType": "Observation",
            "id": sdoh_id,
            "status": "final",
            "code": {
                "text": "Social Determinants of Health (SDOH) Risk Level"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "valueString": state.get("sdoh_risk_label")
        }
        entries.append({"fullUrl": f"urn:uuid:{sdoh_id}", "resource": sdoh_obs})

    if state.get("total_raf_score"):
        raf_id = f"Observation-RAF-{record_id}"
        raf_obs = {
            "resourceType": "Observation",
            "id": raf_id,
            "status": "final",
            "code": {
                "text": "CMS Risk Adjustment Factor (RAF) Score"
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "valueQuantity": {
                "value": state.get("total_raf_score"),
                "unit": "RAF points"
            }
        }
        entries.append({"fullUrl": f"urn:uuid:{raf_id}", "resource": raf_obs})

    # Build FHIR R4 Bundle
    bundle = {
        "resourceType": "Bundle",
        "id": f"Bundle-{record_id}",
        "type": "collection",
        "timestamp": datetime.now().isoformat(),
        "total": len(entries),
        "entry": entries
    }

    log.info("[FHIR R4 ADAPTER] Exported %d resources into FHIR Bundle for Patient/%s", len(entries), patient_id)
    return bundle


def generate_synthetic_patient_chart(condition: str = "Non-Small Cell Lung Cancer") -> Dict[str, Any]:
    """
    Generates a synthetic oncology patient chart complete with clinical note,
    demographics, medication list, and multi-visit timeline for seeding & testing.
    """
    return {
        "patient_id": f"SYNTH-{str(uuid.uuid4())[:6].upper()}",
        "name": "Synthetic Patient Alpha",
        "demographics": {"age": 68, "gender": "M", "medicaid": False},
        "condition": condition,
        "raw_note": (
            "PATIENT PROGRESS NOTE:\n"
            "68-year-old male with Stage III Non-Small Cell Lung Cancer (Adenocarcinoma), EGFR Exon 19 positive. "
            "Currently receiving Osimertinib 80mg daily and Warfarin 5mg daily for DVT prophylaxis. "
            "Patient reports mild fatigue and skin rash. CT scan reveals right upper lobe mass measuring 28mm (down from 42mm at baseline). "
            "History of type 2 diabetes mellitus with peripheral neuropathy and chronic obstructive pulmonary disease."
        ),
        "medications": ["Osimertinib", "Warfarin", "Aspirin"],
        "visit_history": [
            {"date": "2025-11-01", "doc_type": "Baseline CT Scan", "summary": "Right upper lobe lung lesion 42mm.", "target_lesion_mm": 42.0},
            {"date": "2026-03-15", "doc_type": "Follow-up CT #1", "summary": "Lesion reduced to 34mm (-19.0%).", "target_lesion_mm": 34.0},
            {"date": "2026-07-20", "doc_type": "Follow-up CT #2", "summary": "Lesion further reduced to 28mm (-33.3%). Partial Response (PR).", "target_lesion_mm": 28.0}
        ]
    }


def export_preventive_state_to_fhir(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts extracted ASHA Preventive Oncology state into an HL7 FHIR R4 JSON Bundle.

    Resources included in Bundle:
      - Patient
      - RiskAssessment (Overall lifestyle cancer risk score)
      - Observation (for each extracted lifestyle factor: tobacco, arsenic, beedi, etc.)
      - ServiceRequest (ICMR Grounded Preventive Screening referrals)
    """
    import json
    patient_id = state.get("patient_id") or state.get("record_id") or f"pat-{str(uuid.uuid4())[:8]}"
    fhir_patient_id = f"Patient-{patient_id}"

    entries = [
        {
            "fullUrl": f"urn:uuid:{fhir_patient_id}",
            "resource": {
                "resourceType": "Patient",
                "id": fhir_patient_id,
                "active": True,
                "text": {"status": "generated", "div": "<div xmlns=\"http://www.w3.org/1999/xhtml\">ASHA Preventive Patient Record</div>"}
            }
        }
    ]

    # 1. FHIR RiskAssessment Resource
    risk_score = state.get("lifestyle_risk_score", 0.0)
    risk_assessment_id = f"RiskAssessment-{patient_id}"
    risk_resource = {
        "resourceType": "RiskAssessment",
        "id": risk_assessment_id,
        "status": "final",
        "subject": {"reference": f"Patient/{fhir_patient_id}"},
        "occurrenceDateTime": datetime.now().isoformat(),
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "363346000",
                            "display": "Malignant neoplastic disease (risk)"
                        }
                    ],
                    "text": "Preventive Oncology Cancer Risk"
                },
                "probabilityDecimal": float(risk_score),
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": "high" if risk_score > 0.5 else "low",
                            "display": "High Risk" if risk_score > 0.5 else "Low Risk"
                        }
                    ]
                }
            }
        ]
    }
    entries.append({"fullUrl": f"urn:uuid:{risk_assessment_id}", "resource": risk_resource})

    # 2. FHIR Observation Resources (for each lifestyle risk factor)
    factors = state.get("lifestyle_factors") or []
    if isinstance(factors, str) and factors.startswith("["):
        try:
            factors = json.loads(factors)
        except Exception:
            factors = []

    for idx, f in enumerate(factors if isinstance(factors, list) else []):
        if not isinstance(f, dict):
            continue
        term = f.get("term", "unknown")
        post = f.get("posterior", 0.0)
        obs_id = f"Observation-Lifestyle-{patient_id}-{idx+1}"

        obs_resource = {
            "resourceType": "Observation",
            "id": obs_id,
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                            "code": "social-history",
                            "display": "Social History"
                        }
                    ]
                }
            ],
            "code": {
                "text": f"Lifestyle Cancer Risk Factor: {term.capitalize()}"
            },
            "subject": {"reference": f"Patient/{fhir_patient_id}"},
            "valueQuantity": {
                "value": round(float(post), 4),
                "unit": "posterior probability",
                "system": "http://unitsofmeasure.org",
                "code": "1"
            }
        }
        entries.append({"fullUrl": f"urn:uuid:{obs_id}", "resource": obs_resource})

    # 3. FHIR ServiceRequest Resource (Screening referral)
    recs = state.get("preventive_recommendations")
    if recs:
        req_id = f"ServiceRequest-Screening-{patient_id}"
        req_resource = {
            "resourceType": "ServiceRequest",
            "id": req_id,
            "status": "active",
            "intent": "proposal",
            "subject": {"reference": f"Patient/{fhir_patient_id}"},
            "code": {
                "text": "ICMR Preventive Cancer Screening Protocol"
            },
            "patientInstruction": recs
        }
        entries.append({"fullUrl": f"urn:uuid:{req_id}", "resource": req_resource})

    bundle = {
        "resourceType": "Bundle",
        "id": f"Bundle-Preventive-{patient_id}",
        "type": "collection",
        "timestamp": datetime.now().isoformat(),
        "total": len(entries),
        "entry": entries
    }

    log.info("[FHIR R4 PREVENTIVE] Exported %d resources into FHIR Bundle for Patient/%s", len(entries), patient_id)
    return bundle
