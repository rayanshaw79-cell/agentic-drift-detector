"""
clinical/steps/fhir_step.py — LangGraph SMART-on-FHIR R4 Adapter node.

Converts accumulated ClinicalState into an HL7 FHIR R4 JSON Bundle.
"""

import logging
from typing import Dict, Any

from schemas.clinical_state import ClinicalState
from clinical.tools.fhir_adapter import export_clinical_state_to_fhir

log = logging.getLogger(__name__)


def fhir_step(state: ClinicalState) -> dict:
    """
    LangGraph State Node — SMART-on-FHIR R4 Adapter Node.

    Reads:
      - ClinicalState
    Emits:
      - fhir_bundle: dict (HL7 FHIR R4 JSON Bundle)
    """
    bundle = export_clinical_state_to_fhir(dict(state))

    total_resources = bundle.get("total", 0)
    print(f"\n  [FHIR R4 ADAPTER] Bundled {total_resources} HL7 FHIR R4 resources for export.")

    return {
        "current_step": "fhir_export",
        "fhir_bundle": bundle,
        "path_taken": ["fhir_export"]
    }
