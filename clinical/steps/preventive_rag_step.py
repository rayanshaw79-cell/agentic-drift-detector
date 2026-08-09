import logging
from schemas.clinical_state import ClinicalState
from clinical.rag.guideline_store import retrieve_guidelines

log = logging.getLogger(__name__)

def preventive_rag_step(state: ClinicalState) -> ClinicalState:
    log.info("[Preventive] Running Preventive RAG Lookup...")
    
    factors = state.get("lifestyle_factors", [])
    
    recommendations = []
    
    if not factors:
        state["preventive_recommendations"] = "No high-risk lifestyle factors identified. General screening applies."
        return state
        
    for factor in factors:
        term = factor["term"]
        # Use existing RAG but query for "preventive" guidelines to trigger routing
        context = retrieve_guidelines(
            query=f"preventive screening recommendations for {term} users", 
            k=1
        )
        if context:
            recommendations.append(f"- For {term}: {context.strip()}")
            
    if recommendations:
        state["preventive_recommendations"] = "\n".join(recommendations)
    else:
        state["preventive_recommendations"] = "General ICMR screening protocols apply."
        
    return state
