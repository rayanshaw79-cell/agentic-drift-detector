"""
clinical/tools/bayesian_ensemble.py — Bayesian Ensemble Voting for NER.

Implements the "Precision Without Losing Recall" model from Miimansa's research:
  Treat each extractor as an imperfect annotator and estimate
  P(Entity Exists | Votes) using Bayes' Rule.

Core formula (from the slide):
  P(Y=1 | A) = P(A|Y=1) * P(Y=1)
               ─────────────────────────────────────────
               P(A|Y=1)*P(Y=1) + P(A|Y=0)*P(Y=0)

  Under the independence assumption:
  P(A|Y) = ∏ P(A_i | Y)   ← "naive" ensembler

Where:
  Y  = true (hidden) entity status
  A  = observed votes from all extractors
  P(Y=1) = prior probability that a term is a real medical entity
  P(A_i|Y=1) = sensitivity of extractor i (true positive rate)
  P(A_i|Y=0) = 1 - specificity of extractor i (false positive rate)
"""

import math
import logging

log = logging.getLogger(__name__)

# ── Per-extractor accuracy parameters ─────────────────────────────────────────
# These are empirical estimates of each extractor's accuracy characteristics.
# sensitivity = P(extractor votes YES | entity truly exists)  = true positive rate
# fpr         = P(extractor votes YES | entity does NOT exist) = false positive rate
#
# Gemini:  High sensitivity (catches most things), some hallucination → moderate fpr
# Regex:   Very high specificity (conservative), misses novel terms → low sensitivity
# NLM API: Good precision (codebook-grounded), misses abbreviations → moderate sensitivity

_EXTRACTOR_PARAMS = {
    "gemini": {"sensitivity": 0.90, "fpr": 0.15},
    "regex":  {"sensitivity": 0.65, "fpr": 0.02},
    "nlm":    {"sensitivity": 0.75, "fpr": 0.05},
}

# Prior: P(a candidate term is a real medical entity)
# Set moderately high — clinical notes are dense with real conditions
_PRIOR = 0.70


def bayesian_posterior(
    votes: dict[str, bool],
    prior: float = _PRIOR,
    params: dict = _EXTRACTOR_PARAMS,
) -> float:
    """
    Compute P(Entity Exists | Votes) using Bayes' Rule with independence assumption.

    Args:
        votes:  {extractor_name: bool} — did each extractor vote YES for this term?
        prior:  P(Y=1) — prior probability the term is a real entity
        params: Per-extractor {sensitivity, fpr} parameters

    Returns:
        Posterior probability (0.0–1.0) that the term is a real medical entity.
    """
    if not votes:
        return prior

    # Compute log-likelihood ratio for numerical stability
    log_prior_odds = math.log(prior / (1 - prior))
    log_likelihood = 0.0

    for extractor, voted_yes in votes.items():
        p = params.get(extractor, {"sensitivity": 0.70, "fpr": 0.10})
        sens = p["sensitivity"]
        fpr  = p["fpr"]

        if voted_yes:
            # P(vote=YES | entity) / P(vote=YES | no entity)
            log_likelihood += math.log(sens / fpr) if fpr > 0 else math.log(sens / 1e-6)
        else:
            # P(vote=NO | entity) / P(vote=NO | no entity)
            no_entity_no_vote = 1 - fpr
            entity_no_vote    = 1 - sens
            if entity_no_vote > 0 and no_entity_no_vote > 0:
                log_likelihood += math.log(entity_no_vote / no_entity_no_vote)

    log_posterior_odds = log_prior_odds + log_likelihood

    # Convert back to probability via sigmoid
    posterior = 1.0 / (1.0 + math.exp(-log_posterior_odds))
    return round(posterior, 4)


def ensemble_vote(
    gemini_terms:  list[str],
    regex_terms:   list[str],
    nlm_terms:     list[str],
) -> list[dict]:
    """
    Combine three extractor outputs into a Bayesian-scored entity list.

    Each unique term found by ANY extractor gets a posterior confidence score
    based on how many (and which) extractors agreed.

    Args:
        gemini_terms:  Terms extracted by Gemini LLM
        regex_terms:   Terms extracted by the regex dictionary
        nlm_terms:     Terms found via NLM API term search

    Returns:
        List of dicts sorted by posterior confidence (highest first):
        [
          {
            "term":       str,
            "posterior":  float,   # Bayesian P(entity exists | votes)
            "votes": {
              "gemini": bool,
              "regex":  bool,
              "nlm":    bool,
            }
          },
          ...
        ]
    """
    # Normalise all terms to lowercase for comparison
    g_set = {t.lower().strip() for t in gemini_terms}
    r_set = {t.lower().strip() for t in regex_terms}
    n_set = {t.lower().strip() for t in nlm_terms}

    # All unique candidate terms (union of all extractors)
    all_terms = g_set | r_set | n_set
    all_terms.discard("")
    all_terms.discard("unspecified condition")

    results: list[dict] = []
    for term in all_terms:
        votes = {
            "gemini": term in g_set,
            "regex":  term in r_set,
            "nlm":    term in n_set,
        }
        posterior = bayesian_posterior(votes)
        n_votes   = sum(votes.values())

        log.debug(
            "[BAYES] '%s': gemini=%s regex=%s nlm=%s → P=%.3f",
            term, votes["gemini"], votes["regex"], votes["nlm"], posterior,
        )

        results.append({
            "term":      term,
            "posterior": posterior,
            "votes":     votes,
            "n_votes":   n_votes,
        })

    # Sort by posterior confidence descending
    results.sort(key=lambda x: x["posterior"], reverse=True)
    return results
