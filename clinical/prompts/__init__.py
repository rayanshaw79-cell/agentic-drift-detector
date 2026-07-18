"""
clinical/prompts — Few-shot oncology prompt library.

Each module exposes a build_*_prompt(document_type, **kwargs) function
that returns a (system_prompt, few_shot_examples) tuple ready for LLM
invocation.  The document_type argument allows the router (Pillar 1) to
select pathology-specific vs. radiology-specific example sets.
"""
