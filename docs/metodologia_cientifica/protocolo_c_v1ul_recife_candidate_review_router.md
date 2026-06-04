# Protocolo C v1ul - Recife Candidate Review Router

## Scope
- Reads v1uk registries only.
- Routes candidates for supervisor review without promotion.
- Evaluates overlay-readiness preconditions without executing overlay.
- Public outputs contain ids, hashes, flags, counts, and redacted summaries only.

## Guardrails
- ground_truth_operational=false
- can_create_ground_reference=false
- can_create_training_label=false
- can_reopen_protocol_b=false
- dino_usage=SUPPORT_ONLY
- no_overlay_executed=true
- no_coordinates_invented=true
- supervisor_review_completed=false
- max_status=RECIFE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW
