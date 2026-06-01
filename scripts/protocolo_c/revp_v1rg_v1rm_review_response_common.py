"""Shared helpers for REV-P Protocol C v1rg-v1rm.

Manual double-review response workflow + supervisor gate. Transforms the P0
double-review packets into an auditable flow of human A/B responses and a
supervisor decision. Even when a case reaches C3-candidate state it stays
review-only: never an operational label, target, or field-validated ground
truth. Fail-closed whenever no real manual review exists.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (  # noqa: F401
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    assert_no_forbidden_true,
    classify_source_family,
    composite_score,
    detect_absolute_path,
    detect_local_runs_exposure,
    guardrail_row,
    hash_short,
    mask_path,
    normalize_event_id,
    normalize_patch_id,
    normalize_region,
    read_csv_safe,
    safe_relpath,
    score_source_reliability,
    score_spatial_precision,
    score_temporal_precision,
    write_csv_with_header,
    write_doc,
    write_json_safe,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Env vars
# ---------------------------------------------------------------------------

ENV_RESPONSES = "REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH"
ENV_SUPERVISOR = "REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH"
ENV_ALLOW_SYNTHETIC = "REVP_PROTOCOL_C_ALLOW_SYNTHETIC_RESPONSES"


def allow_synthetic() -> bool:
    return os.environ.get(ENV_ALLOW_SYNTHETIC, "false").strip().lower() == "true"


def responses_path() -> Path | None:
    env = os.environ.get(ENV_RESPONSES)
    if env and Path(env).exists():
        return Path(env)
    return None


def supervisor_decisions_path() -> Path | None:
    env = os.environ.get(ENV_SUPERVISOR)
    if env and Path(env).exists():
        return Path(env)
    return None


# ---------------------------------------------------------------------------
# Reviewers / questions / decisions
# ---------------------------------------------------------------------------

REVIEWER_SLOTS = ["REVIEWER_A", "REVIEWER_B"]

REVIEW_QUESTIONS = [
    "evidence_visible",
    "event_supported",
    "location_supported",
    "timing_supported",
    "source_quality",
    "independent_source_present",
    "uncertainty_level",
    "recommended_decision",
    "uncertainty_notes",
]

REQUIRED_QUESTIONS = [
    "evidence_visible", "event_supported", "location_supported",
    "timing_supported", "source_quality", "independent_source_present",
    "recommended_decision",
]

# Allowed review/adjudication decisions
KEEP_C1_CONTEXTUAL = "KEEP_C1_CONTEXTUAL"
KEEP_C2_REVIEW_ONLY = "KEEP_C2_REVIEW_ONLY"
C3_NEEDS_SUPERVISOR = "C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR"
BLOCK_C3_TEMPORAL = "BLOCK_C3_TEMPORAL_PRECISION"
BLOCK_C3_SPATIAL = "BLOCK_C3_SPATIAL_PRECISION"
BLOCK_C3_SOURCE = "BLOCK_C3_SOURCE_WEAK"
BLOCK_C3_DISAGREEMENT = "BLOCK_C3_REVIEW_DISAGREEMENT"
BLOCK_C4_NO_FORMAL_NEGATIVE = "BLOCK_C4_NO_FORMAL_NEGATIVE_SOURCE"

ALLOWED_DECISIONS = frozenset([
    KEEP_C1_CONTEXTUAL, KEEP_C2_REVIEW_ONLY, C3_NEEDS_SUPERVISOR,
    BLOCK_C3_TEMPORAL, BLOCK_C3_SPATIAL, BLOCK_C3_SOURCE,
    BLOCK_C3_DISAGREEMENT, BLOCK_C4_NO_FORMAL_NEGATIVE,
])

# Supervisor actions
SUP_APPROVE_C3 = "APPROVE_C3_CANDIDATE_REVIEW_ONLY"
SUP_KEEP_C2 = "KEEP_C2_REVIEW_ONLY"
SUP_BLOCK_SOURCE = "BLOCK_C3_NEEDS_MORE_SOURCE"
SUP_BLOCK_TEMPORAL = "BLOCK_C3_NEEDS_BETTER_TEMPORAL_PRECISION"
SUP_BLOCK_SPATIAL = "BLOCK_C3_NEEDS_BETTER_SPATIAL_PRECISION"
SUP_REQUEST_MORE = "REQUEST_ADDITIONAL_REVIEW"

ALLOWED_SUPERVISOR_ACTIONS = frozenset([
    SUP_APPROVE_C3, SUP_KEEP_C2, SUP_BLOCK_SOURCE,
    SUP_BLOCK_TEMPORAL, SUP_BLOCK_SPATIAL, SUP_REQUEST_MORE,
])


def normalize_reviewer_slot(raw: str) -> str:
    s = str(raw or "").strip().upper().replace(" ", "_")
    if s in ("A", "REVIEWER_A", "REVIEWERA"):
        return "REVIEWER_A"
    if s in ("B", "REVIEWER_B", "REVIEWERB"):
        return "REVIEWER_B"
    return "UNKNOWN_SLOT"


def normalize_decision(raw: str) -> str:
    s = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    if not s:
        return ""
    if s in ALLOWED_DECISIONS:
        return s
    if "DISAGREE" in s:
        return BLOCK_C3_DISAGREEMENT
    if "C3" in s:
        return C3_NEEDS_SUPERVISOR
    if "C2" in s:
        return KEEP_C2_REVIEW_ONLY
    if "C1" in s or "CONTEXT" in s:
        return KEEP_C1_CONTEXTUAL
    if "TEMPORAL" in s:
        return BLOCK_C3_TEMPORAL
    if "SPATIAL" in s:
        return BLOCK_C3_SPATIAL
    if "SOURCE" in s or "WEAK" in s or "BLOCK" in s or "REJECT" in s:
        return BLOCK_C3_SOURCE
    return ""


def normalize_supervisor_action(raw: str) -> str:
    s = str(raw or "").strip().upper().replace("-", "_").replace(" ", "_")
    if s in ALLOWED_SUPERVISOR_ACTIONS:
        return s
    if "APPROVE" in s and "C3" in s:
        return SUP_APPROVE_C3
    if "C2" in s or "KEEP" in s:
        return SUP_KEEP_C2
    if "TEMPORAL" in s:
        return SUP_BLOCK_TEMPORAL
    if "SPATIAL" in s:
        return SUP_BLOCK_SPATIAL
    if "SOURCE" in s:
        return SUP_BLOCK_SOURCE
    if "REQUEST" in s or "MORE_REVIEW" in s or "ADDITIONAL" in s:
        return SUP_REQUEST_MORE
    return ""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _yes(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("sim", "yes", "true", "1", "y")


def gather_responses(rows: list[dict[str, str]]) -> dict[str, dict[str, dict[str, str]]]:
    """sample_id -> reviewer_slot -> {question_id: answer_value}."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for r in rows:
        rsid = str(r.get("review_sample_id", "")).strip()
        slot = normalize_reviewer_slot(r.get("reviewer_slot", ""))
        qid = str(r.get("question_id", "") or r.get("question_key", "")).strip()
        val = str(r.get("answer_value", "") or r.get("response_value", "")).strip()
        if not rsid or slot == "UNKNOWN_SLOT" or not qid:
            continue
        out.setdefault(rsid, {}).setdefault(slot, {})[qid] = val
    return out


def filled_response_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if str(r.get("answer_value", "") or r.get("response_value", "")).strip()]


# ---------------------------------------------------------------------------
# P0/P1 context loader
# ---------------------------------------------------------------------------

def load_p0_p1_context(datasets: Path | None = None) -> dict[str, list[dict[str, str]]]:
    ds = datasets or DATASETS
    files = {
        "packet_manifest": "protocol_c_double_review_packet_manifest_v1qw.csv",
        "review_forms": "protocol_c_double_review_forms_v1qw.csv",
        "review_sample": "protocol_c_event_patch_review_sample_v1qv.csv",
        "observational_scores": "protocol_c_observational_evidence_scores_v1qx.csv",
        "adjudication": "protocol_c_ground_reference_adjudication_registry_v1qy.csv",
        "partial_summary": "protocol_c_ground_reference_partial_scientific_summary_v1qz.csv",
        "intake_validation": "protocol_c_external_document_intake_validation_v1rc.csv",
        "external_candidates": "protocol_c_external_event_candidates_v1rd.csv",
        "external_links": "protocol_c_external_event_patch_link_candidates_v1re.csv",
    }
    return {k: read_csv_safe(ds / v) for k, v in files.items()}


# ---------------------------------------------------------------------------
# Completed-review scoring
# ---------------------------------------------------------------------------

_REVIEW_WEIGHTS = {
    "evidence_support_score": 0.2,
    "temporal_support_score": 0.2,
    "spatial_support_score": 0.2,
    "source_support_score": 0.25,
    "reviewer_agreement_score": 0.15,
}


def _support_from_two(a_yes: bool, b_yes: bool, both: float = 1.0, one: float = 0.5) -> float:
    if a_yes and b_yes:
        return both
    if a_yes or b_yes:
        return one
    return 0.0


def score_agreement(decision_a: str, decision_b: str) -> float:
    a = normalize_decision(decision_a)
    b = normalize_decision(decision_b)
    if a and b:
        return 1.0 if a == b else 0.3
    if a or b:
        return 0.5
    return 0.0


def compute_completed_score(rsid: str, slots: dict[str, dict[str, str]],
                            meta: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    a = slots.get("REVIEWER_A", {})
    b = slots.get("REVIEWER_B", {})
    a_present = "true" if a else "false"
    b_present = "true" if b else "false"

    evidence = _support_from_two(_yes(a.get("event_supported")), _yes(b.get("event_supported")))
    temporal = _support_from_two(_yes(a.get("timing_supported")), _yes(b.get("timing_supported")), both=1.0, one=0.6)
    spatial = _support_from_two(_yes(a.get("location_supported")), _yes(b.get("location_supported")), both=1.0, one=0.6)

    quality_txt = (a.get("source_quality", "") + " " + b.get("source_quality", "") + " "
                   + meta.get("evidence_status", ""))
    family = classify_source_family(quality_txt)
    source = score_source_reliability(family)

    agreement = score_agreement(a.get("recommended_decision", ""), b.get("recommended_decision", ""))

    scores = {
        "evidence_support_score": evidence,
        "temporal_support_score": temporal,
        "spatial_support_score": spatial,
        "source_support_score": source,
        "reviewer_agreement_score": agreement,
    }
    composite = round(sum(_REVIEW_WEIGHTS[k] * v for k, v in scores.items()), 4)

    da = normalize_decision(a.get("recommended_decision", ""))
    db = normalize_decision(b.get("recommended_decision", ""))
    both_complete = bool(a) and bool(b)
    disagree = bool(da and db and da != db)
    disagreement_type = ""
    if not both_complete:
        disagreement_type = "INCOMPLETE_REVIEW"
    elif disagree:
        disagreement_type = "DECISION_MISMATCH"

    blocked_reason = ""
    if not both_complete:
        decision = KEEP_C2_REVIEW_ONLY
        blocked_reason = "REVIEW_INCOMPLETE_SINGLE_REVIEWER"
    elif disagree:
        decision = BLOCK_C3_DISAGREEMENT
        blocked_reason = "REVIEWERS_DISAGREE"
    elif source < 0.75:
        decision = BLOCK_C3_SOURCE
        blocked_reason = "SOURCE_RELIABILITY_BELOW_THRESHOLD"
    elif temporal < 0.6:
        decision = BLOCK_C3_TEMPORAL
        blocked_reason = "TEMPORAL_SUPPORT_BELOW_THRESHOLD"
    elif spatial < 0.6:
        decision = BLOCK_C3_SPATIAL
        blocked_reason = "SPATIAL_SUPPORT_BELOW_THRESHOLD"
    elif da == C3_NEEDS_SUPERVISOR and db == C3_NEEDS_SUPERVISOR:
        decision = C3_NEEDS_SUPERVISOR
    elif composite >= 0.75:
        decision = C3_NEEDS_SUPERVISOR
    else:
        decision = KEEP_C2_REVIEW_ONLY

    supervisor_required = "true" if decision == C3_NEEDS_SUPERVISOR else "false"

    row = {
        "completed_score_id": f"V1RI_CS_{rsid}",
        "review_sample_id": rsid,
        "event_id": meta.get("event_id", ""),
        "patch_id": meta.get("patch_id", ""),
        "region": normalize_region(meta.get("region", "")),
        "reviewer_a_present": a_present,
        "reviewer_b_present": b_present,
        "reviewer_agreement_score": f"{agreement:.3f}",
        "disagreement_flag": "true" if disagree else "false",
        "disagreement_type": disagreement_type,
        "evidence_support_score": f"{evidence:.3f}",
        "temporal_support_score": f"{temporal:.3f}",
        "spatial_support_score": f"{spatial:.3f}",
        "source_support_score": f"{source:.3f}",
        "composite_review_score": f"{composite:.3f}",
        "recommended_decision": decision,
        "supervisor_review_required": supervisor_required,
        "blocked_reason": blocked_reason,
        "notes": "",
    }
    row.update(guardrail_row())

    disagreement_row = None
    if disagree or not both_complete:
        disagreement_row = {
            "disagreement_id": f"V1RI_DIS_{rsid}",
            "review_sample_id": rsid,
            "decision_a": da, "decision_b": db,
            "disagreement_type": disagreement_type,
            "reviewer_agreement_score": f"{agreement:.3f}",
            "needs_third_reviewer": "true" if disagree else "false",
            "review_only": "true", "notes": "",
        }
    return row, disagreement_row
