"""REV-P v1ow — Evidence strength and precision scoring.

Scores each event/evidence row from v1ov with auditable criteria.
Does NOT transform scores into labels. can_promote_to_label=false always.
can_train_model=false always.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS, read_csv
from revp_v1ou_v1pa_common import (
    _p,
    assert_no_forbidden_true,
    require_no_abs_paths_in_rows,
    write_csv_safe,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Output paths (env-overridable)
# ---------------------------------------------------------------------------

OUT_SCORING = _p("REVP_V1OW_OUT_SCORING", DATASETS / "recife_ground_reference_evidence_scoring_v1ow.csv")
OUT_SUMMARY = _p("REVP_V1OW_OUT_SUMMARY", DATASETS / "recife_ground_reference_evidence_scoring_summary_v1ow.csv")
SCHEMA_SCORING = _p("REVP_V1OW_SCHEMA_SCORING", SCHEMAS / "recife_ground_reference_evidence_scoring_v1ow_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1OW_SCHEMA_SUMMARY", SCHEMAS / "recife_ground_reference_evidence_scoring_summary_v1ow_schema.csv")
DOC = _p("REVP_V1OW_DOC", DOCS / "revp_v1ow_evidence_strength_precision_scoring.md")
IN_V1OV = _p("REVP_V1OW_IN_V1OV", DATASETS / "recife_ground_reference_observed_event_registry_v1ov.csv")

SCORING_FIELDS = [
    "evidence_id",
    "event_id",
    "source_candidate_id",
    "temporal_precision_score",
    "spatial_precision_score",
    "source_reliability_score",
    "event_specificity_score",
    "independence_score",
    "total_review_score",
    "evidence_tier",
    "allowed_use",
    "can_promote_to_label",
    "can_train_model",
    "blocked_reason",
    "reason_codes",
    "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

# Tiers
TIER_STRONG = "STRONG_REVIEW_ONLY"
TIER_MODERATE = "MODERATE_REVIEW_ONLY"
TIER_LIMITED = "LIMITED_CONTEXTUAL"
TIER_CONTEXTUAL = "CONTEXTUAL_GAP"
TIER_BLOCKED = "BLOCKED"

# Score mappings
TEMPORAL_SCORES = {
    "HIGH": 3,
    "MODERATE": 2,
    "LOW": 1,
    "NONE": 0,
}

SPATIAL_SCORES = {
    "POINT_EXPLICIT": 3,
    "ADDRESS_LEVEL": 2,
    "ADMINISTRATIVE": 1,
    "NONE": 0,
}

RELIABILITY_SCORES = {
    "OFFICIAL_HIGH": 3,
    "NEWS_LIMITED": 1,
    "CONTEXTUAL_LIMITED": 1,
    "UNKNOWN": 0,
}

# Event status specificity
SPECIFICITY_SCORES = {
    "OBSERVED_EVENT_CONFIRMED_REVIEW_ONLY": 3,
    "OBSERVED_EVENT_PROBABLE_REVIEW_ONLY": 2,
    "CONTEXTUAL_EVIDENCE_ONLY": 1,
    "BLOCKED_INSUFFICIENT_EVIDENCE": 0,
    "BLOCKED_FIXTURE_OR_SYNTHETIC": 0,
}


def _score_row(event_row: dict[str, str], seq: int) -> dict[str, Any]:
    """Score a single event row from v1ov."""
    event_id = event_row.get("event_id", f"UNKNOWN_{seq}")
    src_id = event_row.get("source_candidate_id", "")

    time_prec = event_row.get("event_time_precision", "NONE")
    spatial_prec = event_row.get("spatial_precision_level", "NONE")
    reliability = event_row.get("source_reliability_level", "UNKNOWN")
    obs_status = event_row.get("observed_event_status", "BLOCKED_INSUFFICIENT_EVIDENCE")
    allowed_use = event_row.get("allowed_use", "BLOCKED_INSUFFICIENT_EVIDENCE")
    blocked_in = event_row.get("blocked_reason", "")

    t_score = TEMPORAL_SCORES.get(time_prec, 0)
    s_score = SPATIAL_SCORES.get(spatial_prec, 0)
    r_score = RELIABILITY_SCORES.get(reliability, 0)
    e_score = SPECIFICITY_SCORES.get(obs_status, 0)

    # Independence: we only have one source per event → low independence
    indep_score = 0

    total = t_score + s_score + r_score + e_score + indep_score

    # Determine tier
    reason_codes = []
    if t_score == 0:
        reason_codes.append("NO_TEMPORAL_PRECISION")
    if s_score == 0:
        reason_codes.append("NO_SPATIAL_PRECISION")
    if r_score == 0:
        reason_codes.append("NO_SOURCE_RELIABILITY")
    if e_score == 0:
        reason_codes.append("NO_EVENT_SPECIFICITY")
    if indep_score == 0:
        reason_codes.append("NO_INDEPENDENT_SOURCE")

    if allowed_use.startswith("BLOCKED") or obs_status.startswith("BLOCKED"):
        tier = TIER_BLOCKED
        blocked_reason = blocked_in or allowed_use or "BLOCKED_BY_V1OV"
    elif total >= 10:
        tier = TIER_STRONG
        blocked_reason = ""
    elif total >= 6:
        tier = TIER_MODERATE
        blocked_reason = ""
    elif total >= 3:
        tier = TIER_LIMITED
        blocked_reason = "LIMITED_EVIDENCE_ONLY"
    elif total >= 1:
        tier = TIER_CONTEXTUAL
        blocked_reason = "CONTEXTUAL_GAP_ONLY"
    else:
        tier = TIER_BLOCKED
        blocked_reason = "ZERO_SCORE_ALL_DIMENSIONS"

    if tier == TIER_BLOCKED and not blocked_reason:
        blocked_reason = "SCORE_BELOW_THRESHOLD"

    return {
        "evidence_id": f"V1OW_SCORE_{seq:04d}",
        "event_id": event_id,
        "source_candidate_id": src_id,
        "temporal_precision_score": str(t_score),
        "spatial_precision_score": str(s_score),
        "source_reliability_score": str(r_score),
        "event_specificity_score": str(e_score),
        "independence_score": str(indep_score),
        "total_review_score": str(total),
        "evidence_tier": tier,
        "allowed_use": allowed_use,
        "can_promote_to_label": "false",
        "can_train_model": "false",
        "blocked_reason": blocked_reason,
        "reason_codes": ";".join(reason_codes),
        "notes": "",
    }


def run() -> None:
    event_rows = read_csv(IN_V1OV) if IN_V1OV.exists() else []

    if not event_rows:
        # Generate empty registry with header
        rows: list[dict[str, Any]] = []
        status = "NO_V1OV_INPUT_FOUND"
    else:
        rows = [_score_row(r, i) for i, r in enumerate(event_rows)]
        status = "SCORED"

    assert_no_forbidden_true(rows, "v1ow_scoring")
    require_no_abs_paths_in_rows(rows, "v1ow_scoring")

    write_csv_safe(OUT_SCORING, rows, SCORING_FIELDS)
    write_schema_safe(SCHEMA_SCORING, SCORING_FIELDS, "v1ow_evidence_strength_precision_scoring")

    tier_counts = {TIER_STRONG: 0, TIER_MODERATE: 0, TIER_LIMITED: 0, TIER_CONTEXTUAL: 0, TIER_BLOCKED: 0}
    for r in rows:
        tier_counts[r["evidence_tier"]] = tier_counts.get(r["evidence_tier"], 0) + 1

    summary_rows = [
        {"stat_key": "total_scored", "stat_value": str(len(rows))},
        {"stat_key": "tier_strong_review_only", "stat_value": str(tier_counts[TIER_STRONG])},
        {"stat_key": "tier_moderate_review_only", "stat_value": str(tier_counts[TIER_MODERATE])},
        {"stat_key": "tier_limited_contextual", "stat_value": str(tier_counts[TIER_LIMITED])},
        {"stat_key": "tier_contextual_gap", "stat_value": str(tier_counts[TIER_CONTEXTUAL])},
        {"stat_key": "tier_blocked", "stat_value": str(tier_counts[TIER_BLOCKED])},
        {"stat_key": "can_promote_to_label_any", "stat_value": "false"},
        {"stat_key": "can_train_model_any", "stat_value": "false"},
        {"stat_key": "scoring_status", "stat_value": status},
        {"stat_key": "stage", "stat_value": "v1ow"},
    ]
    write_csv_safe(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ow_summary")

    write_doc(
        DOC,
        "v1ow — Evidence Strength and Precision Scoring",
        [
            "## Objetivo",
            "Pontuar cada evento/evidência do v1ov com critérios auditáveis de precisão "
            "temporal, precisão espacial, confiabilidade da fonte, especificidade do evento "
            "e independência. Não transforma score em label.",
            "## Critérios de pontuação",
            "- Temporal: HIGH=3, MODERATE=2, LOW=1, NONE=0\n"
            "- Espacial: POINT_EXPLICIT=3, ADDRESS=2, ADMINISTRATIVE=1, NONE=0\n"
            "- Confiabilidade: OFFICIAL_HIGH=3, NEWS/CONTEXTUAL=1, UNKNOWN=0\n"
            "- Especificidade: CONFIRMED=3, PROBABLE=2, CONTEXTUAL=1, BLOCKED=0\n"
            "- Independência: 0 (fonte única disponível)",
            "## Tiers",
            "STRONG_REVIEW_ONLY (≥10), MODERATE_REVIEW_ONLY (≥6), "
            "LIMITED_CONTEXTUAL (≥3), CONTEXTUAL_GAP (≥1), BLOCKED (0).",
            "## Guardrails",
            "can_promote_to_label=false e can_train_model=false em todos os registros. "
            "Score alto não autoriza label ou treino.",
        ],
    )

    print(f"[v1ow] {len(rows)} scored: {tier_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1ow evidence strength and precision scoring")
    parser.parse_args()
    run()
