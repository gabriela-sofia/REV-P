"""REV-P v1oy — Ground truth candidate decision audit.

Consolidates C1/C2/C3/C4 decisions from v1ov-v1ox and v1og-v1ot.
Explains why C3+ is not reached without confirmed scene_date.
Explains why C4 is closed without formal negative.

can_be_used_for_training=false, can_create_operational_label=false — always.
C3+ only if scene_date confirmed AND temporal criteria met.
C4 only if formal_negative_available=true.
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

OUT_AUDIT = _p("REVP_V1OY_OUT_AUDIT", DATASETS / "recife_ground_truth_candidate_decision_audit_v1oy.csv")
OUT_SUMMARY = _p("REVP_V1OY_OUT_SUMMARY", DATASETS / "recife_ground_truth_candidate_decision_summary_v1oy.csv")
SCHEMA_AUDIT = _p("REVP_V1OY_SCHEMA_AUDIT", SCHEMAS / "recife_ground_truth_candidate_decision_audit_v1oy_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1OY_SCHEMA_SUMMARY", SCHEMAS / "recife_ground_truth_candidate_decision_summary_v1oy_schema.csv")
DOC = _p("REVP_V1OY_DOC", DOCS / "revp_v1oy_ground_truth_candidate_decision_audit.md")
IN_V1OV = _p("REVP_V1OY_IN_V1OV", DATASETS / "recife_ground_reference_observed_event_registry_v1ov.csv")
IN_V1OW = _p("REVP_V1OY_IN_V1OW", DATASETS / "recife_ground_reference_evidence_scoring_v1ow.csv")
IN_V1OX = _p("REVP_V1OY_IN_V1OX", DATASETS / "recife_event_patch_linkage_registry_v1ox.csv")
IN_V1OT_SUMMARY = _p(
    "REVP_V1OY_IN_V1OT_SUMMARY",
    DATASETS / "recife_scene_date_recovery_final_scientific_summary_v1ot.csv",
)

AUDIT_FIELDS = [
    "decision_id",
    "event_id",
    "patch_id",
    "candidate_level",
    "decision_status",
    "evidence_tier",
    "spatial_status",
    "temporal_status",
    "source_reliability_level",
    "can_be_reviewed",
    "can_be_used_for_training",
    "can_create_operational_label",
    "formal_negative_available",
    "blocked_reason",
    "decision_rationale",
    "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

# Candidate levels
C1 = "C1_CONTEXTUAL"
C2 = "C2_REVIEW_ONLY_CANDIDATE"
C3_BLOCKED = "C3_BLOCKED_TEMPORAL"
C3_NOT_REACHED = "C3_PLUS_NOT_REACHED"
C4_CLOSED = "C4_CLOSED_NO_FORMAL_NEGATIVE"


def _get_formal_negative_count(v1ot_summary: list[dict[str, str]]) -> int:
    for row in v1ot_summary:
        if row.get("metric") in ("formal_negative_count", "c4_formal_negatives"):
            try:
                return int(row.get("value", "0"))
            except ValueError:
                pass
    return 0


def _decide_c_level(
    event_row: dict[str, str],
    scoring_row: dict[str, str] | None,
    linkage_row: dict[str, str] | None,
    scene_date_confirmed: bool,
    formal_negative_count: int,
) -> dict[str, Any]:
    """Determine C-level for an event."""
    obs_status = event_row.get("observed_event_status", "BLOCKED_INSUFFICIENT_EVIDENCE")
    tier = scoring_row.get("evidence_tier", "BLOCKED") if scoring_row else "BLOCKED"
    temporal_status = linkage_row.get("temporal_linkage_status", "UNKNOWN") if linkage_row else "UNKNOWN"
    spatial_status = linkage_row.get("spatial_linkage_status", "UNKNOWN") if linkage_row else "UNKNOWN"
    source_rel = event_row.get("source_reliability_level", "UNKNOWN")
    patch_id = linkage_row.get("patch_id", "") if linkage_row else ""
    blocked_reason_link = linkage_row.get("blocked_reason", "") if linkage_row else ""

    # C4: only with formal negative
    if formal_negative_count > 0:
        # Would evaluate C4 here — but we still can't without scene_date
        pass

    # C3+: requires scene_date confirmed AND temporal criteria
    if scene_date_confirmed and not temporal_status.startswith("BLOCKED"):
        candidate_level = C2  # Could escalate with more evidence
        decision_status = "C2_TEMPORAL_NOT_VERIFIED"
        blocked_reason = "TEMPORAL_VERIFICATION_PENDING"
        rationale = (
            "Scene date confirmed but temporal delta not verified against event date. "
            "C3+ not reached without explicit temporal window match."
        )
    elif temporal_status.startswith("BLOCKED") or not scene_date_confirmed:
        if tier in ("CONTEXTUAL_GAP", "BLOCKED"):
            candidate_level = C1
            decision_status = "C1_CONTEXTUAL_ONLY"
            blocked_reason = f"SCENE_DATE_FAIL_CLOSED;TIER_{tier}"
            rationale = (
                "No confirmed Sentinel scene date (v1og-v1ot: TEMPORAL_RECOVERY_FAIL_CLOSED). "
                "Evidence tier below threshold for C2. Event is contextual-only reference."
            )
        elif obs_status in ("BLOCKED_INSUFFICIENT_EVIDENCE",):
            candidate_level = C1
            decision_status = "C1_BLOCKED_EVIDENCE"
            blocked_reason = f"BLOCKED_EVIDENCE;SCENE_DATE_FAIL_CLOSED"
            rationale = (
                "Source not acquired (decree/COMPDEC not available). "
                "No Sentinel scene date confirmation. C1 contextual only."
            )
        else:
            candidate_level = C2
            decision_status = "C2_REVIEW_ONLY_TEMPORAL_BLOCKED"
            blocked_reason = "BLOCKED_SENTINEL_SCENE_DATE_MISSING"
            rationale = (
                "Event has some contextual evidence (dossier, gap registry, decree reference) "
                "but Sentinel scene date recovery failed (TEMPORAL_RECOVERY_FAIL_CLOSED). "
                "Cannot escalate to C3+ without confirmed temporal chain."
            )
    else:
        candidate_level = C1
        decision_status = "C1_CONTEXTUAL_DEFAULT"
        blocked_reason = "INSUFFICIENT_EVIDENCE_FOR_C2"
        rationale = "Insufficient evidence to reach C2 or above."

    # C3_PLUS_NOT_REACHED annotation
    c3_not_reached_note = (
        "C3+ not reached: requires confirmed Sentinel scene_date (product_dates_confirmed_real=0 "
        "from v1ot) AND temporal window match. Neither condition is met."
    )

    # C4 annotation
    c4_note = (
        f"C4 closed: formal_negative_count={formal_negative_count}. "
        "C4 requires explicit formal negative statement from official source."
    )

    return {
        "candidate_level": candidate_level,
        "decision_status": decision_status,
        "evidence_tier": tier,
        "spatial_status": spatial_status,
        "temporal_status": temporal_status,
        "source_reliability_level": source_rel,
        "patch_id": patch_id,
        "formal_negative_available": "true" if formal_negative_count > 0 else "false",
        "blocked_reason": blocked_reason,
        "decision_rationale": rationale,
        "notes": f"{c3_not_reached_note} | {c4_note}",
    }


def run() -> None:
    event_rows = read_csv(IN_V1OV) if IN_V1OV.exists() else []
    scoring_rows = read_csv(IN_V1OW) if IN_V1OW.exists() else []
    linkage_rows = read_csv(IN_V1OX) if IN_V1OX.exists() else []
    v1ot_summary = read_csv(IN_V1OT_SUMMARY) if IN_V1OT_SUMMARY.exists() else []

    formal_neg_count = _get_formal_negative_count(v1ot_summary)
    scene_date_confirmed = False  # v1ot: product_dates_confirmed_real=0

    # Check v1ot for actual product date count
    for row in v1ot_summary:
        if row.get("metric") == "product_dates_confirmed_real":
            try:
                scene_date_confirmed = int(row.get("value", "0")) > 0
            except ValueError:
                pass

    # Build indexes
    scoring_index: dict[str, dict[str, str]] = {r.get("event_id", ""): r for r in scoring_rows}
    linkage_index: dict[str, dict[str, str]] = {}
    for lr in linkage_rows:
        eid = lr.get("event_id", "")
        if eid not in linkage_index:
            linkage_index[eid] = lr

    rows: list[dict[str, Any]] = []
    c_level_counts: dict[str, int] = {C1: 0, C2: 0, C3_BLOCKED: 0, C3_NOT_REACHED: 0, C4_CLOSED: 0}

    for i, event_row in enumerate(event_rows):
        eid = event_row.get("event_id", f"UNKNOWN_{i}")
        scoring_row = scoring_index.get(eid)
        linkage_row = linkage_index.get(eid)

        decision = _decide_c_level(
            event_row, scoring_row, linkage_row,
            scene_date_confirmed, formal_neg_count,
        )

        level = decision["candidate_level"]
        c_level_counts[level] = c_level_counts.get(level, 0) + 1

        row: dict[str, Any] = {
            "decision_id": f"V1OY_DEC_{i:04d}",
            "event_id": eid,
            "patch_id": decision["patch_id"],
            "candidate_level": level,
            "decision_status": decision["decision_status"],
            "evidence_tier": decision["evidence_tier"],
            "spatial_status": decision["spatial_status"],
            "temporal_status": decision["temporal_status"],
            "source_reliability_level": decision["source_reliability_level"],
            "can_be_reviewed": "true",
            "can_be_used_for_training": "false",
            "can_create_operational_label": "false",
            "formal_negative_available": decision["formal_negative_available"],
            "blocked_reason": decision["blocked_reason"],
            "decision_rationale": decision["decision_rationale"],
            "notes": decision["notes"],
        }
        rows.append(row)

    # Add explicit C3_PLUS_NOT_REACHED and C4_CLOSED summary rows
    rows.append({
        "decision_id": "V1OY_C3_PLUS_NOT_REACHED",
        "event_id": "ALL_RECIFE_EVENTS",
        "patch_id": "",
        "candidate_level": C3_NOT_REACHED,
        "decision_status": "C3_PLUS_NOT_REACHED_TEMPORAL_FAIL_CLOSED",
        "evidence_tier": "BLOCKED",
        "spatial_status": "NOT_APPLICABLE",
        "temporal_status": "TEMPORAL_RECOVERY_FAIL_CLOSED",
        "source_reliability_level": "NOT_APPLICABLE",
        "can_be_reviewed": "false",
        "can_be_used_for_training": "false",
        "can_create_operational_label": "false",
        "formal_negative_available": "false",
        "blocked_reason": "PRODUCT_DATES_CONFIRMED_REAL_EQUALS_ZERO",
        "decision_rationale": (
            "C3+ requires confirmed Sentinel scene_date. "
            "v1og-v1ot: product_dates_confirmed_real=0 across 2,654 patches evaluated. "
            "C3+ cannot be reached without temporal chain."
        ),
        "notes": "Explicit C3_PLUS_NOT_REACHED declaration for Recife block v1ou-v1pa.",
    })

    rows.append({
        "decision_id": "V1OY_C4_CLOSED",
        "event_id": "ALL_RECIFE_EVENTS",
        "patch_id": "",
        "candidate_level": C4_CLOSED,
        "decision_status": "C4_CLOSED_NO_FORMAL_NEGATIVE",
        "evidence_tier": "BLOCKED",
        "spatial_status": "NOT_APPLICABLE",
        "temporal_status": "NOT_APPLICABLE",
        "source_reliability_level": "NOT_APPLICABLE",
        "can_be_reviewed": "false",
        "can_be_used_for_training": "false",
        "can_create_operational_label": "false",
        "formal_negative_available": "false",
        "blocked_reason": f"FORMAL_NEGATIVE_COUNT_EQUALS_{formal_neg_count}",
        "decision_rationale": (
            f"C4 requires formal negative statement (formal_negative_count={formal_neg_count}). "
            "No official declaration of non-occurrence has been confirmed for Recife. "
            "C4 remains closed."
        ),
        "notes": "Explicit C4_CLOSED declaration for Recife block v1ou-v1pa.",
    })

    assert_no_forbidden_true(rows, "v1oy_audit")
    require_no_abs_paths_in_rows(rows, "v1oy_audit")

    write_csv_safe(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_schema_safe(SCHEMA_AUDIT, AUDIT_FIELDS, "v1oy_ground_truth_candidate_decision_audit")

    event_rows_only = [r for r in rows if r["event_id"] != "ALL_RECIFE_EVENTS"]
    c1_count = sum(1 for r in event_rows_only if r["candidate_level"] == C1)
    c2_count = sum(1 for r in event_rows_only if r["candidate_level"] == C2)

    summary_rows = [
        {"stat_key": "total_decisions", "stat_value": str(len(rows))},
        {"stat_key": "c1_contextual", "stat_value": str(c1_count)},
        {"stat_key": "c2_review_only_candidate", "stat_value": str(c2_count)},
        {"stat_key": "c3_plus_not_reached", "stat_value": "true"},
        {"stat_key": "c3_plus_reason", "stat_value": "TEMPORAL_RECOVERY_FAIL_CLOSED"},
        {"stat_key": "c4_closed", "stat_value": "true"},
        {"stat_key": "c4_formal_negative_count", "stat_value": str(formal_neg_count)},
        {"stat_key": "scene_date_confirmed", "stat_value": str(scene_date_confirmed).lower()},
        {"stat_key": "can_be_used_for_training_any", "stat_value": "false"},
        {"stat_key": "can_create_operational_label_any", "stat_value": "false"},
        {"stat_key": "stage", "stat_value": "v1oy"},
    ]
    write_csv_safe(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1oy_summary")

    write_doc(
        DOC,
        "v1oy — Ground Truth Candidate Decision Audit",
        [
            "## Objetivo",
            "Consolidar decisões C1/C2/C3/C4 com base em v1ov-v1ox e v1og-v1ot. "
            "Gerar decisão auditável para cada evento/patch/linkage.",
            "## Resultado",
            f"C1 (contextual): {c1_count}. "
            f"C2 (review-only candidate): {c2_count}. "
            f"C3+ não alcançado. C4 fechado.",
            "## Por que C3+ não é alcançado",
            "C3+ requer scene_date Sentinel confirmada. v1og-v1ot: product_dates_confirmed_real=0 "
            "em 2.654 patches. Sem cadeia temporal confirmada, C3+ é metodologicamente impossível.",
            "## Por que C4 permanece fechado",
            f"C4 requer negativo formal explícito (formal_negative_count={formal_neg_count}). "
            "Nenhuma declaração oficial de não-ocorrência confirmada para Recife.",
            "## Guardrails",
            "can_be_used_for_training=false, can_create_operational_label=false em todos os registros.",
        ],
    )

    print(f"[v1oy] C1={c1_count}, C2={c2_count}, C3+=NOT_REACHED, C4=CLOSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1oy ground truth candidate decision audit")
    parser.parse_args()
    run()
