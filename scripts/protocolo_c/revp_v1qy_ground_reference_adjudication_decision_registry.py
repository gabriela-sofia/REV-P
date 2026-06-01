"""REV-P v1qy — Ground reference adjudication decision registry.

Consolidates v1qx observational scores into adjudicable decisions without
creating any label. C3 candidates always require supervisor review. C4 is
never opened without an explicit formal negative source (expected: none).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1qu_v1qz_ground_reference_common import (
    _p,
    assert_clean_rows,
    guardrail_row,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_SCORES = _p("REVP_V1QY_IN_SCORES", DATASETS / "protocol_c_observational_evidence_scores_v1qx.csv")
OUT_REGISTRY = _p("REVP_V1QY_OUT_REGISTRY", DATASETS / "protocol_c_ground_reference_adjudication_registry_v1qy.csv")
OUT_SUMMARY = _p("REVP_V1QY_OUT_SUMMARY", DATASETS / "protocol_c_ground_reference_adjudication_summary_v1qy.csv")
SCHEMA_REGISTRY = _p("REVP_V1QY_SCHEMA_REGISTRY", SCHEMAS / "protocol_c_ground_reference_adjudication_registry_v1qy_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1QY_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_ground_reference_adjudication_summary_v1qy_schema.csv")
DOC = _p("REVP_V1QY_DOC", DOCS / "revp_v1qy_ground_reference_adjudication_decision_registry.md")

# Adjudication decisions
KEEP_C1 = "KEEP_C1_CONTEXTUAL"
KEEP_C2 = "KEEP_C2_REVIEW_ONLY"
PROMOTE_C3 = "PROMOTE_TO_C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR"
BLOCK_C3_TEMPORAL = "BLOCK_C3_INSUFFICIENT_TEMPORAL_PRECISION"
BLOCK_C3_SPATIAL = "BLOCK_C3_INSUFFICIENT_SPATIAL_PRECISION"
BLOCK_C3_SOURCE = "BLOCK_C3_SOURCE_WEAK"
BLOCK_C4_NO_FORMAL_NEGATIVE = "BLOCK_C4_NO_FORMAL_NEGATIVE_SOURCE"

REGISTRY_FIELDS = [
    "adjudication_id", "review_sample_id", "event_id", "patch_id", "region",
    "input_decision", "composite_observational_score",
    "temporal_precision_score", "spatial_precision_score", "source_reliability_score",
    "adjudication_decision", "supervisor_review_required",
    "formal_negative", "blocked_reason", "adjudication_rationale",
    "review_only", "dino_validates_event", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "absence_as_negative", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _f(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def adjudicate(score_row: dict[str, str]) -> dict[str, Any]:
    decision_in = score_row.get("recommended_protocol_c_decision", "")
    temp = _f(score_row.get("temporal_precision_score", "0"))
    spat = _f(score_row.get("spatial_precision_score", "0"))
    rel = _f(score_row.get("source_reliability_score", "0"))
    comp = _f(score_row.get("composite_observational_score", "0"))

    supervisor_required = "false"
    formal_negative = "false"
    blocked_reason = ""
    rationale = ""

    if "C3" in decision_in:
        if rel < 0.75:
            adj = BLOCK_C3_SOURCE
            blocked_reason = "SOURCE_RELIABILITY_BELOW_OFFICIAL_THRESHOLD"
        elif temp < 0.6:
            adj = BLOCK_C3_TEMPORAL
            blocked_reason = "TEMPORAL_PRECISION_BELOW_THRESHOLD"
        elif spat < 0.5:
            adj = BLOCK_C3_SPATIAL
            blocked_reason = "SPATIAL_PRECISION_BELOW_THRESHOLD"
        else:
            adj = PROMOTE_C3
            supervisor_required = "true"
            rationale = "official_source_sufficient_precision_needs_supervisor"
    elif "C2" in decision_in or "REVIEW_ONLY" in decision_in:
        adj = KEEP_C2
        rationale = "review_only_candidate_retained"
    elif "C1" in decision_in or "CONTEXTUAL" in decision_in:
        adj = KEEP_C1
        rationale = "contextual_only_retained"
    elif "BLOCKED" in decision_in:
        adj = BLOCK_C3_SOURCE
        blocked_reason = score_row.get("blocked_reason", "") or "INSUFFICIENT_EVIDENCE"
        rationale = "insufficient_evidence_blocked"
    else:
        adj = KEEP_C2
        rationale = "default_review_only"

    row = {
        "adjudication_id": f"V1QY_ADJ_{score_row.get('review_sample_id','')}",
        "review_sample_id": score_row.get("review_sample_id", ""),
        "event_id": score_row.get("event_id", ""),
        "patch_id": score_row.get("patch_id", ""),
        "region": score_row.get("region", ""),
        "input_decision": decision_in,
        "composite_observational_score": f"{comp:.3f}",
        "temporal_precision_score": f"{temp:.3f}",
        "spatial_precision_score": f"{spat:.3f}",
        "source_reliability_score": f"{rel:.3f}",
        "adjudication_decision": adj,
        "supervisor_review_required": supervisor_required,
        "formal_negative": formal_negative,
        "blocked_reason": blocked_reason,
        "adjudication_rationale": rationale,
        "notes": "",
    }
    row.update(guardrail_row())
    # supervisor flag is an extra column, not a guardrail; keep formal_negative false
    row["formal_negative"] = formal_negative
    return row


def run(datasets: Path | None = None) -> dict[str, Any]:
    scores = read_csv_safe(IN_SCORES)
    rows = [adjudicate(s) for s in scores]
    assert_clean_rows(rows, "v1qy_adjudication")

    write_csv_with_header(OUT_REGISTRY, rows, REGISTRY_FIELDS)
    write_schema_safe(SCHEMA_REGISTRY, REGISTRY_FIELDS, "v1qy_adjudication")

    c3_supervisor = sum(1 for r in rows if r["adjudication_decision"] == PROMOTE_C3)
    kept_c1 = sum(1 for r in rows if r["adjudication_decision"] == KEEP_C1)
    kept_c2 = sum(1 for r in rows if r["adjudication_decision"] == KEEP_C2)
    blocked_c3 = sum(1 for r in rows if r["adjudication_decision"].startswith("BLOCK_C3"))
    c4_open = sum(1 for r in rows if r["formal_negative"] == "true")

    summary = [
        {"stat_key": "adjudicated_rows", "stat_value": str(len(rows))},
        {"stat_key": "kept_c1_contextual", "stat_value": str(kept_c1)},
        {"stat_key": "kept_c2_review_only", "stat_value": str(kept_c2)},
        {"stat_key": "promote_c3_needs_supervisor", "stat_value": str(c3_supervisor)},
        {"stat_key": "blocked_c3", "stat_value": str(blocked_c3)},
        {"stat_key": "c4_formal_negatives_opened", "stat_value": str(c4_open)},
        {"stat_key": "formal_negative_count", "stat_value": str(c4_open)},
        {"stat_key": "stage", "stat_value": "v1qy"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1qy_summary")

    write_doc(
        DOC,
        "v1qy — Ground Reference Adjudication Decision Registry",
        [
            "## Objetivo",
            "Consolidar scores observacionais em decisoes adjudicaveis sem criar label. "
            "Todo candidato C3 exige supervisor_review_required=true. C4 nunca e aberto sem "
            "fonte formal negativa explicita (esperado: formal_negative=false).",
            "## Decisoes",
            "KEEP_C1_CONTEXTUAL, KEEP_C2_REVIEW_ONLY, "
            "PROMOTE_TO_C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR, "
            "BLOCK_C3_INSUFFICIENT_TEMPORAL_PRECISION, BLOCK_C3_INSUFFICIENT_SPATIAL_PRECISION, "
            "BLOCK_C3_SOURCE_WEAK, BLOCK_C4_NO_FORMAL_NEGATIVE_SOURCE.",
            "## Resultado",
            f"Adjudicados: {len(rows)}. C3 needing supervisor: {c3_supervisor}. "
            f"Blocked C3: {blocked_c3}. C4 formal negatives: {c4_open}.",
            "## Guardrails",
            "can_create_operational_label=false em todas as linhas. C4 fechado sem fonte "
            "formal. Nenhum target, ground truth ou label criado.",
        ],
    )
    print(f"[v1qy] adjudicated={len(rows)} c3_supervisor={c3_supervisor} c4={c4_open}")
    return {"adjudicated": len(rows), "c3_supervisor": c3_supervisor, "c4": c4_open}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qy adjudication registry").parse_args()
    run()
