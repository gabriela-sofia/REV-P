"""REV-P v1oz — DINO review-only representation queue.

Generates a queue for DINO embedding / visual review only.
Includes patches/events with some contextual value but without labels.
Queue can be empty if no sufficient linkage.

dino_can_create_label=false, dino_can_train_model=false,
dino_target_field_created=false — always.
dino_allowed_use=REVIEW_ONLY_REPRESENTATION only.
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

OUT_QUEUE = _p("REVP_V1OZ_OUT_QUEUE", DATASETS / "recife_dino_review_only_representation_queue_v1oz.csv")
OUT_SUMMARY = _p("REVP_V1OZ_OUT_SUMMARY", DATASETS / "recife_dino_review_only_representation_summary_v1oz.csv")
SCHEMA_QUEUE = _p("REVP_V1OZ_SCHEMA_QUEUE", SCHEMAS / "recife_dino_review_only_representation_queue_v1oz_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1OZ_SCHEMA_SUMMARY", SCHEMAS / "recife_dino_review_only_representation_summary_v1oz_schema.csv")
DOC = _p("REVP_V1OZ_DOC", DOCS / "revp_v1oz_dino_review_only_representation_queue.md")
IN_V1OY = _p("REVP_V1OZ_IN_V1OY", DATASETS / "recife_ground_truth_candidate_decision_audit_v1oy.csv")
IN_V1OX = _p("REVP_V1OZ_IN_V1OX", DATASETS / "recife_event_patch_linkage_registry_v1ox.csv")

QUEUE_FIELDS = [
    "queue_id",
    "event_id",
    "patch_id",
    "alias",
    "region",
    "representation_reason",
    "evidence_tier",
    "linkage_status",
    "dino_allowed_use",
    "dino_can_create_label",
    "dino_can_train_model",
    "dino_target_field_created",
    "blocked_reason",
    "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

DINO_ALLOWED_USE = "REVIEW_ONLY_REPRESENTATION"


def _can_enter_dino_queue(audit_row: dict[str, str], linkage_row: dict[str, str] | None) -> tuple[bool, str]:
    """Determine if a candidate can enter DINO queue, and why (blocked_reason)."""
    level = audit_row.get("candidate_level", "")
    temporal = (linkage_row.get("temporal_linkage_status", "") if linkage_row else "")
    tier = audit_row.get("evidence_tier", "BLOCKED")

    # Never add synthetic/fixture
    if "FIXTURE" in tier or "SYNTHETIC" in tier:
        return False, "BLOCKED_FIXTURE_OR_SYNTHETIC"

    # Only C2+ can enter queue
    if level == "C1_CONTEXTUAL":
        return False, "BLOCKED_C1_CONTEXTUAL_INSUFFICIENT_FOR_DINO_QUEUE"

    # If temporal is completely blocked and no spatial patch
    if temporal.startswith("BLOCKED") and not (linkage_row and linkage_row.get("patch_id")):
        return False, "BLOCKED_NO_PATCH_AND_TEMPORAL_BLOCKED"

    # C3_PLUS_NOT_REACHED and C4_CLOSED are summary rows — skip
    if level in ("C3_PLUS_NOT_REACHED", "C4_CLOSED_NO_FORMAL_NEGATIVE"):
        return False, "BLOCKED_SUMMARY_ROW_NOT_EVENT_CANDIDATE"

    # C2 with some spatial context → can enter as contextual review
    if level == "C2_REVIEW_ONLY_CANDIDATE":
        if tier in ("CONTEXTUAL_GAP", "BLOCKED"):
            return False, "BLOCKED_EVIDENCE_TIER_TOO_LOW"
        return True, ""

    return False, "BLOCKED_CANDIDATE_LEVEL_NOT_ELIGIBLE"


def run() -> None:
    audit_rows = read_csv(IN_V1OY) if IN_V1OY.exists() else []
    linkage_rows = read_csv(IN_V1OX) if IN_V1OX.exists() else []

    # Build linkage index by event_id
    linkage_index: dict[str, dict[str, str]] = {}
    for lr in linkage_rows:
        eid = lr.get("event_id", "")
        if eid and eid not in linkage_index:
            linkage_index[eid] = lr

    rows: list[dict[str, Any]] = []
    seq = 0
    blocked_count = 0

    for audit_row in audit_rows:
        eid = audit_row.get("event_id", "")
        patch_id = audit_row.get("patch_id", "")
        linkage_row = linkage_index.get(eid)
        tier = audit_row.get("evidence_tier", "BLOCKED")

        can_queue, blocked_reason = _can_enter_dino_queue(audit_row, linkage_row)

        if not can_queue:
            blocked_count += 1
            continue

        alias = linkage_row.get("alias", patch_id) if linkage_row else patch_id
        region = audit_row.get("region", "RECIFE") if audit_row.get("region") else "RECIFE"
        linkage_status = (
            linkage_row.get("temporal_linkage_status", "UNKNOWN") if linkage_row else "NO_LINKAGE"
        )

        rows.append({
            "queue_id": f"V1OZ_QUEUE_{seq:04d}",
            "event_id": eid,
            "patch_id": patch_id,
            "alias": alias,
            "region": region,
            "representation_reason": (
                "C2 candidate with contextual spatial linkage — review-only DINO representation"
            ),
            "evidence_tier": tier,
            "linkage_status": linkage_status,
            "dino_allowed_use": DINO_ALLOWED_USE,
            "dino_can_create_label": "false",
            "dino_can_train_model": "false",
            "dino_target_field_created": "false",
            "blocked_reason": "",
            "notes": (
                "DINO may be used for structural/visual representation only. "
                "No label, no training target, no ground truth derived from DINO output."
            ),
        })
        seq += 1

    assert_no_forbidden_true(rows, "v1oz_queue")
    require_no_abs_paths_in_rows(rows, "v1oz_queue")

    write_csv_safe(OUT_QUEUE, rows, QUEUE_FIELDS)
    write_schema_safe(SCHEMA_QUEUE, QUEUE_FIELDS, "v1oz_dino_review_only_representation_queue")

    status = "DINO_QUEUE_EMPTY_INSUFFICIENT_LINKAGE" if not rows else "DINO_QUEUE_POPULATED_REVIEW_ONLY"

    summary_rows = [
        {"stat_key": "total_queue_entries", "stat_value": str(len(rows))},
        {"stat_key": "blocked_not_queued", "stat_value": str(blocked_count)},
        {"stat_key": "dino_allowed_use", "stat_value": DINO_ALLOWED_USE},
        {"stat_key": "dino_can_create_label_any", "stat_value": "false"},
        {"stat_key": "dino_can_train_model_any", "stat_value": "false"},
        {"stat_key": "dino_target_field_created_any", "stat_value": "false"},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "training_targets_created", "stat_value": "0"},
        {"stat_key": "queue_status", "stat_value": status},
        {"stat_key": "stage", "stat_value": "v1oz"},
    ]
    write_csv_safe(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1oz_summary")

    write_doc(
        DOC,
        "v1oz — DINO Review-Only Representation Queue",
        [
            "## Objetivo",
            "Gerar fila para representação visual/embedding DINO review-only. "
            "Inclui patches/eventos com valor contextual mas sem label. "
            "Fila pode ser vazia se não houver linkage suficiente.",
            "## Resultado",
            f"Entradas na fila DINO: {len(rows)}. "
            f"Bloqueados/não incluídos: {blocked_count}. "
            f"Status: {status}.",
            "## Papel do DINO",
            "DINO é usado exclusivamente para representação estrutural visual. "
            "Nenhum target é criado. Nenhum label é derivado. "
            "dino_allowed_use=REVIEW_ONLY_REPRESENTATION.",
            "## Guardrails",
            "dino_can_create_label=false, dino_can_train_model=false, "
            "dino_target_field_created=false em todos os registros.",
        ],
    )

    print(f"[v1oz] DINO queue: {len(rows)} entries, {blocked_count} blocked. Status: {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1oz DINO review-only representation queue")
    parser.parse_args()
    run()
