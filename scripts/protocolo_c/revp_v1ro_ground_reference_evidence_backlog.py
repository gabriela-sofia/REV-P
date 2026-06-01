"""REV-P v1ro — Ground reference evidence backlog.

Consolidates P0/P1/P2 outputs into a prioritized, auditable backlog: which
source is missing, which patch/event/region, which blocker, the next action,
and whether it blocks C3/C4. Review-only; never creates labels or ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1qu_v1qz_ground_reference_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    guardrail_row,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_REQUIREMENTS = _p("REVP_V1RO_IN_REQUIREMENTS", DATASETS / "protocol_c_official_evidence_source_requirements_v1qu.csv")
IN_PRIORITIES = _p("REVP_V1RO_IN_PRIORITIES", DATASETS / "protocol_c_ground_reference_external_collection_priorities_v1qz.csv")
IN_PARTIAL_SUMMARY = _p("REVP_V1RO_IN_PARTIAL_SUMMARY", DATASETS / "protocol_c_ground_reference_partial_scientific_summary_v1qz.csv")
IN_GATE_SUMMARY = _p("REVP_V1RO_IN_GATE_SUMMARY", DATASETS / "protocol_c_review_supervisor_gate_scientific_summary_v1rm.csv")
IN_INTAKE_SUMMARY = _p("REVP_V1RO_IN_INTAKE_SUMMARY", DATASETS / "protocol_c_external_intake_scientific_summary_v1rf.csv")

OUT_BACKLOG = _p("REVP_V1RO_OUT_BACKLOG", DATASETS / "protocol_c_ground_reference_evidence_backlog_v1ro.csv")
OUT_SUMMARY = _p("REVP_V1RO_OUT_SUMMARY", DATASETS / "protocol_c_ground_reference_evidence_backlog_summary_v1ro.csv")
SCHEMA_BACKLOG = _p("REVP_V1RO_SCHEMA_BACKLOG", SCHEMAS / "protocol_c_ground_reference_evidence_backlog_v1ro_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RO_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_ground_reference_evidence_backlog_summary_v1ro_schema.csv")
DOC = _p("REVP_V1RO_DOC", DOCS / "revp_v1ro_ground_reference_evidence_backlog.md")

BACKLOG_FIELDS = [
    "backlog_id", "region", "event_id", "patch_id", "hazard_type",
    "missing_source_family", "missing_source_name", "blocker", "next_action",
    "blocks_c3", "blocks_c4", "priority", "current_state", "status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]


def _stat(rows: list[dict[str, str]], key: str, default: str = "") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def build_rows(req: list[dict[str, str]], priorities: list[dict[str, str]],
               partial: list[dict[str, str]], gate: list[dict[str, str]],
               intake: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    i = 0

    # 1) Missing external sources (from v1qu requirements)
    for r in req:
        if r.get("collection_status") != "SOURCE_REQUIRED_NOT_LOCAL":
            continue
        row = {
            "backlog_id": f"V1RO_BL_{i:04d}", "region": r.get("region", ""),
            "event_id": "", "patch_id": "", "hazard_type": r.get("hazard_type", ""),
            "missing_source_family": r.get("preferred_source_family", ""),
            "missing_source_name": r.get("preferred_source_name", ""),
            "blocker": "EXTERNAL_SOURCE_NOT_LOCAL",
            "next_action": "MANUAL_EXTERNAL_COLLECTION (ver v1ra task board)",
            "blocks_c3": r.get("blocks_c3", "false"), "blocks_c4": r.get("blocks_c4", "false"),
            "priority": r.get("source_priority", "P2"),
            "current_state": "BLOCKED_INSUFFICIENT_EVIDENCE",
            "status": "OPEN", "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
        i += 1

    # 2) Workflow blockers (review not completed / supervisor pending)
    review_completed = _stat(partial, "completed_reviews", "0")
    gate_status = _stat(gate, "final_status", "")
    if review_completed == "0":
        row = {
            "backlog_id": f"V1RO_BL_{i:04d}", "region": "ALL", "event_id": "",
            "patch_id": "", "hazard_type": "ALL",
            "missing_source_family": "", "missing_source_name": "",
            "blocker": "DOUBLE_REVIEW_NOT_COMPLETED",
            "next_action": "PREENCHER respostas A/B (v1rg) e validar (v1rh)",
            "blocks_c3": "true", "blocks_c4": "false", "priority": "P0",
            "current_state": "C2_REVIEW_ONLY_CANDIDATE", "status": "OPEN", "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
        i += 1
    if gate_status in ("REVIEW_SUPERVISOR_GATE_SUPERVISOR_PACKETS_READY",):
        row = {
            "backlog_id": f"V1RO_BL_{i:04d}", "region": "ALL", "event_id": "",
            "patch_id": "", "hazard_type": "ALL",
            "missing_source_family": "", "missing_source_name": "",
            "blocker": "SUPERVISOR_DECISION_PENDING",
            "next_action": "PREENCHER decisao supervisor (v1rk) e validar (v1rl)",
            "blocks_c3": "true", "blocks_c4": "false", "priority": "P1",
            "current_state": "C3_REFERENCE_CANDIDATE_NEEDS_SUPERVISOR", "status": "OPEN", "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
        i += 1

    # 3) Manual intake pending (P1)
    intake_status = _stat(intake, "final_status", "")
    if intake_status in ("EXTERNAL_INTAKE_TASK_BOARD_READY", "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS"):
        row = {
            "backlog_id": f"V1RO_BL_{i:04d}", "region": "ALL", "event_id": "",
            "patch_id": "", "hazard_type": "ALL",
            "missing_source_family": "", "missing_source_name": "",
            "blocker": "EXTERNAL_DOCUMENTS_NOT_INTAKEN",
            "next_action": "COLETAR documentos (v1rb template) e validar (v1rc)",
            "blocks_c3": "true", "blocks_c4": "false", "priority": "P1",
            "current_state": "BLOCKED_INSUFFICIENT_EVIDENCE", "status": "OPEN", "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
        i += 1

    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    req = read_csv_safe(IN_REQUIREMENTS)
    priorities = read_csv_safe(IN_PRIORITIES)
    partial = read_csv_safe(IN_PARTIAL_SUMMARY)
    gate = read_csv_safe(IN_GATE_SUMMARY)
    intake = read_csv_safe(IN_INTAKE_SUMMARY)

    rows = build_rows(req, priorities, partial, gate, intake)
    assert_clean_rows(rows, "v1ro_backlog")
    write_csv_with_header(OUT_BACKLOG, rows, BACKLOG_FIELDS)
    write_schema_safe(SCHEMA_BACKLOG, BACKLOG_FIELDS, "v1ro_backlog")

    blocks_c3 = sum(1 for r in rows if r["blocks_c3"] == "true")
    p0 = sum(1 for r in rows if r["priority"] == "P0")
    by_blocker: dict[str, int] = {}
    for r in rows:
        by_blocker[r["blocker"]] = by_blocker.get(r["blocker"], 0) + 1

    summary = [
        {"stat_key": "backlog_items", "stat_value": str(len(rows))},
        {"stat_key": "items_blocking_c3", "stat_value": str(blocks_c3)},
        {"stat_key": "p0_items", "stat_value": str(p0)},
    ]
    for blocker, n in sorted(by_blocker.items()):
        summary.append({"stat_key": f"blocker_{blocker.lower()}", "stat_value": str(n)})
    summary.append({"stat_key": "stage", "stat_value": "v1ro"})
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ro_summary")

    write_doc(
        DOC,
        "v1ro — Ground Reference Evidence Backlog",
        [
            "## Objetivo",
            "Consolidar P0/P1/P2 num backlog priorizado de evidencia: fonte faltante, "
            "patch/evento/regiao, blocker, proxima acao e se bloqueia C3/C4.",
            "## Resultado",
            f"Itens de backlog: {len(rows)}. Bloqueiam C3: {blocks_c3}. P0: {p0}.",
            "## Guardrails",
            "Backlog e review-only. Nenhum item cria label, target ou ground truth "
            "operacional. Ausencia de fonte nunca vira negativo formal.",
        ],
    )
    print(f"[v1ro] backlog={len(rows)} blocks_c3={blocks_c3} p0={p0}")
    return {"backlog": len(rows), "blocks_c3": blocks_c3, "p0": p0}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ro ground reference backlog").parse_args()
    run()
