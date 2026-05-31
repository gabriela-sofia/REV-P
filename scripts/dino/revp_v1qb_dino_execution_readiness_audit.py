"""REV-P v1qb — DINO execution readiness audit.

Crosses v1qa queue with v1pp backend probe to audit readiness for dry-run
and real execution. Never executes embeddings. Never creates labels/targets.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qa_v1qf_execution_bridge_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, read_backend_probe, read_csv,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

IN_QUEUE = _p("REVP_V1QB_IN_QUEUE",
              DATASETS / "dino_execution_queue_from_visual_expansion_v1qa.csv")
IN_BACKEND = _p("REVP_V1QB_IN_BACKEND",
                DATASETS / "dino_backend_model_probe_summary_v1pp.csv")
OUT_AUDIT = _p("REVP_V1QB_OUT_AUDIT",
               DATASETS / "dino_execution_readiness_audit_v1qb.csv")
OUT_SUM = _p("REVP_V1QB_OUT_SUM",
             DATASETS / "dino_execution_readiness_summary_v1qb.csv")
SCH_AUDIT = _p("REVP_V1QB_SCH_AUDIT",
               SCHEMAS / "dino_execution_readiness_audit_v1qb_schema.csv")
SCH_SUM = _p("REVP_V1QB_SCH_SUM",
             SCHEMAS / "dino_execution_readiness_summary_v1qb_schema.csv")
DOC = _p("REVP_V1QB_DOC", DOCS / "revp_v1qb_dino_execution_readiness_audit.md")

AUDIT_FIELDS = [
    "readiness_id", "execution_queue_id", "patch_id", "region", "visual_type",
    "backend_status", "model_available", "dry_run_allowed", "real_execution_allowed",
    "readiness_status", "readiness_reason", "can_create_label", "can_train_model",
    "target_created", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _readiness_status(model_ok: bool, valid: bool, blocked: str) -> tuple[str, str]:
    if not valid:
        return ("BLOCKED_INVALID_QUEUE_ITEM", blocked)
    if model_ok:
        return ("READY_FOR_LOCAL_MODEL_EXECUTION", "model_available_and_queue_valid")
    return ("READY_FOR_DRY_RUN_ONLY", "model_unavailable_dry_run_permitted")


def build_audit() -> tuple[list[dict[str, Any]], bool, str]:
    queue = read_csv(IN_QUEUE)
    backend = read_backend_probe(IN_BACKEND)
    model_ok = backend.get("can_execute_embeddings", "false") == "true"
    backend_status = backend.get("final_status", "DINO_BACKEND_MODEL_UNAVAILABLE_FAIL_CLOSED")
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(queue, 1):
        valid = r.get("ready_for_dry_run", "false") == "true"
        blocked = r.get("blocked_reason", "")
        status, reason = _readiness_status(model_ok, valid, blocked)
        rows.append({
            "readiness_id": f"V1QB_RD_{i:05d}",
            "execution_queue_id": r.get("execution_queue_id", ""),
            "patch_id": r.get("patch_id", ""),
            "region": r.get("region", ""),
            "visual_type": r.get("visual_type", ""),
            "backend_status": backend_status,
            "model_available": str(model_ok).lower(),
            "dry_run_allowed": "true" if valid else "false",
            "real_execution_allowed": str(model_ok and valid).lower(),
            "readiness_status": status,
            "readiness_reason": reason,
            "can_create_label": "false",
            "can_train_model": "false",
            "target_created": "false",
            "blocked_reason": blocked if not valid else "",
            "notes": "",
        })
    return rows, model_ok, backend_status


def run() -> None:
    rows, model_ok, backend_status = build_audit()
    require_no_abs_paths(rows, "v1qb")
    assert_no_forbidden_true(rows, "v1qb")
    dry = sum(1 for r in rows if str(r.get("readiness_status", "")) == "READY_FOR_DRY_RUN_ONLY")
    real = sum(1 for r in rows if str(r.get("readiness_status", "")) == "READY_FOR_LOCAL_MODEL_EXECUTION")
    blocked = sum(1 for r in rows if str(r.get("readiness_status", "")).startswith("BLOCKED"))
    summary = [
        {"stat_key": "rows_audited", "stat_value": str(len(rows))},
        {"stat_key": "ready_dry_run_only", "stat_value": str(dry)},
        {"stat_key": "ready_local_model_execution", "stat_value": str(real)},
        {"stat_key": "blocked", "stat_value": str(blocked)},
        {"stat_key": "backend_status", "stat_value": backend_status},
        {"stat_key": "model_available", "stat_value": str(model_ok).lower()},
        {"stat_key": "labels_created", "stat_value": "0"},
    ]
    write_csv(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_AUDIT, AUDIT_FIELDS, "v1qb_dino_execution_readiness_audit")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qb_dino_execution_readiness_summary")
    write_doc(DOC, "v1qb — DINO Execution Readiness Audit", [
        "## Objetivo",
        "Auditar prontidão da fila v1qa para dry-run e execução real cruzando "
        "com probe de backend v1pp. Não executa embedding.",
        "## Status",
        "READY_FOR_DRY_RUN_ONLY: modelo ausente. "
        "READY_FOR_LOCAL_MODEL_EXECUTION: modelo configurado localmente.",
        f"## Resultado",
        f"Total: {len(rows)}. Dry-run only: {dry}. Real: {real}. "
        f"Backend: {backend_status}.",
    ])
    print(f"[v1qb] rows={len(rows)} dry={dry} real={real} backend={backend_status}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qb dino execution readiness audit").parse_args()
    run()
