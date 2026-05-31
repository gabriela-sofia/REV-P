"""REV-P v1qf — DINO execution bridge bundle.

Consolidates v1qa-v1qe into manifest, scientific summary and final doc.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qa_v1qf_execution_bridge_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, read_csv, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)
from revp_v1pg_v1pm_dino_representation_common import read_csv_header

OUT_MANIFEST = _p("REVP_V1QF_OUT_MANIFEST",
                  DATASETS / "dino_execution_bridge_manifest_v1qf.csv")
OUT_SUM = _p("REVP_V1QF_OUT_SUM",
             DATASETS / "dino_execution_bridge_scientific_summary_v1qf.csv")
SCH_MAN = _p("REVP_V1QF_SCH_MAN",
             SCHEMAS / "dino_execution_bridge_manifest_v1qf_schema.csv")
SCH_SUM = _p("REVP_V1QF_SCH_SUM",
             SCHEMAS / "dino_execution_bridge_scientific_summary_v1qf_schema.csv")
DOC = _p("REVP_V1QF_DOC", DOCS / "revp_v1qf_dino_execution_bridge_bundle.md")

MANIFEST_FIELDS = ["artifact_id", "stage", "filename", "rows", "header_present", "role"]
SUM_FIELDS = ["summary_id", "metric", "value", "interpretation", "methodological_status", "writing_use"]

ARTIFACTS = [
    ("v1qa", "dino_execution_queue_from_visual_expansion_v1qa.csv", "execution_queue_bridge"),
    ("v1qa", "dino_execution_queue_from_visual_expansion_summary_v1qa.csv", "execution_queue_summary"),
    ("v1qb", "dino_execution_readiness_audit_v1qb.csv", "readiness_audit"),
    ("v1qb", "dino_execution_readiness_summary_v1qb.csv", "readiness_summary"),
    ("v1qc", "dino_dry_run_execution_plan_v1qc.csv", "dry_run_plan"),
    ("v1qc", "dino_dry_run_execution_commands_v1qc.csv", "execution_commands"),
    ("v1qd", "dino_executor_compatibility_report_v1qd.csv", "compatibility_report"),
    ("v1qd", "dino_executor_compatibility_summary_v1qd.csv", "compatibility_summary"),
    ("v1qe", "dino_tcc_table_execution_readiness_v1qe.csv", "tcc_readiness_table"),
    ("v1qe", "dino_tcc_table_execution_safety_v1qe.csv", "tcc_safety_table"),
]


def _stat(fname: str, key: str) -> str:
    for r in read_csv(DATASETS / fname):
        if r.get("stat_key") == key:
            return r.get("stat_value", "0")
    return "0"


def _count(fname: str) -> str:
    p = DATASETS / fname
    if not p.exists():
        return "MISSING"
    return str(len(read_csv(p)))


def build_manifest() -> list[dict[str, Any]]:
    return [{
        "artifact_id": f"V1QF_ART_{i:03d}",
        "stage": stage, "filename": fname,
        "rows": _count(fname),
        "header_present": str(bool(read_csv_header(DATASETS / fname))).lower(),
        "role": role,
    } for i, (stage, fname, role) in enumerate(ARTIFACTS, 1)]


def build_summary() -> tuple[list[dict[str, Any]], str]:
    imported = _stat("dino_execution_queue_from_visual_expansion_summary_v1qa.csv", "queue_rows_imported")
    dry_ready = _stat("dino_execution_queue_from_visual_expansion_summary_v1qa.csv", "ready_for_dry_run")
    real_ready = _stat("dino_execution_queue_from_visual_expansion_summary_v1qa.csv", "ready_for_real_execution")
    blocked_q = _stat("dino_execution_queue_from_visual_expansion_summary_v1qa.csv", "blocked_rows")
    backend = _stat("dino_execution_readiness_summary_v1qb.csv", "backend_status")
    model_avail = _stat("dino_execution_readiness_summary_v1qb.csv", "model_available")
    cmds = _count("dino_dry_run_execution_commands_v1qc.csv")
    compat = _stat("dino_executor_compatibility_summary_v1qd.csv", "compatibility_status")

    dr = int(dry_ready or "0")
    rr = int(real_ready or "0")
    if rr > 0:
        final = "DINO_EXECUTION_BRIDGE_READY_LOCAL_MODEL"
    elif dr > 0:
        final = "DINO_EXECUTION_BRIDGE_READY_DRY_RUN_MODEL_MISSING"
    else:
        final = "DINO_EXECUTION_BRIDGE_EMPTY_FAIL_CLOSED"

    def s(i: int, m: str, v: str, interp: str, ms: str = "RESULTADO_FINAL",
          use: str = "resultado_negativo_auditavel") -> dict[str, Any]:
        return {"summary_id": f"V1QF_S{i:03d}", "metric": m, "value": v,
                "interpretation": interp, "methodological_status": ms, "writing_use": use}

    rows = [
        s(1, "expanded_queue_rows_imported", imported, "Itens da fila expandida importados para bridge", "AUDITAVEL", "metodologia_auditoria"),
        s(2, "dry_run_ready_rows", dry_ready, "Itens prontos para dry-run (modelo pode ser ausente)"),
        s(3, "real_execution_ready_rows", real_ready, "Itens prontos para execução real (requer modelo local)"),
        s(4, "blocked_rows", blocked_q, "Itens bloqueados por guardrails ou fila inválida"),
        s(5, "backend_status", backend, "Status do backend de execução DINO"),
        s(6, "model_available", model_avail, "Modelo local configurado e disponível"),
        s(7, "commands_generated", cmds, "Comandos PowerShell gerados para execução reprodutível", "AUDITAVEL", "metodologia_auditoria"),
        s(8, "compatibility_status", compat, "Status de compatibilidade v1pq ↔ v1qa", "AUDITAVEL", "metodologia_auditoria"),
        s(9, "labels_created", "0", "Labels operacionais criadas — 0 por design"),
        s(10, "targets_created", "0", "Targets de treinamento criados — 0 por design"),
        s(11, "final_status", final, "Status final da ponte de execução DINO", "RESULTADO_FINAL", "conclusao_auditavel"),
    ]
    return rows, final


def run() -> None:
    manifest = build_manifest()
    summary, final = build_summary()
    for label, rows in (("v1qf_manifest", manifest), ("v1qf_summary", summary)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_csv(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCH_MAN, MANIFEST_FIELDS, "v1qf_execution_bridge_manifest")
    write_schema(SCH_SUM, SUM_FIELDS, "v1qf_execution_bridge_scientific_summary")
    write_doc(DOC, "v1qf — DINO Execution Bridge Bundle", [
        "## Objetivo",
        "Consolidar v1qa-v1qe em manifest, summary e doc final.",
        "## Princípio",
        "A fila expandida está pronta para dry-run. Execução real exige modelo local. "
        "DINO é representação visual review-only — não valida evento, não cria rótulo.",
        f"## Status final",
        f"**{final}**.",
    ])
    print(f"[v1qf] final={final}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qf dino execution bridge bundle").parse_args()
    run()
