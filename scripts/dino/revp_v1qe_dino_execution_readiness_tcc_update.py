"""REV-P v1qe — DINO execution readiness TCC update.

Generates TCC-ready tables for execution readiness and safety guardrails.
Never creates labels, targets, or ground truth.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qa_v1qf_execution_bridge_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, read_csv, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

IN_READINESS = _p("REVP_V1QE_IN_READINESS",
                  DATASETS / "dino_execution_readiness_audit_v1qb.csv")
IN_COMMANDS = _p("REVP_V1QE_IN_COMMANDS",
                 DATASETS / "dino_dry_run_execution_commands_v1qc.csv")
OUT_T_READ = _p("REVP_V1QE_OUT_T_READ",
                DATASETS / "dino_tcc_table_execution_readiness_v1qe.csv")
OUT_T_SAFE = _p("REVP_V1QE_OUT_T_SAFE",
                DATASETS / "dino_tcc_table_execution_safety_v1qe.csv")
SCH_READ = _p("REVP_V1QE_SCH_READ",
              SCHEMAS / "dino_tcc_table_execution_readiness_v1qe_schema.csv")
SCH_SAFE = _p("REVP_V1QE_SCH_SAFE",
              SCHEMAS / "dino_tcc_table_execution_safety_v1qe_schema.csv")
DOC = _p("REVP_V1QE_DOC", DOCS / "revp_v1qe_dino_execution_readiness_tcc_update.md")

T_READ_FIELDS = [
    "readiness_id", "patch_id", "region", "backend_status",
    "dry_run_allowed", "real_execution_allowed", "readiness_status",
]
T_SAFE_FIELDS = [
    "command_id", "command_type", "safety_note", "requires_manual_confirmation",
]

TCC_TEXT = (
    "A fila DINO expandida foi convertida em um pacote de execução reprodutível, "
    "mas permanece em modo dry-run enquanto não houver modelo local configurado. "
    "Essa prontidão operacional não altera o status científico dos patches: os "
    "vetores, quando gerados, serão usados apenas como representação "
    "auto-supervisionada review-only."
)


def _project(rows: list[dict[str, str]], fields: list[str]) -> list[dict[str, Any]]:
    return [{f: r.get(f, "") for f in fields} for r in rows]


def run() -> None:
    readiness = read_csv(IN_READINESS)
    commands = read_csv(IN_COMMANDS)
    t_read = _project(readiness, T_READ_FIELDS)
    t_safe = _project(commands, T_SAFE_FIELDS)
    for label, rows in (("v1qe_read", t_read), ("v1qe_safe", t_safe)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_T_READ, t_read, T_READ_FIELDS)
    write_csv(OUT_T_SAFE, t_safe, T_SAFE_FIELDS)
    write_schema(SCH_READ, T_READ_FIELDS, "v1qe_tcc_execution_readiness")
    write_schema(SCH_SAFE, T_SAFE_FIELDS, "v1qe_tcc_execution_safety")
    write_doc(DOC, "v1qe — DINO Execution Readiness TCC Update", [
        "## Objetivo",
        "Tabelas TCC-ready para prontidão de execução e guardrails de segurança.",
        "## Texto para o TCC",
        TCC_TEXT,
        "## Guardrails",
        "DINO é representação visual auto-supervisionada review-only. "
        "Prontidão operacional ≠ validação científica de evento.",
        f"## Resultado",
        f"Linhas readiness: {len(t_read)}. Linhas safety: {len(t_safe)}.",
    ])
    print(f"[v1qe] readiness={len(t_read)} safety={len(t_safe)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qe dino execution readiness tcc update").parse_args()
    run()
