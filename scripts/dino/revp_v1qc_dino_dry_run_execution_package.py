"""REV-P v1qc — DINO dry-run execution package.

Generates a reproducible execution package: item list, PowerShell commands,
env vars, and safety limits. Never runs model. Never downloads. Never infers.
"""
from __future__ import annotations

import argparse
import os
from typing import Any

from revp_v1qa_v1qf_execution_bridge_common import (
    DATASETS, DOCS, SCHEMAS,
    _p, assert_no_forbidden_true, make_powershell_command, read_csv,
    require_no_abs_paths, write_csv, write_doc, write_schema,
)

IN_QUEUE = _p("REVP_V1QC_IN_QUEUE",
              DATASETS / "dino_execution_queue_from_visual_expansion_v1qa.csv")
OUT_PLAN = _p("REVP_V1QC_OUT_PLAN",
              DATASETS / "dino_dry_run_execution_plan_v1qc.csv")
OUT_CMDS = _p("REVP_V1QC_OUT_CMDS",
              DATASETS / "dino_dry_run_execution_commands_v1qc.csv")
SCH_PLAN = _p("REVP_V1QC_SCH_PLAN",
              SCHEMAS / "dino_dry_run_execution_plan_v1qc_schema.csv")
SCH_CMDS = _p("REVP_V1QC_SCH_CMDS",
              SCHEMAS / "dino_dry_run_execution_commands_v1qc_schema.csv")
DOC = _p("REVP_V1QC_DOC", DOCS / "revp_v1qc_dino_dry_run_execution_package.md")

MAX_EXECUTE = int(os.environ.get("REVP_DINO_MAX_EXECUTE", "5"))

PLAN_FIELDS = [
    "plan_id", "execution_queue_id", "patch_id", "region", "relative_path",
    "dry_run_command_id", "execution_order", "max_execute", "model_path_required",
    "allow_download_default", "expected_output", "guardrail_summary", "notes",
]
CMD_FIELDS = [
    "command_id", "command_type", "powershell_command",
    "safety_note", "requires_manual_confirmation",
]


def build_plan(queue: list[dict[str, str]]) -> list[dict[str, Any]]:
    ready = [r for r in queue if r.get("ready_for_dry_run") == "true"]
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(ready[:MAX_EXECUTE], 1):
        rows.append({
            "plan_id": f"V1QC_PLAN_{i:05d}",
            "execution_queue_id": r.get("execution_queue_id", ""),
            "patch_id": r.get("patch_id", ""),
            "region": r.get("region", ""),
            "relative_path": r.get("relative_path", ""),
            "dry_run_command_id": f"V1QC_CMD_005",
            "execution_order": str(i),
            "max_execute": str(MAX_EXECUTE),
            "model_path_required": "true",
            "allow_download_default": "false",
            "expected_output": "EMBEDDING_SKIPPED_DRY_RUN_or_EMBEDDING_EXECUTED_REVIEW_ONLY",
            "guardrail_summary": "can_create_label=false|can_train_model=false|target_created=false",
            "notes": "",
        })
    return rows


def build_commands() -> list[dict[str, Any]]:
    cmd_defs = [
        ("set_model_path", {"model_path": "<path_to_local_dino_model>"}, True),
        ("keep_download_false", {}, False),
        ("enable_dry_run", {}, False),
        ("set_queue_path", {"queue_path": "datasets/dino_execution_queue_from_visual_expansion_v1qa.csv"}, False),
        ("run_v1pq", {}, True),
        ("run_v1pr", {}, False),
        ("run_v1ps", {}, False),
        ("run_v1pt", {}, False),
        ("disable_dry_run_manual", {}, True),
    ]
    rows: list[dict[str, Any]] = []
    for i, (cmd_type, params, manual_confirm) in enumerate(cmd_defs, 1):
        ps_cmd, safety = make_powershell_command(cmd_type, params)
        rows.append({
            "command_id": f"V1QC_CMD_{i:03d}",
            "command_type": cmd_type,
            "powershell_command": ps_cmd,
            "safety_note": safety,
            "requires_manual_confirmation": str(manual_confirm).lower(),
        })
    return rows


def run() -> None:
    queue = read_csv(IN_QUEUE)
    plan = build_plan(queue)
    cmds = build_commands()
    for label, rows in (("v1qc_plan", plan), ("v1qc_cmds", cmds)):
        require_no_abs_paths(rows, label)
        assert_no_forbidden_true(rows, label)
    write_csv(OUT_PLAN, plan, PLAN_FIELDS)
    write_csv(OUT_CMDS, cmds, CMD_FIELDS)
    write_schema(SCH_PLAN, PLAN_FIELDS, "v1qc_dry_run_execution_plan")
    write_schema(SCH_CMDS, CMD_FIELDS, "v1qc_dry_run_execution_commands")
    write_doc(DOC, "v1qc — DINO Dry-Run Execution Package", [
        "## Objetivo",
        "Gerar pacote reprodutível para execução futura de embeddings DINO. "
        "Não roda modelo. Não baixa. Não infere.",
        "## Comandos PowerShell",
        "Ver datasets/dino_dry_run_execution_commands_v1qc.csv. "
        "REVP_DINO_ALLOW_DOWNLOAD=false por padrão. "
        "REVP_DINO_DRY_RUN=false requer confirmação manual.",
        "## Guardrails",
        "can_create_label, can_train_model e target_created sempre false.",
        f"## Resultado",
        f"Itens no plano: {len(plan)}. Comandos gerados: {len(cmds)}.",
    ])
    print(f"[v1qc] plan={len(plan)} commands={len(cmds)}")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qc dino dry-run execution package").parse_args()
    run()
