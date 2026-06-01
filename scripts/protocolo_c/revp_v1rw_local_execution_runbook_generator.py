"""REV-P v1rw — Local execution runbook generator.

Generates a practical runbook for the next real phase: configuring local model
and Sentinel roots, dry-running DINO, and checking guardrails before any real
embedding execution. No science, no downloads, no model loading.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DOCS, SCHEMAS, _p,
    write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs"

OUT_RUNBOOK = _p("REVP_V1RW_OUT_RUNBOOK", DOCS / "revp_v1rw_local_execution_runbook.md")
OUT_STEPS = _p("REVP_V1RW_OUT_STEPS", CONFIGS / "revp_local_execution_steps_v1rw.csv")
OUT_ENVVARS = _p("REVP_V1RW_OUT_ENVVARS", CONFIGS / "revp_required_local_env_vars_v1rw.csv")
SCHEMA_STEPS = _p("REVP_V1RW_SCHEMA_STEPS", SCHEMAS / "revp_local_execution_steps_v1rw_schema.csv")
SCHEMA_ENVVARS = _p("REVP_V1RW_SCHEMA_ENVVARS", SCHEMAS / "revp_required_local_env_vars_v1rw_schema.csv")

STEPS_FIELDS = ["step_id", "phase", "description", "command_or_action",
                "expected_result", "on_fail", "notes"]
ENVVARS_FIELDS = ["env_var", "required", "description", "example_value", "notes"]

_ENV_VARS = [
    ("REVP_DINO_MODEL_PATH", "yes", "Path to local DINOv2+registers model dir", "/path/to/dinov2-registers", "Never commit; set in local shell"),
    ("REVP_SENTINEL_LOCAL_ROOT", "yes", "Root dir of local Sentinel TIF files", "/path/to/sentinel_root", "Never commit"),
    ("REVP_DINO_VISUAL_ROOT", "yes", "Root dir of visual assets for DINO queue", "/path/to/visual_root", "Never commit"),
    ("REVP_DINO_ASSET_ROOT", "yes", "Alternate root for DINO assets", "/path/to/asset_root", "Never commit"),
    ("HF_HUB_OFFLINE", "yes", "Disable HuggingFace Hub downloads", "1", "Set to 1 before all DINO runs"),
    ("REVP_DINO_ALLOW_DOWNLOAD", "yes", "Forbid automatic model download", "false", "Always false"),
    ("REVP_DINO_DRY_RUN", "yes", "Run in dry-run mode first", "true", "Set false only after confirming dry-run"),
    ("REVP_PROTOCOL_C_REVIEW_RESPONSES_PATH", "no", "Path to filled A/B review CSV", "/path/to/responses.csv", "Fill v1rg template first"),
    ("REVP_PROTOCOL_C_SUPERVISOR_DECISIONS_PATH", "no", "Path to filled supervisor decision CSV", "/path/to/decisions.csv", "Fill v1rk template first"),
    ("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH", "no", "Path to filled external intake CSV", "/path/to/intake.csv", "Fill v1rb template first"),
]

_STEPS = [
    ("S01", "CONFIG", "Configure all required env vars in local shell profile", "export REVP_DINO_MODEL_PATH=... (see env vars table)", "env vars readable", "Abort; never commit paths", ""),
    ("S02", "CONFIG", "Set HF_HUB_OFFLINE=1 and REVP_DINO_ALLOW_DOWNLOAD=false", "export HF_HUB_OFFLINE=1; export REVP_DINO_ALLOW_DOWNLOAD=false", "No internet calls", "Abort", ""),
    ("S03", "DRYRUN", "Run local readiness audit in dry-run mode", "python scripts/dino/revp_v1qn_local_root_environment_audit.py", "LOCAL_DINO_READINESS status reported", "Check missing roots/model", "REVP_DINO_DRY_RUN=true"),
    ("S04", "DRYRUN", "Run smoke asset reconciliation", "python scripts/dino/revp_v1qo_smoke_asset_local_reconciliation.py", "Assets resolved or repair suggestions generated", "Check repair suggestions", ""),
    ("S05", "DRYRUN", "Run local smoke run readiness gate", "python scripts/dino/revp_v1qr_local_smoke_run_readiness_gate.py", "GATE_PASS or GATE_FAIL with reasons", "Fix blockers before continuing", ""),
    ("S06", "VALIDATE", "Confirm dry-run is still true; review outputs", "echo $REVP_DINO_DRY_RUN", "true", "Do NOT proceed if false unintentionally", ""),
    ("S07", "EXECUTE", "Set dry_run=false and run controlled smoke embedding", "export REVP_DINO_DRY_RUN=false; python scripts/dino/revp_v1qj_controlled_real_smoke_embedding_executor.py", "Embeddings written to gitignored output dir only", "Check error log; never commit embeddings", "Only after full dry-run validation"),
    ("S08", "VALIDATE", "Run guardrail scan on any new CSV outputs", "python scripts/protocolo_c/revp_v1ru_cross_block_guardrail_audit.py", "GUARDRAIL_CLEAN", "Fix violations before staging", ""),
    ("S09", "COMMIT_PREP", "Run commit readiness package", "python scripts/protocolo_c/revp_v1rv_commit_readiness_package.py", "Recommended/excluded files classified", "Review ambiguous files manually", "Never stage .tif/.npy/.npz"),
    ("S10", "NEVER", "Never commit raster or embedding files", "—", "—", "—", ".tif .tiff .npy .npz must stay in gitignored output dirs (see .gitignore)"),
]


def run(datasets: Path | None = None) -> dict[str, Any]:
    steps = [
        {"step_id": s[0], "phase": s[1], "description": s[2],
         "command_or_action": s[3], "expected_result": s[4],
         "on_fail": s[5], "notes": s[6]}
        for s in _STEPS
    ]
    envvars = [
        {"env_var": e[0], "required": e[1], "description": e[2],
         "example_value": e[3], "notes": e[4]}
        for e in _ENV_VARS
    ]

    write_csv_with_header(OUT_STEPS, steps, STEPS_FIELDS)
    write_csv_with_header(OUT_ENVVARS, envvars, ENVVARS_FIELDS)
    write_schema_safe(SCHEMA_STEPS, STEPS_FIELDS, "v1rw_steps")
    write_schema_safe(SCHEMA_ENVVARS, ENVVARS_FIELDS, "v1rw_envvars")

    write_doc(OUT_RUNBOOK, "v1rw — Local Execution Runbook", [
        "## Objetivo",
        "Guia prático para a próxima fase real: configurar modelo local e roots Sentinel, "
        "rodar DINO em dry-run e validar antes de qualquer embedding real.",
        "## Variáveis de ambiente obrigatórias",
        "| Var | Obrigatório | Descrição |",
        "| --- | --- | --- |",
        *[f"| `{e[0]}` | {e[1]} | {e[2]} |" for e in _ENV_VARS],
        "## Passos",
        *[f"**{s[0]}** [{s[1]}] {s[2]}\n- Comando: `{s[3]}`\n- Esperado: {s[4]}\n- Em falha: {s[5]}" for s in _STEPS],
        "## Regras críticas",
        "1. `REVP_DINO_DRY_RUN=true` PRIMEIRO. Só mude para false depois de validar dry-run.",
        "2. `HF_HUB_OFFLINE=1` SEMPRE antes de rodar qualquer script DINO.",
        "3. NUNCA commitar `.tif`, `.tiff`, `.npy`, `.npz`, paths absolutos ou `local_runs/`.",
        "4. Outputs reais de embeddings ficam em `local_runs/` (gitignored).",
        "5. Rodar guardrail scan (v1ru) antes de qualquer staging.",
    ])

    print(f"[v1rw] steps={len(steps)} envvars={len(envvars)}")
    return {"steps": len(steps), "envvars": len(envvars)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rw local execution runbook").parse_args()
    run()
