"""REV-P v1qq — Local execution configuration template.

Generates safe, versionable configuration templates for real DINO smoke
embedding execution. Templates use placeholders — no real absolute paths.
"""
from __future__ import annotations

import argparse
from typing import Any

from revp_v1qn_v1qt_local_readiness_common import (
    DOCS, ROOT, SCHEMAS,
    _p, assert_no_forbidden_true, require_no_abs_paths,
    write_csv, write_doc, write_schema,
)

CONFIGS = ROOT / "configs"

OUT_DOC      = _p("REVP_V1QQ_OUT_DOC",      DOCS / "revp_v1qq_local_dino_execution_config_template.md")
OUT_ENV_EX   = _p("REVP_V1QQ_OUT_ENV_EX",   CONFIGS / "revp_dino_local_execution.env.example")
OUT_CHECKLIST= _p("REVP_V1QQ_OUT_CL",       ROOT / "configs" / "revp_dino_local_execution_checklist_v1qq.csv")
SCH_CL       = _p("REVP_V1QQ_SCH_CL",       SCHEMAS / "revp_dino_local_execution_checklist_v1qq_schema.csv")

CHECKLIST_FIELDS = [
    "step_id", "step_name", "command_or_action", "requires_manual",
    "safety_note", "expected_outcome", "review_only",
]

ENV_EXAMPLE_CONTENT = """\
# REV-P DINO local execution environment — DO NOT COMMIT WITH REAL PATHS
# Copy this file, fill in real paths locally, and source before running v1qj.
# Never version a copy that contains real absolute paths.

# ── Required for real execution ──────────────────────────────────────────
REVP_DINO_MODEL_PATH=<ABSOLUTE_LOCAL_MODEL_PATH_NOT_VERSIONED>
REVP_SENTINEL_LOCAL_ROOT=<ABSOLUTE_LOCAL_SENTINEL_ROOT_NOT_VERSIONED>
REVP_DINO_VISUAL_ROOT=<ABSOLUTE_LOCAL_VISUAL_ROOT_NOT_VERSIONED>
REVP_DINO_ASSET_ROOT=<ABSOLUTE_LOCAL_ASSET_ROOT_NOT_VERSIONED>

# ── Queue pointer (relative, safe to version) ─────────────────────────────
REVP_V1PQ_QUEUE_PATH=datasets/dino_execution_queue_from_visual_expansion_v1qa.csv

# ── Safety gates (defaults — change with care) ────────────────────────────
REVP_DINO_ALLOW_DOWNLOAD=false
REVP_DINO_DRY_RUN=true
REVP_DINO_PIXEL_READ_ALLOWED=false
HF_HUB_OFFLINE=1

# ── Execution tuning ──────────────────────────────────────────────────────
REVP_DINO_BATCH_SIZE=4
REVP_DINO_MAX_EXECUTE=32
"""

PS_STEPS = [
    ("set_model_path",
     '$env:REVP_DINO_MODEL_PATH = "<ABSOLUTE_LOCAL_MODEL_PATH>"',
     "Set to local DINOv2 model directory. Never commit this value.",
     True),
    ("set_sentinel_root",
     '$env:REVP_SENTINEL_LOCAL_ROOT = "<ABSOLUTE_LOCAL_SENTINEL_ROOT>"',
     "Root directory containing Sentinel TIF files locally.",
     True),
    ("lock_download",
     '$env:REVP_DINO_ALLOW_DOWNLOAD = "false"',
     "Never change to true — prevents accidental model downloads.",
     False),
    ("enable_offline",
     '$env:HF_HUB_OFFLINE = "1"',
     "Force HuggingFace hub offline mode.",
     False),
    ("run_v1qn",
     "python scripts/dino/revp_v1qn_local_root_environment_audit.py",
     "Audit local roots and model directory.",
     False),
    ("run_v1qo",
     "python scripts/dino/revp_v1qo_smoke_asset_local_reconciliation.py",
     "Reconcile smoke sample with local files.",
     False),
    ("run_v1qg",
     "python scripts/dino/revp_v1qg_local_dino_model_offline_audit.py",
     "Verify model config/weights without loading.",
     False),
    ("run_v1qi",
     "python scripts/dino/revp_v1qi_local_asset_preprocessing_audit.py",
     "Audit local asset metadata (no pixel read unless env set).",
     False),
    ("dry_run_v1qj",
     'python scripts/dino/revp_v1qj_controlled_real_smoke_embedding_executor.py',
     "Dry-run by default. Review outputs before enabling real execution.",
     False),
    ("enable_real_execution_manual",
     '# MANUAL: $env:REVP_DINO_DRY_RUN = "false"  # Only after reviewing all gate outputs',
     "Enable real embedding execution ONLY after all gates pass. Manual step.",
     True),
    ("enable_pixel_read_manual",
     '# MANUAL: $env:REVP_DINO_PIXEL_READ_ALLOWED = "true"  # Required for real execution',
     "Enable pixel reading ONLY for real execution run. Manual step.",
     True),
]


def build_checklist() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, (name, cmd, note, manual) in enumerate(PS_STEPS, 1):
        rows.append({
            "step_id": f"V1QQ_STEP_{i:02d}",
            "step_name": name,
            "command_or_action": cmd,
            "requires_manual": str(manual).lower(),
            "safety_note": note,
            "expected_outcome": "see doc",
            "review_only": "true",
        })
    return rows


def run() -> None:
    checklist = build_checklist()
    require_no_abs_paths(checklist, "v1qq_checklist")
    assert_no_forbidden_true(checklist, "v1qq_checklist")

    CONFIGS.mkdir(exist_ok=True)
    OUT_ENV_EX.parent.mkdir(parents=True, exist_ok=True)
    OUT_ENV_EX.write_text(ENV_EXAMPLE_CONTENT, encoding="utf-8")

    write_csv(OUT_CHECKLIST, checklist, CHECKLIST_FIELDS)
    write_schema(SCH_CL, CHECKLIST_FIELDS, "v1qq_local_execution_checklist")

    write_doc(OUT_DOC, "v1qq — Local DINO Execution Config Template", [
        "## Objetivo",
        "Gerar templates seguros para configurar execução smoke real de DINO. "
        "Nenhum path absoluto real é versionado.",
        "## Como usar",
        "1. Copie `configs/revp_dino_local_execution.env.example` para um arquivo "
        "local (não versionado).\n"
        "2. Preencha os campos `<ABSOLUTE_...>` com caminhos reais locais.\n"
        "3. Execute os passos do checklist em ordem.\n"
        "4. Nunca comite o arquivo com paths reais preenchidos.",
        "## PowerShell steps",
        "\n".join(f"- `{s['step_name']}`: {s['safety_note']}"
                  for s in checklist),
    ])
    print(f"[v1qq] checklist={len(checklist)} env_example=written")


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1qq local execution config template").parse_args()
    run()
