"""REV-P v1rv — Commit readiness package.

Runs read-only git queries to classify files for staging. Never runs git add,
commit or push. Generates PowerShell git add commands for each commit block.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row,
    write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_RECOMMENDED = _p("REVP_V1RV_OUT_RECOMMENDED", DATASETS / "protocol_c_commit_readiness_recommended_files_v1rv.csv")
OUT_EXCLUDED = _p("REVP_V1RV_OUT_EXCLUDED", DATASETS / "protocol_c_commit_readiness_excluded_files_v1rv.csv")
OUT_AMBIGUOUS = _p("REVP_V1RV_OUT_AMBIGUOUS", DATASETS / "protocol_c_commit_readiness_ambiguous_files_v1rv.csv")
SCHEMA_REC = _p("REVP_V1RV_SCHEMA_REC", SCHEMAS / "protocol_c_commit_readiness_recommended_v1rv_schema.csv")
SCHEMA_EXC = _p("REVP_V1RV_SCHEMA_EXC", SCHEMAS / "protocol_c_commit_readiness_excluded_v1rv_schema.csv")
SCHEMA_AMB = _p("REVP_V1RV_SCHEMA_AMB", SCHEMAS / "protocol_c_commit_readiness_ambiguous_v1rv_schema.csv")
DOC = _p("REVP_V1RV_DOC", DOCS / "revp_v1rv_commit_readiness_package.md")

FILE_FIELDS = ["file_id", "filepath", "commit_block", "reason", "git_command", "review_only", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

# Patterns that must NEVER be staged
_EXCLUDE_PATTERNS = [
    re.compile(r"\.(tif|tiff|geotiff|npy|npz)$", re.IGNORECASE),
    re.compile(r"local_runs/"),
    re.compile(r"data/"),
    re.compile(r"\.env"),
]
# Patterns that are ambiguous (need manual review)
_AMBIGUOUS_PATTERNS = [
    re.compile(r"\.log$"),
    re.compile(r"__pycache__"),
    re.compile(r"\.pyc$"),
]

# Block assignments
_BLOCK_RULES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"v1p[g-z]|v1q[a-m]"), "A", "DINO_REPRESENTATION_EXECUTION_V1PG_V1QT"),
    (re.compile(r"v1q[n-z]"), "B", "GROUND_REF_P0_V1QU_V1QZ"),
    (re.compile(r"v1r[a-f]"), "C", "EXTERNAL_INTAKE_P1_V1RA_V1RF"),
    (re.compile(r"v1r[g-m]"), "D", "REVIEW_GATE_P2_V1RG_V1RM"),
    (re.compile(r"v1r[n-r]"), "E", "DASHBOARD_P3_V1RN_V1RR"),
    (re.compile(r"v1r[s-z]"), "F", "INTEGRATION_V1RS_V1RZ"),
]


def _git(*args: str) -> list[str]:
    try:
        r = subprocess.run(
            ["git"] + list(args),
            cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        return [l.strip() for l in r.stdout.splitlines() if l.strip()]
    except Exception:
        return []


def _classify(fp: str) -> tuple[str, str]:
    """Return (block_letter, block_name)."""
    low = fp.lower()
    for pattern, letter, name in _BLOCK_RULES:
        if pattern.search(low):
            return letter, name
    return "X", "UNCATEGORIZED"


def run(datasets: Path | None = None) -> dict[str, Any]:
    modified = _git("diff", "--name-only")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    staged = _git("diff", "--cached", "--name-only")

    all_files = list(dict.fromkeys(modified + untracked))

    recommended: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []

    for i, fp in enumerate(all_files):
        if any(p.search(fp) for p in _EXCLUDE_PATTERNS):
            row = {"file_id": f"V1RV_E{i:04d}", "filepath": fp,
                   "commit_block": "EXCLUDED", "reason": "BINARY_OR_SENSITIVE_OR_DATA",
                   "git_command": "", "review_only": "true", "notes": "do_not_stage"}
            excluded.append(row)
        elif any(p.search(fp) for p in _AMBIGUOUS_PATTERNS):
            row = {"file_id": f"V1RV_B{i:04d}", "filepath": fp,
                   "commit_block": "AMBIGUOUS", "reason": "NEEDS_MANUAL_REVIEW",
                   "git_command": "", "review_only": "true", "notes": ""}
            ambiguous.append(row)
        else:
            letter, name = _classify(fp)
            cmd = f"git add '{fp}'" if fp else ""
            row = {"file_id": f"V1RV_R{i:04d}", "filepath": fp,
                   "commit_block": f"{letter}_{name}", "reason": "CODE_OR_CSV_OR_DOC",
                   "git_command": cmd, "review_only": "true", "notes": ""}
            recommended.append(row)

    write_csv_with_header(OUT_RECOMMENDED, recommended, FILE_FIELDS)
    write_csv_with_header(OUT_EXCLUDED, excluded, FILE_FIELDS)
    write_csv_with_header(OUT_AMBIGUOUS, ambiguous, FILE_FIELDS)
    write_schema_safe(SCHEMA_REC, FILE_FIELDS, "v1rv_recommended")
    write_schema_safe(SCHEMA_EXC, FILE_FIELDS, "v1rv_excluded")
    write_schema_safe(SCHEMA_AMB, FILE_FIELDS, "v1rv_ambiguous")

    # Build PowerShell commit blocks for the doc
    by_block: dict[str, list[str]] = {}
    for r in recommended:
        blk = r["commit_block"]
        by_block.setdefault(blk, []).append(r["git_command"])

    ps_lines: list[str] = []
    commit_messages = {
        "A_DINO_REPRESENTATION_EXECUTION_V1PG_V1QT": "DINO representation/execution readiness v1pg-v1qt",
        "B_GROUND_REF_P0_V1QU_V1QZ": "Ground reference partial workbench P0 v1qu-v1qz",
        "C_EXTERNAL_INTAKE_P1_V1RA_V1RF": "External evidence intake P1 v1ra-v1rf",
        "D_REVIEW_GATE_P2_V1RG_V1RM": "Review/supervisor gate P2 v1rg-v1rm",
        "E_DASHBOARD_P3_V1RN_V1RR": "Protocol C dashboard/roadmap P3 v1rn-v1rr",
        "F_INTEGRATION_V1RS_V1RZ": "Integration/hardening v1rs-v1rz",
    }
    for blk, cmds in sorted(by_block.items()):
        msg = commit_messages.get(blk, blk)
        ps_lines.append(f"# Block {blk}")
        ps_lines.extend(cmds[:20])  # cap per block
        ps_lines.append(f'git commit -m "{msg}"')
        ps_lines.append("")

    write_doc(DOC, "v1rv — Commit Readiness Package", [
        "## Objetivo",
        "Preparar pacote de commit por blocos sem executar `git add`. "
        "Roda apenas comandos git de leitura.",
        "## Arquivos recomendados para staging",
        f"Total: {len(recommended)}. Excluídos: {len(excluded)}. Ambíguos: {len(ambiguous)}.",
        "## Staged atual",
        f"Staged now: {len(staged)} arquivos.",
        "## Blocos de commit sugeridos (PowerShell)",
        "```powershell",
        "# NUNCA executar git add sem revisar primeiro",
        *ps_lines[:80],
        "```",
        "## Regras",
        "Nunca stagear: .tif/.tiff/.npy/.npz, local_runs/, data/, .env. "
        "Revisar manualmente arquivos ambíguos (.log, .pyc, __pycache__).",
    ])

    print(f"[v1rv] recommended={len(recommended)} excluded={len(excluded)} ambiguous={len(ambiguous)}")
    return {"recommended": len(recommended), "excluded": len(excluded), "ambiguous": len(ambiguous)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rv commit readiness").parse_args()
    run()
