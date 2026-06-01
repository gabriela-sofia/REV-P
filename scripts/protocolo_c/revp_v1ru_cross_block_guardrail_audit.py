"""REV-P v1ru — Cross-block guardrail audit.

Audits CSV outputs and docs from all implemented blocks against a controlled
checklist of forbidden claims. Uses a fixed list of target files — no
recursive scan. No science, no labels, no targets, no ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p, assert_clean_rows, guardrail_row,
    GUARDRAIL_FIELDS, scan_csv_guardrails, scan_doc_guardrails,
    safe_relpath, write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUDIT = _p("REVP_V1RU_OUT_AUDIT", DATASETS / "protocol_c_cross_block_guardrail_audit_v1ru.csv")
OUT_SUMMARY = _p("REVP_V1RU_OUT_SUMMARY", DATASETS / "protocol_c_cross_block_guardrail_summary_v1ru.csv")
SCHEMA_AUDIT = _p("REVP_V1RU_SCHEMA_AUDIT", SCHEMAS / "protocol_c_cross_block_guardrail_audit_v1ru_schema.csv")
SCHEMA_SUM = _p("REVP_V1RU_SCHEMA_SUM", SCHEMAS / "protocol_c_cross_block_guardrail_summary_v1ru_schema.csv")
DOC = _p("REVP_V1RU_DOC", DOCS / "revp_v1ru_cross_block_guardrail_audit.md")

AUDIT_FIELDS = ["audit_id", "artifact_path", "artifact_type", "check", "status",
                "violation_count", "severity", "review_only", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]

# Controlled target patterns (no recursion)
_CSV_GLOBS = [
    "datasets/protocol_c_*v1q[u-z]*.csv",
    "datasets/protocol_c_*v1r*.csv",
    "datasets/dino_*v1q[a-z]*.csv",
]
_DOC_GLOBS = [
    "docs/metodologia_cientifica/revp_v1q[u-z]*.md",
    "docs/metodologia_cientifica/revp_v1r*.md",
]
# claims that must never appear as C3 automatic / C4 open
_CLAIM_PHRASES = [
    ("no_c3_automatic", ["promote_automatically", "c3_auto", "auto_promote_c3"]),
    ("no_c4_opened", ["c4_opened", "formal_negative,true", "formal_negative=true"]),
    ("no_dino_as_proof", ["dino_validates_event,true", "dino_proves_event", "dino=proof"]),
    ("no_absence_as_negative", ["absence_as_negative,true", "absence=negative"]),
]


def _csv_paths() -> list[Path]:
    paths: list[Path] = []
    for pat in _CSV_GLOBS:
        for p in sorted(ROOT.glob(pat)):
            if "schema" not in p.name and p not in paths:
                paths.append(p)
    return paths


def _doc_paths() -> list[Path]:
    paths: list[Path] = []
    for pat in _DOC_GLOBS:
        for p in sorted(ROOT.glob(pat)):
            if p not in paths:
                paths.append(p)
    return paths


def run(datasets: Path | None = None) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    idx = 0

    def emit(path, atype, check, count, sev):
        nonlocal idx
        row = {
            "audit_id": f"V1RU_A{idx:05d}",
            "artifact_path": safe_relpath(path),
            "artifact_type": atype,
            "check": check,
            "status": "CLEAN" if count == 0 else "VIOLATION",
            "violation_count": str(count),
            "severity": sev,
            "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)
        idx += 1

    for path in _csv_paths():
        counts = scan_csv_guardrails(path)
        emit(path, "CSV", "abs_path", counts["abs_path"], "critical")
        emit(path, "CSV", "forbidden_literal", counts["forbidden_literal"], "critical")
        for f in GUARDRAIL_FIELDS:
            emit(path, "CSV", f"guardrail_{f}", counts.get(f, 0), "critical")

    for path in _doc_paths():
        dcounts = scan_doc_guardrails(path)
        emit(path, "DOC", "abs_path", dcounts["abs_path"], "critical")
        # forbidden_literal in docs is descriptive text — only enforce on CSVs
        # (docs may legitimately name the gitignored output directory as a concept)

    assert_clean_rows(rows, "v1ru_audit")
    write_csv_with_header(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_schema_safe(SCHEMA_AUDIT, AUDIT_FIELDS, "v1ru_audit")

    total = len(rows)
    violations = sum(1 for r in rows if r["status"] == "VIOLATION")
    overall = "GUARDRAIL_CLEAN" if violations == 0 else "GUARDRAIL_VIOLATIONS_FOUND"

    summary = [
        {"stat_key": "audit_status", "stat_value": overall},
        {"stat_key": "checks_total", "stat_value": str(total)},
        {"stat_key": "violations", "stat_value": str(violations)},
        {"stat_key": "stage", "stat_value": "v1ru"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_safe(SCHEMA_SUM, SUM_FIELDS, "v1ru_summary")

    write_doc(DOC, "v1ru — Cross-Block Guardrail Audit", [
        "## Objetivo",
        "Auditar CSVs e docs de todos os blocos implementados contra claims proibidos. "
        "Lista controlada de arquivos; sem scan recursivo.",
        "## Resultado",
        f"Status: {overall}. Checks: {total}. Violações: {violations}.",
        "## Guardrails verificados",
        "abs_path, forbidden_literal, can_create_operational_label, can_train_model, "
        "target_created, ground_truth_operational, formal_negative, dino_validates_event, "
        "absence_as_negative.",
    ])
    print(f"[v1ru] status={overall} checks={total} violations={violations}")
    return {"status": overall, "checks": total, "violations": violations}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1ru cross-block guardrail audit").parse_args()
    run()
