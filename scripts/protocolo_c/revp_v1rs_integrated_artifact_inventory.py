"""REV-P v1rs — Integrated artifact inventory.

Inventories datasets/schemas/docs/tests across all implemented blocks
(DINO v1pg-v1qt, Protocol C P0-P3, integration v1rs-v1rz).
Read-only; no science, no labels, no ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from revp_v1rs_v1rz_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p, assert_clean_rows, guardrail_row,
    collect_artifacts_by_patterns, classify_artifact_status,
    count_rows, infer_block_from_filename, infer_stage_from_filename,
    safe_relpath, write_csv_with_header, write_doc, write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_INVENTORY = _p("REVP_V1RS_OUT_INVENTORY", DATASETS / "protocol_c_integrated_artifact_inventory_v1rs.csv")
OUT_SUMMARY = _p("REVP_V1RS_OUT_SUMMARY", DATASETS / "protocol_c_integrated_artifact_inventory_summary_v1rs.csv")
SCHEMA_INV = _p("REVP_V1RS_SCHEMA_INV", SCHEMAS / "protocol_c_integrated_artifact_inventory_v1rs_schema.csv")
SCHEMA_SUM = _p("REVP_V1RS_SCHEMA_SUM", SCHEMAS / "protocol_c_integrated_artifact_inventory_summary_v1rs_schema.csv")
DOC = _p("REVP_V1RS_DOC", DOCS / "revp_v1rs_integrated_artifact_inventory.md")

INV_FIELDS = [
    "artifact_id", "block", "stage", "artifact_type", "artifact_path",
    "artifact_exists", "row_count", "has_schema", "has_doc", "has_test",
    "status", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

# glob patterns covering all blocks
_CSV_PATTERNS = [
    "datasets/dino_*v1p[g-z]*.csv",
    "datasets/dino_*v1q[a-z]*.csv",
    "datasets/protocol_c_*v1q*.csv",
    "datasets/protocol_c_*v1r*.csv",
]
_DOC_PATTERNS = [
    "docs/metodologia_cientifica/revp_v1p[g-z]*.md",
    "docs/metodologia_cientifica/revp_v1q*.md",
    "docs/metodologia_cientifica/revp_v1r*.md",
]
_SCHEMA_PATTERNS = ["datasets/schemas/*v1p[g-z]*.csv",
                    "datasets/schemas/*v1q*.csv",
                    "datasets/schemas/*v1r*.csv"]
_TEST_PATTERNS = ["tests/test_revp_v1p*.py", "tests/test_revp_v1q*.py",
                  "tests/test_revp_v1r*.py"]
_SCRIPT_PATTERNS = ["scripts/protocolo_c/revp_v1p[g-z]*.py",
                    "scripts/protocolo_c/revp_v1q*.py",
                    "scripts/protocolo_c/revp_v1r*.py",
                    "scripts/dino/revp_v1p*.py", "scripts/dino/revp_v1q*.py"]


def _schema_exists_for(path: Path) -> bool:
    stem = path.stem
    for p in SCHEMAS.glob(f"*{stem}*schema.csv"):
        return True
    return False


def _doc_exists_for(path: Path) -> bool:
    name = path.stem
    # strip leading stage tokens; look for matching doc
    tokens = name.split("_")
    for tok in tokens[1:3]:  # rough match
        for p in DOCS.glob(f"revp_*{tok}*.md"):
            return True
    return False


def _test_exists_for(block: str) -> bool:
    for tp in ROOT.glob("tests/test_revp_v1*.py"):
        low = tp.stem.lower()
        # map block to expected test tokens
        if "v1qu_v1qz" in low and "P0" in block:
            return True
        if "v1ra_v1rf" in low and "P1" in block:
            return True
        if "v1rg_v1rm" in low and "P2" in block:
            return True
        if "v1rn_v1rr" in low and "P3" in block:
            return True
        if "v1qn_v1qt" in low and "LOCAL" in block:
            return True
        if "v1qg_v1qm" in low and "SMOKE" in block:
            return True
        if "v1qa_v1qf" in low and "BRIDGE" in block:
            return True
    return False


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 0

    def add(type_: str, path: Path, extra: dict | None = None):
        nonlocal idx
        rel = safe_relpath(path)
        block = infer_block_from_filename(path.name)
        stage = infer_stage_from_filename(path.name)
        status = classify_artifact_status(path)
        row = {
            "artifact_id": f"V1RS_A{idx:04d}",
            "block": block, "stage": stage,
            "artifact_type": type_,
            "artifact_path": rel,
            "artifact_exists": "true" if path.exists() else "false",
            "row_count": str(count_rows(path)) if type_ in ("CSV", "SCHEMA") else "N/A",
            "has_schema": "true" if _schema_exists_for(path) else "false",
            "has_doc": "true" if _doc_exists_for(path) else "false",
            "has_test": "true" if _test_exists_for(block) else "false",
            "status": status,
            "notes": extra.get("notes", "") if extra else "",
        }
        row.update(guardrail_row())
        rows.append(row)
        idx += 1

    for p in collect_artifacts_by_patterns(_CSV_PATTERNS, ROOT):
        if "schema" not in p.name:
            add("CSV", p)
    for p in collect_artifacts_by_patterns(_SCHEMA_PATTERNS, ROOT):
        add("SCHEMA", p)
    for p in collect_artifacts_by_patterns(_DOC_PATTERNS, ROOT):
        add("DOC", p)
    for p in collect_artifacts_by_patterns(_SCRIPT_PATTERNS, ROOT):
        add("SCRIPT", p)
    return rows


def run(datasets: Path | None = None) -> dict[str, Any]:
    rows = build_rows()
    assert_clean_rows(rows, "v1rs_inventory")
    write_csv_with_header(OUT_INVENTORY, rows, INV_FIELDS)
    write_schema_safe(SCHEMA_INV, INV_FIELDS, "v1rs_inventory")

    total = len(rows)
    csvs = sum(1 for r in rows if r["artifact_type"] == "CSV")
    missing = sum(1 for r in rows if r["artifact_exists"] == "false")
    no_schema = sum(1 for r in rows if r["artifact_type"] == "CSV" and r["has_schema"] == "false")
    no_doc = sum(1 for r in rows if r["artifact_type"] in ("CSV", "SCRIPT") and r["has_doc"] == "false")
    no_test = sum(1 for r in rows if r["artifact_type"] == "CSV" and r["has_test"] == "false")

    summary = [
        {"stat_key": "total_artifacts", "stat_value": str(total)},
        {"stat_key": "csv_artifacts", "stat_value": str(csvs)},
        {"stat_key": "missing_artifacts", "stat_value": str(missing)},
        {"stat_key": "csv_missing_schema", "stat_value": str(no_schema)},
        {"stat_key": "csv_missing_doc", "stat_value": str(no_doc)},
        {"stat_key": "csv_missing_test", "stat_value": str(no_test)},
        {"stat_key": "stage", "stat_value": "v1rs"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_safe(SCHEMA_SUM, SUM_FIELDS, "v1rs_summary")

    write_doc(DOC, "v1rs — Integrated Artifact Inventory", [
        "## Objetivo",
        "Inventariar todos os artefatos (CSV, schema, doc, script) dos blocos DINO v1pg-v1qt "
        "e Protocolo C P0-P3 (v1qu-v1rr), coletando existência, row count, has_schema, "
        "has_doc, has_test e status.",
        "## Resultado",
        f"Total: {total}. CSVs: {csvs}. Ausentes: {missing}. "
        f"CSV sem schema: {no_schema}. CSV sem doc: {no_doc}. CSV sem teste: {no_test}.",
    ])
    print(f"[v1rs] total={total} csvs={csvs} missing={missing}")
    return {"total": total, "csvs": csvs, "missing": missing,
            "no_schema": no_schema, "no_doc": no_doc, "no_test": no_test}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rs artifact inventory").parse_args()
    run()
