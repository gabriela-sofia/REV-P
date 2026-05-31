"""REV-P v1ou — External evidence source inventory.

Scans existing repo files (metadata-only: headers + first rows) for terms
associated with events, sources, and external evidence. Does not use
the internet, does not download anything, does not OCR documents.

All detected candidates are classified with allowed_use; no operational
labels or ground truth flags are created.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

from revp_v1lj_v1lq_common import DATASETS, DOCS, SCHEMAS
from revp_v1ou_v1pa_common import (
    EVIDENCE_TERMS,
    _p,
    assert_no_forbidden_true,
    classify_evidence_use,
    classify_source_reliability,
    is_fixture_or_synthetic,
    normalize_event_date,
    normalize_region,
    path_hash_sanitized,
    require_no_abs_paths_in_rows,
    write_csv_safe,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Output paths (env-overridable)
# ---------------------------------------------------------------------------

OUT_INVENTORY = _p("REVP_V1OU_OUT_INVENTORY", DATASETS / "recife_external_evidence_source_inventory_v1ou.csv")
OUT_SUMMARY = _p("REVP_V1OU_OUT_SUMMARY", DATASETS / "recife_external_evidence_source_inventory_summary_v1ou.csv")
SCHEMA_INVENTORY = _p("REVP_V1OU_SCHEMA_INVENTORY", SCHEMAS / "recife_external_evidence_source_inventory_v1ou_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1OU_SCHEMA_SUMMARY", SCHEMAS / "recife_external_evidence_source_inventory_summary_v1ou_schema.csv")
DOC = _p("REVP_V1OU_DOC", DOCS / "revp_v1ou_external_evidence_source_inventory.md")

INVENTORY_FIELDS = [
    "source_candidate_id",
    "region",
    "file_path_hash",
    "relative_path",
    "file_type",
    "evidence_terms_found",
    "candidate_event_id",
    "candidate_source_type",
    "candidate_source_name",
    "candidate_date_raw",
    "candidate_location_raw",
    "confidence_preliminary",
    "is_fixture_or_synthetic",
    "allowed_for_event_registry",
    "blocked_reason",
    "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

# ---------------------------------------------------------------------------
# Directories to scan
# ---------------------------------------------------------------------------

SCAN_CSV_DIRS = [
    DATASETS,
]
SCAN_MD_DIRS = [
    ROOT / "docs" / "metodologia_cientifica",
    ROOT / "docs",
]
EXCLUDE_DIRS = {
    "schemas", "__pycache__", "local_runs",
    ".git", ".claude", "node_modules",
}

# Files that are only internal bookkeeping — not external evidence
INTERNAL_ONLY_NAMES = {
    # These files are summaries/metrics of our own pipeline, not external evidence
    "recife_scene_date_recovery_final_manifest_v1ot.csv",
    "recife_scene_date_recovery_final_quality_checks_v1ot.csv",
    "recife_scene_date_recovery_final_scientific_summary_v1ot.csv",
    "recife_fixture_contamination_audit_v1os.csv",
    "recife_fixture_contamination_summary_v1os.csv",
}

# Columns whose presence strongly suggests external evidence content
EVIDENCE_COLUMNS = {
    "event_id", "source_url", "data_evento", "event_date", "candidate_date_raw",
    "candidate_source_name", "candidate_event_id", "observed_event_id",
    "source_institution", "reliability", "spatial_precision", "temporal_precision",
    "dossier_status", "current_blocker", "needed_id", "gap_id",
    "evidence_type", "evidence_tier", "source_reliability_level",
    "candidate_event_period", "event_or_survey_date", "event_name",
}

# Terms to look for in text
_LOWER_TERMS = [t.lower() for t in EVIDENCE_TERMS]


def _contains_evidence_terms(text: str) -> list[str]:
    lo = text.lower()
    return [t for t in _LOWER_TERMS if t in lo]


def _scan_csv_file(path: Path) -> dict[str, Any] | None:
    """Return metadata about a CSV file if it contains evidence content."""
    if path.name in INTERNAL_ONLY_NAMES:
        return None
    try:
        import csv as _csv
        with path.open(encoding="utf-8-sig", errors="replace", newline="") as fh:
            reader = _csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows: list[dict[str, str]] = []
            for i, row in enumerate(reader):
                if i >= 4:
                    break
                rows.append(dict(row))
    except Exception:
        return None

    if not fields:
        return None

    # Check column names for evidence terms
    cols_lower = " ".join(fields).lower()
    terms_in_cols = _contains_evidence_terms(cols_lower)

    # Check if any evidence columns present
    fields_set = {f.lower() for f in fields}
    evidence_col_matches = [c for c in EVIDENCE_COLUMNS if c in fields_set]

    # Check values in sample rows
    sample_text = " ".join(
        " ".join(str(v) for v in row.values()) for row in rows
    )
    terms_in_values = _contains_evidence_terms(sample_text)

    all_terms = list(dict.fromkeys(terms_in_cols + terms_in_values))
    if not all_terms and not evidence_col_matches:
        return None

    # Extract candidate info from first rows
    candidate_event_id = ""
    candidate_source_name = ""
    candidate_date_raw = ""
    candidate_location_raw = ""
    candidate_source_type = ""
    region_raw = ""

    for row in rows:
        for k, v in row.items():
            kl = k.lower()
            if not candidate_event_id and ("event_id" in kl or "dossier_id" in kl or "observed_event" in kl):
                candidate_event_id = str(v).strip()[:60]
            if not candidate_source_name and ("source_institution" in kl or "source_name" in kl or "institution" in kl):
                candidate_source_name = str(v).strip()[:80]
            if not candidate_date_raw and ("date" in kl or "periodo" in kl or "period" in kl):
                val = str(v).strip()
                if val and val.lower() not in ("unknown", "n/a", ""):
                    candidate_date_raw = val[:40]
            if not candidate_location_raw and ("municipality" in kl or "region" in kl or "locality" in kl):
                candidate_location_raw = str(v).strip()[:60]
            if not region_raw and "region" in kl:
                region_raw = str(v).strip()

    region = normalize_region(region_raw) if region_raw else "UNKNOWN"
    if not candidate_source_type:
        fname = path.name.lower()
        if "event" in fname:
            candidate_source_type = "EVENT_REGISTRY"
        elif "evidence" in fname or "external" in fname:
            candidate_source_type = "EVIDENCE_REGISTRY"
        elif "gap" in fname or "needed" in fname:
            candidate_source_type = "GAP_REGISTRY"
        elif "patch" in fname:
            candidate_source_type = "PATCH_REGISTRY"
        else:
            candidate_source_type = "DATASET_CSV"

    return {
        "file_type": "CSV",
        "fields": fields,
        "evidence_terms": all_terms,
        "evidence_col_matches": evidence_col_matches,
        "candidate_event_id": candidate_event_id,
        "candidate_source_type": candidate_source_type,
        "candidate_source_name": candidate_source_name,
        "candidate_date_raw": candidate_date_raw,
        "candidate_location_raw": candidate_location_raw,
        "region": region,
        "rows_sample": rows,
    }


def _scan_md_file(path: Path) -> dict[str, Any] | None:
    """Return metadata about an MD file if it contains evidence content."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception:
        return None

    terms_found = _contains_evidence_terms(text)
    if not terms_found:
        return None

    region = "UNKNOWN"
    for line in text.splitlines()[:30]:
        lo = line.lower()
        if "recife" in lo:
            region = "RECIFE"
            break
        if "petrópolis" in lo or "petropolis" in lo:
            region = "PET"
            break

    return {
        "file_type": "MD",
        "fields": [],
        "evidence_terms": terms_found,
        "evidence_col_matches": [],
        "candidate_event_id": "",
        "candidate_source_type": "METHODOLOGY_DOC",
        "candidate_source_name": path.stem,
        "candidate_date_raw": "",
        "candidate_location_raw": region,
        "region": region,
        "rows_sample": [],
    }


def _is_excluded(path: Path) -> bool:
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def scan_repo(root: Path) -> list[dict[str, Any]]:
    """Scan the repo for external evidence source candidates."""
    results = []
    seen_paths: set[str] = set()

    # Scan CSV files in datasets/
    for csv_dir in SCAN_CSV_DIRS:
        if not csv_dir.exists():
            continue
        for path in sorted(csv_dir.glob("*.csv")):
            if _is_excluded(path):
                continue
            rel = path.relative_to(root).as_posix()
            if rel in seen_paths:
                continue
            meta = _scan_csv_file(path)
            if meta:
                meta["path"] = path
                meta["rel"] = rel
                seen_paths.add(rel)
                results.append(meta)

    # Scan MD files in docs/
    for md_dir in SCAN_MD_DIRS:
        if not md_dir.exists():
            continue
        for path in sorted(md_dir.glob("*.md")):
            if _is_excluded(path):
                continue
            rel = path.relative_to(root).as_posix()
            if rel in seen_paths:
                continue
            meta = _scan_md_file(path)
            if meta:
                meta["path"] = path
                meta["rel"] = rel
                seen_paths.add(rel)
                results.append(meta)

    return results


def build_inventory_row(idx: int, meta: dict[str, Any]) -> dict[str, Any]:
    rel = meta["rel"]
    ph = path_hash_sanitized(rel)
    candidate_id = f"V1OU_{idx:04d}"

    # Build a synthetic evidence row to classify
    evidence_row_for_classify = {
        "candidate_source_name": meta["candidate_source_name"],
        "candidate_date_raw": meta["candidate_date_raw"],
        "candidate_location_raw": meta["candidate_location_raw"],
        "region": meta["region"],
        "current_blocker": "",
        "confidence_preliminary": "",
    }

    allowed_use = classify_evidence_use(evidence_row_for_classify)

    # Determine preliminary confidence
    n_terms = len(meta["evidence_terms"])
    n_cols = len(meta["evidence_col_matches"])
    if n_terms >= 4 and n_cols >= 2:
        confidence = "MODERATE"
    elif n_terms >= 2 or n_cols >= 1:
        confidence = "LIMITED"
    else:
        confidence = "NONE"

    # Fixture check
    fixture_reason = ""
    for row in meta.get("rows_sample", []):
        reason = is_fixture_or_synthetic(row)
        if reason:
            fixture_reason = reason
            break

    is_fixture = "true" if fixture_reason else "false"
    if fixture_reason:
        allowed_use = "BLOCKED_FIXTURE_OR_SYNTHETIC"
        blocked_reason = fixture_reason
    elif allowed_use.startswith("BLOCKED"):
        blocked_reason = allowed_use
    else:
        blocked_reason = ""

    allowed_for_registry = "false" if allowed_use.startswith("BLOCKED") else "true"

    return {
        "source_candidate_id": candidate_id,
        "region": meta["region"],
        "file_path_hash": ph,
        "relative_path": rel,
        "file_type": meta["file_type"],
        "evidence_terms_found": ";".join(meta["evidence_terms"][:8]),
        "candidate_event_id": meta["candidate_event_id"],
        "candidate_source_type": meta["candidate_source_type"],
        "candidate_source_name": meta["candidate_source_name"],
        "candidate_date_raw": meta["candidate_date_raw"],
        "candidate_location_raw": meta["candidate_location_raw"],
        "confidence_preliminary": confidence,
        "is_fixture_or_synthetic": is_fixture,
        "allowed_for_event_registry": allowed_for_registry,
        "blocked_reason": blocked_reason,
        "notes": "",
    }


def run(root: Path | None = None) -> None:
    root = root or ROOT
    scan_results = scan_repo(root)

    rows: list[dict[str, Any]] = []
    for i, meta in enumerate(scan_results):
        row = build_inventory_row(i, meta)
        rows.append(row)

    assert_no_forbidden_true(rows, "v1ou_inventory")
    require_no_abs_paths_in_rows(rows, "v1ou_inventory")

    write_csv_safe(OUT_INVENTORY, rows, INVENTORY_FIELDS)
    write_schema_safe(SCHEMA_INVENTORY, INVENTORY_FIELDS, "v1ou_external_evidence_source_inventory")

    # Summary
    total = len(rows)
    allowed = sum(1 for r in rows if r["allowed_for_event_registry"] == "true")
    blocked = total - allowed
    fixture = sum(1 for r in rows if r["is_fixture_or_synthetic"] == "true")
    recife = sum(1 for r in rows if r["region"] == "RECIFE")
    pet = sum(1 for r in rows if r["region"] == "PET")

    summary_rows = [
        {"stat_key": "total_source_candidates_found", "stat_value": str(total)},
        {"stat_key": "allowed_for_event_registry", "stat_value": str(allowed)},
        {"stat_key": "blocked", "stat_value": str(blocked)},
        {"stat_key": "fixture_or_synthetic_excluded", "stat_value": str(fixture)},
        {"stat_key": "region_recife", "stat_value": str(recife)},
        {"stat_key": "region_pet", "stat_value": str(pet)},
        {"stat_key": "scan_status", "stat_value": "METADATA_ONLY_NO_INTERNET_NO_OCR"},
        {"stat_key": "stage", "stat_value": "v1ou"},
    ]
    write_csv_safe(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1ou_summary")

    write_doc(
        DOC,
        "v1ou — External Evidence Source Inventory",
        [
            "## Objetivo",
            "Escanear arquivos existentes do repositório para identificar candidatos a fontes "
            "e evidências externas de eventos observados. Não usa internet, não baixa nada, "
            "não executa OCR. Lê apenas headers + primeiras linhas (metadata-only).",
            "## Resultado",
            f"Total de candidatos encontrados: {total}. "
            f"Permitidos para registro de eventos: {allowed}. "
            f"Bloqueados: {blocked}. "
            f"Fixture/sintético excluídos: {fixture}.",
            "## Guardrails",
            "Nenhum candidato é promovido a ground truth operacional. "
            "allowed_for_event_registry=true significa apenas que o arquivo contém "
            "termos relevantes e pode ser inspecionado para construir o registro de eventos. "
            "Não implica confirmação de evento, label ou target.",
            "## Relação com v1og-v1ot",
            "v1og-v1ot confirmou TEMPORAL_RECOVERY_FAIL_CLOSED para Recife: "
            "0 product_dates confirmadas, 0 C3+ candidates, 0 formal negatives. "
            "v1ou não tenta destravar temporal artificialmente.",
        ],
    )
    print(f"[v1ou] {total} source candidates: {allowed} allowed, {blocked} blocked")
    print(f"[v1ou] Output: {OUT_INVENTORY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v1ou external evidence source inventory")
    parser.parse_args()
    run()
