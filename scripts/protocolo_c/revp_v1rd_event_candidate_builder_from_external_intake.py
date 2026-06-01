"""REV-P v1rd — Event candidate builder from external intake.

If a validated external intake exists, builds review-only event candidates.
Joins v1rc validation (valid rows) with the manually-filled intake for date /
location. Candidates are review-only; never labels, targets or ground truth.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from revp_v1ra_v1rf_external_intake_common import (
    DATASETS,
    DOCS,
    SCHEMAS,
    _p,
    assert_clean_rows,
    guardrail_row,
    hash_short,
    normalize_region,
    read_csv_safe,
    write_csv_with_header,
    write_doc,
    write_schema_safe,
)

ROOT = Path(__file__).resolve().parents[2]

IN_VALIDATION = _p("REVP_V1RD_IN_VALIDATION", DATASETS / "protocol_c_external_document_intake_validation_v1rc.csv")
OUT_CANDIDATES = _p("REVP_V1RD_OUT_CANDIDATES", DATASETS / "protocol_c_external_event_candidates_v1rd.csv")
OUT_SUMMARY = _p("REVP_V1RD_OUT_SUMMARY", DATASETS / "protocol_c_external_event_candidates_summary_v1rd.csv")
SCHEMA_CANDIDATES = _p("REVP_V1RD_SCHEMA_CANDIDATES", SCHEMAS / "protocol_c_external_event_candidates_v1rd_schema.csv")
SCHEMA_SUMMARY = _p("REVP_V1RD_SCHEMA_SUMMARY", SCHEMAS / "protocol_c_external_event_candidates_summary_v1rd_schema.csv")
DOC = _p("REVP_V1RD_DOC", DOCS / "revp_v1rd_event_candidate_builder_from_external_intake.md")

CANDIDATE_FIELDS = [
    "event_candidate_id", "document_id", "region", "hazard_type",
    "source_name", "source_family", "event_date_text", "event_location_text",
    "temporal_precision_claim", "spatial_precision_claim", "license_status",
    "candidate_status", "review_only", "dino_validates_event",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "absence_as_negative", "notes",
]

SUMMARY_FIELDS = ["stat_key", "stat_value"]

NO_INTAKE = "EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS"
CANDIDATES_READY = "EXTERNAL_INTAKE_CANDIDATES_READY_REVIEW_ONLY"


def _intake_index() -> dict[str, dict[str, str]]:
    env = os.environ.get("REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH")
    if not env or not Path(env).exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    for r in read_csv_safe(Path(env)):
        did = str(r.get("document_id", "")).strip()
        if did:
            out[did] = r
    return out


def _passing_documents(validation: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Group v1rc per-check rows by document_id; keep docs with no C3-blocking FAIL."""
    by_doc: dict[str, dict[str, str]] = {}
    blocked: set[str] = set()
    for v in validation:
        did = str(v.get("document_id", "")).strip()
        if not did:
            continue
        by_doc.setdefault(did, v)  # first row carries doc-level fields
        if v.get("status") == "FAIL" and v.get("blocks_c3") == "true":
            blocked.add(did)
    return {did: row for did, row in by_doc.items() if did not in blocked}


def run(datasets: Path | None = None) -> dict[str, Any]:
    validation = read_csv_safe(IN_VALIDATION)
    intake = _intake_index()

    rows: list[dict[str, Any]] = []
    for did, v in _passing_documents(validation).items():
        src = intake.get(did, {})
        region = normalize_region(v.get("region", ""))
        date_text = src.get("event_date_text", "")
        loc_text = src.get("event_location_text", "")
        cand = {
            "event_candidate_id": f"V1RD_CAND_{hash_short(did + region, 10)}",
            "document_id": did,
            "region": region,
            "hazard_type": v.get("hazard_type", ""),
            "source_name": v.get("source_name", ""),
            "source_family": v.get("source_family", ""),
            "event_date_text": date_text,
            "event_location_text": loc_text,
            "temporal_precision_claim": str(src.get("temporal_precision_claim", "")).upper(),
            "spatial_precision_claim": str(src.get("spatial_precision_claim", "")).upper(),
            "license_status": src.get("license_note", ""),
            "candidate_status": "REVIEW_ONLY_EXTERNAL_CANDIDATE",
            "notes": "",
        }
        cand.update(guardrail_row())
        rows.append(cand)

    assert_clean_rows(rows, "v1rd_candidates")
    write_csv_with_header(OUT_CANDIDATES, rows, CANDIDATE_FIELDS)
    write_schema_safe(SCHEMA_CANDIDATES, CANDIDATE_FIELDS, "v1rd_candidates")

    status = CANDIDATES_READY if rows else NO_INTAKE
    by_region: dict[str, int] = {}
    for r in rows:
        by_region[r["region"]] = by_region.get(r["region"], 0) + 1

    summary = [
        {"stat_key": "candidate_status", "stat_value": status},
        {"stat_key": "event_candidates", "stat_value": str(len(rows))},
    ]
    for region, n in sorted(by_region.items()):
        summary.append({"stat_key": f"region_{region.lower()}", "stat_value": str(n)})
    summary.append({"stat_key": "stage", "stat_value": "v1rd"})
    write_csv_with_header(OUT_SUMMARY, summary, SUMMARY_FIELDS)
    write_schema_safe(SCHEMA_SUMMARY, SUMMARY_FIELDS, "v1rd_summary")

    write_doc(
        DOC,
        "v1rd — Event Candidate Builder from External Intake",
        [
            "## Objetivo",
            "Construir candidatos de evento review-only a partir do intake externo validado "
            "(v1rc). Sem intake valido, fica EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS.",
            "## Resultado",
            f"Status: {status}. Candidatos de evento: {len(rows)}.",
            "## Guardrails",
            "candidate_status=REVIEW_ONLY_EXTERNAL_CANDIDATE. dino_validates_event=false. "
            "absence_as_negative=false. Nenhum label/target/ground truth operacional.",
        ],
    )
    print(f"[v1rd] status={status} candidates={len(rows)}")
    return {"status": status, "candidates": len(rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1rd event candidate builder").parse_args()
    run()
