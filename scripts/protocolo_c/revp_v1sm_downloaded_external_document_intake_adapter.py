"""REV-P v1sm — Downloaded external document intake adapter.

Transforms official download manifests into intake drafts compatible with
v1rb template. Hydrometeorological evidence becomes HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, read_csv_safe, forbidden_guardrail_scan,
    hash_short, classify_source_family, infer_region_from_text,
)

ROOT = Path(__file__).resolve().parents[2]

IN_ORCHESTRATOR = _p("REVP_V1SM_IN_ORCHESTRATOR", DATASETS / "protocol_c_official_download_orchestrator_manifest_v1sl.csv")
IN_INMET_STATIONS = _p("REVP_V1SM_IN_STATIONS", DATASETS / "protocol_c_inmet_station_candidates_v1si.csv")

OUT_DRAFT = _p("REVP_V1SM_OUT_DRAFT", DATASETS / "protocol_c_downloaded_external_document_intake_draft_v1sm.csv")
OUT_SUMMARY = _p("REVP_V1SM_OUT_SUMMARY", DATASETS / "protocol_c_downloaded_external_document_intake_summary_v1sm.csv")
SCHEMA_D = _p("REVP_V1SM_SCHEMA_D", SCHEMAS / "protocol_c_downloaded_external_document_intake_draft_v1sm_schema.csv")
SCHEMA_S = _p("REVP_V1SM_SCHEMA_S", SCHEMAS / "protocol_c_downloaded_external_document_intake_summary_v1sm_schema.csv")
DOC = _p("REVP_V1SM_DOC", DOCS / "revp_v1sm_downloaded_external_document_intake_adapter.md")

DRAFT_FIELDS = [
    "intake_draft_id", "document_id", "source_name", "source_family", "region",
    "hazard_type", "event_date_text", "event_location_text", "url_or_reference",
    "evidence_type", "temporal_precision_claim", "spatial_precision_claim",
    "license_note", "manual_review_required", "intake_status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run(datasets: Path | None = None) -> dict[str, Any]:
    orchestrator = read_csv_safe(IN_ORCHESTRATOR)
    stations = read_csv_safe(IN_INMET_STATIONS)
    station_regions = {s.get("station_code", ""): s.get("region_candidate", "") for s in stations}

    rows: list[dict[str, Any]] = []
    for i, r in enumerate(orchestrator):
        if r.get("downloaded") != "true":
            continue
        source = r.get("source_name", "")
        family = classify_source_family(source)
        region = infer_region_from_text(source + " " + r.get("source_block", ""))
        row = {
            "intake_draft_id": f"V1SM_ID_{i:04d}",
            "document_id": f"AUTO_{hash_short(source + str(i), 10)}",
            "source_name": source, "source_family": family,
            "region": region, "hazard_type": "HYDROMETEOROLOGICAL",
            "event_date_text": "", "event_location_text": "",
            "url_or_reference": "official_download_see_manifest",
            "evidence_type": "HYDROMETEOROLOGICAL_CONTEXT_REVIEW_ONLY",
            "temporal_precision_claim": "DAY",
            "spatial_precision_claim": "ADMINISTRATIVE",
            "license_note": "PUBLIC_OFFICIAL_SOURCE_NEEDS_LICENSE_REVIEW",
            "manual_review_required": "true",
            "intake_status": "DRAFT_FROM_DOWNLOAD_NEEDS_REVIEW_GATE",
            "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)

    forbidden_guardrail_scan(rows, "v1sm_draft")
    write_csv_with_header(OUT_DRAFT, rows, DRAFT_FIELDS)
    write_schema_for(SCHEMA_D, DRAFT_FIELDS, "v1sm_draft")

    summary = [
        {"stat_key": "intake_draft_rows", "stat_value": str(len(rows))},
        {"stat_key": "manual_review_required", "stat_value": str(len(rows))},
        {"stat_key": "stage", "stat_value": "v1sm"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sm_summary")

    write_doc(DOC, "v1sm — Downloaded External Document Intake Adapter", [
        "## Objetivo",
        "Transformar downloads oficiais em drafts de intake compativeis com v1rb.",
        f"## Resultado\nDrafts: {len(rows)}. Todos requerem revisao manual.",
    ])
    print(f"[v1sm] drafts={len(rows)}")
    return {"drafts": len(rows)}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sm intake adapter").parse_args()
    run()
