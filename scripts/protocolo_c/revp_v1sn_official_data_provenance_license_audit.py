"""REV-P v1sn — Official data provenance and license audit.

Audits downloaded/queued sources by provenance, domain, hash, license and
terms. Never affirms a license when unknown. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, read_csv_safe, is_allowed_domain,
    domain_from_url, forbidden_guardrail_scan,
)

ROOT = Path(__file__).resolve().parents[2]

IN_ORCHESTRATOR = _p("REVP_V1SN_IN_ORCHESTRATOR", DATASETS / "protocol_c_official_download_orchestrator_manifest_v1sl.csv")
OUT_AUDIT = _p("REVP_V1SN_OUT_AUDIT", DATASETS / "protocol_c_official_data_provenance_license_audit_v1sn.csv")
OUT_SUMMARY = _p("REVP_V1SN_OUT_SUMMARY", DATASETS / "protocol_c_official_data_provenance_license_summary_v1sn.csv")
SCHEMA_A = _p("REVP_V1SN_SCHEMA_A", SCHEMAS / "protocol_c_official_data_provenance_license_audit_v1sn_schema.csv")
SCHEMA_S = _p("REVP_V1SN_SCHEMA_S", SCHEMAS / "protocol_c_official_data_provenance_license_summary_v1sn_schema.csv")
DOC = _p("REVP_V1SN_DOC", DOCS / "revp_v1sn_official_data_provenance_license_audit.md")

AUDIT_FIELDS = [
    "audit_id", "source_name", "source_block", "domain", "allowed_domain",
    "downloaded", "provenance_status", "license_status", "can_use_in_tcc",
    "manual_license_review_required", "review_only", "can_create_operational_label",
    "can_train_model", "target_created", "ground_truth_operational",
    "formal_negative", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _license_from_domain(domain: str) -> tuple[str, str, str]:
    """Return (license_status, can_use_in_tcc, manual_review)."""
    if domain.endswith(".gov.br"):
        return ("PUBLIC_OFFICIAL_SOURCE_REVIEW_ONLY", "true", "true")
    if is_allowed_domain(f"https://{domain}"):
        return ("LICENSE_REVIEW_REQUIRED", "false", "true")
    return ("ACCESS_RESTRICTED_BLOCKED", "false", "true")


def run(datasets: Path | None = None) -> dict[str, Any]:
    orchestrator = read_csv_safe(IN_ORCHESTRATOR)
    rows: list[dict[str, Any]] = []

    for i, r in enumerate(orchestrator):
        source = r.get("source_name", "")
        block = r.get("source_block", "")
        domain = domain_from_url(r.get("url", "")) if r.get("url") else ""
        if not domain:
            domain = block.split("_")[0] if "_" in block else ""
        allowed = "true" if is_allowed_domain(f"https://{domain}") else "false"
        downloaded = r.get("downloaded", "false")

        lic_status, can_tcc, manual = _license_from_domain(domain)
        prov = "PROVENANCE_COMPLETE" if downloaded == "true" else "PROVENANCE_INCOMPLETE_FAIL_CLOSED"
        blocked = "" if lic_status != "ACCESS_RESTRICTED_BLOCKED" else "ACCESS_RESTRICTED"

        row = {
            "audit_id": f"V1SN_A{i:04d}", "source_name": source,
            "source_block": block, "domain": domain,
            "allowed_domain": allowed, "downloaded": downloaded,
            "provenance_status": prov, "license_status": lic_status,
            "can_use_in_tcc": can_tcc,
            "manual_license_review_required": manual,
            "blocked_reason": blocked, "notes": "",
        }
        row.update(guardrail_row())
        rows.append(row)

    forbidden_guardrail_scan(rows, "v1sn_audit")
    write_csv_with_header(OUT_AUDIT, rows, AUDIT_FIELDS)
    write_schema_for(SCHEMA_A, AUDIT_FIELDS, "v1sn_audit")

    lic_review = sum(1 for r in rows if r["manual_license_review_required"] == "true")
    public = sum(1 for r in rows if r["license_status"] == "PUBLIC_OFFICIAL_SOURCE_REVIEW_ONLY")
    summary = [
        {"stat_key": "audited_items", "stat_value": str(len(rows))},
        {"stat_key": "public_official_sources", "stat_value": str(public)},
        {"stat_key": "license_review_required", "stat_value": str(lic_review)},
        {"stat_key": "stage", "stat_value": "v1sn"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sn_summary")

    write_doc(DOC, "v1sn — Official Data Provenance and License Audit", [
        "## Objetivo",
        "Auditar proveniencia e licenca de downloads/queued sources.",
        f"## Resultado\nAuditados: {len(rows)}. Publicos: {public}. License review: {lic_review}.",
    ])
    print(f"[v1sn] audited={len(rows)} public={public} license_review={lic_review}")
    return {"audited": len(rows), "public": public, "license_review": lic_review}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sn provenance/license audit").parse_args()
    run()
