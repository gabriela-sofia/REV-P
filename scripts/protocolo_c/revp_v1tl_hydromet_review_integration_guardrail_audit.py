"""REV-P v1tl — Hydromet review integration guardrail audit.

Audits all v1tg-v1tk CSVs for standard violations plus the two new
hydromet-specific fields (hydromet_validates_event, hydromet_is_negative_evidence).
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1tg_v1tm_hydromet_review_integration_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    read_csv_safe, write_csv_with_header, write_schema, write_doc,
    scan_guardrails_extended,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUD  = _p("REVP_V1TL_OUT_AUD",  DATASETS / "protocol_c_hydromet_review_integration_guardrail_audit_v1tl.csv")
OUT_SUM  = _p("REVP_V1TL_OUT_SUM",  DATASETS / "protocol_c_hydromet_review_integration_guardrail_summary_v1tl.csv")
SCHEMA_A = _p("REVP_V1TL_SCHEMA_A", SCHEMAS  / "protocol_c_hydromet_review_integration_guardrail_audit_v1tl_schema.csv")
SCHEMA_S = _p("REVP_V1TL_SCHEMA_S", SCHEMAS  / "protocol_c_hydromet_review_integration_guardrail_summary_v1tl_schema.csv")
DOC      = _p("REVP_V1TL_DOC",      DOCS     / "revp_v1tl_hydromet_review_integration_guardrail_audit.md")

AUDIT_FIELDS = ["audit_id", "csv_file", "row_count", "violations_found",
                "violation_details", "audit_status", "notes"]
SUM_FIELDS   = ["stat_key", "stat_value"]

_TARGETS = [
    "protocol_c_hydromet_evidence_packet_registry_v1tg.csv",
    "protocol_c_hydromet_evidence_packet_summary_v1tg.csv",
    "protocol_c_hydromet_double_review_addendum_manifest_v1th.csv",
    "protocol_c_hydromet_double_review_addendum_forms_v1th.csv",
    "protocol_c_hydromet_review_scores_v1ti.csv",
    "protocol_c_supervisor_hydromet_addendum_v1tj.csv",
    "protocol_c_tcc_table_hydromet_review_packets_v1tk.csv",
    "protocol_c_tcc_table_hydromet_supervisor_addendum_v1tk.csv",
    "protocol_c_tcc_table_hydromet_overclaim_controls_v1tk.csv",
]


def run() -> dict[str, Any]:
    audit_rows: list[dict[str, Any]] = []
    total_violations = 0

    for i, fname in enumerate(_TARGETS):
        rows = read_csv_safe(DATASETS / fname)
        if not rows:
            audit_rows.append({
                "audit_id": f"V1TL_A{i:02d}", "csv_file": fname,
                "row_count": "0", "violations_found": "0",
                "violation_details": "", "audit_status": "EMPTY_OR_MISSING", "notes": "",
            })
            continue

        issues = scan_guardrails_extended(rows, fname)
        n_viol = len(issues)
        total_violations += n_viol
        audit_rows.append({
            "audit_id":         f"V1TL_A{i:02d}",
            "csv_file":         fname,
            "row_count":        str(len(rows)),
            "violations_found": str(n_viol),
            "violation_details": "; ".join(issues[:3]) if issues else "",
            "audit_status":     "PASS" if n_viol == 0 else "FAIL",
            "notes":            "",
        })

    write_csv_with_header(OUT_AUD, audit_rows, AUDIT_FIELDS)
    write_schema(SCHEMA_A, AUDIT_FIELDS, "v1tl_audit")

    passes = sum(1 for r in audit_rows if r["audit_status"] == "PASS")
    fails  = sum(1 for r in audit_rows if r["audit_status"] == "FAIL")
    final  = "GUARDRAIL_PASS_ALL" if fails == 0 else "GUARDRAIL_FAIL_CLOSED"
    summary = [
        {"stat_key": "files_audited",    "stat_value": str(len(audit_rows))},
        {"stat_key": "files_pass",       "stat_value": str(passes)},
        {"stat_key": "files_fail",       "stat_value": str(fails)},
        {"stat_key": "total_violations", "stat_value": str(total_violations)},
        {"stat_key": "audit_status",     "stat_value": final},
        {"stat_key": "stage",            "stat_value": "v1tl"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1tl_summary")

    write_doc(DOC, "v1tl — Hydromet Review Integration Guardrail Audit", [
        "## Objetivo",
        "Auditar todos os CSVs v1tg-v1tk: path absoluto, local_runs, "
        "labels/targets/ground_truth/formal_negative/dino/absence/hydromet_validates/hydromet_negative.",
        f"## Resultado\nAuditados: {len(audit_rows)}. Pass: {passes}. Status: {final}.",
    ])
    print(f"[v1tl] audited={len(audit_rows)} pass={passes} fail={fails} violations={total_violations}")
    return {"audited": len(audit_rows), "pass": passes, "fail": fails,
            "violations": total_violations, "status": final}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1tl guardrail audit").parse_args()
    run()
