"""REV-P v1sx — Hydrometeorological context guardrail audit.

Scans all v1sr-v1sw CSVs for forbidden field values, absolute paths and
local_runs exposure. Returns a pass/fail audit table.
"""
from __future__ import annotations
import argparse
import glob
from pathlib import Path
from typing import Any

from revp_v1sr_v1sz_hydromet_context_common import (
    DATASETS, DOCS, SCHEMAS, _p,
    write_csv_with_header, write_schema, write_doc,
    read_csv_safe, scan_guardrails,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_AUDIT = _p("REVP_V1SX_OUT_AUDIT", DATASETS / "protocol_c_hydromet_context_guardrail_audit_v1sx.csv")
OUT_SUM   = _p("REVP_V1SX_OUT_SUM",   DATASETS / "protocol_c_hydromet_context_guardrail_summary_v1sx.csv")
SCHEMA_A  = _p("REVP_V1SX_SCHEMA_A",  SCHEMAS  / "protocol_c_hydromet_context_guardrail_audit_v1sx_schema.csv")
SCHEMA_S  = _p("REVP_V1SX_SCHEMA_S",  SCHEMAS  / "protocol_c_hydromet_context_guardrail_summary_v1sx_schema.csv")
DOC       = _p("REVP_V1SX_DOC",       DOCS     / "revp_v1sx_hydromet_context_guardrail_audit.md")

AUDIT_FIELDS = ["audit_id", "csv_file", "row_count", "violations_found",
                "violation_details", "audit_status", "notes"]
SUM_FIELDS   = ["stat_key", "stat_value"]

# All v1sr-v1sw output CSVs (pattern match in datasets/)
_V1SR_V1SW_PATTERNS = [
    "protocol_c_inmet_station_region_proximity_v1sr.csv",
    "protocol_c_event_date_windows_v1ss.csv",
    "protocol_c_inmet_precipitation_event_window_context_v1st.csv",
    "protocol_c_rolling_rainfall_context_features_v1su.csv",
    "protocol_c_hydromet_evidence_intake_crosswalk_v1sv.csv",
    "protocol_c_tcc_table_hydromet_station_coverage_v1sw.csv",
    "protocol_c_tcc_table_hydromet_event_windows_v1sw.csv",
    "protocol_c_tcc_table_hydromet_context_limitations_v1sw.csv",
]


def run() -> dict[str, Any]:
    audit_rows: list[dict[str, Any]] = []
    total_violations = 0

    for i, fname in enumerate(_V1SR_V1SW_PATTERNS):
        path = DATASETS / fname
        rows = read_csv_safe(path)
        if not rows:
            audit_rows.append({
                "audit_id": f"V1SX_A{i:02d}", "csv_file": fname,
                "row_count": "0", "violations_found": "0",
                "violation_details": "",
                "audit_status": "EMPTY_OR_MISSING",
                "notes": "",
            })
            continue

        issues = scan_guardrails(rows, fname)
        n_viol = len(issues)
        total_violations += n_viol
        audit_rows.append({
            "audit_id":          f"V1SX_A{i:02d}",
            "csv_file":          fname,
            "row_count":         str(len(rows)),
            "violations_found":  str(n_viol),
            "violation_details": "; ".join(issues[:3]) if issues else "",
            "audit_status":      "PASS" if n_viol == 0 else "FAIL",
            "notes":             "",
        })

    write_csv_with_header(OUT_AUDIT, audit_rows, AUDIT_FIELDS)
    write_schema(SCHEMA_A, AUDIT_FIELDS, "v1sx_audit")

    passes = sum(1 for r in audit_rows if r["audit_status"] == "PASS")
    fails  = sum(1 for r in audit_rows if r["audit_status"] == "FAIL")
    final  = "GUARDRAIL_PASS_ALL" if fails == 0 else "GUARDRAIL_FAIL_CLOSED"
    summary = [
        {"stat_key": "files_audited",       "stat_value": str(len(audit_rows))},
        {"stat_key": "files_pass",          "stat_value": str(passes)},
        {"stat_key": "files_fail",          "stat_value": str(fails)},
        {"stat_key": "total_violations",    "stat_value": str(total_violations)},
        {"stat_key": "audit_status",        "stat_value": final},
        {"stat_key": "stage",               "stat_value": "v1sx"},
    ]
    write_csv_with_header(OUT_SUM, summary, SUM_FIELDS)
    write_schema(SCHEMA_S, SUM_FIELDS, "v1sx_summary")

    write_doc(DOC, "v1sx — Hydromet Context Guardrail Audit", [
        "## Objetivo",
        "Varrer todos os CSVs v1sr-v1sw por violacoes de guardrail: path absoluto, "
        "local_runs, labels/targets/ground_truth/formal_negative/dino/absence=true.",
        f"## Resultado\nAuditados: {len(audit_rows)}. Pass: {passes}. Fail: {fails}. Status: {final}.",
    ])
    print(f"[v1sx] audited={len(audit_rows)} pass={passes} fail={fails} violations={total_violations}")
    return {"audited": len(audit_rows), "pass": passes, "fail": fails,
            "violations": total_violations, "status": final}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sx guardrail audit").parse_args()
    run()
