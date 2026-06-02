"""REV-P v1so — Official evidence readiness gate.

Consolidates v1sg-v1sn to determine if sufficient official data exists
to start manual validation. Never calls v1rc automatically; never creates labels.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, read_csv_safe, downloads_enabled,
    forbidden_guardrail_scan,
)

ROOT = Path(__file__).resolve().parents[2]

IN_ENDPOINTS = _p("REVP_V1SO_IN_ENDPOINTS", DATASETS / "protocol_c_official_source_endpoint_summary_v1sg.csv")
IN_ORCH_SUMMARY = _p("REVP_V1SO_IN_ORCH", DATASETS / "protocol_c_official_download_orchestrator_summary_v1sl.csv")
IN_DRAFT_SUMMARY = _p("REVP_V1SO_IN_DRAFT", DATASETS / "protocol_c_downloaded_external_document_intake_summary_v1sm.csv")
IN_LICENSE_SUMMARY = _p("REVP_V1SO_IN_LICENSE", DATASETS / "protocol_c_official_data_provenance_license_summary_v1sn.csv")

OUT_GATE = _p("REVP_V1SO_OUT_GATE", DATASETS / "protocol_c_official_evidence_readiness_gate_v1so.csv")
OUT_SUMMARY = _p("REVP_V1SO_OUT_SUMMARY", DATASETS / "protocol_c_official_evidence_readiness_summary_v1so.csv")
SCHEMA_G = _p("REVP_V1SO_SCHEMA_G", SCHEMAS / "protocol_c_official_evidence_readiness_gate_v1so_schema.csv")
SCHEMA_S = _p("REVP_V1SO_SCHEMA_S", SCHEMAS / "protocol_c_official_evidence_readiness_summary_v1so_schema.csv")
DOC = _p("REVP_V1SO_DOC", DOCS / "revp_v1so_official_evidence_readiness_gate.md")

GATE_FIELDS = [
    "gate_check_id", "check_name", "status", "value", "severity",
    "review_only", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

ST_READY = "OFFICIAL_EVIDENCE_READY_FOR_MANUAL_INTAKE"
ST_QUEUE = "OFFICIAL_EVIDENCE_DOWNLOADS_DISABLED_QUEUE_READY"
ST_PARTIAL = "OFFICIAL_EVIDENCE_PARTIAL_READY_REVIEW_ONLY"
ST_BLOCKED = "OFFICIAL_EVIDENCE_BLOCKED_NO_SOURCES"
ST_GUARDRAIL = "OFFICIAL_EVIDENCE_GUARDRAIL_FAIL_CLOSED"

def _stat(rows: list[dict[str, str]], key: str, default: str = "0") -> str:
    for r in rows:
        if r.get("stat_key") == key:
            return r.get("stat_value", default)
    return default


def run(datasets: Path | None = None) -> dict[str, Any]:
    endpoints = read_csv_safe(IN_ENDPOINTS)
    orch = read_csv_safe(IN_ORCH_SUMMARY)
    draft = read_csv_safe(IN_DRAFT_SUMMARY)
    license_ = read_csv_safe(IN_LICENSE_SUMMARY)

    enabled = downloads_enabled()
    downloaded = int(_stat(orch, "files_downloaded", "0") or 0)
    drafts = int(_stat(draft, "intake_draft_rows", "0") or 0)
    lic_review = int(_stat(license_, "license_review_required", "0") or 0)
    ep_ready = int(_stat(endpoints, "endpoints_ready", "0") or 0)

    checks: list[dict[str, Any]] = []
    def chk(name, ok, val, sev):
        checks.append({
            "gate_check_id": f"V1SO_GC{len(checks):02d}", "check_name": name,
            "status": "PASS" if ok else "FAIL", "value": str(val),
            "severity": sev, "review_only": "true", "notes": "",
        })

    chk("endpoints_configured", ep_ready > 0, ep_ready, "high")
    chk("downloads_attempted", enabled or downloaded > 0, f"enabled={enabled} downloaded={downloaded}", "medium")
    chk("intake_drafts_generated", drafts > 0 or not enabled, drafts, "medium")
    chk("license_review_pending", True, lic_review, "medium")  # informational

    write_csv_with_header(OUT_GATE, checks, GATE_FIELDS)
    write_schema_for(SCHEMA_G, GATE_FIELDS, "v1so_gate")

    fails = sum(1 for c in checks if c["status"] == "FAIL" and c["severity"] in ("critical", "high"))
    if fails > 0 and ep_ready == 0:
        final = ST_BLOCKED
    elif downloaded > 0 and drafts > 0:
        final = ST_READY
    elif downloaded > 0:
        final = ST_PARTIAL
    elif not enabled and ep_ready > 0:
        final = ST_QUEUE
    else:
        final = ST_BLOCKED

    summary = [
        {"stat_key": "readiness_status", "stat_value": final},
        {"stat_key": "endpoints_ready", "stat_value": str(ep_ready)},
        {"stat_key": "files_downloaded", "stat_value": str(downloaded)},
        {"stat_key": "intake_drafts", "stat_value": str(drafts)},
        {"stat_key": "license_review_required", "stat_value": str(lic_review)},
        {"stat_key": "downloads_enabled", "stat_value": str(enabled).lower()},
        {"stat_key": "stage", "stat_value": "v1so"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1so_summary")

    write_doc(DOC, "v1so — Official Evidence Readiness Gate", [
        "## Objetivo",
        "Consolidar v1sg-v1sn e determinar se ha dados oficiais suficientes "
        "para iniciar validacao manual. Nao chama v1rc automaticamente.",
        f"## Resultado\nStatus: {final}. Downloaded: {downloaded}. Drafts: {drafts}.",
    ])
    print(f"[v1so] status={final} downloaded={downloaded} drafts={drafts}")
    return {"status": final, "downloaded": downloaded, "drafts": drafts}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1so evidence readiness gate").parse_args()
    run()
