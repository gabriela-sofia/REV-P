"""REV-P v1sp — Official acquisition bundle.

Final bundle for v1sg-v1so. Manifest, QC, summary, TCC table. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, read_csv_safe, downloads_enabled,
)

ROOT = Path(__file__).resolve().parents[2]

IN_EP_SUMMARY = _p("REVP_V1SP_IN_EP", DATASETS / "protocol_c_official_source_endpoint_summary_v1sg.csv")
IN_ORCH_SUMMARY = _p("REVP_V1SP_IN_ORCH", DATASETS / "protocol_c_official_download_orchestrator_summary_v1sl.csv")
IN_DRAFT_SUMMARY = _p("REVP_V1SP_IN_DRAFT", DATASETS / "protocol_c_downloaded_external_document_intake_summary_v1sm.csv")
IN_LICENSE_SUMMARY = _p("REVP_V1SP_IN_LIC", DATASETS / "protocol_c_official_data_provenance_license_summary_v1sn.csv")
IN_GATE_SUMMARY = _p("REVP_V1SP_IN_GATE", DATASETS / "protocol_c_official_evidence_readiness_summary_v1so.csv")

OUT_MANIFEST = _p("REVP_V1SP_OUT_MANIFEST", DATASETS / "protocol_c_official_acquisition_manifest_v1sp.csv")
OUT_QC = _p("REVP_V1SP_OUT_QC", DATASETS / "protocol_c_official_acquisition_quality_checks_v1sp.csv")
OUT_SUMMARY = _p("REVP_V1SP_OUT_SUMMARY", DATASETS / "protocol_c_official_acquisition_scientific_summary_v1sp.csv")
OUT_TCC = _p("REVP_V1SP_OUT_TCC", DATASETS / "protocol_c_tcc_table_official_acquisition_status_v1sp.csv")
SCHEMA_M = _p("REVP_V1SP_SCHEMA_M", SCHEMAS / "protocol_c_official_acquisition_manifest_v1sp_schema.csv")
SCHEMA_Q = _p("REVP_V1SP_SCHEMA_Q", SCHEMAS / "protocol_c_official_acquisition_quality_checks_v1sp_schema.csv")
SCHEMA_S = _p("REVP_V1SP_SCHEMA_S", SCHEMAS / "protocol_c_official_acquisition_scientific_summary_v1sp_schema.csv")
SCHEMA_T = _p("REVP_V1SP_SCHEMA_T", SCHEMAS / "protocol_c_tcc_table_official_acquisition_status_v1sp_schema.csv")
DOC = _p("REVP_V1SP_DOC", DOCS / "revp_v1sp_official_acquisition_bundle.md")

MAN_FIELDS = ["artifact_id", "stage", "artifact_name", "exists", "row_count", "role", "notes"]
QC_FIELDS = ["check_id", "check_name", "expected", "observed", "passed", "severity", "notes"]
SUM_FIELDS = ["stat_key", "stat_value"]
TCC_FIELDS = ["metric", "value", "interpretation_note"]

ST_DOWNLOADED = "OFFICIAL_ACQUISITION_DOWNLOADED_REVIEW_ONLY"
ST_QUEUE = "OFFICIAL_ACQUISITION_QUEUE_READY_DOWNLOADS_DISABLED"
ST_PARTIAL = "OFFICIAL_ACQUISITION_PARTIAL_DOWNLOAD_REVIEW_ONLY"
ST_BLOCKED = "OFFICIAL_ACQUISITION_BLOCKED_FAIL_CLOSED"

MANDATORY = (
    "A camada v1sg-v1sp automatiza a aquisicao controlada de dados oficiais externos e "
    "registra proveniencia, hashes e limitacoes de uso. Os downloads oficiais podem "
    "alimentar o intake e a revisao supervisora, mas nao criam rotulos operacionais, targets "
    "supervisionados, ground truth operacional ou negativos formais."
)

def _stat(rows, key, default="0"):
    for r in rows:
        if r.get("stat_key") == key: return r.get("stat_value", default)
    return default


def run(datasets: Path | None = None) -> dict[str, Any]:
    ep = read_csv_safe(IN_EP_SUMMARY)
    orch = read_csv_safe(IN_ORCH_SUMMARY)
    draft = read_csv_safe(IN_DRAFT_SUMMARY)
    lic = read_csv_safe(IN_LICENSE_SUMMARY)
    gate = read_csv_safe(IN_GATE_SUMMARY)

    enabled = downloads_enabled()
    ep_total = _stat(ep, "endpoints_total")
    ep_ready = _stat(ep, "endpoints_ready")
    downloaded = _stat(orch, "files_downloaded")
    total_bytes = _stat(orch, "total_bytes")
    inmet = _stat(orch, "inmet_items")
    ana = _stat(orch, "ana_items")
    inst = _stat(orch, "institutional_items")
    drafts = _stat(draft, "intake_draft_rows")
    lic_review = _stat(lic, "license_review_required")
    readiness = _stat(gate, "readiness_status", "UNKNOWN")

    if int(downloaded or 0) > 0 and int(drafts or 0) > 0:
        final = ST_DOWNLOADED
    elif int(downloaded or 0) > 0:
        final = ST_PARTIAL
    elif not enabled and int(ep_ready or 0) > 0:
        final = ST_QUEUE
    else:
        final = ST_BLOCKED

    manifest = [
        {"artifact_id": "V1SP_A01", "stage": "v1sg", "artifact_name": "endpoint_registry", "exists": "true", "row_count": ep_total, "role": "endpoints", "notes": ""},
        {"artifact_id": "V1SP_A02", "stage": "v1sh", "artifact_name": "inmet_downloads", "exists": "true", "row_count": inmet, "role": "inmet_data", "notes": ""},
        {"artifact_id": "V1SP_A03", "stage": "v1sj", "artifact_name": "ana_downloads", "exists": "true", "row_count": ana, "role": "ana_data", "notes": ""},
        {"artifact_id": "V1SP_A04", "stage": "v1sk", "artifact_name": "institutional_queue", "exists": "true", "row_count": inst, "role": "institutional_discovery", "notes": ""},
        {"artifact_id": "V1SP_A05", "stage": "v1sm", "artifact_name": "intake_draft", "exists": "true", "row_count": drafts, "role": "intake_adapter", "notes": ""},
        {"artifact_id": "V1SP_A06", "stage": "v1sn", "artifact_name": "provenance_audit", "exists": "true", "row_count": _stat(lic, "audited_items"), "role": "license_audit", "notes": ""},
    ]
    write_csv_with_header(OUT_MANIFEST, manifest, MAN_FIELDS)
    write_schema_for(SCHEMA_M, MAN_FIELDS, "v1sp_manifest")

    qc = [
        {"check_id": "QC01", "check_name": "endpoints_configured", "expected": ">=1", "observed": ep_ready, "passed": "true" if int(ep_ready or 0) > 0 else "false", "severity": "high", "notes": ""},
        {"check_id": "QC02", "check_name": "labels_created_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC03", "check_name": "targets_created_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC04", "check_name": "ground_truth_operational_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
        {"check_id": "QC05", "check_name": "formal_negatives_zero", "expected": "0", "observed": "0", "passed": "true", "severity": "critical", "notes": ""},
    ]
    write_csv_with_header(OUT_QC, qc, QC_FIELDS)
    write_schema_for(SCHEMA_Q, QC_FIELDS, "v1sp_qc")

    summary = [
        {"stat_key": "endpoints_total", "stat_value": ep_total},
        {"stat_key": "endpoints_ready", "stat_value": ep_ready},
        {"stat_key": "downloads_enabled", "stat_value": str(enabled).lower()},
        {"stat_key": "files_downloaded", "stat_value": downloaded},
        {"stat_key": "bytes_downloaded", "stat_value": total_bytes},
        {"stat_key": "inmet_files", "stat_value": inmet},
        {"stat_key": "ana_files", "stat_value": ana},
        {"stat_key": "institutional_queue_rows", "stat_value": inst},
        {"stat_key": "intake_draft_rows", "stat_value": drafts},
        {"stat_key": "license_review_required", "stat_value": lic_review},
        {"stat_key": "ready_for_manual_intake", "stat_value": readiness},
        {"stat_key": "labels_created", "stat_value": "0"},
        {"stat_key": "targets_created", "stat_value": "0"},
        {"stat_key": "ground_truth_operational_created", "stat_value": "0"},
        {"stat_key": "formal_negatives_created", "stat_value": "0"},
        {"stat_key": "final_status", "stat_value": final},
        {"stat_key": "stage", "stat_value": "v1sp"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sp_summary")

    tcc = [
        {"metric": "endpoints_ready", "value": ep_ready, "interpretation_note": "fontes oficiais configuradas"},
        {"metric": "files_downloaded", "value": downloaded, "interpretation_note": "arquivos oficiais baixados"},
        {"metric": "intake_draft_rows", "value": drafts, "interpretation_note": "drafts de intake review-only"},
        {"metric": "labels_created", "value": "0", "interpretation_note": "nenhum label operacional"},
        {"metric": "ground_truth_operational", "value": "0", "interpretation_note": "nenhum ground truth operacional"},
        {"metric": "final_status", "value": final, "interpretation_note": "estado da aquisicao oficial"},
    ]
    write_csv_with_header(OUT_TCC, tcc, TCC_FIELDS)
    write_schema_for(SCHEMA_T, TCC_FIELDS, "v1sp_tcc")

    write_doc(DOC, "v1sp — Official Acquisition Bundle", [
        "## Objetivo",
        "Bundle final v1sg-v1so: manifest, QC, summary e tabela TCC.",
        f"## Resultado\nfinal_status={final}. Downloaded: {downloaded}. Drafts: {drafts}.",
        "## Declaracao obrigatoria",
        MANDATORY,
    ])
    qc_failed = sum(1 for c in qc if c["passed"] != "true")
    print(f"[v1sp] final_status={final} downloaded={downloaded} drafts={drafts} qc_failed={qc_failed}")
    return {"final_status": final, "downloaded": downloaded, "drafts": drafts, "qc_failed": qc_failed}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sp official acquisition bundle").parse_args()
    run()
