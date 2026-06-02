"""REV-P v1sj — ANA/HidroWeb acquisition planner/downloader.

Registers ANA/HidroWeb/Telemetria endpoints and attempts light discovery.
When no simple direct URL exists, generates a manual query queue. Review-only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, forbidden_guardrail_scan,
    downloads_enabled, download_text, is_allowed_domain, raw_root,
    download_file, ensure_dir, safe_relpath, hash_short,
)

ROOT = Path(__file__).resolve().parents[2]

OUT_QUEUE = _p("REVP_V1SJ_OUT_QUEUE", DATASETS / "protocol_c_ana_hidroweb_acquisition_queue_v1sj.csv")
OUT_MANIFEST = _p("REVP_V1SJ_OUT_MANIFEST", DATASETS / "protocol_c_ana_hidroweb_download_manifest_v1sj.csv")
OUT_SUMMARY = _p("REVP_V1SJ_OUT_SUMMARY", DATASETS / "protocol_c_ana_hidroweb_acquisition_summary_v1sj.csv")
SCHEMA_Q = _p("REVP_V1SJ_SCHEMA_Q", SCHEMAS / "protocol_c_ana_hidroweb_acquisition_queue_v1sj_schema.csv")
SCHEMA_M = _p("REVP_V1SJ_SCHEMA_M", SCHEMAS / "protocol_c_ana_hidroweb_download_manifest_v1sj_schema.csv")
SCHEMA_S = _p("REVP_V1SJ_SCHEMA_S", SCHEMAS / "protocol_c_ana_hidroweb_acquisition_summary_v1sj_schema.csv")
DOC = _p("REVP_V1SJ_DOC", DOCS / "revp_v1sj_ana_hidroweb_acquisition.md")

QUEUE_FIELDS = [
    "queue_id", "source_name", "region", "endpoint_url", "query_type",
    "auto_download_allowed", "requires_manual_query", "priority",
    "review_only", "blocked_reason", "notes",
]
MANIFEST_FIELDS = [
    "download_id", "source_name", "region", "url", "downloaded",
    "download_attempted", "raw_relative_path", "file_sha256_short",
    "file_size_bytes", "download_status", "review_only",
    "can_create_operational_label", "can_train_model", "target_created",
    "ground_truth_operational", "formal_negative", "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]

# ANA/HidroWeb requires interactive form or SOAP API; direct CSV download
# is not trivially automatable. We register the endpoints and generate
# manual query suggestions.
_ENDPOINTS = [
    ("ANA_HIDROWEB_SERIES", "ANA HidroWeb", "ALL",
     "https://www.snirh.gov.br/hidroweb/seriesHistoricas",
     "WEB_FORM", "false", "true", "P0",
     "HidroWeb requer consulta interativa por estacao/periodo"),
    ("ANA_TELEMETRIA_RECIFE", "ANA Telemetria", "RECIFE",
     "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?codEstacao=39270000&dataInicio=01/01/2020&dataFim=31/12/2024",
     "SOAP_API", "true", "false", "P0",
     "Exemplo estacao Recife; pode precisar ajuste de codigo"),
    ("ANA_TELEMETRIA_PET", "ANA Telemetria", "PET",
     "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?codEstacao=58974000&dataInicio=01/01/2020&dataFim=31/12/2024",
     "SOAP_API", "true", "false", "P1",
     "Exemplo estacao Piabanha/Petropolis"),
    ("ANA_TELEMETRIA_CWB", "ANA Telemetria", "CURITIBA",
     "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos?codEstacao=65017006&dataInicio=01/01/2020&dataFim=31/12/2024",
     "SOAP_API", "true", "false", "P1",
     "Exemplo estacao Bacia Iguacu/Curitiba"),
]


def run(datasets: Path | None = None) -> dict[str, Any]:
    enabled = downloads_enabled()
    queue: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []

    for i, ep in enumerate(_ENDPOINTS):
        eid, name, region, url, qtype, auto, manual, prio, notes = ep
        blocked = ""
        if not is_allowed_domain(url):
            blocked = "DOMAIN_NOT_ALLOWED"
        elif auto == "false":
            blocked = "ANA_MANUAL_QUERY_REQUIRED"
        qrow = {
            "queue_id": f"V1SJ_Q{i:03d}", "source_name": name, "region": region,
            "endpoint_url": url, "query_type": qtype,
            "auto_download_allowed": auto, "requires_manual_query": manual,
            "priority": prio, "review_only": "true",
            "blocked_reason": blocked, "notes": notes,
        }
        queue.append(qrow)

        mrow = {
            "download_id": f"V1SJ_DL_{i:03d}", "source_name": name,
            "region": region, "url": url, "downloaded": "false",
            "download_attempted": "false", "raw_relative_path": "",
            "file_sha256_short": "", "file_size_bytes": "0",
            "download_status": "ANA_DOWNLOAD_DISABLED_QUEUE_ONLY",
            "blocked_reason": blocked or ("DOWNLOADS_DISABLED" if not enabled else ""),
            "notes": notes,
        }
        mrow.update(guardrail_row())

        if enabled and auto == "true" and not blocked:
            dest = ensure_dir(raw_root() / "ana") / f"ana_{region.lower()}_{i:03d}.xml"
            result = download_file(url, dest, max_bytes=50_000_000)
            ok = result["downloaded"] == "true"
            mrow.update({
                "downloaded": result["downloaded"],
                "download_attempted": result.get("download_attempted", "false"),
                "download_status": "ANA_DOWNLOADED_OK" if ok else result["download_status"],
                "file_sha256_short": result["file_sha256_short"],
                "file_size_bytes": result["file_size_bytes"],
                "raw_relative_path": safe_relpath(dest) if ok else "",
                "blocked_reason": "" if ok else result["download_status"],
            })
        manifest.append(mrow)

    forbidden_guardrail_scan(manifest, "v1sj_manifest")
    write_csv_with_header(OUT_QUEUE, queue, QUEUE_FIELDS)
    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_schema_for(SCHEMA_Q, QUEUE_FIELDS, "v1sj_queue")
    write_schema_for(SCHEMA_M, MANIFEST_FIELDS, "v1sj_manifest")

    downloaded = sum(1 for r in manifest if r["downloaded"] == "true")
    manual_req = sum(1 for q in queue if q["requires_manual_query"] == "true")
    summary = [
        {"stat_key": "queue_size", "stat_value": str(len(queue))},
        {"stat_key": "downloads_enabled", "stat_value": str(enabled).lower()},
        {"stat_key": "files_downloaded", "stat_value": str(downloaded)},
        {"stat_key": "manual_queries_required", "stat_value": str(manual_req)},
        {"stat_key": "stage", "stat_value": "v1sj"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sj_summary")

    write_doc(DOC, "v1sj — ANA/HidroWeb Acquisition Planner", [
        "## Objetivo",
        "Registrar endpoints ANA/HidroWeb/Telemetria e gerar queue de aquisicao. "
        "HidroWeb requer consulta interativa; Telemetria usa SOAP API direta.",
        "## Resultado",
        f"Queue: {len(queue)}. Downloaded: {downloaded}. Manual: {manual_req}.",
    ])
    print(f"[v1sj] queue={len(queue)} downloaded={downloaded} manual={manual_req}")
    return {"queue": len(queue), "downloaded": downloaded, "manual": manual_req}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sj ANA/HidroWeb acquisition").parse_args()
    run()
