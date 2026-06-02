"""REV-P v1sl — Official download orchestrator.

Consolidates download queues and manifests from v1sg/v1sh/v1sj/v1sk into a
unified view. Re-reads existing outputs rather than calling subprocesses.
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

IN_INMET_MANIFEST = _p("REVP_V1SL_IN_INMET", DATASETS / "protocol_c_inmet_download_manifest_v1sh.csv")
IN_ANA_MANIFEST = _p("REVP_V1SL_IN_ANA", DATASETS / "protocol_c_ana_hidroweb_download_manifest_v1sj.csv")
IN_INSTITUTIONAL = _p("REVP_V1SL_IN_INST", DATASETS / "protocol_c_institutional_document_discovery_queue_v1sk.csv")

OUT_MANIFEST = _p("REVP_V1SL_OUT_MANIFEST", DATASETS / "protocol_c_official_download_orchestrator_manifest_v1sl.csv")
OUT_SUMMARY = _p("REVP_V1SL_OUT_SUMMARY", DATASETS / "protocol_c_official_download_orchestrator_summary_v1sl.csv")
SCHEMA_M = _p("REVP_V1SL_SCHEMA_M", SCHEMAS / "protocol_c_official_download_orchestrator_manifest_v1sl_schema.csv")
SCHEMA_S = _p("REVP_V1SL_SCHEMA_S", SCHEMAS / "protocol_c_official_download_orchestrator_summary_v1sl_schema.csv")
DOC = _p("REVP_V1SL_DOC", DOCS / "revp_v1sl_official_download_orchestrator.md")

MANIFEST_FIELDS = [
    "orchestrator_id", "source_block", "source_name", "download_status",
    "downloaded", "file_size_bytes", "requires_manual", "review_only", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def run(datasets: Path | None = None) -> dict[str, Any]:
    inmet = read_csv_safe(IN_INMET_MANIFEST)
    ana = read_csv_safe(IN_ANA_MANIFEST)
    inst = read_csv_safe(IN_INSTITUTIONAL)
    enabled = downloads_enabled()

    rows: list[dict[str, Any]] = []
    idx = 0
    for r in inmet:
        rows.append({
            "orchestrator_id": f"V1SL_O{idx:04d}", "source_block": "v1sh_INMET",
            "source_name": r.get("source_name", "INMET"),
            "download_status": r.get("download_status", ""),
            "downloaded": r.get("downloaded", "false"),
            "file_size_bytes": r.get("file_size_bytes", "0"),
            "requires_manual": "false", "review_only": "true", "notes": "",
        })
        idx += 1
    for r in ana:
        rows.append({
            "orchestrator_id": f"V1SL_O{idx:04d}", "source_block": "v1sj_ANA",
            "source_name": r.get("source_name", "ANA"),
            "download_status": r.get("download_status", ""),
            "downloaded": r.get("downloaded", "false"),
            "file_size_bytes": r.get("file_size_bytes", "0"),
            "requires_manual": r.get("requires_manual_query", "true") if "requires_manual_query" in r else "true",
            "review_only": "true", "notes": "",
        })
        idx += 1
    for r in inst:
        rows.append({
            "orchestrator_id": f"V1SL_O{idx:04d}", "source_block": "v1sk_INSTITUTIONAL",
            "source_name": r.get("source_name", ""),
            "download_status": "MANUAL_QUEUE",
            "downloaded": "false", "file_size_bytes": "0",
            "requires_manual": "true", "review_only": "true", "notes": "",
        })
        idx += 1

    write_csv_with_header(OUT_MANIFEST, rows, MANIFEST_FIELDS)
    write_schema_for(SCHEMA_M, MANIFEST_FIELDS, "v1sl_manifest")

    downloaded_count = sum(1 for r in rows if r["downloaded"] == "true")
    total_bytes = sum(int(r.get("file_size_bytes", "0") or 0) for r in rows)
    manual = sum(1 for r in rows if r["requires_manual"] == "true")
    summary = [
        {"stat_key": "total_items", "stat_value": str(len(rows))},
        {"stat_key": "downloads_enabled", "stat_value": str(enabled).lower()},
        {"stat_key": "files_downloaded", "stat_value": str(downloaded_count)},
        {"stat_key": "total_bytes", "stat_value": str(total_bytes)},
        {"stat_key": "manual_required", "stat_value": str(manual)},
        {"stat_key": "inmet_items", "stat_value": str(len(inmet))},
        {"stat_key": "ana_items", "stat_value": str(len(ana))},
        {"stat_key": "institutional_items", "stat_value": str(len(inst))},
        {"stat_key": "stage", "stat_value": "v1sl"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sl_summary")

    write_doc(DOC, "v1sl — Official Download Orchestrator", [
        "## Objetivo",
        "Consolidar queues e manifests de v1sh/v1sj/v1sk num unico manifesto orquestrado.",
        "## Resultado",
        f"Total: {len(rows)}. Downloaded: {downloaded_count}. Manual: {manual}. "
        f"Bytes: {total_bytes}.",
    ])
    print(f"[v1sl] total={len(rows)} downloaded={downloaded_count} bytes={total_bytes}")
    return {"total": len(rows), "downloaded": downloaded_count, "bytes": total_bytes}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sl download orchestrator").parse_args()
    run()
