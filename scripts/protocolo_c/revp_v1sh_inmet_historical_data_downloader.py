"""REV-P v1sh — INMET historical data downloader.

Downloads annual ZIP/CSV files from INMET's official historical data page.
When REVP_ENABLE_OFFICIAL_DOWNLOADS is false, generates queue only. Saves
raw files to data/external_raw/inmet/historical/. Review-only.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Any

from revp_v1sg_v1sz_official_download_common import (
    DATASETS, DOCS, SCHEMAS, _p, guardrail_row, write_csv_with_header,
    write_doc, write_schema_for, forbidden_guardrail_scan,
    downloads_enabled, download_mode, raw_root, max_files, max_bytes_per_file,
    download_file, download_text, hash_short, is_allowed_domain,
    classify_document_type, sha256_file_short, ensure_dir, safe_relpath,
)

ROOT = Path(__file__).resolve().parents[2]
INMET_BASE = "https://portal.inmet.gov.br/uploads/dadoshistoricos/"

OUT_QUEUE = _p("REVP_V1SH_OUT_QUEUE", DATASETS / "protocol_c_inmet_download_queue_v1sh.csv")
OUT_MANIFEST = _p("REVP_V1SH_OUT_MANIFEST", DATASETS / "protocol_c_inmet_download_manifest_v1sh.csv")
OUT_SUMMARY = _p("REVP_V1SH_OUT_SUMMARY", DATASETS / "protocol_c_inmet_download_summary_v1sh.csv")
SCHEMA_Q = _p("REVP_V1SH_SCHEMA_Q", SCHEMAS / "protocol_c_inmet_download_queue_v1sh_schema.csv")
SCHEMA_M = _p("REVP_V1SH_SCHEMA_M", SCHEMAS / "protocol_c_inmet_download_manifest_v1sh_schema.csv")
SCHEMA_S = _p("REVP_V1SH_SCHEMA_S", SCHEMAS / "protocol_c_inmet_download_summary_v1sh_schema.csv")
DOC = _p("REVP_V1SH_DOC", DOCS / "revp_v1sh_inmet_historical_data_downloader.md")

QUEUE_FIELDS = ["queue_id", "source_name", "year", "url", "domain", "priority",
                "expected_file_type", "download_enabled", "review_only", "notes"]
MANIFEST_FIELDS = [
    "download_id", "source_name", "year", "url", "domain", "downloaded",
    "download_attempted", "raw_relative_path", "file_sha256_short",
    "file_size_bytes", "content_type", "http_status", "download_status",
    "review_only", "can_create_operational_label", "can_train_model",
    "target_created", "ground_truth_operational", "formal_negative",
    "blocked_reason", "notes",
]
SUM_FIELDS = ["stat_key", "stat_value"]


def _discover_annual_links() -> list[tuple[str, str]]:
    """Try to discover annual ZIP links from the INMET historical data page."""
    links: list[tuple[str, str]] = []
    # Try to fetch the listing page
    text, status = download_text(INMET_BASE, timeout=30, max_bytes=500_000)
    if status == 200 and text:
        for m in re.finditer(r'href=["\']([^"\']*?(\d{4})[^"\']*?\.zip)["\']', text, re.IGNORECASE):
            url = m.group(1)
            year = m.group(2)
            if not url.startswith("http"):
                url = INMET_BASE + url.lstrip("./")
            links.append((year, url))
    # Fallback: generate known pattern
    if not links:
        for year in range(2000, 2027):
            url = f"{INMET_BASE}{year}.zip"
            links.append((str(year), url))
    return sorted(set(links))


def _filter_years(links: list[tuple[str, str]]) -> list[tuple[str, str]]:
    mode = download_mode()
    if mode == "minimal":
        target_years = {str(y) for y in range(2020, 2027)}
        return [(y, u) for y, u in links if y in target_years]
    if mode == "regional":
        target_years = {str(y) for y in range(2000, 2027)}
        return [(y, u) for y, u in links if y in target_years]
    return links  # full_official_bounded


def run(datasets: Path | None = None) -> dict[str, Any]:
    all_links = _discover_annual_links()
    filtered = _filter_years(all_links)
    enabled = downloads_enabled()

    queue: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    total_bytes = 0
    file_budget = max_files()
    dest_dir = raw_root() / "inmet" / "historical"

    for i, (year, url) in enumerate(filtered):
        dest = dest_dir / f"inmet_{year}.zip"
        qrow = {
            "queue_id": f"V1SH_Q{i:04d}", "source_name": "INMET",
            "year": year, "url": url, "domain": "portal.inmet.gov.br",
            "priority": "P0", "expected_file_type": "ZIP",
            "download_enabled": "true" if enabled else "false",
            "review_only": "true", "notes": "",
        }
        queue.append(qrow)

        mrow = {
            "download_id": f"V1SH_DL_{i:04d}", "source_name": "INMET",
            "year": year, "url": url, "domain": "portal.inmet.gov.br",
            "downloaded": "false", "download_attempted": "false",
            "raw_relative_path": "", "file_sha256_short": "",
            "file_size_bytes": "0", "content_type": "", "http_status": "",
            "download_status": "DOWNLOAD_DISABLED_QUEUE_ONLY",
            "blocked_reason": "" if enabled else "DOWNLOADS_DISABLED",
            "notes": "",
        }
        mrow.update(guardrail_row())

        # download_file is idempotent: it hashes pre-existing files (e.g. from a
        # prior interrupted run), fails closed when downloads are disabled, and
        # only fetches when enabled and within the per-run file budget.
        attempt_new = enabled and i < file_budget
        if attempt_new or dest.exists():
            result = download_file(url, dest, max_bytes=max_bytes_per_file())
            mrow.update({
                "downloaded": result["downloaded"],
                "download_attempted": result.get("download_attempted", "false"),
                "download_status": result["download_status"],
                "file_sha256_short": result["file_sha256_short"],
                "file_size_bytes": result["file_size_bytes"],
                "content_type": result.get("content_type", ""),
                "http_status": result.get("http_status", ""),
                "raw_relative_path": safe_relpath(dest) if result["file_size_bytes"] not in ("", "0") else "",
                "blocked_reason": "" if result["download_status"] in ("DOWNLOADED_OK", "ALREADY_EXISTS_HASHED") else result["download_status"],
            })
            if result["downloaded"] == "true":
                total_bytes += int(result["file_size_bytes"] or 0)
        elif enabled:
            mrow["download_status"] = "SKIPPED_MAX_FILES"
            mrow["blocked_reason"] = "MAX_FILES_REACHED"

        manifest.append(mrow)

    forbidden_guardrail_scan(manifest, "v1sh_manifest")
    write_csv_with_header(OUT_QUEUE, queue, QUEUE_FIELDS)
    write_csv_with_header(OUT_MANIFEST, manifest, MANIFEST_FIELDS)
    write_schema_for(SCHEMA_Q, QUEUE_FIELDS, "v1sh_queue")
    write_schema_for(SCHEMA_M, MANIFEST_FIELDS, "v1sh_manifest")

    downloaded = sum(1 for r in manifest if r["downloaded"] == "true")
    already = sum(1 for r in manifest if r["download_status"] == "ALREADY_EXISTS_HASHED")
    partial = sum(1 for r in manifest if r["download_status"] == "PARTIAL_OR_EMPTY_FILE_FAIL_CLOSED")
    summary = [
        {"stat_key": "queue_size", "stat_value": str(len(queue))},
        {"stat_key": "downloads_enabled", "stat_value": "true" if enabled else "false"},
        {"stat_key": "files_downloaded", "stat_value": str(downloaded)},
        {"stat_key": "already_exists_hashed", "stat_value": str(already)},
        {"stat_key": "partial_or_empty_files", "stat_value": str(partial)},
        {"stat_key": "total_bytes_downloaded", "stat_value": str(total_bytes)},
        {"stat_key": "links_discovered", "stat_value": str(len(all_links))},
        {"stat_key": "stage", "stat_value": "v1sh"},
    ]
    write_csv_with_header(OUT_SUMMARY, summary, SUM_FIELDS)
    write_schema_for(SCHEMA_S, SUM_FIELDS, "v1sh_summary")

    write_doc(DOC, "v1sh — INMET Historical Data Downloader", [
        "## Objetivo",
        "Baixar ZIPs/CSVs anuais de dados historicos do INMET (portal oficial). "
        "Downloads desativados por padrao; gerar queue apenas.",
        "## Resultado",
        f"Links: {len(all_links)}. Queue: {len(queue)}. Downloaded: {downloaded}. "
        f"Bytes: {total_bytes}.",
    ])
    print(f"[v1sh] queue={len(queue)} downloaded={downloaded} bytes={total_bytes}")
    return {"queue": len(queue), "downloaded": downloaded, "bytes": total_bytes}


if __name__ == "__main__":
    argparse.ArgumentParser(description="v1sh INMET downloader").parse_args()
    run()
