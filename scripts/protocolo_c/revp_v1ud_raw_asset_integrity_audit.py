#!/usr/bin/env python3
"""
v1ud — Raw Asset Integrity Audit

Verifies downloaded files: existence, hash, extension vs mime,
content probing (PDF, ZIP, geodata, HTML).
Outputs raw_asset_integrity_registry.csv and gate_delta_registry.csv.
"""

import argparse
import csv
import hashlib
import json
import os
import zipfile
from pathlib import Path

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

PROTOCOL_VERSION = "v1ud"

INTEGRITY_COLUMNS = [
    "integrity_id", "extraction_id", "source_id", "event_id",
    "local_raw_path", "file_exists", "expected_sha256", "actual_sha256",
    "hash_match", "file_size_bytes", "extension", "detected_mime",
    "extension_mime_consistent", "file_category",
    "probe_status", "probe_detail",
    "geometry_type", "crs", "feature_count", "bounds",
    "pdf_page_count", "pdf_text_sample",
    "zip_entry_count", "zip_relevant_files",
    "html_link_count", "html_relevant_links",
    "notes",
]

GATE_DELTA_COLUMNS = [
    "delta_id", "evidence_id", "source_id", "event_id",
    "v1uc_status", "v1ud_status", "gained_hash", "gained_pdf_text",
    "gained_geometry", "gained_html_links", "still_blocked",
    "blocking_reason", "can_create_ground_reference", "notes",
]

GEODATA_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".gml"}


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_mime(filepath: str) -> str:
    import mimetypes
    mt, _ = mimetypes.guess_type(filepath)
    return mt or "application/octet-stream"


def probe_pdf(filepath: str) -> dict:
    if pypdf is None:
        return {"status": "PDF_BACKEND_MISSING", "pages": 0, "text": ""}
    try:
        reader = pypdf.PdfReader(filepath)
        text = ""
        if reader.pages:
            text = reader.pages[0].extract_text()[:500]
        return {"status": "PDF_PARSED", "pages": len(reader.pages), "text": text}
    except Exception as e:
        return {"status": "PDF_PARSE_ERROR", "pages": 0, "text": str(e)[:200]}


def probe_zip(filepath: str) -> dict:
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            names = zf.namelist()
            relevant = [
                n for n in names
                if any(n.lower().endswith(ext) for ext in
                       [".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".pdf", ".csv", ".xlsx"])
            ]
            return {"status": "ZIP_OK", "count": len(names), "relevant": relevant[:20]}
    except Exception as e:
        return {"status": "ZIP_ERROR", "count": 0, "relevant": [], "error": str(e)[:200]}


def probe_geodata(filepath: str) -> dict:
    if gpd is None:
        return {"status": "GEODATA_BACKEND_MISSING"}
    try:
        gdf = gpd.read_file(filepath)
        geom_types = []
        if "geometry" in gdf.columns and len(gdf) > 0:
            geom_types = gdf.geometry.geom_type.unique().tolist()
        return {
            "status": "GEO_PARSED",
            "geometry_type": str(geom_types),
            "crs": str(gdf.crs) if gdf.crs else "UNKNOWN",
            "feature_count": len(gdf),
            "bounds": list(gdf.total_bounds) if len(gdf) > 0 else [],
            "columns": list(gdf.columns)[:20],
        }
    except Exception as e:
        return {"status": "GEO_PARSE_ERROR", "error": str(e)[:200]}


def probe_html(filepath: str) -> dict:
    if BeautifulSoup is None:
        return {"status": "HTML_BACKEND_MISSING", "links": 0, "relevant": []}
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        soup = BeautifulSoup(content, "html.parser")
        all_links = soup.find_all("a", href=True)
        relevant = []
        for a in all_links:
            href = a["href"].lower()
            if any(ext in href for ext in [".pdf", ".zip", ".csv", ".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".xlsx"]):
                relevant.append({"href": a["href"], "text": a.get_text(strip=True)[:100]})
        return {"status": "HTML_PARSED", "links": len(all_links), "relevant": relevant[:30]}
    except Exception as e:
        return {"status": "HTML_PARSE_ERROR", "error": str(e)[:200]}


def classify_link(href: str) -> str:
    lower = href.lower()
    if any(kw in lower for kw in ["download", "dados", "historico", "serie"]):
        return "download_candidate"
    if any(ext in lower for ext in [".pdf", ".zip", ".csv", ".shp", ".gpkg"]):
        return "download_candidate"
    if any(kw in lower for kw in ["solicit", "requer", "contato", "formulario"]):
        return "formal_request_candidate"
    if any(kw in lower for kw in ["evento", "desastre", "inundacao", "flood"]):
        return "event_specific_candidate"
    return "portal_generic"


def main():
    parser = argparse.ArgumentParser(description="v1ud — Raw Asset Integrity Audit")
    parser.add_argument("--extraction-registry", default="datasets/protocolo_c/v1ud_evidence_extraction_registry.csv")
    parser.add_argument("--v1uc-evidence", default="datasets/protocolo_c/evidence_source_registry.csv")
    parser.add_argument("--out-integrity", default="datasets/protocolo_c/v1ud_raw_asset_integrity_registry.csv")
    parser.add_argument("--out-gate-delta", default="datasets/protocolo_c/v1ud_gate_delta_registry.csv")
    args = parser.parse_args()

    extractions = []
    if os.path.exists(args.extraction_registry):
        with open(args.extraction_registry, "r", encoding="utf-8") as f:
            extractions = list(csv.DictReader(f))

    v1uc_evidence = []
    if os.path.exists(args.v1uc_evidence):
        with open(args.v1uc_evidence, "r", encoding="utf-8") as f:
            v1uc_evidence = list(csv.DictReader(f))

    v1uc_map = {}
    for ev in v1uc_evidence:
        key = f"{ev.get('source_id', '')}_{ev.get('event_id', '')}"
        if key not in v1uc_map:
            v1uc_map[key] = ev

    integrity_rows = []
    delta_rows = []

    for idx, ext in enumerate(extractions):
        local_path = ext.get("local_raw_path", "")
        source_id = ext.get("source_id", "")
        event_id = ext.get("event_id", "")
        expected_hash = ext.get("sha256", "")

        row = {
            "integrity_id": f"INT_{PROTOCOL_VERSION}_{idx:04d}",
            "extraction_id": ext.get("extraction_id", ""),
            "source_id": source_id,
            "event_id": event_id,
            "local_raw_path": local_path,
            "file_exists": "false",
            "expected_sha256": expected_hash,
            "actual_sha256": "",
            "hash_match": "",
            "file_size_bytes": "",
            "extension": "",
            "detected_mime": "",
            "extension_mime_consistent": "",
            "file_category": "",
            "probe_status": "",
            "probe_detail": "",
            "geometry_type": "",
            "crs": "",
            "feature_count": "",
            "bounds": "",
            "pdf_page_count": "",
            "pdf_text_sample": "",
            "zip_entry_count": "",
            "zip_relevant_files": "",
            "html_link_count": "",
            "html_relevant_links": "",
            "notes": "",
        }

        if ext.get("acquisition_status") not in ("DOWNLOADED",):
            row["probe_status"] = "NOT_DOWNLOADED"
            row["notes"] = f"Status: {ext.get('acquisition_status', '')}"
            integrity_rows.append(row)

            key = f"{source_id}_{event_id}"
            v1uc_entry = v1uc_map.get(key, {})
            delta_rows.append({
                "delta_id": f"DEL_{PROTOCOL_VERSION}_{idx:04d}",
                "evidence_id": ext.get("extraction_id", ""),
                "source_id": source_id,
                "event_id": event_id,
                "v1uc_status": v1uc_entry.get("acquisition_status", "UNKNOWN"),
                "v1ud_status": ext.get("acquisition_status", ""),
                "gained_hash": "false",
                "gained_pdf_text": "false",
                "gained_geometry": "false",
                "gained_html_links": "false",
                "still_blocked": "true",
                "blocking_reason": ext.get("acquisition_status", ""),
                "can_create_ground_reference": "false",
                "notes": "",
            })
            continue

        if not local_path or not os.path.exists(local_path):
            row["probe_status"] = "FILE_NOT_FOUND"
            integrity_rows.append(row)
            continue

        row["file_exists"] = "true"
        row["file_size_bytes"] = str(os.path.getsize(local_path))
        row["extension"] = Path(local_path).suffix.lower()
        row["detected_mime"] = detect_mime(local_path)

        actual_hash = sha256_file(local_path)
        row["actual_sha256"] = actual_hash
        row["hash_match"] = str(expected_hash == actual_hash) if expected_hash else "NO_EXPECTED"

        ext_lower = row["extension"]
        mime_lower = row["detected_mime"].lower()
        consistent = True
        if ext_lower == ".pdf" and "pdf" not in mime_lower:
            consistent = False
        if ext_lower == ".html" and "html" not in mime_lower:
            consistent = False
        if ext_lower == ".zip" and "zip" not in mime_lower:
            consistent = False
        row["extension_mime_consistent"] = str(consistent)

        gained_hash = bool(actual_hash)
        gained_pdf = False
        gained_geo = False
        gained_html = False

        if ext_lower == ".pdf":
            row["file_category"] = "PDF"
            probe = probe_pdf(local_path)
            row["probe_status"] = probe["status"]
            row["pdf_page_count"] = str(probe.get("pages", 0))
            row["pdf_text_sample"] = probe.get("text", "")[:200]
            gained_pdf = probe["status"] == "PDF_PARSED" and bool(probe.get("text"))

        elif ext_lower == ".zip":
            row["file_category"] = "ZIP"
            probe = probe_zip(local_path)
            row["probe_status"] = probe["status"]
            row["zip_entry_count"] = str(probe.get("count", 0))
            row["zip_relevant_files"] = json.dumps(probe.get("relevant", []), ensure_ascii=False)[:500]

        elif ext_lower in GEODATA_EXTENSIONS:
            row["file_category"] = "GEODATA"
            probe = probe_geodata(local_path)
            row["probe_status"] = probe.get("status", "")
            if probe.get("status") == "GEO_PARSED":
                row["geometry_type"] = probe.get("geometry_type", "")
                row["crs"] = probe.get("crs", "")
                row["feature_count"] = str(probe.get("feature_count", 0))
                row["bounds"] = str(probe.get("bounds", ""))
                gained_geo = True

        elif ext_lower == ".html" or "html" in mime_lower:
            row["file_category"] = "HTML"
            probe = probe_html(local_path)
            row["probe_status"] = probe.get("status", "")
            row["html_link_count"] = str(probe.get("links", 0))
            relevant = probe.get("relevant", [])
            row["html_relevant_links"] = json.dumps(relevant, ensure_ascii=False)[:500]
            gained_html = len(relevant) > 0

        elif ext_lower in (".csv", ".json", ".xml", ".xlsx"):
            row["file_category"] = ext_lower.upper().lstrip(".")
            row["probe_status"] = "TABULAR_DETECTED"

        else:
            row["file_category"] = "OTHER"
            row["probe_status"] = "NO_SPECIFIC_PROBE"

        integrity_rows.append(row)

        key = f"{source_id}_{event_id}"
        v1uc_entry = v1uc_map.get(key, {})
        delta_rows.append({
            "delta_id": f"DEL_{PROTOCOL_VERSION}_{idx:04d}",
            "evidence_id": ext.get("extraction_id", ""),
            "source_id": source_id,
            "event_id": event_id,
            "v1uc_status": v1uc_entry.get("acquisition_status", "UNKNOWN"),
            "v1ud_status": "DOWNLOADED",
            "gained_hash": str(gained_hash).lower(),
            "gained_pdf_text": str(gained_pdf).lower(),
            "gained_geometry": str(gained_geo).lower(),
            "gained_html_links": str(gained_html).lower(),
            "still_blocked": "true",
            "blocking_reason": "G10_G11_always_fail_v1ud",
            "can_create_ground_reference": "false",
            "notes": "",
        })

    os.makedirs(os.path.dirname(args.out_integrity) or ".", exist_ok=True)
    with open(args.out_integrity, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INTEGRITY_COLUMNS)
        writer.writeheader()
        writer.writerows(integrity_rows)

    with open(args.out_gate_delta, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GATE_DELTA_COLUMNS)
        writer.writeheader()
        writer.writerows(delta_rows)

    print(f"[Integrity Audit v1ud] {len(integrity_rows)} entries")
    print(f"  Downloaded: {sum(1 for r in integrity_rows if r['file_exists'] == 'true')}")
    print(f"  Not downloaded: {sum(1 for r in integrity_rows if r['file_exists'] == 'false')}")
    print(f"  can_create_ground_reference = false (all)")
    print(f"\n  Integrity: {args.out_integrity}")
    print(f"  Gate Delta: {args.out_gate_delta}")


if __name__ == "__main__":
    main()
