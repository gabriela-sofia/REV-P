#!/usr/bin/env python3
"""
v1ui — Public Artifact Inventory

Inventories downloaded public artifacts: ZIP contents, PDF pages,
CSV columns, geodata metadata. Falls back gracefully when deps missing.
"""

import argparse
import csv
import hashlib
import os
import zipfile

PROTOCOL_VERSION = "v1ui"

INVENTORY_COLUMNS = [
    "inventory_id", "artifact_id", "event_id", "source_id",
    "container_type", "internal_path", "asset_type", "extension",
    "file_size_bytes", "sha256", "has_geometry", "geometry_type",
    "crs", "feature_count", "has_prj", "has_attribute_table",
    "columns_detected", "pdf_pages", "text_extract_status",
    "event_term_detected", "hazard_term_detected", "locality_term_detected",
    "inventory_status", "notes",
]

GEO_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".kml", ".kmz"}
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
HAZARD_TERMS = {"inundacao", "alagamento", "enchente", "deslizamento",
                "flood", "landslide", "movimento", "transbordamento"}


def sha256_file(path):
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_asset_type(ext):
    if ext in GEO_EXTENSIONS:
        return "geospatial_vector"
    if ext in TABULAR_EXTENSIONS:
        return "tabular"
    if ext == ".pdf":
        return "document"
    if ext in (".png", ".jpg", ".jpeg"):
        return "static_map"
    if ext == ".zip":
        return "archive"
    if ext in (".json", ".xml"):
        return "data_structured"
    return "unknown"


def detect_terms(text):
    lower = text.lower()
    event = any(t in lower for t in ["2022", "2024", "petropolis", "recife"])
    hazard = any(t in lower for t in HAZARD_TERMS)
    locality = any(t in lower for t in ["petropolis", "recife", "centro",
                                         "boa viagem", "alto da serra"])
    return event, hazard, locality


def inventory_zip(filepath, artifact_id, seq):
    rows = []
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            shp_bases = set()
            prj_bases = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                ext = os.path.splitext(name)[1].lower()
                if ext == ".shp":
                    shp_bases.add(os.path.splitext(name)[0])
                if ext == ".prj":
                    prj_bases.add(os.path.splitext(name)[0])

                ev, hz, loc = detect_terms(name)
                rows.append({
                    "inventory_id": f"INV_{PROTOCOL_VERSION}_{seq:04d}",
                    "artifact_id": artifact_id,
                    "event_id": "", "source_id": "",
                    "container_type": "zip", "internal_path": name,
                    "asset_type": detect_asset_type(ext), "extension": ext,
                    "file_size_bytes": str(info.file_size), "sha256": "",
                    "has_geometry": str(ext in GEO_EXTENSIONS).lower(),
                    "geometry_type": "", "crs": "", "feature_count": "",
                    "has_prj": "", "has_attribute_table": str(ext in (".shp", ".gpkg", ".geojson")).lower(),
                    "columns_detected": "", "pdf_pages": "",
                    "text_extract_status": "",
                    "event_term_detected": str(ev).lower(),
                    "hazard_term_detected": str(hz).lower(),
                    "locality_term_detected": str(loc).lower(),
                    "inventory_status": "INVENTORIED", "notes": "",
                })
                seq += 1

            for base in shp_bases:
                has_prj = base in prj_bases
                for r in rows:
                    if r["internal_path"].endswith(".shp") and \
                       os.path.splitext(r["internal_path"])[0] == base:
                        r["has_prj"] = str(has_prj).lower()
    except Exception as e:
        rows.append({
            "inventory_id": f"INV_{PROTOCOL_VERSION}_{seq:04d}",
            "artifact_id": artifact_id,
            "event_id": "", "source_id": "",
            "container_type": "zip", "internal_path": "",
            "asset_type": "corrupted_archive", "extension": ".zip",
            "file_size_bytes": "", "sha256": "", "has_geometry": "false",
            "geometry_type": "", "crs": "", "feature_count": "",
            "has_prj": "", "has_attribute_table": "",
            "columns_detected": "", "pdf_pages": "",
            "text_extract_status": "",
            "event_term_detected": "false",
            "hazard_term_detected": "false",
            "locality_term_detected": "false",
            "inventory_status": "ERROR", "notes": str(e)[:200],
        })
        seq += 1
    return rows, seq


def inventory_standalone(filepath, artifact_id, seq):
    ext = os.path.splitext(filepath)[1].lower()
    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    sha = sha256_file(filepath)
    ev, hz, loc = detect_terms(os.path.basename(filepath))

    row = {
        "inventory_id": f"INV_{PROTOCOL_VERSION}_{seq:04d}",
        "artifact_id": artifact_id,
        "event_id": "", "source_id": "",
        "container_type": "standalone", "internal_path": os.path.basename(filepath),
        "asset_type": detect_asset_type(ext), "extension": ext,
        "file_size_bytes": str(size), "sha256": sha,
        "has_geometry": str(ext in GEO_EXTENSIONS).lower(),
        "geometry_type": "", "crs": "", "feature_count": "",
        "has_prj": "", "has_attribute_table": str(ext in (".shp", ".gpkg", ".geojson")).lower(),
        "columns_detected": "", "pdf_pages": "", "text_extract_status": "",
        "event_term_detected": str(ev).lower(),
        "hazard_term_detected": str(hz).lower(),
        "locality_term_detected": str(loc).lower(),
        "inventory_status": "INVENTORIED", "notes": "",
    }

    if ext == ".csv" and os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    row["columns_detected"] = "|".join(header[:20])
        except Exception:
            pass

    return row


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="v1ui — Public Artifact Inventory")
    parser.add_argument("--downloads", default="datasets/protocolo_c/v1ui_public_artifact_inventory_download.csv")
    parser.add_argument("--raw-dir", default="local_only/protocolo_c/public_official_artifacts/raw/v1ui")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_public_artifact_inventory.csv")
    args = parser.parse_args()

    downloads = load_csv(args.downloads)
    rows = []
    seq = 0

    for dl in downloads:
        if dl.get("download_status") not in ("DOWNLOADED", "ALREADY_EXISTS"):
            continue
        artifact_id = dl.get("artifact_id", "")
        source_id = dl.get("source_id", "")
        event_id = dl.get("event_id", "")

        local_dir = os.path.join(args.raw_dir, source_id, event_id)
        if not os.path.isdir(local_dir):
            continue

        for fname in os.listdir(local_dir):
            fpath = os.path.join(local_dir, fname)
            fext = os.path.splitext(fname)[1].lower()
            if fext == ".zip":
                zip_rows, seq = inventory_zip(fpath, artifact_id, seq)
                for r in zip_rows:
                    r["event_id"] = event_id
                    r["source_id"] = source_id
                rows.extend(zip_rows)
            else:
                row = inventory_standalone(fpath, artifact_id, seq)
                row["event_id"] = event_id
                row["source_id"] = source_id
                rows.append(row)
                seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Public Artifact Inventory v1ui] {len(rows)} assets inventoried")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
