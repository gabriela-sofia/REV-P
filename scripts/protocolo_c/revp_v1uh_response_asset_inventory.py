#!/usr/bin/env python3
"""
v1uh — Response Asset Inventory

Inventories content of accepted responses: ZIP contents, PDF pages,
spreadsheet columns, geodata properties. If dependencies are missing,
records DEPENDENCY_MISSING without breaking.
"""

import argparse
import csv
import hashlib
import os
import zipfile

PROTOCOL_VERSION = "v1uh"

ASSET_COLUMNS = [
    "asset_id", "response_id", "event_id", "institution",
    "container_type", "internal_path", "asset_type", "extension",
    "file_size_bytes", "sha256", "has_geometry", "geometry_type",
    "crs", "feature_count", "has_prj", "has_attribute_table",
    "columns_detected", "pdf_pages", "text_extract_status",
    "inventory_status", "notes",
]

GEOSPATIAL_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".kml", ".kmz"}
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
SHAPEFILE_COMPANIONS = {".shx", ".dbf", ".prj", ".cpg"}


def sha256_file(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_asset_type(ext: str) -> str:
    if ext in GEOSPATIAL_EXTENSIONS:
        return "geospatial_vector"
    if ext in TABULAR_EXTENSIONS:
        return "tabular"
    if ext == ".pdf":
        return "document"
    if ext in (".png", ".jpg", ".jpeg"):
        return "static_map"
    if ext == ".zip":
        return "archive"
    if ext == ".json":
        return "data_json"
    if ext in SHAPEFILE_COMPANIONS:
        return "shapefile_companion"
    return "unknown"


def inventory_zip(filepath: str, response_id: str, seq_start: int) -> tuple[list[dict], int]:
    rows = []
    seq = seq_start
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            shp_names = set()
            prj_names = set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                ext = os.path.splitext(name)[1].lower()
                atype = detect_asset_type(ext)
                if ext == ".shp":
                    shp_names.add(os.path.splitext(name)[0])
                if ext == ".prj":
                    prj_names.add(os.path.splitext(name)[0])

                rows.append({
                    "asset_id": f"ASSET_{PROTOCOL_VERSION}_{seq:04d}",
                    "response_id": response_id,
                    "event_id": "",
                    "institution": "",
                    "container_type": "zip",
                    "internal_path": name,
                    "asset_type": atype,
                    "extension": ext,
                    "file_size_bytes": str(info.file_size),
                    "sha256": "",
                    "has_geometry": str(ext in GEOSPATIAL_EXTENSIONS).lower(),
                    "geometry_type": "",
                    "crs": "",
                    "feature_count": "",
                    "has_prj": "",
                    "has_attribute_table": str(ext in (".shp", ".gpkg", ".geojson")).lower(),
                    "columns_detected": "",
                    "pdf_pages": "",
                    "text_extract_status": "",
                    "inventory_status": "INVENTORIED",
                    "notes": "",
                })
                seq += 1

            for shp_base in shp_names:
                has_prj = shp_base in prj_names
                for r in rows:
                    if r["internal_path"].endswith(".shp") and \
                       os.path.splitext(r["internal_path"])[0] == shp_base:
                        r["has_prj"] = str(has_prj).lower()
    except (zipfile.BadZipFile, Exception) as e:
        rows.append({
            "asset_id": f"ASSET_{PROTOCOL_VERSION}_{seq:04d}",
            "response_id": response_id,
            "event_id": "", "institution": "",
            "container_type": "zip", "internal_path": "",
            "asset_type": "corrupted_archive", "extension": ".zip",
            "file_size_bytes": "", "sha256": "", "has_geometry": "false",
            "geometry_type": "", "crs": "", "feature_count": "",
            "has_prj": "", "has_attribute_table": "",
            "columns_detected": "", "pdf_pages": "",
            "text_extract_status": "", "inventory_status": "ERROR",
            "notes": str(e)[:200],
        })
        seq += 1
    return rows, seq


def inventory_standalone(filepath: str, response: dict, seq: int) -> dict:
    ext = os.path.splitext(filepath)[1].lower()
    atype = detect_asset_type(ext)
    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    sha = sha256_file(filepath)

    row = {
        "asset_id": f"ASSET_{PROTOCOL_VERSION}_{seq:04d}",
        "response_id": response.get("response_id", ""),
        "event_id": response.get("event_id", ""),
        "institution": response.get("institution", ""),
        "container_type": "standalone",
        "internal_path": os.path.basename(filepath),
        "asset_type": atype,
        "extension": ext,
        "file_size_bytes": str(size),
        "sha256": sha,
        "has_geometry": str(ext in GEOSPATIAL_EXTENSIONS).lower(),
        "geometry_type": "",
        "crs": "",
        "feature_count": "",
        "has_prj": "",
        "has_attribute_table": str(ext in (".shp", ".gpkg", ".geojson")).lower(),
        "columns_detected": "",
        "pdf_pages": "",
        "text_extract_status": "",
        "inventory_status": "INVENTORIED",
        "notes": "",
    }

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(filepath)
            row["pdf_pages"] = str(len(reader.pages))
            row["text_extract_status"] = "AVAILABLE"
        except ImportError:
            row["text_extract_status"] = "DEPENDENCY_MISSING_pypdf"
        except Exception as e:
            row["text_extract_status"] = f"ERROR: {str(e)[:100]}"

    if ext in TABULAR_EXTENSIONS and ext == ".csv":
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    row["columns_detected"] = "|".join(header[:20])
        except Exception:
            pass

    return row


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="v1uh — Response Asset Inventory")
    parser.add_argument("--responses",
                        default="datasets/protocolo_c/v1uh_formal_response_registry.csv")
    parser.add_argument("--staging",
                        default="local_only/protocolo_c/formal_responses/staging")
    parser.add_argument("--out",
                        default="datasets/protocolo_c/v1uh_response_asset_inventory.csv")
    args = parser.parse_args()

    responses = load_csv(args.responses)
    all_rows = []
    seq = 0

    for resp in responses:
        if resp.get("intake_status") != "ACCEPTED":
            continue
        filename = resp.get("original_filename", "")
        filepath = os.path.join(args.staging, filename)
        ext = os.path.splitext(filename)[1].lower()

        if ext == ".zip" and os.path.exists(filepath):
            zip_rows, seq = inventory_zip(filepath, resp["response_id"], seq)
            all_rows.extend(zip_rows)
        elif os.path.exists(filepath):
            row = inventory_standalone(filepath, resp, seq)
            all_rows.append(row)
            seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ASSET_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[Response Asset Inventory v1uh] {len(all_rows)} assets inventoried")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
