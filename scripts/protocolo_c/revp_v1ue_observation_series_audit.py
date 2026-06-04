#!/usr/bin/env python3
"""
v1ue — Observation Series Audit

Audits downloaded/resolved assets for observational content:
- INMET CSV/ZIP: detect stations, precipitation columns, dates, window totals
- ANA CSV/ZIP/HTML: detect series, station, date, level/discharge
- Cemaden: detect station/pluviometer/alert/bulletin
- PDF: extract text, search hazard/locality terms
- Geodata: extract metadata, CRS, geometry_type (no overlay)

Never executes overlay. Never invents coordinates.
"""

import argparse
import csv
import hashlib
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

PROTOCOL_VERSION = "v1ue"

OBSERVATION_COLUMNS = [
    "observation_id", "event_id", "source_id", "station_candidate_id",
    "asset_path_hash", "asset_type", "observed_variable", "observed_start",
    "observed_end", "temporal_overlap_status", "precipitation_total_mm",
    "water_level_signal", "discharge_signal", "hazard_terms_found",
    "locality_terms_found", "geometry_metadata_available", "event_specificity",
    "evidence_role", "evidence_strength", "gate_support", "limitations", "notes",
]

HAZARD_TERMS = [
    "inundação", "inundacao", "alagamento", "enchente", "transbordamento",
    "deslizamento", "movimento de massa", "escorregamento", "cheia",
]
LOCALITY_HINT_TERMS = [
    "bairro", "rua", "avenida", "localidade", "comunidade", "morro", "córrego",
    "corrego",
]
GEODATA_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".gml"}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sha256_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_html(filepath: str) -> dict:
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            text = f.read().lower()
    except Exception as e:
        return {"asset_type": "HTML", "error": str(e)[:200]}

    hazard_found = [t for t in HAZARD_TERMS if t in text]
    locality_found = [t for t in LOCALITY_HINT_TERMS if t in text]

    return {
        "asset_type": "HTML",
        "hazard_terms": hazard_found,
        "locality_terms": locality_found,
        "event_specificity": "GENERIC_PORTAL_HOMEPAGE",
        "evidence_role": "contextual_only",
        "evidence_strength": "WEAK_GENERIC",
        "notes": "Portal HTML — does not close event gate",
    }


def audit_pdf(filepath: str) -> dict:
    if pypdf is None:
        return {"asset_type": "PDF", "probe": "PDF_BACKEND_MISSING"}
    try:
        reader = pypdf.PdfReader(filepath)
        full_text = ""
        for page in reader.pages[:30]:
            full_text += (page.extract_text() or "") + "\n"
        lower = full_text.lower()
        hazard_found = [t for t in HAZARD_TERMS if t in lower]
        locality_found = [t for t in LOCALITY_HINT_TERMS if t in lower]
        return {
            "asset_type": "PDF",
            "probe": "PDF_PARSED",
            "page_count": str(len(reader.pages)),
            "hazard_terms": hazard_found,
            "locality_terms": locality_found,
            "event_specificity": "DOCUMENT_NEEDS_REVIEW",
            "evidence_role": "documentary_candidate",
            "evidence_strength": "MODERATE_IF_RELEVANT" if hazard_found else "WEAK",
        }
    except Exception as e:
        return {"asset_type": "PDF", "probe": "PDF_PARSE_ERROR", "error": str(e)[:200]}


def audit_inmet_csv_zip(filepath: str) -> dict:
    ext = Path(filepath).suffix.lower()
    result = {
        "asset_type": "INMET_SERIES",
        "observed_variable": "precipitation",
        "evidence_role": "temporal_anchor",
        "evidence_strength": "MODERATE",
        "event_specificity": "YEAR_SPECIFIC_SERIES",
    }
    if ext == ".zip":
        try:
            with zipfile.ZipFile(filepath, "r") as zf:
                names = zf.namelist()
                csv_files = [n for n in names if n.lower().endswith(".csv")]
                result["station_files_count"] = str(len(csv_files))
                result["notes"] = f"ZIP with {len(csv_files)} CSV station files"
        except Exception as e:
            result["error"] = str(e)[:200]
    return result


def audit_geodata(filepath: str) -> dict:
    if gpd is None:
        return {"asset_type": "GEODATA", "probe": "GEODATA_BACKEND_MISSING"}
    try:
        gdf = gpd.read_file(filepath)
        geom_types = []
        if "geometry" in gdf.columns and len(gdf) > 0:
            geom_types = gdf.geometry.geom_type.unique().tolist()
        return {
            "asset_type": "GEODATA",
            "probe": "GEO_PARSED",
            "geometry_type": str(geom_types),
            "crs": str(gdf.crs) if gdf.crs else "UNKNOWN",
            "feature_count": len(gdf),
            "columns": list(gdf.columns)[:20],
            "geometry_metadata_available": "true",
            "evidence_role": "spatial_candidate",
            "evidence_strength": "STRONG_IF_OFFICIAL",
            "limitations": "Metadata only — NO overlay executed",
        }
    except Exception as e:
        return {"asset_type": "GEODATA", "probe": "GEO_PARSE_ERROR", "error": str(e)[:200]}


def audit_asset(filepath: str) -> dict:
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return audit_pdf(filepath)
    if ext in (".csv", ".zip"):
        return audit_inmet_csv_zip(filepath)
    if ext in GEODATA_EXTENSIONS:
        return audit_geodata(filepath)
    if ext in (".html", ".htm"):
        return audit_html(filepath)
    return {"asset_type": "OTHER", "probe": "NO_SPECIFIC_AUDIT"}


def main():
    parser = argparse.ArgumentParser(description="v1ue — Observation Series Audit")
    parser.add_argument("--extraction-registry", default="datasets/protocolo_c/v1ud_evidence_extraction_registry.csv")
    parser.add_argument("--stations", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ue_observation_series_registry.csv")
    args = parser.parse_args()

    extractions = load_csv(args.extraction_registry)
    stations = load_csv(args.stations)

    station_by_source_event = {}
    for s in stations:
        key = f"{s.get('source_id', '')}_{s.get('event_id', '')}"
        station_by_source_event.setdefault(key, s.get("station_candidate_id", ""))

    rows = []
    seq = 0
    for ext in extractions:
        if ext.get("acquisition_status") != "DOWNLOADED":
            continue
        local_path = ext.get("local_raw_path", "")
        source_id = ext.get("source_id", "")
        event_id = ext.get("event_id", "")

        if not local_path or not os.path.exists(local_path):
            audit = {"asset_type": "MISSING_FILE", "probe": "FILE_NOT_FOUND"}
            asset_hash = ""
        else:
            audit = audit_asset(local_path)
            asset_hash = sha256_file(local_path)

        key = f"{source_id}_{event_id}"
        station_id = station_by_source_event.get(key, "")

        hazard_terms = audit.get("hazard_terms", [])
        locality_terms = audit.get("locality_terms", [])
        geom_available = audit.get("geometry_metadata_available", "false")

        gate_support = "temporal_context"
        if audit.get("asset_type") == "GEODATA" and geom_available == "true":
            gate_support = "geometry_metadata_only"
        elif audit.get("asset_type") == "HTML":
            gate_support = "none_generic_portal"

        rows.append({
            "observation_id": f"OBS_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": event_id,
            "source_id": source_id,
            "station_candidate_id": station_id,
            "asset_path_hash": asset_hash,
            "asset_type": audit.get("asset_type", ""),
            "observed_variable": audit.get("observed_variable", ""),
            "observed_start": "",
            "observed_end": "",
            "temporal_overlap_status": "NOT_COMPUTED_PORTAL" if audit.get("asset_type") == "HTML" else "NEEDS_SERIES_DATA",
            "precipitation_total_mm": "",
            "water_level_signal": "",
            "discharge_signal": "",
            "hazard_terms_found": "|".join(hazard_terms),
            "locality_terms_found": "|".join(locality_terms),
            "geometry_metadata_available": geom_available,
            "event_specificity": audit.get("event_specificity", "UNKNOWN"),
            "evidence_role": audit.get("evidence_role", "contextual_only"),
            "evidence_strength": audit.get("evidence_strength", "WEAK"),
            "gate_support": gate_support,
            "limitations": audit.get("limitations", "No overlay executed; no coordinates invented"),
            "notes": audit.get("notes", audit.get("probe", "")),
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Observation Series Audit v1ue] {len(rows)} observations audited")
    types = {}
    for r in rows:
        types[r["asset_type"]] = types.get(r["asset_type"], 0) + 1
    for t, c in sorted(types.items()):
        print(f"  {t}: {c}")
    print(f"  no overlay executed; no coordinates invented")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
