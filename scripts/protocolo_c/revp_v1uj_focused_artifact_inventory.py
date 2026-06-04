#!/usr/bin/env python3
"""
v1uj — Focused Artifact Inventory

Inventaria os downloads v1uj em local_only: ZIP / GPKG / GeoJSON / KML / KMZ /
SHP / CSV / XLSX / PDF. Detecta geometry / CRS / fields / feature_count e
termos de data / fenomeno / localidade.

Classifica observado vs contexto (suscetibilidade) vs document_only.
ZIP so com PDF = DOCUMENT_ONLY. Sem rede. Fallback gracioso sem deps.
"""

import argparse
import csv
import hashlib
import json
import os
import unicodedata
import zipfile

PROTOCOL_VERSION = "v1uj"

INVENTORY_COLUMNS = [
    "inventory_id", "source_tag", "event_id", "container_type",
    "internal_path", "asset_type", "extension", "file_size_bytes", "sha256",
    "has_geometry", "geometry_type", "crs", "feature_count", "has_prj",
    "columns_detected", "date_term_detected", "hazard_term_detected",
    "locality_term_detected", "susceptibility_term_detected",
    "classification", "inventory_status", "notes",
]

GEO_EXTENSIONS = {".shp", ".gpkg", ".geojson", ".kml", ".kmz", ".gml"}
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
HAZARD_TERMS = {"inundacao", "alagamento", "enchente", "deslizamento",
                "flood", "landslide", "movimento", "transbordamento",
                "enxurrada", "ocorrencia", "solicitacao", "solicitacoes",
                "atendimento", "atendimentos", "vistoria", "vistorias",
                "defesa civil"}
SUSCEPT_TERMS = {"suscetibilidade", "susceptibilidade", "risco", "modelagem",
                 "carta geotecnica", "area de risco", "areas de risco"}
LOCALITY_TERMS = {"petropolis", "recife", "centro", "boa viagem", "alto da serra",
                  "pernambuco"}
DATE_TERMS = {"2022", "2024", "data", "data_ocorr", "data_registro"}
COORD_TERMS = {"latitude", "longitude", "lat", "lon", "lng", "x", "y",
               "coordenada", "coordenadas", "geometria", "geometry"}


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


def normalize_text(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def detect_terms(text):
    lower = normalize_text(text)
    date = any(t in lower for t in DATE_TERMS)
    hazard = any(t in lower for t in HAZARD_TERMS)
    locality = any(t in lower for t in LOCALITY_TERMS)
    suscept = any(t in lower for t in SUSCEPT_TERMS)
    return date, hazard, locality, suscept


def has_coordinate_columns(columns):
    tokens = {normalize_text(c).strip() for c in (columns or "").replace("|", ",").split(",")}
    if {"latitude", "longitude"}.issubset(tokens):
        return True
    if {"lat", "lon"}.issubset(tokens) or {"lat", "lng"}.issubset(tokens):
        return True
    if {"x", "y"}.issubset(tokens):
        return True
    return any(t in " ".join(tokens) for t in COORD_TERMS if len(t) > 2)


def classify_asset(asset_type, hazard, suscept, has_geometry, has_coordinates=False,
                   has_date=False):
    """Classificacao read-only do asset. Funcao pura."""
    if asset_type == "geospatial_vector" or has_geometry:
        if hazard:
            return "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW"
        if suscept:
            return "CONTEXT_ONLY"
        return "CONTEXTUAL_OFFICIAL_LAYER"
    if asset_type == "tabular":
        if hazard and has_coordinates:
            return "TABLE_WITH_COORDINATES_CANDIDATE_FOR_REVIEW"
        if hazard and (has_date or not has_coordinates):
            return "DOCUMENTED_OCCURRENCE_TABLE_NO_GEOMETRY"
        if suscept:
            return "CONTEXT_ONLY"
        return "CONTEXTUAL_OFFICIAL_LAYER"
    if asset_type == "document":
        return "document_only"
    if asset_type == "static_map":
        return "static_map"
    return "unknown"


def parse_geojson_meta(raw_bytes):
    """Extrai (geometry_type, crs, feature_count) de bytes GeoJSON. Pura."""
    try:
        doc = json.loads(raw_bytes.decode("utf-8", errors="replace"))
    except Exception:
        return "", "", ""
    crs = ""
    crs_obj = doc.get("crs", {})
    if isinstance(crs_obj, dict):
        crs = str(crs_obj.get("properties", {}).get("name", ""))
    feats = doc.get("features", [])
    gtype = ""
    if feats:
        geom = feats[0].get("geometry", {})
        gtype = geom.get("type", "") if isinstance(geom, dict) else ""
    return gtype, crs, str(len(feats)) if feats else ""


def inventory_zip(filepath, source_tag, event_id, seq):
    rows = []
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            shp_bases, prj_bases = set(), set()
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                ext = os.path.splitext(name)[1].lower()
                base = os.path.splitext(name)[0]
                if ext == ".shp":
                    shp_bases.add(base)
                if ext == ".prj":
                    prj_bases.add(base)

                gtype = crs = fcount = ""
                if ext == ".geojson":
                    try:
                        gtype, crs, fcount = parse_geojson_meta(zf.read(name))
                    except Exception:
                        pass

                d, hz, loc, su = detect_terms(name)
                atype = detect_asset_type(ext)
                has_geom = ext in GEO_EXTENSIONS
                rows.append({
                    "inventory_id": f"FINV_{PROTOCOL_VERSION}_{seq:04d}",
                    "source_tag": source_tag, "event_id": event_id,
                    "container_type": "zip", "internal_path": name,
                    "asset_type": atype, "extension": ext,
                    "file_size_bytes": str(info.file_size), "sha256": "",
                    "has_geometry": str(has_geom).lower(),
                    "geometry_type": gtype, "crs": crs, "feature_count": fcount,
                    "has_prj": "", "columns_detected": "",
                    "date_term_detected": str(d).lower(),
                    "hazard_term_detected": str(hz).lower(),
                    "locality_term_detected": str(loc).lower(),
                    "susceptibility_term_detected": str(su).lower(),
                    "classification": classify_asset(atype, hz, su, has_geom,
                                                     False, d),
                    "inventory_status": "INVENTORIED", "notes": "",
                })
                seq += 1
            for r in rows:
                if r["internal_path"].endswith(".shp"):
                    base = os.path.splitext(r["internal_path"])[0]
                    r["has_prj"] = str(base in prj_bases).lower()
    except Exception as e:
        rows.append({
            "inventory_id": f"FINV_{PROTOCOL_VERSION}_{seq:04d}",
            "source_tag": source_tag, "event_id": event_id,
            "container_type": "zip", "internal_path": "",
            "asset_type": "corrupted_archive", "extension": ".zip",
            "file_size_bytes": "", "sha256": "", "has_geometry": "false",
            "geometry_type": "", "crs": "", "feature_count": "", "has_prj": "",
            "columns_detected": "", "date_term_detected": "false",
            "hazard_term_detected": "false", "locality_term_detected": "false",
            "susceptibility_term_detected": "false",
            "classification": "unknown", "inventory_status": "ERROR",
            "notes": str(e)[:200],
        })
        seq += 1
    return rows, seq


def inventory_standalone(filepath, source_tag, event_id, seq):
    ext = os.path.splitext(filepath)[1].lower()
    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    sha = sha256_file(filepath)
    name = os.path.basename(filepath)

    gtype = crs = fcount = ""
    columns = ""
    if ext == ".geojson" and os.path.exists(filepath):
        with open(filepath, "rb") as f:
            gtype, crs, fcount = parse_geojson_meta(f.read())
    if ext == ".csv" and os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                header = next(csv.reader(f), None)
                if header:
                    columns = "|".join(header[:25])
        except Exception:
            pass

    text_for_terms = name + " " + columns
    d, hz, loc, su = detect_terms(text_for_terms)
    atype = detect_asset_type(ext)
    has_geom = ext in GEO_EXTENSIONS
    has_coords = has_coordinate_columns(columns)

    return {
        "inventory_id": f"FINV_{PROTOCOL_VERSION}_{seq:04d}",
        "source_tag": source_tag, "event_id": event_id,
        "container_type": "standalone", "internal_path": name,
        "asset_type": atype, "extension": ext,
        "file_size_bytes": str(size), "sha256": sha,
        "has_geometry": str(has_geom).lower(),
        "geometry_type": gtype, "crs": crs, "feature_count": fcount,
        "has_prj": "", "columns_detected": columns,
        "date_term_detected": str(d).lower(),
        "hazard_term_detected": str(hz).lower(),
        "locality_term_detected": str(loc).lower(),
        "susceptibility_term_detected": str(su).lower(),
        "classification": classify_asset(atype, hz, su, has_geom, has_coords, d),
        "inventory_status": "INVENTORIED", "notes": "",
    }


def derive_source_event(root, raw_dir):
    rel = os.path.relpath(root, raw_dir).replace("\\", "/").split("/")
    source_tag = rel[0] if rel and rel[0] != "." else ""
    event_id = ""
    for p in rel:
        if p.startswith(("PET_", "REC_")):
            event_id = p
            break
    return source_tag, event_id


def is_safe_v1uj_filename(fname):
    parts = fname.split("__")
    if len(parts) < 5:
        return False
    url_hash = parts[-2]
    return len(url_hash) == 12 and all(ch in "0123456789abcdef" for ch in url_hash)


def main():
    parser = argparse.ArgumentParser(description="v1uj — Focused Artifact Inventory")
    parser.add_argument("--raw-dir", default="local_only/protocolo_c/focused_public_artifacts/raw/v1uj")
    parser.add_argument("--out", default="datasets/protocolo_c/v1uj_focused_artifact_inventory.csv")
    args = parser.parse_args()

    rows = []
    seq = 0
    if os.path.isdir(args.raw_dir):
        for root, _dirs, files in os.walk(args.raw_dir):
            source_tag, event_id = derive_source_event(root, args.raw_dir)
            safe_files = [f for f in files if is_safe_v1uj_filename(f)]
            scan_files = safe_files if safe_files else files
            for fname in scan_files:
                fpath = os.path.join(root, fname)
                ext = os.path.splitext(fname)[1].lower()
                if ext == ".zip":
                    zip_rows, seq = inventory_zip(fpath, source_tag, event_id, seq)
                    rows.extend(zip_rows)
                else:
                    rows.append(inventory_standalone(fpath, source_tag, event_id, seq))
                    seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=INVENTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    observed = sum(1 for r in rows
                   if r["classification"] == "OBSERVED_GEOMETRY_CANDIDATE_FOR_REVIEW")
    print(f"[Focused Artifact Inventory v1uj] {len(rows)} assets | observed_geometry={observed}")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
