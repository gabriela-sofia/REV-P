"""
SUSC-07B Real Coordinate Extraction (offline, traceable-only)

Extracts ONLY explicit geometry (lat/lon, bbox, GeoJSON, KML Placemark, WKT
POINT/POLYGON, CSV lat/lon) from the real-coordinate candidate files found by the
07B scan and from any manually placed sources. Region is assigned from the WGS84
coordinate bounds (a sanity check, NOT a coordinate invention). Centroid /
geocoding / unreferenced sources are rejected.

Reads:
  - datasets/suscetibilidade/susc_07_event_evidence_catalog_v1.csv
  - outputs_public/suscetibilidade/SUSC_07B_real_coordinate_source_scan.csv
  - manifests/suscetibilidade/susc_07b_external_coordinate_acquisition_manifest_v1.csv
  - datasets/suscetibilidade/external_event_geometry_sources/

Writes:
  - datasets/suscetibilidade/susc_07b_real_event_coordinates_v1.csv
  - outputs_public/suscetibilidade/SUSC_07B_real_coordinate_extraction_audit.csv
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_07_event_evidence_catalog_v1.csv"
SCAN = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_source_scan.csv"
ACQ_MANIFEST = (
    ROOT / "manifests" / "suscetibilidade" / "susc_07b_external_coordinate_acquisition_manifest_v1.csv"
)
DL_DIR = ROOT / "datasets" / "suscetibilidade" / "external_event_geometry_sources"

OUT_COORDS = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
OUT_AUDIT = (
    ROOT / "outputs_public" / "suscetibilidade" / "SUSC_07B_real_coordinate_extraction_audit.csv"
)

REGION_BOUNDS = {
    "recife": (-35.20, -34.80, -8.50, -7.85),
    "petropolis": (-43.45, -43.00, -22.75, -22.20),
    "curitiba": (-49.55, -49.00, -25.75, -25.20),
}
FORBIDDEN_NAME_HINTS = ("centroid", "centroide", "geocod", "placeholder", "missing", "empty",
                        "template", "example", "synthetic", "scaffold")
EVENT_TOKEN_RE = re.compile(r"\b((REC|PET|CUR|CTB)[-_]?\d{4}[-_]\d{2}[-_]\d{2}(?:[-_]\d{2})?)\b", re.IGNORECASE)
WKT_RE = re.compile(r"\b(POINT|POLYGON|MULTIPOLYGON|MULTIPOINT)\s*\(", re.IGNORECASE)

COORD_FIELDS = [
    "coordinate_id", "evidence_id", "region", "event_type", "date_or_period",
    "source_path_or_url", "source_name", "geometry_type", "lat", "lon", "bbox",
    "wkt", "geojson_ref", "coordinate_crs", "coordinate_precision",
    "extraction_method", "source_confidence", "can_link_to_patch",
    "can_be_ground_truth", "allowed_for_training", "review_only",
    "requires_manual_review", "notes",
]


def region_of(lon, lat):
    for reg, (lo, hi, la, ha) in REGION_BOUNDS.items():
        if lo <= lon <= hi and la <= lat <= ha:
            return reg
    return "unknown"


def iter_coords_geojson(obj):
    """Yield (lon, lat) and the geometry types seen from a GeoJSON object."""
    geoms = []
    if isinstance(obj, dict) and obj.get("type") == "FeatureCollection":
        for ft in obj.get("features", []):
            g = ft.get("geometry")
            if g:
                geoms.append(g)
    elif isinstance(obj, dict) and obj.get("type") == "Feature":
        if obj.get("geometry"):
            geoms.append(obj["geometry"])
    elif isinstance(obj, dict) and "coordinates" in obj:
        geoms.append(obj)
    types = []
    coords = []

    def walk(c):
        if isinstance(c, list):
            if c and isinstance(c[0], (int, float)) and len(c) >= 2:
                coords.append((float(c[0]), float(c[1])))
            else:
                for x in c:
                    walk(x)

    for g in geoms:
        if not g:
            continue
        types.append(g.get("type", "unknown"))
        walk(g.get("coordinates", []))
    return coords, types


def parse_wkt(text):
    coords = []
    types = []
    for m in WKT_RE.finditer(text):
        types.append(m.group(1).upper())
    for m in re.finditer(r"(-?\d{1,3}\.\d+)\s+(-?\d{1,3}\.\d+)", text):
        coords.append((float(m.group(1)), float(m.group(2))))
    return coords, types


def parse_csv_latlon(path):
    coords = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            latc = next((cols[k] for k in cols if k in ("lat", "latitude")), None)
            lonc = next((cols[k] for k in cols if k in ("lon", "long", "longitude")), None)
            if not latc or not lonc:
                return coords
            for r in reader:
                try:
                    coords.append((float(r[lonc]), float(r[latc])))
                except (ValueError, TypeError, KeyError):
                    continue
    except OSError:
        pass
    return coords


def summarize(coords, types):
    """Return one extracted-geometry record dict (geometry-level) or None."""
    in_region_coords = [(lo, la) for lo, la in coords if region_of(lo, la) != "unknown"]
    if not in_region_coords:
        return None
    lons = [c[0] for c in in_region_coords]
    lats = [c[1] for c in in_region_coords]
    bbox = [round(min(lons), 6), round(min(lats), 6), round(max(lons), 6), round(max(lats), 6)]
    region = region_of(in_region_coords[0][0], in_region_coords[0][1])
    poly = any(t.upper() in {"POLYGON", "MULTIPOLYGON"} for t in types)
    n = len(in_region_coords)
    if poly:
        geometry_type = "polygon"
        precision = "polygon_extent"
        lat = lon = ""
    elif n == 1:
        geometry_type = "point"
        precision = "point"
        lon, lat = in_region_coords[0]
        lon, lat = round(lon, 6), round(lat, 6)
    else:
        geometry_type = "point_set"
        precision = "point_set_extent"
        lat = lon = ""
    return {
        "region": region, "geometry_type": geometry_type, "bbox": bbox,
        "lat": lat, "lon": lon, "precision": precision, "n_coords": n,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-07B Real Coordinate Extraction (traceable-only)")
    print("=" * 60)

    if not SCAN.exists():
        print(f"STOP: scan not found: {SCAN}")
        return 1
    with SCAN.open("r", encoding="utf-8", newline="") as f:
        scan = list(csv.DictReader(f))

    candidate_files = [r["file_path"] for r in scan if r["candidate_status"] == "real_coordinate_candidate"]
    # include manually placed sources
    if DL_DIR.exists():
        for p in DL_DIR.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".geojson", ".json", ".csv", ".wkt", ".kml"}:
                candidate_files.append(str(p.relative_to(ROOT)).replace("\\", "/"))

    coord_rows = []
    audit_rows = []
    cid = 0
    for rel_path in sorted(set(candidate_files)):
        p = ROOT / rel_path
        ext = p.suffix.lower()
        name_l = p.name.lower()
        rejected = next((h for h in FORBIDDEN_NAME_HINTS if h in name_l), None)
        if rejected:
            audit_rows.append({"source_path": rel_path, "result": "rejected_non_real_source",
                               "reason": f"name_hint:{rejected}", "n_coords": 0})
            continue
        try:
            if ext in {".geojson", ".json"}:
                obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
                coords, types = iter_coords_geojson(obj)
                method = "geojson_geometry"
            elif ext in {".csv", ".tsv"}:
                coords = parse_csv_latlon(p)
                types = ["POINT"] * (1 if coords else 0)
                method = "csv_lat_lon"
            elif ext in {".wkt", ".txt"}:
                coords, types = parse_wkt(p.read_text(encoding="utf-8", errors="ignore"))
                method = "wkt"
            elif ext == ".kml":
                txt = p.read_text(encoding="utf-8", errors="ignore")
                coords = [(float(a), float(b)) for a, b in
                          re.findall(r"(-?\d{1,3}\.\d+),(-?\d{1,3}\.\d+)", txt)]
                types = ["POINT"]
                method = "kml_placemark"
            else:
                audit_rows.append({"source_path": rel_path, "result": "skipped_no_offline_parser",
                                   "reason": ext, "n_coords": 0})
                continue
        except (OSError, ValueError, json.JSONDecodeError) as e:
            audit_rows.append({"source_path": rel_path, "result": "parse_error",
                               "reason": str(e)[:60], "n_coords": 0})
            continue

        summ = summarize(coords, types)
        if not summ:
            audit_rows.append({"source_path": rel_path, "result": "no_in_region_coordinate",
                               "reason": "no_coord_in_region_bounds", "n_coords": len(coords)})
            continue

        token_m = EVENT_TOKEN_RE.search(p.name) or EVENT_TOKEN_RE.search(rel_path)
        event_token = token_m.group(1).upper().replace("-", "_") if token_m else ""
        evidence_id = event_token if event_token else f"REGION_LEVEL_{summ['region']}"
        # evidence tie is loose unless an explicit event token is in the path
        evid_traceable = bool(event_token)

        cid += 1
        coord_rows.append({
            "coordinate_id": f"COORD_{cid:04d}",
            "evidence_id": evidence_id,
            "region": summ["region"],
            "event_type": "documented_geometry_source",
            "date_or_period": event_token.replace("_", "-")[3:] if event_token else "unknown",
            "source_path_or_url": rel_path,
            "source_name": p.name,
            "geometry_type": summ["geometry_type"],
            "lat": summ["lat"],
            "lon": summ["lon"],
            "bbox": ";".join(str(x) for x in summ["bbox"]),
            "wkt": "",
            "geojson_ref": rel_path if ext in {".geojson", ".json"} else "",
            "coordinate_crs": "EPSG:4326",
            "coordinate_precision": summ["precision"],
            "extraction_method": method,
            "source_confidence": "traceable_local_vector_review_only",
            # patch-link is spatial: geometry explicit + region + CRS known
            "can_link_to_patch": "true",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "requires_manual_review": "false" if evid_traceable else "true",
            "notes": (
                f"{summ['n_coords']} coord(s); evidence tie "
                + ("explicit_event_token" if evid_traceable else "region_level_only")
                + "; geometry source review-only, NOT event ground truth."
            ),
        })
        audit_rows.append({"source_path": rel_path, "result": "extracted",
                           "reason": f"{summ['geometry_type']}:{summ['region']}",
                           "n_coords": summ["n_coords"]})

    OUT_COORDS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_COORDS.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COORD_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(coord_rows)

    with OUT_AUDIT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source_path", "result", "reason", "n_coords"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(audit_rows)

    from collections import Counter
    print(f"\nextracted coordinate records: {len(coord_rows)}")
    print("by region:", dict(Counter(r["region"] for r in coord_rows)))
    print("by geometry:", dict(Counter(r["geometry_type"] for r in coord_rows)))
    print("audit:", dict(Counter(r["result"] for r in audit_rows)))
    print("\nOnly traceable explicit geometry. No centroid/geocoding. review-only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
