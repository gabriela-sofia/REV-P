"""
SUSC-09B External Vector Parser (offline, traceable-only)

Parses vector files acquired/placed in the SUSC-09B download dir: GeoJSON, JSON,
CSV lat/lon, WKT, KML (stdlib XML best-effort), KMZ (stdlib zip+XML). Shapefile/
PDF are recorded as metadata-only (blocked without a library) — never invented.

Writes:
  - datasets/suscetibilidade/susc_09b_external_geometries_parsed_v1.csv
  - outputs_public/suscetibilidade/SUSC_09B_external_vector_parse_audit.csv
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, rel  # noqa: E402
from susc_geometry import geojson_points, region_of_coord, bbox_of_points, in_brazil  # noqa: E402

DL_DIR = ROOT / "datasets" / "suscetibilidade" / "external_event_geometry_sources_susc09b"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_09b_external_geometries_parsed_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_parse_audit.csv"

PARSED_FIELDS = ["geometry_id", "source_file", "region", "geometry_type", "n_coords",
                 "bbox", "coordinate_crs", "parse_method", "source_confidence",
                 "can_be_ground_truth", "allowed_for_training", "review_only",
                 "requires_manual_review", "notes"]
COORD_RE = re.compile(r"(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")


def parse_geojson(p):
    obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    return geojson_points(obj)


def parse_csv_latlon(p):
    pts, types = [], ["POINT"]
    rows = read_csv(p)
    if not rows:
        return [], []
    cols = {c.lower(): c for c in rows[0].keys()}
    latc = next((cols[k] for k in cols if k in ("lat", "latitude")), None)
    lonc = next((cols[k] for k in cols if k in ("lon", "long", "longitude")), None)
    if not latc or not lonc:
        return [], []
    for r in rows:
        try:
            pts.append((float(r[lonc]), float(r[latc])))
        except (ValueError, TypeError, KeyError):
            continue
    return pts, types


def parse_wkt_or_text(p):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    types = re.findall(r"\b(POINT|POLYGON|MULTIPOLYGON|MULTIPOINT)\b", txt, re.IGNORECASE)
    pts = [(float(a), float(b)) for a, b in COORD_RE.findall(txt)]
    return pts, [t.upper() for t in types]


def parse_kml_text(text):
    pts = []
    for m in re.findall(r"<coordinates>(.*?)</coordinates>", text, re.DOTALL | re.IGNORECASE):
        for token in m.replace("\n", " ").split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return pts, ["KML"]


def main() -> int:
    print("=" * 60)
    print("SUSC-09B External Vector Parser (offline)")
    print("=" * 60)

    parsed = []
    audit = []
    gid = 0
    files = [p for p in (DL_DIR.rglob("*") if DL_DIR.exists() else []) if p.is_file() and p.name != "README.md"]
    for p in files:
        ext = p.suffix.lower()
        try:
            if ext in {".geojson", ".json"}:
                pts, types = parse_geojson(p); method = "geojson"
            elif ext in {".csv", ".tsv"}:
                pts, types = parse_csv_latlon(p); method = "csv_lat_lon"
            elif ext in {".wkt", ".txt"}:
                pts, types = parse_wkt_or_text(p); method = "wkt_text"
            elif ext == ".kml":
                pts, types = parse_kml_text(p.read_text(encoding="utf-8", errors="ignore")); method = "kml"
            elif ext == ".kmz":
                with zipfile.ZipFile(p) as z:
                    name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
                    pts, types = (parse_kml_text(z.read(name).decode("utf-8", "ignore")) if name else ([], []))
                method = "kmz"
            elif ext in {".shp", ".gpkg", ".dbf", ".shx", ".prj"}:
                audit.append({"source_file": rel(p), "result": "metadata_only_no_offline_parser",
                              "reason": ext, "n_coords": 0}); continue
            elif ext == ".pdf":
                audit.append({"source_file": rel(p), "result": "pdf_text_skip_no_library",
                              "reason": "pdf", "n_coords": 0}); continue
            else:
                audit.append({"source_file": rel(p), "result": "unsupported_ext",
                              "reason": ext, "n_coords": 0}); continue
        except (OSError, ValueError, json.JSONDecodeError, ET.ParseError, zipfile.BadZipFile) as e:
            audit.append({"source_file": rel(p), "result": "parse_error", "reason": str(e)[:50], "n_coords": 0})
            continue

        in_reg = [(lo, la) for lo, la in pts if in_brazil(lo, la)]
        if not in_reg:
            audit.append({"source_file": rel(p), "result": "no_valid_coordinate", "reason": "none_in_brazil", "n_coords": len(pts)})
            continue
        bbox = bbox_of_points(in_reg)
        region = region_of_coord(in_reg[0][0], in_reg[0][1])
        poly = any(t.upper() in {"POLYGON", "MULTIPOLYGON"} for t in types)
        gtype = "polygon" if poly else ("point" if len(in_reg) == 1 else "point_set")
        gid += 1
        parsed.append({
            "geometry_id": f"EXTGEO_{gid:04d}", "source_file": rel(p), "region": region,
            "geometry_type": gtype, "n_coords": len(in_reg),
            "bbox": ";".join(str(round(x, 6)) for x in bbox), "coordinate_crs": "EPSG:4326",
            "parse_method": method, "source_confidence": "external_official_traceable_review_only",
            "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
            "requires_manual_review": "true",
            "notes": "parsed from external acquired vector; review-only, not ground truth",
        })
        audit.append({"source_file": rel(p), "result": "parsed", "reason": f"{gtype}:{region}", "n_coords": len(in_reg)})

    write_csv(PARSED, parsed, PARSED_FIELDS)
    write_csv(AUDIT, audit, ["source_file", "result", "reason", "n_coords"])
    from collections import Counter
    print(f"\nfiles in susc09b dir: {len(files)} | parsed geometries: {len(parsed)}")
    print("audit:", dict(Counter(a["result"] for a in audit)) if audit else "{} (no external files yet)")
    print("No coordinate invented. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
