"""
SUSC-11B Observed Event Parser (offline, traceable-only)

Builds a raw catalog of OBSERVED flood/inundation events from two traceable
sources, never inventing coordinates:

  1. Files placed/downloaded in the SUSC-11B acquisition dir
     (GeoJSON/JSON, CSV lat/lon, WKT/TXT, KML, KMZ). Shapefile/PDF without a
     library -> recorded as metadata-only and blocked.
  2. The already-traceable local event-geometry sources catalogued in
     susc_07b_real_event_coordinates_v1.csv (real in-repo vectors with bbox and
     source path: Charter758 digitized candidate, Defesa Civil risk locations,
     occurrence registries, station catalogs).

Each record gets a deterministic source_role and a first-pass evidence_level.
The canonical classifier lives here and is reused by the catalog builder.

Writes:
  - datasets/suscetibilidade/susc_11b_observed_events_raw_parsed_v1.csv
  - outputs_public/suscetibilidade/SUSC_11B_observed_event_parse_audit.csv
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, rel  # noqa: E402
from susc_geometry import geojson_points, region_of_coord, bbox_of_points, in_brazil  # noqa: E402

DL_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc11b"
COORDS_07B = ROOT / "datasets" / "suscetibilidade" / "susc_07b_real_event_coordinates_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_events_raw_parsed_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_11B_observed_event_parse_audit.csv"

FLOOD_TYPES = {"flood", "inundation", "urban_flooding", "flash_flood", "landslide_related_flood"}
REGION_SUBTYPE = {"recife": "urban_flooding", "curitiba": "urban_flooding", "petropolis": "flash_flood"}
MUNICIPALITY = {"recife": "Recife", "petropolis": "Petropolis", "curitiba": "Curitiba"}
COORD_RE = re.compile(r"(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")
DATE_RE = re.compile(r"(20\d{2})[_-](\d{2})[_-](\d{2})(?:[_-](\d{2}))?")

PARSED_FIELDS = [
    "event_id", "region", "municipality", "event_type", "date_or_period", "event_date",
    "event_period_start", "event_period_end", "source_name", "source_type",
    "source_url_or_path", "geometry_type", "lat", "lon", "bbox", "wkt", "geojson_ref",
    "crs", "n_coords", "source_role", "evidence_level", "source_confidence",
    "can_link_to_patch", "requires_manual_review", "can_be_ground_truth",
    "allowed_for_training", "review_only", "notes",
]


# --------------------------------------------------------------------------- #
# Canonical helpers (reused by the catalog builder)
# --------------------------------------------------------------------------- #
def derive_source_role(name: str, path: str = "") -> str:
    """Deterministic role from filename/source semantics.

    The filename (basename) takes precedence over the path, and the most
    conservative roles (risk/alert) are checked first so a risk/alert layer is
    never mislabeled as an observed flood event just because it sits under an
    event-named folder.
    """
    n = name.lower()
    p = path.lower()
    if any(k in n for k in ("risk", "risco")):
        return "risk_area"
    if any(k in n for k in ("alert", "alerta", "aviso")):
        return "alert"
    if any(k in n for k in ("charter", "event_polygon", "mancha", "_flood", "inunda", "enchente")):
        return "flood_extent_map"
    if any(k in n for k in ("ground_reference", "occurrence", "ocorrenc", "event_registry",
                            "multi_anchor", "coordinate_recovery")):
        return "occurrence_registry"
    if any(k in n for k in ("station", "inmet", "estac", "catalog")):
        return "station_registry"
    if any(k in n for k in ("patch_boundary", "boundary", "aoi", "digitization")):
        return "patch_geometry_context"
    # fall back to path only for explicit flood-extent context
    if any(k in p for k in ("charter", "event_polygon", "mancha")):
        return "flood_extent_map"
    return "unknown"


def derive_event_type(role: str, region: str) -> str:
    sub = REGION_SUBTYPE.get(region, "flood")
    if role in ("flood_extent_map", "occurrence_registry"):
        return sub
    return "unknown"


def derive_source_type(role: str, name: str) -> str:
    t = name.lower()
    if "charter" in t:
        return "news"
    if role == "patch_geometry_context":
        return "technical_report"
    if role in ("occurrence_registry", "station_registry"):
        return "registry"
    if role in ("risk_area", "alert", "flood_extent_map"):
        return "official"
    return "unknown"


def classify_evidence(role: str, geometry_type: str, has_explicit_geom: bool,
                      has_date: bool, event_type: str) -> str:
    """Canonical evidence-level classifier.

    *_strong is reserved for explicit geometry of confirmed flood (point) or a
    validated polygon footprint. Digitized/candidate polygons are *_moderate.
    Risk areas, alerts, stations and patch context never become observed events.
    """
    if not has_explicit_geom:
        return "documentary_context_only" if role != "unknown" else "insufficient"
    if role == "flood_extent_map":
        if geometry_type in ("polygon", "multipolygon", "bbox"):
            return "observed_flood_bbox_moderate"
        if geometry_type == "point":
            # a single confirmed flood point is strong only when also dated
            return "observed_flood_point_strong" if has_date else "observed_flood_bbox_moderate"
        return "observed_flood_bbox_moderate"  # point_set extent
    if role == "risk_area":
        return "risk_area_context"
    if role == "alert":
        return "alert_only"
    if role == "occurrence_registry":
        return "official_occurrence_point_moderate"
    if role == "station_registry":
        return "administrative_record_only"
    if role == "patch_geometry_context":
        return "documentary_context_only"
    # unknown role but explicit geometry present
    if event_type in FLOOD_TYPES:
        if geometry_type == "point":
            return "observed_flood_point_strong" if has_date else "observed_flood_bbox_moderate"
        if geometry_type in ("polygon", "multipolygon", "bbox"):
            return "observed_flood_bbox_moderate"
    return "documentary_context_only"


def confidence_for(level: str) -> str:
    return {
        "observed_flood_polygon_strong": "high",
        "observed_flood_point_strong": "high",
        "observed_flood_bbox_moderate": "medium",
        "official_occurrence_point_moderate": "medium",
        "risk_area_context": "low",
        "alert_only": "low",
        "administrative_record_only": "low",
        "documentary_context_only": "low",
        "insufficient": "insufficient",
    }.get(level, "insufficient")


def extract_dates(text: str):
    """Return (event_date, period_start, period_end) parsed from text, else None."""
    m = DATE_RE.search(text or "")
    if not m:
        return None, None, None
    y, mo, d1, d2 = m.group(1), m.group(2), m.group(3), m.group(4)
    start = f"{y}-{mo}-{d1}"
    if d2:
        return start, start, f"{y}-{mo}-{d2}"
    return start, None, None


# --------------------------------------------------------------------------- #
# File parsers for the acquisition dir
# --------------------------------------------------------------------------- #
def _parse_geojson(p):
    obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    return geojson_points(obj)


def _parse_csv_latlon(p):
    rows = read_csv(p)
    if not rows:
        return [], []
    cols = {c.lower(): c for c in rows[0].keys()}
    latc = next((cols[k] for k in cols if k in ("lat", "latitude")), None)
    lonc = next((cols[k] for k in cols if k in ("lon", "long", "longitude")), None)
    if not latc or not lonc:
        return [], []
    pts = []
    for r in rows:
        try:
            pts.append((float(r[lonc]), float(r[latc])))
        except (ValueError, TypeError, KeyError):
            continue
    return pts, ["POINT"]


def _parse_wkt_text(p):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    types = re.findall(r"\b(POINT|POLYGON|MULTIPOLYGON|MULTIPOINT)\b", txt, re.IGNORECASE)
    pts = [(float(a), float(b)) for a, b in COORD_RE.findall(txt)]
    return pts, [t.upper() for t in types]


def _parse_kml_text(text):
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


def _geometry_type_from(pts, types):
    poly = any(t.upper() in {"POLYGON", "MULTIPOLYGON"} for t in types)
    if poly:
        return "polygon"
    return "point" if len(pts) == 1 else "point_set"


def parse_acquisition_files(audit):
    """Parse traceable files dropped/downloaded in the acquisition dir."""
    out = []
    files = [p for p in (DL_DIR.rglob("*") if DL_DIR.exists() else [])
             if p.is_file() and p.name != "README.md"]
    eid = 0
    for p in files:
        ext = p.suffix.lower()
        try:
            if ext in {".geojson", ".json"}:
                pts, types = _parse_geojson(p)
            elif ext in {".csv", ".tsv"}:
                pts, types = _parse_csv_latlon(p)
            elif ext in {".wkt", ".txt"}:
                pts, types = _parse_wkt_text(p)
            elif ext == ".kml":
                pts, types = _parse_kml_text(p.read_text(encoding="utf-8", errors="ignore"))
            elif ext == ".kmz":
                with zipfile.ZipFile(p) as z:
                    name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
                    pts, types = (_parse_kml_text(z.read(name).decode("utf-8", "ignore"))
                                  if name else ([], []))
            elif ext in {".shp", ".gpkg", ".dbf", ".shx", ".prj"}:
                audit.append({"source_file": rel(p), "result": "metadata_only_no_offline_parser",
                              "reason": ext, "n_coords": 0})
                continue
            elif ext == ".pdf":
                audit.append({"source_file": rel(p), "result": "pdf_text_skip_no_library",
                              "reason": "pdf", "n_coords": 0})
                continue
            else:
                audit.append({"source_file": rel(p), "result": "unsupported_ext",
                              "reason": ext, "n_coords": 0})
                continue
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            audit.append({"source_file": rel(p), "result": "parse_error",
                          "reason": str(e)[:50], "n_coords": 0})
            continue

        in_reg = [(lo, la) for lo, la in pts if in_brazil(lo, la)]
        if not in_reg:
            audit.append({"source_file": rel(p), "result": "no_valid_coordinate",
                          "reason": "none_in_brazil", "n_coords": len(pts)})
            continue
        bbox = bbox_of_points(in_reg) or []
        region = region_of_coord(in_reg[0][0], in_reg[0][1])
        gtype = _geometry_type_from(in_reg, types)
        role = derive_source_role(p.name, rel(p))
        etype = derive_event_type(role, region)
        edate, pstart, pend = extract_dates(p.name)
        level = classify_evidence(role, gtype, True, bool(edate), etype)
        eid += 1
        out.append(_event_row(
            f"OBSEV_DL_{eid:04d}", region, etype, edate, pstart, pend, p.name,
            derive_source_type(role, p.name), rel(p), gtype,
            (in_reg[0][1] if gtype == "point" else None),
            (in_reg[0][0] if gtype == "point" else None),
            ";".join(str(round(x, 6)) for x in bbox),
            "", (rel(p) if ext in {".geojson", ".json"} else ""), len(in_reg), role, level,
            "parsed from acquired traceable vector"))
        audit.append({"source_file": rel(p), "result": "parsed",
                      "reason": f"{gtype}:{region}:{level}", "n_coords": len(in_reg)})
    return out


def parse_local_traceable_coords(audit):
    """Ingest the already-traceable local event-geometry sources from SUSC-07B."""
    out = []
    if not COORDS_07B.exists():
        audit.append({"source_file": rel(COORDS_07B), "result": "absent_local_07b_coords",
                      "reason": "no_prior_extraction", "n_coords": 0})
        return out
    for i, c in enumerate(read_csv(COORDS_07B)):
        region = (c.get("region") or "").strip().lower()
        name = c.get("source_name", "")
        path = c.get("source_path_or_url", "")
        gtype = (c.get("geometry_type") or "unknown").strip().lower()
        bbox = (c.get("bbox") or "").strip()
        lat = (c.get("lat") or "").strip()
        lon = (c.get("lon") or "").strip()
        geojson_ref = (c.get("geojson_ref") or "").strip()
        has_geom = bool(bbox or (lat and lon))
        role = derive_source_role(name, path)
        etype = derive_event_type(role, region)
        edate, pstart, pend = extract_dates(f"{name} {path}")
        level = classify_evidence(role, gtype, has_geom, bool(edate), etype)
        n_coords = ""
        notes = c.get("notes", "")
        m = re.search(r"(\d+)\s*coord", notes)
        if m:
            n_coords = m.group(1)
        out.append(_event_row(
            f"OBSEV_LOC_{i:04d}", region, etype, edate, pstart, pend, name,
            derive_source_type(role, name), path, gtype, lat or None, lon or None,
            bbox, c.get("wkt", ""), geojson_ref, n_coords, role, level,
            "ingested traceable local SUSC-07B geometry source"))
        audit.append({"source_file": path, "result": "ingested_local",
                      "reason": f"{gtype}:{region}:{level}", "n_coords": n_coords or 0})
    return out


def _event_row(eid, region, etype, edate, pstart, pend, sname, stype, spath,
               gtype, lat, lon, bbox, wkt, geojson_ref, n_coords, role, level, notes):
    has_geom = bool(bbox or (lat and lon))
    date_or_period = edate or "unknown"
    if pend:
        date_or_period = f"{pstart or edate}..{pend}"
    return {
        "event_id": eid, "region": region, "municipality": MUNICIPALITY.get(region, region or "unknown"),
        "event_type": etype, "date_or_period": date_or_period,
        "event_date": edate or "", "event_period_start": pstart or "", "event_period_end": pend or "",
        "source_name": sname, "source_type": stype, "source_url_or_path": spath,
        "geometry_type": gtype, "lat": lat or "", "lon": lon or "", "bbox": bbox,
        "wkt": wkt or "", "geojson_ref": geojson_ref or "", "crs": "EPSG:4326",
        "n_coords": n_coords, "source_role": role, "evidence_level": level,
        "source_confidence": confidence_for(level),
        "can_link_to_patch": "true" if has_geom else "false",
        "requires_manual_review": "true", "can_be_ground_truth": "false",
        "allowed_for_training": "false", "review_only": "true", "notes": notes,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-11B Observed Event Parser (offline, traceable-only)")
    print("=" * 60)

    audit: list[dict] = []
    events = parse_acquisition_files(audit) + parse_local_traceable_coords(audit)

    for row in audit:
        row["review_only"] = "true"
    write_csv(PARSED, events, PARSED_FIELDS)
    write_csv(AUDIT, audit, ["source_file", "result", "reason", "n_coords", "review_only"])

    print(f"\nparsed observed-event records: {len(events)}")
    print("by evidence_level:", dict(Counter(e["evidence_level"] for e in events)))
    print("by region:", dict(Counter(e["region"] for e in events)))
    print("No coordinate invented. review-only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
