"""
SUSC-13A strong observed-event parser (review-only).

Reads controlled SUSC-13A manual/downloaded sources plus SUSC-11B traceable
catalog records. It strengthens classification where possible, but fails closed:
risk, alert, administrative and documentary context never become observed
events; strong levels require explicit geometry plus date/period.

Writes:
  - datasets/suscetibilidade/susc_13a_strong_observed_events_parsed_v1.csv
  - outputs_public/suscetibilidade/SUSC_13A_strong_event_parse_audit.csv
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
from susc_geometry import bbox_of_points, geojson_points, in_brazil, region_of_coord  # noqa: E402
from susc_io import read_csv, rel, write_csv  # noqa: E402

DROP_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13a"
DL_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13a_strong_event_download_manifest_v1.csv"
LEGACY_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc11b"
LEGACY_CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_event_catalog_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_observed_events_parsed_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13A_strong_event_parse_audit.csv"

MUNICIPALITY = {"recife": "Recife", "petropolis": "Petropolis", "curitiba": "Curitiba", "unknown": "unknown"}
REGION_EVENT_TYPE = {"recife": "urban_flooding", "petropolis": "flash_flood", "curitiba": "urban_flooding"}
DATE_RE = re.compile(r"(20\d{2})[_-](\d{2})[_-](\d{2})(?:[_-](\d{2}))?")
COORD_RE = re.compile(r"(-?\d{1,3}\.\d+)[,\s]+(-?\d{1,3}\.\d+)")

FIELDS = [
    "event_id",
    "region",
    "municipality",
    "event_type",
    "event_date",
    "event_period_start",
    "event_period_end",
    "source_name",
    "source_institution",
    "source_type",
    "source_url_or_path",
    "evidence_level",
    "geometry_type",
    "lat",
    "lon",
    "bbox",
    "wkt",
    "geojson_ref",
    "crs",
    "spatial_precision",
    "temporal_precision",
    "is_observed_event",
    "is_risk_area",
    "is_alert",
    "is_administrative_only",
    "can_link_to_patch",
    "requires_manual_review",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "notes",
]

AUDIT_FIELDS = [
    "source_file",
    "source_origin",
    "parse_status",
    "classification_status",
    "n_records",
    "n_coordinates",
    "requires_manual_review",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "notes",
]


def _date_from_text(text: str) -> tuple[str, str, str]:
    m = DATE_RE.search(text or "")
    if not m:
        return "", "", ""
    start = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if m.group(4):
        return start, start, f"{m.group(1)}-{m.group(2)}-{m.group(4)}"
    return start, "", ""


def _geometry_type(pts: list[tuple[float, float]], types: list[str]) -> str:
    if any(t.upper() in {"POLYGON", "MULTIPOLYGON"} for t in types):
        return "polygon"
    if len(pts) == 1:
        return "point"
    if len(pts) > 1:
        return "point_set"
    return "unknown"


def _parse_geojson(p: Path):
    obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    return geojson_points(obj)


def _read_csv_sniffed(p: Path):
    """Read a CSV/TSV trying common delimiters (',', ';', tab). Brazilian official
    data is frequently ';'-delimited. Returns list[dict] with None keys dropped."""
    import csv as _csv
    text = p.read_text(encoding="utf-8", errors="ignore")
    sample = text[:4096]
    delim = ","
    counts = {d: sample.count(d) for d in (",", ";", "\t")}
    delim = max(counts, key=lambda d: counts[d]) if any(counts.values()) else ","
    reader = _csv.DictReader(text.splitlines(), delimiter=delim)
    out = []
    for row in reader:
        out.append({k: v for k, v in row.items() if k is not None})
    return out


def _parse_csv_latlon(p: Path):
    rows = _read_csv_sniffed(p)
    if not rows:
        return [], []
    cols = {c.lower(): c for c in rows[0].keys() if c}
    latc = next((cols[c] for c in cols if c in {"lat", "latitude", "y", "coord_y"}), None)
    lonc = next((cols[c] for c in cols if c in {"lon", "long", "lng", "longitude", "x", "coord_x"}), None)
    if not latc or not lonc:
        return [], []
    pts = []
    for row in rows:
        try:
            lon = float(str(row.get(lonc, "")).replace(",", "."))
            lat = float(str(row.get(latc, "")).replace(",", "."))
        except ValueError:
            continue
        pts.append((lon, lat))
    return pts, ["POINT"]


def _parse_xlsx_latlon(p: Path):
    try:
        import openpyxl  # type: ignore
    except Exception:
        return [], ["XLSX_NO_OPENPYXL"]
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return [], ["XLSX_EMPTY"]
    header = [str(v or "").strip().lower() for v in rows[0]]
    lat_i = next((i for i, c in enumerate(header) if c in {"lat", "latitude", "y"}), None)
    lon_i = next((i for i, c in enumerate(header) if c in {"lon", "long", "longitude", "x"}), None)
    if lat_i is None or lon_i is None:
        return [], ["XLSX_NO_LATLON"]
    pts = []
    for row in rows[1:]:
        try:
            pts.append((float(row[lon_i]), float(row[lat_i])))
        except (TypeError, ValueError):
            continue
    return pts, ["POINT"]


def _parse_wkt_text(p: Path):
    txt = p.read_text(encoding="utf-8", errors="ignore")
    types = re.findall(r"\b(POINT|POLYGON|MULTIPOLYGON|MULTIPOINT)\b", txt, re.IGNORECASE)
    pts = [(float(a), float(b)) for a, b in COORD_RE.findall(txt)]
    return pts, [t.upper() for t in types]


def _parse_kml_text(text: str):
    pts = []
    for block in re.findall(r"<coordinates>(.*?)</coordinates>", text, re.DOTALL | re.IGNORECASE):
        for token in block.replace("\n", " ").split():
            parts = token.split(",")
            if len(parts) >= 2:
                try:
                    pts.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return pts, ["KML"]


def _text_from_pdf(p: Path) -> str:
    try:
        import pypdf  # type: ignore
    except Exception:
        return ""
    try:
        reader = pypdf.PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in reader.pages[:10])
    except Exception:
        return ""


def _role_from_text(name: str, path: str = "") -> str:
    low = f"{name} {path}".lower()
    if any(k in low for k in ("risco", "risk", "suscept", "suscet")):
        return "risk"
    if any(k in low for k in ("alert", "alerta", "aviso", "warning", "previs")):
        return "alert"
    if any(k in low for k in ("station", "estacao", "inmet", "catalog", "cadastro", "administr")):
        return "administrative"
    if any(k in low for k in ("charter", "mancha", "footprint", "flood_extent", "inunda", "alag")):
        return "flood_extent"
    if any(k in low for k in ("occurrence", "ocorrenc", "registro", "disaster", "desastre")):
        return "occurrence"
    if any(k in low for k in ("boundary", "aoi", "limite", "patch")):
        return "documentary"
    return "unknown"


def _institution_from(source_name: str, source_path: str) -> str:
    text = f"{source_name} {source_path}".lower()
    if "apac" in text:
        return "APAC"
    if "defesa" in text or "codecir" in text:
        return "Defesa Civil"
    if "cprm" in text or "sgb" in text:
        return "CPRM/SGB"
    if "inea" in text:
        return "INEA/RJ"
    if "ippuc" in text or "geocuritiba" in text:
        return "GeoCuritiba/IPPUC"
    if "s2id" in text:
        return "S2iD"
    if "charter" in text:
        return "International Charter"
    if "inmet" in text:
        return "INMET"
    if "manual" in text:
        return "manual_placement"
    return "unknown"


def _spatial_precision(geometry_type: str, has_geom: bool, level: str) -> str:
    if not has_geom:
        return "region"
    if geometry_type == "point":
        return "street"
    if geometry_type in {"polygon", "multipolygon", "bbox", "point_set"}:
        return "neighborhood"
    if level in {"documentary_only", "weak_administrative_context"}:
        return "municipality"
    return "unknown"


def _temporal_precision(edate: str, start: str, end: str) -> str:
    if start and end:
        return "period"
    if edate:
        return "day"
    return "unknown"


def _classify(role: str, geometry_type: str, has_geom: bool, has_date: bool, source_type: str) -> str:
    if role == "risk":
        return "weak_risk_area_context"
    if role == "alert":
        return "weak_alert_context"
    if role == "administrative":
        return "weak_administrative_context"
    if role == "documentary" or not has_geom:
        return "documentary_only" if role != "unknown" else "rejected_not_observed_event"
    if role == "flood_extent":
        if geometry_type in {"polygon", "multipolygon"} and has_date and source_type != "news":
            return "strong_observed_flood_polygon"
        if geometry_type == "point" and has_date and source_type != "news":
            return "strong_observed_flood_point"
        return "moderate_official_flood_bbox"
    if role == "occurrence":
        if geometry_type == "point" and has_date and source_type in {"official", "registry", "municipal", "state", "federal"}:
            return "strong_observed_flood_point"
        return "moderate_official_occurrence_point"
    return "documentary_only"


def _flags(level: str) -> dict[str, str]:
    return {
        "is_observed_event": "true" if level in {
            "strong_observed_flood_polygon",
            "strong_observed_flood_point",
            "moderate_official_occurrence_point",
            "moderate_official_flood_bbox",
        } else "false",
        "is_risk_area": "true" if level == "weak_risk_area_context" else "false",
        "is_alert": "true" if level == "weak_alert_context" else "false",
        "is_administrative_only": "true" if level == "weak_administrative_context" else "false",
    }


def _row(idx: int, *, region: str, event_type: str, edate: str, start: str, end: str,
         source_name: str, source_path: str, source_type: str, geometry_type: str,
         lat: str, lon: str, bbox: str, wkt: str, geojson_ref: str, notes: str) -> dict:
    region = (region or "unknown").lower()
    role = _role_from_text(source_name, source_path)
    has_geom = bool(bbox or (lat and lon))
    has_date = bool(edate or start)
    level = _classify(role, geometry_type, has_geom, has_date, source_type)
    flags = _flags(level)
    return {
        "event_id": f"SUSC13A_{idx:05d}",
        "region": region,
        "municipality": MUNICIPALITY.get(region, region or "unknown"),
        "event_type": event_type or REGION_EVENT_TYPE.get(region, "unknown"),
        "event_date": edate,
        "event_period_start": start,
        "event_period_end": end,
        "source_name": source_name,
        "source_institution": _institution_from(source_name, source_path),
        "source_type": source_type or "unknown",
        "source_url_or_path": source_path,
        "evidence_level": level,
        "geometry_type": geometry_type or "unknown",
        "lat": lat,
        "lon": lon,
        "bbox": bbox,
        "wkt": wkt,
        "geojson_ref": geojson_ref,
        "crs": "EPSG:4326",
        "spatial_precision": _spatial_precision(geometry_type, has_geom, level),
        "temporal_precision": _temporal_precision(edate, start, end),
        **flags,
        "can_link_to_patch": "true" if has_geom and level not in {"documentary_only", "rejected_not_observed_event"} else "false",
        "requires_manual_review": "true",
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
        "notes": notes,
    }


def _parse_manual_file(p: Path, audit: list[dict], start_idx: int) -> list[dict]:
    ext = p.suffix.lower()
    pts: list[tuple[float, float]] = []
    types: list[str] = []
    wkt = ""
    try:
        if ext in {".geojson", ".json"}:
            pts, types = _parse_geojson(p)
        elif ext in {".csv", ".tsv"}:
            pts, types = _parse_csv_latlon(p)
        elif ext == ".xlsx":
            pts, types = _parse_xlsx_latlon(p)
        elif ext in {".wkt", ".txt"}:
            pts, types = _parse_wkt_text(p)
            wkt = p.read_text(encoding="utf-8", errors="ignore")[:500]
        elif ext == ".kml":
            pts, types = _parse_kml_text(p.read_text(encoding="utf-8", errors="ignore"))
        elif ext == ".kmz":
            with zipfile.ZipFile(p) as z:
                name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
                if name:
                    pts, types = _parse_kml_text(z.read(name).decode("utf-8", "ignore"))
        elif ext == ".pdf":
            text = _text_from_pdf(p)
            pts = [(float(a), float(b)) for a, b in COORD_RE.findall(text)]
            types = ["PDF_TEXT"] if text else ["PDF_NO_TEXT_LIBRARY"]
        elif ext in {".zip", ".gpkg", ".shp"}:
            audit.append(_audit(rel(p), "susc13a_manual", "metadata_only_vector_container", "requires_manual_vector_review", 0, 0, f"{ext} needs GIS/manual review"))
            return []
        else:
            audit.append(_audit(rel(p), "susc13a_manual", "unsupported_extension", "not_classified", 0, 0, ext))
            return []
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        audit.append(_audit(rel(p), "susc13a_manual", "parse_error", "not_classified", 0, 0, str(exc)[:80]))
        return []

    valid = [(lon, lat) for lon, lat in pts if in_brazil(lon, lat)]
    if not valid:
        audit.append(_audit(rel(p), "susc13a_manual", "no_valid_coordinate", "not_classified", 0, len(pts), "no coordinate in Brazil bounds"))
        return []
    bbox = bbox_of_points(valid) or []
    region = region_of_coord(valid[0][0], valid[0][1])
    edate, start, end = _date_from_text(p.name)
    gtype = _geometry_type(valid, types)
    row = _row(
        start_idx,
        region=region,
        event_type=REGION_EVENT_TYPE.get(region, "unknown"),
        edate=edate,
        start=start,
        end=end,
        source_name=p.name,
        source_path=rel(p),
        source_type="manual",
        geometry_type=gtype,
        lat=str(valid[0][1]) if gtype == "point" else "",
        lon=str(valid[0][0]) if gtype == "point" else "",
        bbox=";".join(str(round(x, 6)) for x in bbox),
        wkt=wkt,
        geojson_ref=rel(p) if ext in {".geojson", ".json"} else "",
        notes="parsed from SUSC-13A manual/download directory",
    )
    audit.append(_audit(rel(p), "susc13a_manual", "parsed", row["evidence_level"], 1, len(valid), "review-only parsed manual source"))
    return [row]


def _map_legacy_level(level: str, source_name: str, source_type: str, geom: str, has_date: bool) -> str:
    if level == "observed_flood_polygon_strong":
        return "strong_observed_flood_polygon" if has_date and source_type != "news" else "moderate_official_flood_bbox"
    if level == "observed_flood_point_strong":
        return "strong_observed_flood_point" if has_date and source_type != "news" else "moderate_official_occurrence_point"
    if level == "observed_flood_bbox_moderate":
        return "moderate_official_flood_bbox"
    if level == "official_occurrence_point_moderate":
        return "moderate_official_occurrence_point"
    if level == "risk_area_context":
        return "weak_risk_area_context"
    if level == "alert_only":
        return "weak_alert_context"
    if level == "administrative_record_only":
        return "weak_administrative_context"
    if level == "documentary_context_only":
        return "documentary_only"
    role = _role_from_text(source_name)
    return _classify(role, geom, True, has_date, source_type)


def _legacy_rows(audit: list[dict], start_idx: int) -> list[dict]:
    if not LEGACY_CATALOG.exists():
        audit.append(_audit(rel(LEGACY_CATALOG), "susc11b_catalog", "missing", "not_classified", 0, 0, "legacy catalog absent"))
        return []
    out = []
    for i, r in enumerate(read_csv(LEGACY_CATALOG), start=start_idx):
        geom = (r.get("geometry_type") or "unknown").lower()
        edate = (r.get("event_date") or "").strip()
        start = (r.get("event_period_start") or "").strip()
        end = (r.get("event_period_end") or "").strip()
        has_date = bool(edate or start)
        level = _map_legacy_level(r.get("evidence_level", ""), r.get("source_name", ""), r.get("source_type", ""), geom, has_date)
        flags = _flags(level)
        has_geom = bool(r.get("bbox") or (r.get("lat") and r.get("lon")))
        row = {
            "event_id": f"SUSC13A_{i:05d}",
            "region": (r.get("region") or "unknown").lower(),
            "municipality": r.get("municipality", ""),
            "event_type": r.get("event_type") or REGION_EVENT_TYPE.get((r.get("region") or "").lower(), "unknown"),
            "event_date": edate,
            "event_period_start": start,
            "event_period_end": end,
            "source_name": r.get("source_name", ""),
            "source_institution": _institution_from(r.get("source_name", ""), r.get("source_url_or_path", "")),
            "source_type": r.get("source_type", ""),
            "source_url_or_path": r.get("source_url_or_path", ""),
            "evidence_level": level,
            "geometry_type": geom,
            "lat": r.get("lat", ""),
            "lon": r.get("lon", ""),
            "bbox": r.get("bbox", ""),
            "wkt": r.get("wkt", ""),
            "geojson_ref": r.get("geojson_ref", ""),
            "crs": r.get("crs", "EPSG:4326"),
            "spatial_precision": r.get("spatial_precision", _spatial_precision(geom, has_geom, level)),
            "temporal_precision": r.get("temporal_precision", _temporal_precision(edate, start, end)),
            **flags,
            "can_link_to_patch": "true" if has_geom and level not in {"documentary_only", "rejected_not_observed_event"} else "false",
            "requires_manual_review": "true",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "notes": "reclassified conservatively from SUSC-11B catalog; " + r.get("notes", ""),
        }
        out.append(row)
    audit.append(_audit(rel(LEGACY_CATALOG), "susc11b_catalog", "ingested", "legacy_reclassified", len(out), 0, "SUSC-11B catalog reused conservatively"))
    if LEGACY_DIR.exists():
        audit.append(_audit(rel(LEGACY_DIR), "susc11b_drop_dir", "checked", "no_direct_promotion", 0, 0, "legacy acquisition dir retained as source context"))
    if DL_MANIFEST.exists():
        audit.append(_audit(rel(DL_MANIFEST), "susc13a_download_manifest", "checked", "download_status_available", len(read_csv(DL_MANIFEST)), 0, "download manifest reviewed for local files"))
    return out


def _audit(source_file: str, origin: str, parse_status: str, classification_status: str,
           n_records: int, n_coordinates: int, notes: str) -> dict:
    return {
        "source_file": source_file,
        "source_origin": origin,
        "parse_status": parse_status,
        "classification_status": classification_status,
        "n_records": n_records,
        "n_coordinates": n_coordinates,
        "requires_manual_review": "true",
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
        "notes": notes,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-13A Strong Observed Event Parser")
    print("=" * 60)
    audit: list[dict] = []
    rows: list[dict] = []

    manual_files = sorted(p for p in DROP_DIR.rglob("*") if p.is_file() and p.name != "README.md") if DROP_DIR.exists() else []
    for p in manual_files:
        rows.extend(_parse_manual_file(p, audit, len(rows)))
    rows.extend(_legacy_rows(audit, len(rows)))

    write_csv(PARSED, rows, FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)
    print(f"parsed rows: {len(rows)}")
    print("by evidence_level:", dict(Counter(r["evidence_level"] for r in rows)))
    print("manual files parsed:", len(manual_files))
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
