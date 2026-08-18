"""
SUSC-13B-AUTO universal parser of auto-discovered/official sources.

Parses files acquired or dropped into the SUSC-13B acquisition directory and
classifies each into an evidence level, failing closed: risk, alert,
administrative and documentary context never become observed events; strong
levels require explicit geometry plus date/period. Reuses the SUSC-13A parsing
primitives and adds ArcGIS/WFS GeoJSON, generic JSON geometry and basic XML/GML.
Missing optional parser dependencies are recorded, never auto-installed.

Writes:
  - datasets/suscetibilidade/susc_13b_auto_observed_events_parsed_v1.csv
  - outputs_public/suscetibilidade/SUSC_13B_auto_parse_audit.csv
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
import parse_susc_13a_strong_observed_events as p13a  # noqa: E402
import susc_13b_web_discovery_common as wd  # noqa: E402
from susc_geometry import bbox_of_points, geojson_points, in_brazil, region_of_coord  # noqa: E402
from susc_io import read_csv, rel, write_csv  # noqa: E402

DROP_DIR = ROOT / "datasets" / "suscetibilidade" / "observed_event_sources_susc13b_auto"
DL_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_download_manifest_v1.csv"
DISCOVERED = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_discovered_sources_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_observed_events_parsed_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_parse_audit.csv"

MUNICIPALITY = {"recife": "Recife", "petropolis": "Petropolis", "curitiba": "Curitiba", "unknown": "unknown"}
REGION_EVENT_TYPE = {"recife": "urban_flooding", "petropolis": "flash_flood", "curitiba": "urban_flooding"}

FIELDS = [
    "event_id", "source_discovery_id", "region", "municipality", "event_type",
    "event_date", "event_period_start", "event_period_end", "source_institution",
    "source_title", "source_url", "local_file_path", "source_format", "evidence_level",
    "geometry_type", "lat", "lon", "bbox", "wkt", "geojson_ref", "crs",
    "spatial_precision", "temporal_precision", "is_observed_event", "is_risk_area",
    "is_alert", "is_administrative_only", "source_confidence", "parse_confidence",
    "requires_manual_review", "can_be_ground_truth", "allowed_for_training",
    "review_only", "classification_reason", "notes",
]

AUDIT_FIELDS = [
    "source_file", "discovery_id", "source_format", "parse_status",
    "classification_status", "n_records", "n_coordinates", "missing_dependency",
    "requires_manual_review", "can_be_ground_truth", "allowed_for_training",
    "review_only", "notes",
]


def _audit(source_file, discovery_id, source_format, parse_status, classification_status,
           n_records, n_coordinates, missing_dependency, notes) -> dict:
    return {
        "source_file": source_file, "discovery_id": discovery_id, "source_format": source_format,
        "parse_status": parse_status, "classification_status": classification_status,
        "n_records": n_records, "n_coordinates": n_coordinates,
        "missing_dependency": missing_dependency, "requires_manual_review": "true",
        "can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true",
        "notes": notes,
    }


def _parse_generic_json(p: Path):
    """GeoJSON or generic JSON containing coordinates/geometry."""
    obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    pts, types = geojson_points(obj)
    if pts:
        return pts, types
    # generic search for lat/lon-ish keys
    pts2: list[tuple[float, float]] = []

    def walk(o):
        if isinstance(o, dict):
            lat = o.get("lat") or o.get("latitude") or o.get("y")
            lon = o.get("lon") or o.get("lng") or o.get("longitude") or o.get("x")
            try:
                if lat is not None and lon is not None:
                    pts2.append((float(lon), float(lat)))
            except (TypeError, ValueError):
                pass
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return pts2, ["JSON_GENERIC"] if pts2 else []


def _parse_xml_gml(p: Path):
    text = p.read_text(encoding="utf-8", errors="ignore")
    # gml:pos / gml:coordinates / posList
    pts: list[tuple[float, float]] = []
    for block in re.findall(r"<(?:gml:)?(?:pos|coordinates|posList)>([^<]+)</", text, re.IGNORECASE):
        nums = re.findall(r"-?\d+\.\d+", block)
        for i in range(0, len(nums) - 1, 2):
            try:
                a, b = float(nums[i]), float(nums[i + 1])
                # GML pos is often lat lon; accept whichever lands in Brazil
                if in_brazil(a, b):
                    pts.append((a, b))
                elif in_brazil(b, a):
                    pts.append((b, a))
            except ValueError:
                continue
    types = ["GML"] if pts else []
    return pts, types


def _classify(role: str, geometry_type: str, has_geom: bool, has_date: bool, source_type: str) -> tuple[str, str]:
    """Return (evidence_level, reason)."""
    if role == "risk":
        return "weak_risk_area_context", "fonte indica risco/suscetibilidade; nunca evento observado"
    if role == "alert":
        return "weak_alert_context", "fonte indica alerta/aviso/previsao; nunca ocorrencia observada"
    if role == "administrative":
        return "weak_administrative_context", "cadastro/estacao/limite sem geometria de evento"
    if not has_geom:
        return ("documentary_only", "sem geometria explicita reutilizavel") if role != "unknown" \
            else ("rejected_not_observed_event", "sem geometria nem rastreabilidade de evento")
    if role == "flood_extent":
        if geometry_type in {"polygon", "multipolygon"} and has_date and source_type != "news":
            return "strong_observed_flood_polygon", "footprint poligonal de area alagada com data"
        if geometry_type == "point" and has_date and source_type != "news":
            return "strong_observed_flood_point", "ponto de inundacao ocorrida com data"
        return "moderate_official_flood_bbox", "mancha/area aproximada de inundacao sem validacao forte"
    if role == "occurrence":
        if geometry_type == "point" and has_date and source_type in {"official", "registry", "municipal", "state", "federal", "manual"}:
            return "strong_observed_flood_point", "ocorrencia oficial pontual georreferenciada com data"
        return "moderate_official_occurrence_point", "ocorrencia oficial sem footprint ou com lacuna temporal"
    return "documentary_only", "contexto sem classificacao forte"


def _flags(level: str) -> dict[str, str]:
    return {
        "is_observed_event": "true" if level in {
            "strong_observed_flood_polygon", "strong_observed_flood_point",
            "moderate_official_occurrence_point", "moderate_official_flood_bbox"} else "false",
        "is_risk_area": "true" if level == "weak_risk_area_context" else "false",
        "is_alert": "true" if level == "weak_alert_context" else "false",
        "is_administrative_only": "true" if level == "weak_administrative_context" else "false",
    }


def _source_conf(level: str, temporal_precision: str) -> str:
    if level in {"strong_observed_flood_polygon", "strong_observed_flood_point"}:
        return "high"
    if level in {"moderate_official_occurrence_point", "moderate_official_flood_bbox"}:
        return "medium" if temporal_precision != "unknown" else "medium_temporal_gap"
    return "low"


def _parse_file(p: Path, discovery_id: str, source_url: str, source_title: str,
                source_format: str, audit: list[dict], idx: int) -> list[dict]:
    ext = p.suffix.lower()
    pts: list[tuple[float, float]] = []
    types: list[str] = []
    wkt = ""
    try:
        if ext in {".geojson", ".json"}:
            pts, types = _parse_generic_json(p)
        elif ext in {".csv", ".tsv"}:
            pts, types = p13a._parse_csv_latlon(p)
        elif ext in {".xlsx", ".xls"}:
            pts, types = p13a._parse_xlsx_latlon(p)
            if types and "NO_OPENPYXL" in types[0]:
                audit.append(_audit(rel(p), discovery_id, source_format, "blocked_missing_parser_dependency",
                                    "not_classified", 0, 0, "openpyxl", "XLSX requer openpyxl; nao instalado"))
                return []
        elif ext in {".wkt", ".txt"}:
            pts, types = p13a._parse_wkt_text(p)
            wkt = p.read_text(encoding="utf-8", errors="ignore")[:500]
        elif ext == ".kml":
            pts, types = p13a._parse_kml_text(p.read_text(encoding="utf-8", errors="ignore"))
        elif ext == ".kmz":
            with zipfile.ZipFile(p) as z:
                name = next((n for n in z.namelist() if n.lower().endswith(".kml")), None)
                if name:
                    pts, types = p13a._parse_kml_text(z.read(name).decode("utf-8", "ignore"))
        elif ext in {".xml", ".gml"}:
            pts, types = _parse_xml_gml(p)
        elif ext == ".pdf":
            text = p13a._text_from_pdf(p)
            if not text:
                audit.append(_audit(rel(p), discovery_id, source_format, "blocked_missing_parser_dependency",
                                    "documentary_only", 0, 0, "pypdf", "PDF sem texto extraivel ou sem pypdf"))
                return []
            pts = [(float(a), float(b)) for a, b in p13a.COORD_RE.findall(text)]
            types = ["PDF_TEXT"]
        elif ext in {".gpkg", ".shp"}:
            audit.append(_audit(rel(p), discovery_id, source_format, "blocked_missing_parser_dependency",
                                "requires_manual_vector_review", 0, 0, "geopandas/fiona",
                                f"{ext} requer GIS; revisao manual"))
            return []
        elif ext == ".zip":
            audit.append(_audit(rel(p), discovery_id, source_format, "metadata_only_vector_container",
                                "requires_manual_vector_review", 0, 0, "", "ZIP vetorial; revisao manual"))
            return []
        else:
            audit.append(_audit(rel(p), discovery_id, source_format, "unsupported_extension",
                                "not_classified", 0, 0, "", ext))
            return []
    except (OSError, ValueError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        audit.append(_audit(rel(p), discovery_id, source_format, "parse_error", "not_classified",
                            0, 0, "", str(exc)[:80]))
        return []

    valid = [(lon, lat) for lon, lat in pts if in_brazil(lon, lat)]
    if not valid:
        audit.append(_audit(rel(p), discovery_id, source_format, "no_valid_coordinate",
                            "not_classified", 0, len(pts), "", "sem coordenada dentro do Brasil"))
        return []
    bbox = bbox_of_points(valid) or []
    region = region_of_coord(valid[0][0], valid[0][1])
    edate, start, end = p13a._date_from_text(f"{p.name} {source_title}")
    gtype = p13a._geometry_type(valid, types)
    role = p13a._role_from_text(f"{source_title} {p.name}", source_url)
    has_geom = bool(bbox or valid)
    has_date = bool(edate or start)
    level, reason = _classify(role, gtype, has_geom, has_date, "official")
    flags = _flags(level)
    temporal_precision = p13a._temporal_precision(edate, start, end)
    row = {
        "event_id": f"SUSC13B_{idx:05d}", "source_discovery_id": discovery_id,
        "region": region, "municipality": MUNICIPALITY.get(region, region or "unknown"),
        "event_type": REGION_EVENT_TYPE.get(region, "unknown"), "event_date": edate,
        "event_period_start": start, "event_period_end": end,
        "source_institution": p13a._institution_from(source_title, source_url),
        "source_title": source_title or p.name, "source_url": source_url, "local_file_path": rel(p),
        "source_format": source_format or ext.lstrip("."), "evidence_level": level,
        "geometry_type": gtype, "lat": str(valid[0][1]) if gtype == "point" else "",
        "lon": str(valid[0][0]) if gtype == "point" else "",
        "bbox": ";".join(str(round(x, 6)) for x in bbox), "wkt": wkt,
        "geojson_ref": rel(p) if ext in {".geojson", ".json"} else "", "crs": "EPSG:4326",
        "spatial_precision": p13a._spatial_precision(gtype, has_geom, level),
        "temporal_precision": temporal_precision, **flags,
        "source_confidence": _source_conf(level, temporal_precision),
        "parse_confidence": "medium" if has_date else "low",
        "requires_manual_review": "true", "can_be_ground_truth": "false",
        "allowed_for_training": "false", "review_only": "true",
        "classification_reason": reason,
        "notes": f"parseado do SUSC-13B auto; {len(valid)} coordenada(s) valida(s)",
    }
    audit.append(_audit(rel(p), discovery_id, source_format, "parsed", level, 1, len(valid), "",
                        "fonte parseada review-only"))
    return [row]


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Universal Observed Event Parser")
    print("=" * 60)
    audit: list[dict] = []
    rows: list[dict] = []

    # Map downloaded files to their discovery metadata via the download manifest.
    file_meta: dict[str, dict] = {}
    if DL_MANIFEST.exists():
        for r in read_csv(DL_MANIFEST):
            fp = (r.get("file_path") or "").strip()
            if fp:
                file_meta[fp] = r

    files = sorted(p for p in DROP_DIR.rglob("*") if p.is_file() and p.name != "README.md") if DROP_DIR.exists() else []
    for p in files:
        meta = file_meta.get(rel(p), {})
        rows.extend(_parse_file(
            p, meta.get("discovery_id", "MANUAL"), meta.get("url_or_local_path", rel(p)),
            meta.get("source_title", p.name), meta.get("content_ext", p.suffix.lstrip(".")),
            audit, len(rows)))

    if not files:
        net = wd.network_enabled()
        audit.append(_audit("", "", "", "no_files_to_parse", "no_observed_events",
                            0, 0, "",
                            "diretorio de aquisicao vazio (offline/sem download); rede=%s" % net))

    write_csv(PARSED, rows, FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)
    print(f"parsed rows: {len(rows)}")
    print("by evidence_level:", dict(Counter(r["evidence_level"] for r in rows)) or "none")
    print(f"files seen: {len(files)} | audit rows: {len(audit)}")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
