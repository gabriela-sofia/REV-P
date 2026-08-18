"""
SUSC-14A FASE 4 - parse acquired official reference layers into reference features.

Reads the FASE 3 download manifest, parses each registered/reused official file
(GeoJSON/JSON/CSV/KML/KMZ) into normalized reference features classified as
street_segment, address_point, neighborhood_polygon, flood_point, flood_polygon,
risk_area, drainage_line, river_line or unknown. Address points and street
references carry an official lat/lon used later for geocoding; risk/suscept
layers are classified as risk_area and never become occurrences.

Writes:
  - datasets/suscetibilidade/susc_14a_official_geocoding_reference_features_v1.csv
  - outputs_public/suscetibilidade/SUSC_14A_official_reference_parse_audit.csv
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_geometry import bbox_of_points, in_brazil, region_of_coord  # noqa: E402
from susc_io import read_csv, rel, write_csv  # noqa: E402

MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14a_official_reference_download_manifest_v1.csv"
FEATURES = ROOT / "datasets" / "suscetibilidade" / "susc_14a_official_geocoding_reference_features_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_official_reference_parse_audit.csv"

FIELDS = [
    "reference_feature_id", "region", "source_id", "feature_type", "name",
    "normalized_name", "neighborhood", "normalized_neighborhood", "geometry_type",
    "lat", "lon", "bbox", "wkt", "geojson_ref", "crs", "source_confidence",
    "parse_confidence", "review_only",
]
AUDIT_FIELDS = [
    "source_file", "source_id", "parse_status", "n_features", "n_address_points",
    "n_street", "n_neighborhood", "n_flood", "n_risk", "review_only", "notes",
]

NAME_KEYS = ["endereco", "logradouro", "nome", "name", "descricao", "local", "localidade", "rua", "via"]
HOOD_KEYS = ["bairro", "neighborhood", "regional", "distrito", "localidade"]
LAT_KEYS = ["latitude", "lat", "y", "coord_y"]
LON_KEYS = ["longitude", "lon", "long", "lng", "x", "coord_x"]


def _prop(props_norm: dict, keys: list[str]) -> str:
    for k in keys:
        if k in props_norm and str(props_norm[k]).strip():
            return str(props_norm[k]).strip()
    return ""


def _classify_feature(name: str, neighborhood: str, geom_type: str, source_name: str) -> str:
    text = g14.normalize_text(" ".join([name, neighborhood, source_name]))
    if g14.looks_risk_only(name, source_name) or "risco" in text or "suscet" in text:
        if geom_type in {"polygon", "multipolygon"}:
            return "risk_area"
        # a risk address point still anchors a real official address -> address_point
    if any(t in text for t in ("drenagem", "canal", "galeria")):
        return "drainage_line"
    if any(t in text for t in ("rio", "riacho", "corrego", "igarape")):
        return "river_line"
    if any(t in text for t in ("alagament", "inundac", "enchente", "mancha")):
        return "flood_polygon" if geom_type in {"polygon", "multipolygon"} else "flood_point"
    if geom_type in {"polygon", "multipolygon"} and any(t in text for t in ("bairro", "distrito")):
        return "neighborhood_polygon"
    if geom_type == "point" and (name or neighborhood):
        # an official georeferenced address/point is an address reference
        return "address_point"
    if any(t in text for t in ("logradouro", "eixo", "via", "rua", "avenida", "trecho")):
        return "street_segment"
    if geom_type in {"linestring", "multilinestring"}:
        return "street_segment"
    return "unknown"


def _geom_type_of(geom: dict) -> str:
    return {"Point": "point", "MultiPoint": "point", "LineString": "linestring",
            "MultiLineString": "multilinestring", "Polygon": "polygon",
            "MultiPolygon": "multipolygon"}.get(geom.get("type", ""), "unknown")


def _first_point(coords) -> tuple | None:
    if isinstance(coords, list):
        if coords and isinstance(coords[0], (int, float)) and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
        for c in coords:
            p = _first_point(c)
            if p:
                return p
    return None


def _all_points(coords, out):
    if isinstance(coords, list):
        if coords and isinstance(coords[0], (int, float)) and len(coords) >= 2:
            out.append((float(coords[0]), float(coords[1])))
        else:
            for c in coords:
                _all_points(c, out)


def _parse_geojson(p: Path, source_id: str, source_name: str, audit: list[dict]) -> list[dict]:
    try:
        obj = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, ValueError) as exc:
        audit.append(_audit(rel(p), source_id, "parse_error", 0, 0, 0, 0, 0, 0, str(exc)[:80]))
        return []
    feats = obj.get("features", []) if obj.get("type") == "FeatureCollection" else (
        [obj] if obj.get("type") == "Feature" else [])
    out: list[dict] = []
    for ft in feats:
        geom = ft.get("geometry") or {}
        gtype = _geom_type_of(geom)
        props = ft.get("properties", {}) or {}
        props_norm = {g14.normalize_text(k): v for k, v in props.items()}
        name = _prop(props_norm, NAME_KEYS)
        hood = _prop(props_norm, HOOD_KEYS)
        lat = _prop(props_norm, LAT_KEYS)
        lon = _prop(props_norm, LON_KEYS)
        pt = None
        if lat and lon:
            try:
                pt = (float(str(lon).replace(",", ".")), float(str(lat).replace(",", ".")))
            except ValueError:
                pt = None
        if pt is None:
            pt = _first_point(geom.get("coordinates"))
        if pt is None or not in_brazil(pt[0], pt[1]):
            continue
        allpts: list = []
        _all_points(geom.get("coordinates"), allpts)
        bbox = bbox_of_points([q for q in allpts if in_brazil(q[0], q[1])]) if allpts else None
        ftype = _classify_feature(name, hood, gtype, source_name)
        region = region_of_coord(pt[0], pt[1])
        out.append({
            "region": region, "source_id": source_id, "feature_type": ftype,
            "name": name[:160], "normalized_name": g14.normalize_street(name),
            "neighborhood": hood[:80], "normalized_neighborhood": g14.normalize_text(hood),
            "geometry_type": gtype, "lat": round(pt[1], 6), "lon": round(pt[0], 6),
            "bbox": ";".join(str(round(x, 6)) for x in bbox) if bbox else "",
            "wkt": "", "geojson_ref": rel(p), "crs": "EPSG:4326",
            "source_confidence": "medium", "parse_confidence": "medium" if name else "low",
            "review_only": "true",
        })
    counts = Counter(r["feature_type"] for r in out)
    audit.append(_audit(rel(p), source_id, "parsed", len(out),
                        counts.get("address_point", 0), counts.get("street_segment", 0),
                        counts.get("neighborhood_polygon", 0),
                        counts.get("flood_point", 0) + counts.get("flood_polygon", 0),
                        counts.get("risk_area", 0), "camada oficial parseada review-only"))
    return out


def _parse_csv_points(p: Path, source_id: str, source_name: str, audit: list[dict]) -> list[dict]:
    rows = g14.read_brazilian_csv(p)
    if not rows:
        audit.append(_audit(rel(p), source_id, "empty_or_no_latlon", 0, 0, 0, 0, 0, 0, "csv sem linhas"))
        return []
    keys = list(rows[0].keys())
    latk = next((k for k in keys if k in LAT_KEYS), None)
    lonk = next((k for k in keys if k in LON_KEYS), None)
    if not latk or not lonk:
        audit.append(_audit(rel(p), source_id, "csv_no_official_latlon", 0, 0, 0, 0, 0, 0,
                            "CSV de ocorrencia sem lat/lon: nao vira referencia"))
        return []
    out: list[dict] = []
    for r in rows:
        try:
            lon = float(str(r.get(lonk, "")).replace(",", "."))
            lat = float(str(r.get(latk, "")).replace(",", "."))
        except ValueError:
            continue
        if not in_brazil(lon, lat):
            continue
        name = g14.header_pick(r, *NAME_KEYS)
        hood = g14.header_pick(r, *HOOD_KEYS)
        out.append({
            "region": region_of_coord(lon, lat), "source_id": source_id,
            "feature_type": _classify_feature(name, hood, "point", source_name),
            "name": name[:160], "normalized_name": g14.normalize_street(name),
            "neighborhood": hood[:80], "normalized_neighborhood": g14.normalize_text(hood),
            "geometry_type": "point", "lat": round(lat, 6), "lon": round(lon, 6),
            "bbox": "", "wkt": "", "geojson_ref": rel(p), "crs": "EPSG:4326",
            "source_confidence": "medium", "parse_confidence": "medium" if name else "low",
            "review_only": "true",
        })
    counts = Counter(r["feature_type"] for r in out)
    audit.append(_audit(rel(p), source_id, "parsed", len(out), counts.get("address_point", 0),
                        counts.get("street_segment", 0), counts.get("neighborhood_polygon", 0),
                        counts.get("flood_point", 0) + counts.get("flood_polygon", 0),
                        counts.get("risk_area", 0), "csv georreferenciado oficial parseado"))
    return out


def _audit(source_file, source_id, status, nf, na, ns, nn, nfl, nr, notes) -> dict:
    return {
        "source_file": source_file, "source_id": source_id, "parse_status": status,
        "n_features": nf, "n_address_points": na, "n_street": ns, "n_neighborhood": nn,
        "n_flood": nfl, "n_risk": nr, "review_only": "true", "notes": notes,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Official Reference Parser")
    print("=" * 60)
    audit: list[dict] = []
    features: list[dict] = []
    if not MANIFEST.exists():
        print(f"STOP: missing manifest {rel(MANIFEST)}")
        return 1
    seen: set[str] = set()
    for r in read_csv(MANIFEST):
        fp = (r.get("file_path") or "").strip()
        if not fp or fp in seen:
            continue
        seen.add(fp)
        p = ROOT / fp
        if not p.exists():
            audit.append(_audit(fp, r.get("reference_id", ""), "missing_local_file", 0, 0, 0, 0, 0, 0, "arquivo nao encontrado"))
            continue
        ext = p.suffix.lower()
        sid = r.get("reference_id", "")
        sname = r.get("source_name", "")
        if ext in {".geojson", ".json"}:
            features.extend(_parse_geojson(p, sid, sname, audit))
        elif ext in {".csv", ".tsv"}:
            features.extend(_parse_csv_points(p, sid, sname, audit))
        elif ext in {".kml", ".kmz", ".gpkg", ".zip", ".shp", ".xlsx", ".pdf"}:
            audit.append(_audit(fp, sid, "needs_manual_vector_review", 0, 0, 0, 0, 0, 0,
                                f"{ext} requer revisao GIS/manual; nao parseado automaticamente"))
        else:
            audit.append(_audit(fp, sid, "unsupported_extension", 0, 0, 0, 0, 0, 0, ext))

    for i, f in enumerate(features):
        f["reference_feature_id"] = f"S14AFEAT_{i:06d}"
    features = [{k: f.get(k, "") for k in FIELDS} for f in features]
    write_csv(FEATURES, features, FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)

    by_type = Counter(f["feature_type"] for f in features)
    print(f"reference features: {len(features)} | by type: {dict(by_type)}")
    print(f"audited source files: {len(audit)}")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
