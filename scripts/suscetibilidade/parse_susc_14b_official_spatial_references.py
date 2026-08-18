"""SUSC-14B FASE 3 - parse official spatial references into features."""

from __future__ import annotations

import csv
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

MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14b_official_spatial_reference_download_manifest_v1.csv"
FEATURES = ROOT / "datasets" / "suscetibilidade" / "susc_14b_official_spatial_reference_features_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14B_official_spatial_reference_parse_audit.csv"

FIELDS = [
    "reference_feature_id", "region", "source_id", "feature_type", "name", "normalized_name",
    "street_type", "neighborhood", "normalized_neighborhood", "geometry_type", "lat", "lon",
    "bbox", "wkt", "geojson_ref", "crs", "source_confidence", "parse_confidence",
    "allowed_for_training", "can_be_ground_truth", "can_be_used_as_ground_truth", "review_only",
]
AUDIT_FIELDS = ["source_id", "source_file", "parse_status", "n_features", "by_feature_type", "review_only", "notes"]
MAX_FEATURES_TOTAL = 100000
MAX_FEATURES_PER_SOURCE = 10000
NAME_KEYS = ["endereco", "logradouro", "nome", "name", "descricao", "local", "rua", "via", "titulo"]
HOOD_KEYS = ["bairro", "neighborhood", "regional", "distrito", "localidade"]
LAT_KEYS = ["latitude", "lat", "y", "coord_y"]
LON_KEYS = ["longitude", "lon", "long", "lng", "x", "coord_x"]


def _prop(props, keys):
    norm = {g14.normalize_text(k): v for k, v in props.items()}
    for key in keys:
        nk = g14.normalize_text(key)
        if nk in norm and str(norm[nk]).strip():
            return str(norm[nk]).strip()
    return ""


def _street_type(name):
    n = g14.normalize_text(name)
    if n.startswith("avenida "):
        return "avenida"
    if n.startswith("travessa "):
        return "travessa"
    if n.startswith("estrada "):
        return "estrada"
    if n.startswith("rua "):
        return "rua"
    return ""


def _classify(text, geom_type):
    low = g14.normalize_text(text)
    if any(t in low for t in ("drenagem", "canal", "galeria")):
        return "drainage_line"
    if any(t in low for t in ("hidrografia", "rio", "riacho", "corrego")):
        return "river_line"
    if any(t in low for t in ("alagamento", "inundacao", "enchente", "ponto critico")):
        return "flood_polygon" if geom_type in {"polygon", "multipolygon"} else "flood_point"
    if "risco" in low or "suscet" in low:
        return "risk_area"
    if any(t in low for t in ("bairro", "bairros")) or geom_type in {"polygon", "multipolygon"}:
        return "neighborhood_polygon"
    if any(t in low for t in ("logradouro", "eixo", "via", "rua", "avenida", "travessa", "estrada", "arruamento")):
        return "street_segment"
    if geom_type in {"linestring", "multilinestring"}:
        return "street_segment"
    if geom_type == "point":
        return "address_point"
    return "unknown"


def _geom_type(geom):
    return {
        "Point": "point", "MultiPoint": "point", "LineString": "linestring",
        "MultiLineString": "multilinestring", "Polygon": "polygon", "MultiPolygon": "multipolygon",
    }.get((geom or {}).get("type", ""), "unknown")


def _first_point(coords):
    if isinstance(coords, list):
        if coords and isinstance(coords[0], (int, float)) and len(coords) >= 2:
            return float(coords[0]), float(coords[1])
        for item in coords:
            point = _first_point(item)
            if point:
                return point
    return None


def _all_points(coords, out):
    if isinstance(coords, list):
        if coords and isinstance(coords[0], (int, float)) and len(coords) >= 2:
            out.append((float(coords[0]), float(coords[1])))
        else:
            for item in coords:
                _all_points(item, out)


def _feature_row(source_id, source_file, region, ftype, name, hood, geom_type, lon="", lat="", bbox="", wkt="", conf="low"):
    return {
        "reference_feature_id": "",
        "region": region,
        "source_id": source_id,
        "feature_type": ftype,
        "name": name[:180],
        "normalized_name": g14.normalize_street(name),
        "street_type": _street_type(g14.normalize_street(name)),
        "neighborhood": hood[:120],
        "normalized_neighborhood": g14.normalize_text(hood),
        "geometry_type": geom_type,
        "lat": lat,
        "lon": lon,
        "bbox": bbox,
        "wkt": wkt,
        "geojson_ref": source_file,
        "crs": "EPSG:4326",
        "source_confidence": "medium",
        "parse_confidence": conf,
        "allowed_for_training": "false",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "review_only": "true",
    }


def _parse_geojson(path, source_id, title):
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except ValueError:
        return []
    if isinstance(obj, list):
        out = []
        for item in obj[:50000]:
            if not isinstance(item, dict):
                continue
            name = _prop(item, NAME_KEYS) or title
            hood = _prop(item, HOOD_KEYS)
            lat = _prop(item, LAT_KEYS)
            lon = _prop(item, LON_KEYS)
            flon = flat = ""
            region = ""
            if lat and lon:
                try:
                    flon_f = float(str(lon).replace(",", "."))
                    flat_f = float(str(lat).replace(",", "."))
                    if in_brazil(flon_f, flat_f):
                        flon, flat = round(flon_f, 6), round(flat_f, 6)
                        region = region_of_coord(flon_f, flat_f)
                except ValueError:
                    pass
            geom_type = "point" if flon and flat else "table"
            ftype = _classify(f"{name} {hood} {title}", geom_type)
            if ftype == "unknown":
                continue
            out.append(_feature_row(source_id, rel(path), region, ftype,
                                    name, hood, geom_type, flon, flat, "", "", "medium" if name else "low"))
            if len(out) >= MAX_FEATURES_PER_SOURCE:
                break
        return out
    if not isinstance(obj, dict):
        return []
    feats = obj.get("features", []) if obj.get("type") == "FeatureCollection" else ([obj] if obj.get("type") == "Feature" else [])
    out = []
    for feat in feats[:50000]:
        geom = feat.get("geometry") or {}
        gtype = _geom_type(geom)
        props = feat.get("properties") or {}
        name = _prop(props, NAME_KEYS) or title
        hood = _prop(props, HOOD_KEYS)
        pts = []
        _all_points(geom.get("coordinates"), pts)
        pts = [p for p in pts if in_brazil(p[0], p[1])]
        point = _first_point(geom.get("coordinates"))
        lon = lat = ""
        if point and in_brazil(point[0], point[1]):
            lon, lat = round(point[0], 6), round(point[1], 6)
            region = region_of_coord(point[0], point[1])
        else:
            region = ""
        bbox = bbox_of_points(pts) if pts else None
        ftype = _classify(f"{name} {title}", gtype)
        if ftype == "unknown":
            continue
        out.append(_feature_row(source_id, rel(path), region, ftype, name, hood, gtype,
                                lon, lat, ";".join(str(round(x, 6)) for x in bbox) if bbox else "", "", "medium"))
        if len(out) >= MAX_FEATURES_PER_SOURCE:
            break
    return out


def _read_table(path):
    try:
        return g14.read_brazilian_csv(path)
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            return list(csv.DictReader(handle))


def _parse_csv(path, source_id, title, fallback_region):
    rows = _read_table(path)
    out = []
    for row in rows[:50000]:
        name = g14.header_pick(row, *NAME_KEYS) or title
        hood = g14.header_pick(row, *HOOD_KEYS)
        lat = g14.header_pick(row, *LAT_KEYS)
        lon = g14.header_pick(row, *LON_KEYS)
        flon = flat = ""
        region = fallback_region
        if lat and lon:
            try:
                flon_f = float(str(lon).replace(",", "."))
                flat_f = float(str(lat).replace(",", "."))
                if in_brazil(flon_f, flat_f):
                    flon, flat = round(flon_f, 6), round(flat_f, 6)
                    region = region_of_coord(flon_f, flat_f)
            except ValueError:
                pass
        geom_type = "point" if flon and flat else "table"
        ftype = _classify(f"{name} {hood} {title}", geom_type)
        if ftype == "unknown":
            continue
        out.append(_feature_row(source_id, rel(path), region, ftype, name, hood, geom_type, flon, flat, "", "", "medium" if name else "low"))
        if len(out) >= MAX_FEATURES_PER_SOURCE:
            break
    return out


def main() -> int:
    print("=" * 60)
    print("SUSC-14B Official Spatial Reference Parser")
    print("=" * 60)
    features, audit = [], []
    for row in read_csv(MANIFEST) if MANIFEST.exists() else []:
        fp = row.get("file_path", "")
        sid = row.get("reference_id", "")
        if not fp:
            audit.append({"source_id": sid, "source_file": "", "parse_status": "no_file", "n_features": 0, "by_feature_type": "{}", "review_only": "true", "notes": row.get("blocked_reason", "")})
            continue
        path = ROOT / fp
        if not path.exists():
            audit.append({"source_id": sid, "source_file": fp, "parse_status": "missing", "n_features": 0, "by_feature_type": "{}", "review_only": "true", "notes": "file not found"})
            continue
        if path.name == "susc_14a_official_geocoding_reference_features_v1.csv":
            old = read_csv(path)
            parsed = []
            for old_row in old:
                parsed.append(_feature_row(sid, fp, old_row.get("region", ""), old_row.get("feature_type", "unknown"),
                                           old_row.get("name", ""), old_row.get("neighborhood", ""),
                                           old_row.get("geometry_type", ""), old_row.get("lon", ""), old_row.get("lat", ""),
                                           old_row.get("bbox", ""), old_row.get("wkt", ""), "medium"))
        elif path.suffix.lower() in {".geojson", ".json"}:
            parsed = _parse_geojson(path, sid, row.get("source_title", ""))
        elif path.suffix.lower() in {".csv", ".tsv"}:
            parsed = _parse_csv(path, sid, row.get("source_title", ""), row.get("region", ""))
        else:
            parsed = []
        counts = Counter(r["feature_type"] for r in parsed)
        audit.append({"source_id": sid, "source_file": fp, "parse_status": "parsed" if parsed else "no_supported_features",
                      "n_features": len(parsed), "by_feature_type": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
                      "review_only": "true", "notes": "official/reference parsing review-only"})
        slots = MAX_FEATURES_TOTAL - len(features)
        if slots > 0:
            features.extend(parsed[:slots])
        if len(features) >= MAX_FEATURES_TOTAL:
            audit.append({"source_id": "GLOBAL", "source_file": "", "parse_status": "feature_cap_reached",
                          "n_features": len(features), "by_feature_type": "{}", "review_only": "true",
                          "notes": f"public feature cap {MAX_FEATURES_TOTAL} reached"})
            break
    for idx, row in enumerate(features):
        row["reference_feature_id"] = f"S14BFEAT_{idx:06d}"
    write_csv(FEATURES, [{k: row.get(k, "") for k in FIELDS} for row in features], FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)
    print(f"features -> {FEATURES.relative_to(ROOT)} ({len(features)} rows; {dict(Counter(r['feature_type'] for r in features))})")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
