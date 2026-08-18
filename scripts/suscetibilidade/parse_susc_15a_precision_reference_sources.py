"""SUSC-15A FASE 5 - parse precision reference features."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import parse_susc_14b_official_spatial_references as p14b  # noqa: E402
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
MAN = ROOT / "manifests" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
DOWNLOADS = MAN / "susc_15a_precision_reference_download_manifest_v1.csv"
UNION_14C = DAT / "susc_14c_official_spatial_reference_features_union_v1.csv"
FEATURES = DAT / "susc_15a_precision_reference_features_v1.csv"
AUDIT = OUT / "SUSC_15A_precision_reference_parse_audit.csv"

FIELDS = [
    "reference_feature_id", "region", "source_id", "feature_type", "name", "normalized_name",
    "street_name", "street_number", "number_from_left", "number_to_left",
    "number_from_right", "number_to_right", "cross_street_a", "cross_street_b",
    "neighborhood", "lat", "lon", "bbox", "wkt", "geojson_ref", "crs",
    "source_confidence", "parse_confidence", "allowed_for_training",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "review_only",
]
AUDIT_FIELDS = ["source_id", "source_file", "parse_status", "n_features", "by_feature_type", "review_only", "notes"]
MAX_FEATURES = 140000


def _type(row: dict) -> str:
    ftype = row.get("feature_type", "")
    geom = row.get("geometry_type", "")
    text = g14.normalize_text(f"{row.get('name','')} {row.get('source_id','')}")
    if ftype in {"flood_point"}:
        return "official_event_point" if "ocorr" in text else "flood_point"
    if ftype in {"flood_polygon"}:
        return "official_event_polygon" if "ocorr" in text else "flood_polygon"
    if ftype == "address_point":
        return "address_point"
    if ftype == "neighborhood_polygon":
        return "neighborhood_polygon"
    if ftype == "risk_area":
        return "risk_area"
    if ftype == "river_line":
        return "river_line"
    if ftype == "drainage_line":
        return "drainage_line"
    if ftype == "street_segment":
        has_range = any(row.get(k) for k in ("number_from_left", "number_to_left", "number_from_right", "number_to_right"))
        return "street_segment_with_number_range" if has_range else "street_segment_without_number_range"
    if geom == "point" and any(k in text for k in ("cruzamento", "intersecao", "esquina")):
        return "intersection_point"
    if geom in {"polygon", "multipolygon"} and any(k in text for k in ("lote", "parcela", "cadastro")):
        return "parcel_polygon"
    return "unknown"


def _row(src: dict, idx: int) -> dict:
    name = src.get("name", "")
    return {
        "reference_feature_id": f"S15AFEAT_{idx:06d}",
        "region": src.get("region", ""),
        "source_id": src.get("source_id", ""),
        "feature_type": _type(src),
        "name": name,
        "normalized_name": src.get("normalized_name") or g14.normalize_street(name),
        "street_name": src.get("normalized_name") or g14.normalize_street(name),
        "street_number": "",
        "number_from_left": "",
        "number_to_left": "",
        "number_from_right": "",
        "number_to_right": "",
        "cross_street_a": "",
        "cross_street_b": "",
        "neighborhood": src.get("normalized_neighborhood") or src.get("neighborhood", ""),
        "lat": src.get("lat", ""),
        "lon": src.get("lon", ""),
        "bbox": src.get("bbox", ""),
        "wkt": src.get("wkt", ""),
        "geojson_ref": src.get("geojson_ref", ""),
        "crs": src.get("crs", ""),
        "source_confidence": src.get("source_confidence", "medium"),
        "parse_confidence": src.get("parse_confidence", "medium"),
        "allowed_for_training": "false",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "review_only": "true",
    }


def _parse_downloads(start_idx: int) -> tuple[list[dict], list[dict]]:
    features, audit = [], []
    idx = start_idx
    for dl in read_csv(DOWNLOADS) if DOWNLOADS.exists() else []:
        fp = dl.get("file_path", "")
        if not fp:
            audit.append({"source_id": dl.get("source_id", ""), "source_file": "", "parse_status": "no_file",
                          "n_features": 0, "by_feature_type": "{}", "review_only": "true",
                          "notes": dl.get("blocked_reason", "")})
            continue
        path = ROOT / fp
        parsed = []
        if path.exists() and path.suffix.lower() in {".geojson", ".json"}:
            parsed = p14b._parse_geojson(path, dl.get("source_id", ""), dl.get("institution", ""))
        elif path.exists() and path.suffix.lower() in {".csv", ".tsv"}:
            parsed = p14b._parse_csv(path, dl.get("source_id", ""), dl.get("institution", ""), dl.get("region", ""))
        converted = []
        for row in parsed[:20000]:
            converted.append(_row(row, idx))
            idx += 1
        features.extend(converted)
        counts = Counter(r["feature_type"] for r in converted)
        audit.append({"source_id": dl.get("source_id", ""), "source_file": fp,
                      "parse_status": "parsed" if converted else "no_supported_features",
                      "n_features": len(converted), "by_feature_type": json.dumps(dict(counts), sort_keys=True),
                      "review_only": "true", "notes": "precision parser review-only"})
    return features, audit


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Reference Parser")
    print("=" * 60)
    features = []
    for src in read_csv(UNION_14C) if UNION_14C.exists() else []:
        if len(features) >= MAX_FEATURES:
            break
        features.append(_row(src, len(features)))
    extra, audit = _parse_downloads(len(features))
    features.extend(extra[: max(0, MAX_FEATURES - len(features))])
    counts = Counter(r["feature_type"] for r in features)
    write_csv(FEATURES, [{k: row.get(k, "") for k in FIELDS} for row in features], FIELDS)
    audit.insert(0, {"source_id": "susc_14c_union", "source_file": str(UNION_14C.relative_to(ROOT)).replace("\\", "/"),
                     "parse_status": "parsed", "n_features": len(read_csv(UNION_14C)) if UNION_14C.exists() else 0,
                     "by_feature_type": json.dumps(dict(counts), sort_keys=True), "review_only": "true",
                     "notes": "14C official union recast into precision feature types"})
    write_csv(AUDIT, audit, AUDIT_FIELDS)
    print(f"features: {len(features)} | {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
