"""
SUSC-11B Observed Event Catalog Builder (review-only)

Normalizes the raw parsed observed-event records into the SUSC-11B schema and
re-applies the canonical evidence-level classifier (single source of truth in
parse_susc_11b_observed_events) so the catalog is internally consistent. Adds
spatial/temporal precision and stamps governance. Writes a manifest with the
distribution by region / evidence level / geometry and the standing limitations.

Reads:
  - datasets/suscetibilidade/susc_11b_observed_events_raw_parsed_v1.csv
Writes:
  - datasets/suscetibilidade/susc_11b_observed_event_catalog_v1.csv
  - manifests/suscetibilidade/susc_11b_observed_event_catalog_manifest_v1.json
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, write_csv, write_json, sha256_file, rel  # noqa: E402
import parse_susc_11b_observed_events as parser  # noqa: E402

RAW = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_events_raw_parsed_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_11b_observed_event_catalog_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_11b_observed_event_catalog_manifest_v1.json"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_11b_observed_event_schema_v1.json"

CATALOG_FIELDS = [
    "event_id", "region", "municipality", "event_type", "evidence_level", "event_date",
    "event_period_start", "event_period_end", "source_name", "source_type",
    "source_url_or_path", "geometry_type", "lat", "lon", "bbox", "wkt", "geojson_ref",
    "crs", "spatial_precision", "temporal_precision", "source_confidence", "source_role",
    "n_coords", "can_link_to_patch", "requires_manual_review", "can_be_ground_truth",
    "allowed_for_training", "review_only", "notes",
]

OBSERVED_LEVELS = {"observed_flood_polygon_strong", "observed_flood_point_strong",
                   "observed_flood_bbox_moderate", "official_occurrence_point_moderate"}
STRONG_LEVELS = {"observed_flood_polygon_strong", "observed_flood_point_strong"}


def spatial_precision(geometry_type: str, has_geom: bool) -> str:
    if not has_geom:
        return "region"
    if geometry_type == "point":
        return "street"
    if geometry_type in ("polygon", "multipolygon"):
        return "neighborhood"
    if geometry_type in ("point_set", "bbox"):
        return "neighborhood"
    return "unknown"


def temporal_precision(edate: str, pstart: str, pend: str) -> str:
    if pstart and pend:
        return "period"
    if edate:
        return "day"
    return "unknown"


def main() -> int:
    print("=" * 60)
    print("SUSC-11B Observed Event Catalog (review-only)")
    print("=" * 60)

    if not RAW.exists():
        print(f"STOP: raw parsed events not found: {RAW}")
        return 1
    raw = read_csv(RAW)

    out = []
    for r in raw:
        geom = (r.get("geometry_type") or "unknown").strip().lower()
        bbox = (r.get("bbox") or "").strip()
        lat = (r.get("lat") or "").strip()
        lon = (r.get("lon") or "").strip()
        has_geom = bool(bbox or (lat and lon))
        role = (r.get("source_role") or "unknown").strip()
        etype = (r.get("event_type") or "unknown").strip()
        edate = (r.get("event_date") or "").strip()
        pstart = (r.get("event_period_start") or "").strip()
        pend = (r.get("event_period_end") or "").strip()
        # recompute the canonical evidence level (consistency guard)
        level = parser.classify_evidence(role, geom, has_geom, bool(edate or pstart), etype)
        out.append({
            "event_id": r["event_id"], "region": r.get("region", ""),
            "municipality": r.get("municipality", ""), "event_type": etype,
            "evidence_level": level, "event_date": edate, "event_period_start": pstart,
            "event_period_end": pend, "source_name": r.get("source_name", ""),
            "source_type": r.get("source_type", ""), "source_url_or_path": r.get("source_url_or_path", ""),
            "geometry_type": geom, "lat": lat, "lon": lon, "bbox": bbox,
            "wkt": r.get("wkt", ""), "geojson_ref": r.get("geojson_ref", ""), "crs": r.get("crs", "EPSG:4326"),
            "spatial_precision": spatial_precision(geom, has_geom),
            "temporal_precision": temporal_precision(edate, pstart, pend),
            "source_confidence": parser.confidence_for(level), "source_role": role,
            "n_coords": r.get("n_coords", ""),
            "can_link_to_patch": "true" if has_geom else "false",
            "requires_manual_review": "true", "can_be_ground_truth": "false",
            "allowed_for_training": "false", "review_only": "true", "notes": r.get("notes", ""),
        })

    write_csv(CATALOG, out, CATALOG_FIELDS)

    by_region = Counter(r["region"] for r in out)
    by_level = Counter(r["evidence_level"] for r in out)
    by_geom = Counter(r["geometry_type"] for r in out)
    n_geom = sum(1 for r in out if r["bbox"] or (r["lat"] and r["lon"]))
    n_dated = sum(1 for r in out if r["event_date"] or r["event_period_start"])
    n_observed = sum(1 for r in out if r["evidence_level"] in OBSERVED_LEVELS)
    n_strong = sum(1 for r in out if r["evidence_level"] in STRONG_LEVELS)

    manifest = {
        "artifact": "SUSC-11B observed event catalog",
        "schema_ref": rel(SCHEMA),
        "schema_sha256": sha256_file(SCHEMA) if SCHEMA.exists() else "",
        "catalog_csv": rel(CATALOG),
        "catalog_sha256": sha256_file(CATALOG),
        "n_events": len(out),
        "n_with_geometry": n_geom,
        "n_with_date_or_period": n_dated,
        "n_observed_evidence_levels": n_observed,
        "n_strong_evidence_levels": n_strong,
        "by_region": dict(by_region),
        "by_evidence_level": dict(by_level),
        "by_geometry_type": dict(by_geom),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "observed_event_evidence_catalog_review_only_not_patch_ground_truth",
        "limitations": [
            "Nivel *_strong exige geometria explicita validada; nenhum footprint oficial validado disponivel localmente.",
            "Poligono de mancha disponivel e digitalizacao candidata (Charter), classificada como *_moderate, NAO footprint validado.",
            "Setor/area/ponto de risco da Defesa Civil != ocorrencia de alagamento confirmada (risk_area_context).",
            "Registro de ocorrencia oficial sem footprint != area alagada (official_occurrence_point_moderate).",
            "Estacoes/catalogos administrativos != evento observado (administrative_record_only).",
            "Nenhuma coordenada inventada; nenhum geocoding generico; nenhum centroide de municipio como evento.",
            "Nenhum registro vira ground truth e nada desbloqueia treinamento supervisionado.",
        ],
    }
    write_json(MANIFEST, manifest)

    print(f"\ncatalog -> {rel(CATALOG)} ({len(out)} events)")
    print("by region:", dict(by_region))
    print("by evidence_level:", dict(by_level))
    print(f"with geometry: {n_geom} | with date: {n_dated} | observed-level: {n_observed} | strong: {n_strong}")
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
