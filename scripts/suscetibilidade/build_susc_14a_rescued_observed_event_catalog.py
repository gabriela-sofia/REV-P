"""
SUSC-14A FASE 7 - consolidated rescued observed-event catalog (review-only).

Merges the SUSC-13C consolidated catalog (13A+13B+13C) with the SUSC-14A
geocoded official occurrences and any SUSC-14A flood footprints into one
review-only catalog, deduplicates by region + date/period + approximate
geometry + evidence level, and ranks records by rescue priority:

  footprint oficial com data
    > ponto oficial com lat/lon
    > ocorrencia oficial geocodificada por camada oficial
    > street segment oficial
    > bairro/contexto.

Nothing here is ground truth or training material; no score v7 is created.

Writes:
  - datasets/suscetibilidade/susc_14a_rescued_observed_event_catalog_v1.csv
  - manifests/suscetibilidade/susc_14a_rescued_observed_event_catalog_manifest_v1.json
  - outputs_public/suscetibilidade/SUSC_14A_rescued_event_consolidation_audit.csv
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, rel, write_csv, write_json  # noqa: E402

CAT_13C = ROOT / "datasets" / "suscetibilidade" / "susc_13c_consolidated_observed_event_catalog_v1.csv"
GEOCODED = ROOT / "datasets" / "suscetibilidade" / "susc_14a_geocoded_official_occurrences_v1.csv"
FOOTPRINT_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14a_flood_footprint_manifest_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_14a_rescued_observed_event_catalog_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_14a_rescued_observed_event_catalog_manifest_v1.json"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_rescued_event_consolidation_audit.csv"

FIELDS = [
    "event_id", "source_lineage", "region", "municipality", "event_type",
    "event_date", "event_period_start", "event_period_end", "source_institution",
    "source_title", "source_url", "local_file_path", "evidence_level",
    "georeferencing_status", "geometry_type", "lat", "lon", "bbox", "wkt",
    "geojson_ref", "crs", "spatial_precision", "temporal_precision",
    "is_observed_event", "source_confidence", "geocoding_confidence",
    "priority_rank", "duplicate_group_id", "duplicate_candidate",
    "preferred_record", "requires_manual_review", "can_be_ground_truth",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
    "consolidation_notes",
]
AUDIT_FIELDS = ["event_id", "source_lineage", "region", "evidence_level",
                "georeferencing_status", "preferred_record", "priority_rank",
                "review_only", "notes"]

STRONG = {"strong_observed_flood_polygon", "strong_observed_flood_point"}
MODERATE = {"moderate_official_occurrence_point", "moderate_official_occurrence_geocoded",
            "moderate_official_street_segment_match", "moderate_official_flood_bbox"}

# rescue priority (higher = stronger), per FASE 7
PRIORITY = {
    "strong_observed_flood_polygon": 90, "strong_observed_flood_point": 85,
    "moderate_official_occurrence_point": 70, "moderate_official_occurrence_geocoded": 60,
    "moderate_official_street_segment_match": 55, "moderate_official_flood_bbox": 50,
    "weak_official_neighborhood_context": 20, "weak_risk_area_context": 10,
    "weak_alert_context": 8, "documentary_only": 5, "rejected_not_observed_event": 0,
}


def _has_geom(r: dict) -> bool:
    return bool(r.get("bbox") or (r.get("lat") and r.get("lon")) or r.get("wkt") or r.get("geojson_ref"))


def _geo_key(r: dict) -> str:
    if r.get("lat") and r.get("lon"):
        try:
            return f"{round(float(r['lon']), 2)},{round(float(r['lat']), 2)}"
        except ValueError:
            pass
    if r.get("bbox"):
        parts = str(r["bbox"]).replace(",", ";").split(";")
        if len(parts) == 4:
            try:
                return ";".join(str(round(float(x), 2)) for x in parts)
            except ValueError:
                pass
    return "no_geom"


def _dup_key(r: dict) -> tuple:
    period = r.get("event_date") or r.get("event_period_start") or "no_date"
    return (r.get("region", ""), period, _geo_key(r), r.get("evidence_level", ""),
            r.get("source_lineage", "") if _geo_key(r) == "no_geom" else "")


def _norm_13c(r: dict) -> dict:
    return {
        "source_lineage": r.get("source_lineage", "susc_13c"), "region": r.get("region", ""),
        "municipality": r.get("municipality", ""), "event_type": r.get("event_type", ""),
        "event_date": r.get("event_date", ""), "event_period_start": r.get("event_period_start", ""),
        "event_period_end": r.get("event_period_end", ""),
        "source_institution": r.get("source_institution", ""), "source_title": r.get("source_title", ""),
        "source_url": r.get("source_url", ""), "local_file_path": r.get("local_file_path", ""),
        "evidence_level": r.get("evidence_level", ""),
        "georeferencing_status": "explicit_geometry" if _has_geom(r) else "text_only_no_geometry",
        "geometry_type": r.get("geometry_type", ""), "lat": r.get("lat", ""), "lon": r.get("lon", ""),
        "bbox": r.get("bbox", ""), "wkt": r.get("wkt", ""), "geojson_ref": r.get("geojson_ref", ""),
        "crs": r.get("crs", "EPSG:4326"), "spatial_precision": r.get("spatial_precision", ""),
        "temporal_precision": r.get("temporal_precision", ""), "is_observed_event": r.get("is_observed_event", "false"),
        "source_confidence": r.get("source_confidence", ""), "geocoding_confidence": "",
        "consolidation_notes": f"herdado de 13C ({r.get('source_lineage','')}); {r.get('consolidation_notes','')}"[:200],
    }


def _norm_geocoded(r: dict) -> dict:
    level = r.get("evidence_level", "")
    has_pt = bool(r.get("lat") and r.get("lon"))
    edate = r.get("event_date", "")
    return {
        "source_lineage": "susc_14a_geocoded", "region": r.get("region", ""),
        "municipality": r.get("municipality", ""), "event_type": "urban_flooding",
        "event_date": edate, "event_period_start": "", "event_period_end": "",
        "source_institution": "Defesa Civil Recife (atendimentos)",
        "source_title": (r.get("raw_occurrence_text", "") + " | " + r.get("raw_address", ""))[:160],
        "source_url": "", "local_file_path": "", "evidence_level": level,
        "georeferencing_status": r.get("georeferencing_status", ""),
        "geometry_type": "point" if has_pt else "text_only",
        "lat": r.get("lat", ""), "lon": r.get("lon", ""), "bbox": "", "wkt": "",
        "geojson_ref": "", "crs": "EPSG:4326",
        "spatial_precision": "street" if has_pt else "neighborhood",
        "temporal_precision": "day" if edate else "unknown",
        "is_observed_event": "true" if level in (STRONG | MODERATE) else "false",
        "source_confidence": "medium" if has_pt else "low",
        "geocoding_confidence": r.get("geocoding_confidence", "none"),
        "consolidation_notes": f"ocorrencia oficial geocodificada 14A; ref={r.get('matched_reference_feature_id','')}; bairro={r.get('raw_neighborhood','')}"[:200],
    }


def _footprint_rows() -> list[dict]:
    if not FOOTPRINT_MANIFEST.exists():
        return []
    out = []
    for r in read_csv(FOOTPRINT_MANIFEST):
        if r.get("geometry_available") != "true":
            continue  # no acquired vector footprint -> nothing observed yet
        out.append({
            "source_lineage": "susc_14a_footprint", "region": r.get("region", ""),
            "municipality": "", "event_type": "flood",
            "event_date": "", "event_period_start": r.get("event_period", ""), "event_period_end": "",
            "source_institution": r.get("provider", ""), "source_title": r.get("product", ""),
            "source_url": r.get("source_url", ""), "local_file_path": r.get("file_path", ""),
            "evidence_level": "strong_observed_flood_polygon", "georeferencing_status": "explicit_geometry",
            "geometry_type": "polygon", "lat": "", "lon": "", "bbox": "", "wkt": "",
            "geojson_ref": r.get("file_path", ""), "crs": "EPSG:4326",
            "spatial_precision": "neighborhood", "temporal_precision": "period" if r.get("event_period") else "unknown",
            "is_observed_event": "true", "source_confidence": "high", "geocoding_confidence": "",
            "consolidation_notes": "footprint vetorial publico adquirido 14A",
        })
    return out


def _gov(row: dict) -> dict:
    row["requires_manual_review"] = "true"
    row["can_be_ground_truth"] = "false"
    row["can_be_used_as_ground_truth"] = "false"
    row["allowed_for_training"] = "false"
    row["review_only"] = "true"
    return row


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Rescued Observed Event Catalog")
    print("=" * 60)
    recs: list[dict] = []
    if CAT_13C.exists():
        recs.extend(_norm_13c(r) for r in read_csv(CAT_13C)
                    if r.get("preferred_record", "true") != "false")
    if GEOCODED.exists():
        for r in read_csv(GEOCODED):
            # keep only rescued evidence (geocoded moderate or neighborhood context);
            # pure blocked/rejected rows stay in the geocoding audit, not the catalog.
            if r.get("evidence_level") in (MODERATE | {"weak_official_neighborhood_context"}):
                recs.extend([_norm_geocoded(r)])
    recs.extend(_footprint_rows())

    for r in recs:
        r["priority_rank"] = PRIORITY.get(r["evidence_level"], 0)

    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        groups[_dup_key(r)].append(i)
    gid_of: dict[int, str] = {}
    pref: set[int] = set()
    for gi, members in enumerate(m for _, m in sorted(groups.items(), key=lambda kv: str(kv[0]))):
        gid = f"DUPG14A_{gi:05d}"
        best = max(members, key=lambda i: recs[i]["priority_rank"])
        pref.add(best)
        for i in members:
            gid_of[i] = gid

    rows: list[dict] = []
    audit: list[dict] = []
    for i, r in enumerate(recs):
        is_pref = i in pref
        is_dup = len(groups[_dup_key(r)]) > 1
        row = _gov({**r, "event_id": f"SUSC14A_{i:05d}",
                    "duplicate_group_id": gid_of[i],
                    "duplicate_candidate": "true" if is_dup else "false",
                    "preferred_record": "true" if is_pref else "false"})
        rows.append({k: row.get(k, "") for k in FIELDS})
        audit.append({"event_id": row["event_id"], "source_lineage": r["source_lineage"],
                      "region": r["region"], "evidence_level": r["evidence_level"],
                      "georeferencing_status": r.get("georeferencing_status", ""),
                      "preferred_record": row["preferred_record"], "priority_rank": r["priority_rank"],
                      "review_only": "true", "notes": "consolidado review-only"})

    write_csv(CATALOG, rows, FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)

    by_lineage = Counter(r["source_lineage"] for r in rows)
    by_level = Counter(r["evidence_level"] for r in rows)
    pref_rows = [r for r in rows if r["preferred_record"] == "true"]
    write_json(MANIFEST, {
        "artifact": "SUSC-14A rescued observed event catalog",
        "n_records": len(rows), "n_preferred": len(pref_rows), "n_groups": len(groups),
        "by_source_lineage": dict(by_lineage), "by_evidence_level": dict(by_level),
        "n_strong_preferred": sum(1 for r in pref_rows if r["evidence_level"] in STRONG),
        "n_moderate_preferred": sum(1 for r in pref_rows if r["evidence_level"] in MODERATE),
        "inputs": {"susc_13c_catalog": rel(CAT_13C), "susc_14a_geocoded": rel(GEOCODED),
                   "susc_14a_footprints": rel(FOOTPRINT_MANIFEST)},
        "can_be_ground_truth": False, "can_be_used_as_ground_truth": False,
        "allowed_for_training": False, "review_only": True, "score_v7_created": False,
    })
    print(f"records: {len(rows)} ({len(pref_rows)} preferred) | by lineage: {dict(by_lineage)}")
    print("by level:", dict(by_level))
    print("review-only. No ground truth. No training. No score v7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
