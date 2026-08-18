"""
SUSC-13B-AUTO consolidated observed-event catalog (review-only).

Merges the SUSC-13A parsed catalog with the SUSC-13B auto-parsed catalog,
normalizing both into a common schema, then flags likely duplicates by region +
date/period + approximate geometry + source. The higher-evidence record in each
duplicate group is marked preferred; all records are preserved for audit. No
record becomes ground truth or training material.

Writes:
  - datasets/suscetibilidade/susc_13b_auto_consolidated_observed_event_catalog_v1.csv
  - manifests/suscetibilidade/susc_13b_auto_consolidated_catalog_manifest_v1.json
  - outputs_public/suscetibilidade/SUSC_13B_auto_consolidation_audit.csv
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, rel, write_csv, write_json  # noqa: E402

P13A = ROOT / "datasets" / "suscetibilidade" / "susc_13a_strong_observed_events_parsed_v1.csv"
P13B = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_observed_events_parsed_v1.csv"
CATALOG = ROOT / "datasets" / "suscetibilidade" / "susc_13b_auto_consolidated_observed_event_catalog_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_13b_auto_consolidated_catalog_manifest_v1.json"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_13B_auto_consolidation_audit.csv"

FIELDS = [
    "event_id", "source_lineage", "region", "municipality", "event_type",
    "event_date", "event_period_start", "event_period_end", "source_institution",
    "source_title", "source_url", "local_file_path", "evidence_level", "geometry_type",
    "lat", "lon", "bbox", "wkt", "geojson_ref", "crs", "spatial_precision",
    "temporal_precision", "is_observed_event", "source_confidence", "duplicate_group_id",
    "duplicate_candidate", "deduplication_action", "preferred_record",
    "evidence_upgrade_from_13a", "requires_manual_review", "can_be_ground_truth",
    "allowed_for_training", "review_only", "consolidation_notes",
]

AUDIT_FIELDS = [
    "event_id", "source_lineage", "region", "evidence_level", "duplicate_group_id",
    "duplicate_candidate", "deduplication_action", "preferred_record", "review_only", "notes",
]

EVIDENCE_RANK = {
    "strong_observed_flood_polygon": 6, "strong_observed_flood_point": 5,
    "moderate_official_flood_bbox": 4, "moderate_official_occurrence_point": 3,
    "documentary_only": 2, "weak_administrative_context": 1, "weak_risk_area_context": 1,
    "weak_alert_context": 1, "rejected_not_observed_event": 0,
}


def _norm_13a(r: dict) -> dict:
    return {
        "source_lineage": "susc_13a", "region": (r.get("region") or "").lower(),
        "municipality": r.get("municipality", ""), "event_type": r.get("event_type", ""),
        "event_date": r.get("event_date", ""), "event_period_start": r.get("event_period_start", ""),
        "event_period_end": r.get("event_period_end", ""),
        "source_institution": r.get("source_institution", ""),
        "source_title": r.get("source_name", ""), "source_url": r.get("source_url_or_path", ""),
        "local_file_path": r.get("source_url_or_path", ""), "evidence_level": r.get("evidence_level", ""),
        "geometry_type": r.get("geometry_type", ""), "lat": r.get("lat", ""), "lon": r.get("lon", ""),
        "bbox": r.get("bbox", ""), "wkt": r.get("wkt", ""), "geojson_ref": r.get("geojson_ref", ""),
        "crs": r.get("crs", "EPSG:4326"), "spatial_precision": r.get("spatial_precision", ""),
        "temporal_precision": r.get("temporal_precision", ""), "is_observed_event": r.get("is_observed_event", "false"),
        "source_confidence": "high" if r.get("evidence_level", "").startswith("strong") else "medium",
        "orig_event_id": r.get("event_id", ""),
    }


def _norm_13b(r: dict) -> dict:
    return {
        "source_lineage": "susc_13b_auto", "region": (r.get("region") or "").lower(),
        "municipality": r.get("municipality", ""), "event_type": r.get("event_type", ""),
        "event_date": r.get("event_date", ""), "event_period_start": r.get("event_period_start", ""),
        "event_period_end": r.get("event_period_end", ""),
        "source_institution": r.get("source_institution", ""),
        "source_title": r.get("source_title", ""), "source_url": r.get("source_url", ""),
        "local_file_path": r.get("local_file_path", ""), "evidence_level": r.get("evidence_level", ""),
        "geometry_type": r.get("geometry_type", ""), "lat": r.get("lat", ""), "lon": r.get("lon", ""),
        "bbox": r.get("bbox", ""), "wkt": r.get("wkt", ""), "geojson_ref": r.get("geojson_ref", ""),
        "crs": r.get("crs", "EPSG:4326"), "spatial_precision": r.get("spatial_precision", ""),
        "temporal_precision": r.get("temporal_precision", ""), "is_observed_event": r.get("is_observed_event", "false"),
        "source_confidence": r.get("source_confidence", "low"),
        "orig_event_id": r.get("event_id", ""),
    }


def _geo_key(rec: dict) -> str:
    if rec.get("lat") and rec.get("lon"):
        try:
            return f"{round(float(rec['lon']), 2)},{round(float(rec['lat']), 2)}"
        except ValueError:
            pass
    if rec.get("bbox"):
        parts = str(rec["bbox"]).replace(",", ";").split(";")
        if len(parts) == 4:
            try:
                return ";".join(str(round(float(x), 2)) for x in parts)
            except ValueError:
                pass
    return "no_geom"


def _dup_key(rec: dict) -> tuple:
    period = rec.get("event_date") or rec.get("event_period_start") or "no_date"
    return (rec.get("region", ""), period, _geo_key(rec), rec.get("evidence_level", ""))


def main() -> int:
    print("=" * 60)
    print("SUSC-13B-AUTO Consolidated Event Catalog")
    print("=" * 60)
    recs: list[dict] = []
    if P13A.exists():
        recs.extend(_norm_13a(r) for r in read_csv(P13A))
    if P13B.exists():
        recs.extend(_norm_13b(r) for r in read_csv(P13B))

    # group duplicates
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, rec in enumerate(recs):
        groups[_dup_key(rec)].append(i)

    group_id_of: dict[int, str] = {}
    preferred_idx: set[int] = set()
    for gi, members in enumerate(m for _, m in sorted(groups.items(), key=lambda kv: str(kv[0]))):
        gid = f"DUPG_{gi:04d}"
        best = max(members, key=lambda i: (EVIDENCE_RANK.get(recs[i]["evidence_level"], 0),
                                           recs[i]["source_lineage"] == "susc_13b_auto"))
        preferred_idx.add(best)
        for i in members:
            group_id_of[i] = gid

    rows: list[dict] = []
    audit: list[dict] = []
    for i, rec in enumerate(recs):
        gid = group_id_of[i]
        members = groups[_dup_key(rec)]
        is_pref = i in preferred_idx
        is_dup = len(members) > 1
        # evidence upgrade: a 13b record preferred over a same-group 13a record
        upgrade = "false"
        if is_pref and is_dup and rec["source_lineage"] == "susc_13b_auto":
            if any(recs[j]["source_lineage"] == "susc_13a" and j != i for j in members):
                upgrade = "true"
        action = "keep_preferred" if is_pref else ("drop_duplicate_review" if is_dup else "keep_unique")
        event_id = f"SUSC13BC_{i:05d}"
        row = {
            "event_id": event_id, "source_lineage": rec["source_lineage"], "region": rec["region"],
            "municipality": rec["municipality"], "event_type": rec["event_type"],
            "event_date": rec["event_date"], "event_period_start": rec["event_period_start"],
            "event_period_end": rec["event_period_end"], "source_institution": rec["source_institution"],
            "source_title": rec["source_title"], "source_url": rec["source_url"],
            "local_file_path": rec["local_file_path"], "evidence_level": rec["evidence_level"],
            "geometry_type": rec["geometry_type"], "lat": rec["lat"], "lon": rec["lon"],
            "bbox": rec["bbox"], "wkt": rec["wkt"], "geojson_ref": rec["geojson_ref"],
            "crs": rec["crs"], "spatial_precision": rec["spatial_precision"],
            "temporal_precision": rec["temporal_precision"], "is_observed_event": rec["is_observed_event"],
            "source_confidence": rec["source_confidence"], "duplicate_group_id": gid,
            "duplicate_candidate": "true" if is_dup else "false", "deduplication_action": action,
            "preferred_record": "true" if is_pref else "false", "evidence_upgrade_from_13a": upgrade,
            "requires_manual_review": "true", "can_be_ground_truth": "false",
            "allowed_for_training": "false", "review_only": "true",
            "consolidation_notes": f"origem={rec['source_lineage']}; orig_id={rec['orig_event_id']}",
        }
        rows.append(row)
        audit.append({
            "event_id": event_id, "source_lineage": rec["source_lineage"], "region": rec["region"],
            "evidence_level": rec["evidence_level"], "duplicate_group_id": gid,
            "duplicate_candidate": row["duplicate_candidate"], "deduplication_action": action,
            "preferred_record": row["preferred_record"], "review_only": "true",
            "notes": "upgrade_from_13a" if upgrade == "true" else "consolidado review-only",
        })

    write_csv(CATALOG, rows, FIELDS)
    write_csv(AUDIT, audit, AUDIT_FIELDS)

    by_lineage = Counter(r["source_lineage"] for r in rows)
    by_level = Counter(r["evidence_level"] for r in rows)
    n_dup = sum(1 for r in rows if r["duplicate_candidate"] == "true")
    n_pref = sum(1 for r in rows if r["preferred_record"] == "true")
    manifest = {
        "artifact": "SUSC-13B-AUTO consolidated observed event catalog",
        "n_records": len(rows), "n_groups": len(groups), "n_duplicate_candidates": n_dup,
        "n_preferred_records": n_pref, "by_source_lineage": dict(by_lineage),
        "by_evidence_level": dict(by_level),
        "n_evidence_upgrades_from_13a": sum(1 for r in rows if r["evidence_upgrade_from_13a"] == "true"),
        "inputs": {"susc_13a": rel(P13A), "susc_13b_auto": rel(P13B)},
        "can_be_ground_truth": False, "allowed_for_training": False, "review_only": True,
        "score_v7_created": False,
    }
    write_json(MANIFEST, manifest)

    print(f"records: {len(rows)} | groups: {len(groups)} | duplicates: {n_dup} | preferred: {n_pref}")
    print("by lineage:", dict(by_lineage))
    print("by level:", dict(by_level))
    print("review-only. No ground truth. No training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
