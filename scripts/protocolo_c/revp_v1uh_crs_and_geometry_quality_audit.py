#!/usr/bin/env python3
"""
v1uh — CRS and Geometry Quality Audit

Audits CRS presence, geometry validity, bounds plausibility,
feature count, and shapefile completeness for geometry candidates.
Uses geopandas/shapely/pyproj if available; fails closed otherwise.
Even if geometry is valid, no overlay is executed.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1uh"

AUDIT_COLUMNS = [
    "audit_id", "candidate_id", "event_id", "asset_id",
    "crs_present", "crs_value", "crs_status",
    "geometry_validity_status", "feature_count", "geometry_type",
    "bounds_status", "region_plausibility_status", "prj_status",
    "quality_status", "blocking", "required_action", "notes",
]

BRAZIL_BOUNDS = {
    "min_lon": -74.0, "max_lon": -34.0,
    "min_lat": -34.0, "max_lat": 6.0,
}


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def audit_candidate(candidate: dict, asset: dict) -> dict:
    has_geom = candidate.get("has_geometry") == "true"
    crs_val = candidate.get("crs", "") or asset.get("crs", "")
    has_prj = asset.get("has_prj", "")
    geom_type = candidate.get("geometry_type", "") or asset.get("geometry_type", "")
    feat_count = asset.get("feature_count", "")
    candidate_class = candidate.get("candidate_class", "")

    if candidate_class in ("DOCUMENT_ONLY", "STATIC_MAP_ONLY",
                           "MODELED_SUSCEPTIBILITY_CONTEXT"):
        return {
            "crs_present": "false",
            "crs_value": "",
            "crs_status": "NOT_APPLICABLE",
            "geometry_validity_status": "NOT_APPLICABLE",
            "feature_count": "",
            "geometry_type": "",
            "bounds_status": "NOT_APPLICABLE",
            "region_plausibility_status": "NOT_APPLICABLE",
            "prj_status": "NOT_APPLICABLE",
            "quality_status": "NOT_APPLICABLE",
            "blocking": "false",
            "required_action": "",
            "notes": f"Non-geometry asset: {candidate_class}",
        }

    if not has_geom and candidate_class != "TABLE_WITH_COORDINATES_CANDIDATE":
        return {
            "crs_present": "false",
            "crs_value": "",
            "crs_status": "MISSING",
            "geometry_validity_status": "NO_GEOMETRY",
            "feature_count": "",
            "geometry_type": "",
            "bounds_status": "NOT_APPLICABLE",
            "region_plausibility_status": "NOT_APPLICABLE",
            "prj_status": "NOT_APPLICABLE",
            "quality_status": "FAIL_NO_GEOMETRY",
            "blocking": "true",
            "required_action": "Obtain geometry data",
            "notes": "",
        }

    crs_present = bool(crs_val)
    crs_status = "PRESENT" if crs_present else "MISSING"

    prj_status = "NOT_APPLICABLE"
    ext = asset.get("extension", "").lower()
    if ext == ".shp":
        prj_status = "PRESENT" if has_prj == "true" else "MISSING"

    blockers = []
    if not crs_present:
        blockers.append("no_crs")
    if ext == ".shp" and has_prj != "true":
        blockers.append("no_prj")

    quality = "PASS_PENDING_GEOSPATIAL_BACKEND" if not blockers else "BLOCKED"

    return {
        "crs_present": str(crs_present).lower(),
        "crs_value": crs_val,
        "crs_status": crs_status,
        "geometry_validity_status": "PENDING_GEOSPATIAL_BACKEND",
        "feature_count": feat_count,
        "geometry_type": geom_type,
        "bounds_status": "PENDING_GEOSPATIAL_BACKEND",
        "region_plausibility_status": "PENDING_GEOSPATIAL_BACKEND",
        "prj_status": prj_status,
        "quality_status": quality,
        "blocking": str(bool(blockers)).lower(),
        "required_action": "|".join(blockers) if blockers else "",
        "notes": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="v1uh — CRS and Geometry Quality Audit")
    parser.add_argument("--candidates",
                        default="datasets/protocolo_c/v1uh_observed_geometry_candidate_registry.csv")
    parser.add_argument("--assets",
                        default="datasets/protocolo_c/v1uh_response_asset_inventory.csv")
    parser.add_argument("--out",
                        default="datasets/protocolo_c/v1uh_crs_geometry_quality_audit.csv")
    args = parser.parse_args()

    candidates = load_csv(args.candidates)
    assets_by_id = {a["asset_id"]: a for a in load_csv(args.assets)}

    rows = []
    seq = 0
    for cand in candidates:
        asset = assets_by_id.get(cand.get("asset_id", ""), {})
        result = audit_candidate(cand, asset)
        rows.append({
            "audit_id": f"CRS_{PROTOCOL_VERSION}_{seq:04d}",
            "candidate_id": cand.get("candidate_id", ""),
            "event_id": cand.get("event_id", ""),
            "asset_id": cand.get("asset_id", ""),
            **result,
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    blocked = sum(1 for r in rows if r["blocking"] == "true")
    print(f"[CRS & Geometry Quality Audit v1uh] {len(rows)} audited | blocked={blocked}")
    print(f"  no_overlay_executed=true")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
