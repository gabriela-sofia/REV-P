#!/usr/bin/env python3
"""
v1uh — Observed Geometry Candidate Audit

Classifies inventoried assets as geometry candidates.
can_create_ground_reference is always false.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1uh"

CANDIDATE_COLUMNS = [
    "candidate_id", "event_id", "response_id", "asset_id", "institution",
    "candidate_class", "has_geometry", "geometry_type", "crs",
    "has_event_date_field", "event_date_field",
    "has_hazard_field", "hazard_field",
    "has_locality_field", "locality_field",
    "has_source_field", "source_field",
    "is_modeled_product", "is_observed_occurrence", "is_static_map",
    "can_be_ground_reference_candidate",
    "can_create_ground_reference", "can_create_training_label",
    "blocking_reason", "required_next_action", "notes",
]

SUSCEPTIBILITY_INDICATORS = [
    "suscetibilidade", "susceptibility", "risco", "risk_map",
    "modelo", "model", "zoneamento", "hazard_map",
]


def load_csv(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_asset(asset: dict) -> tuple[str, bool, str, str]:
    atype = asset.get("asset_type", "")
    has_geom = asset.get("has_geometry") == "true"
    internal = asset.get("internal_path", "").lower()
    columns = asset.get("columns_detected", "").lower()

    for ind in SUSCEPTIBILITY_INDICATORS:
        if ind in internal or ind in columns:
            return "MODELED_SUSCEPTIBILITY_CONTEXT", False, \
                "Modeled product — not observed occurrence", ""

    if atype == "static_map":
        return "STATIC_MAP_ONLY", False, \
            "Static map image — not validated geometry", ""

    if atype == "document":
        return "DOCUMENT_ONLY", False, \
            "PDF/document — no vector geometry", ""

    if has_geom:
        crs = asset.get("crs", "")
        blocking = []
        if not crs:
            blocking.append("no_crs")
        return "OBSERVED_EVENT_GEOMETRY_CANDIDATE", True, \
            "|".join(blocking) if blocking else "", \
            "Review CRS and event binding" if blocking else "Ready for field mapping"

    if atype == "tabular":
        has_coords = any(c in columns for c in
                         ["lat", "lon", "latitude", "longitude", "x", "y",
                          "coord", "geometry"])
        if has_coords:
            return "TABLE_WITH_COORDINATES_CANDIDATE", True, \
                "", "Map coordinate columns to canonical fields"
        return "INSUFFICIENT_METADATA", False, \
            "Tabular without coordinate columns", ""

    if atype == "shapefile_companion":
        return "INSUFFICIENT_METADATA", False, \
            "Shapefile companion — needs main .shp", ""

    return "INSUFFICIENT_METADATA", False, \
        "Cannot determine geometry from available metadata", ""


def main():
    parser = argparse.ArgumentParser(
        description="v1uh — Observed Geometry Candidate Audit")
    parser.add_argument("--assets",
                        default="datasets/protocolo_c/v1uh_response_asset_inventory.csv")
    parser.add_argument("--out",
                        default="datasets/protocolo_c/v1uh_observed_geometry_candidate_registry.csv")
    args = parser.parse_args()

    assets = load_csv(args.assets)
    rows = []
    seq = 0

    for asset in assets:
        candidate_class, can_be_candidate, blocking, next_action = classify_asset(asset)
        is_modeled = candidate_class == "MODELED_SUSCEPTIBILITY_CONTEXT"
        is_observed = candidate_class in (
            "OBSERVED_EVENT_GEOMETRY_CANDIDATE",
            "TABLE_WITH_COORDINATES_CANDIDATE",
        )
        is_static = candidate_class == "STATIC_MAP_ONLY"

        rows.append({
            "candidate_id": f"CAND_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": asset.get("event_id", ""),
            "response_id": asset.get("response_id", ""),
            "asset_id": asset.get("asset_id", ""),
            "institution": asset.get("institution", ""),
            "candidate_class": candidate_class,
            "has_geometry": asset.get("has_geometry", "false"),
            "geometry_type": asset.get("geometry_type", ""),
            "crs": asset.get("crs", ""),
            "has_event_date_field": "false",
            "event_date_field": "",
            "has_hazard_field": "false",
            "hazard_field": "",
            "has_locality_field": "false",
            "locality_field": "",
            "has_source_field": "false",
            "source_field": "",
            "is_modeled_product": str(is_modeled).lower(),
            "is_observed_occurrence": str(is_observed).lower(),
            "is_static_map": str(is_static).lower(),
            "can_be_ground_reference_candidate": str(can_be_candidate).lower(),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocking_reason": blocking,
            "required_next_action": next_action,
            "notes": "",
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CANDIDATE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    candidates = sum(1 for r in rows
                     if r["can_be_ground_reference_candidate"] == "true")
    print(f"[Observed Geometry Candidate Audit v1uh] {len(rows)} assets classified")
    print(f"  Candidates for review: {candidates}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
