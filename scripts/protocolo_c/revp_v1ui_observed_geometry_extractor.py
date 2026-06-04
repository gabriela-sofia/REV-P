#!/usr/bin/env python3
"""
v1ui — Observed Geometry Extractor

Classifies inventoried public artifacts as geometry candidates.
Extracts metadata only — no overlay, no labels, no ground reference.
"""

import argparse
import csv
import os

PROTOCOL_VERSION = "v1ui"

EXTRACTION_COLUMNS = [
    "extraction_id", "event_id", "artifact_id", "source_id",
    "geometry_candidate_class", "has_geometry", "geometry_type", "crs",
    "feature_count", "has_date_field", "date_field",
    "has_hazard_field", "hazard_field", "has_locality_field", "locality_field",
    "has_coordinate_fields", "coordinate_fields",
    "is_observed_occurrence", "is_modeled_product", "is_event_specific",
    "can_be_observed_geometry_candidate",
    "can_create_ground_reference", "can_create_training_label",
    "blocking_reason", "required_next_action", "notes",
]

SUSCEPTIBILITY_INDICATORS = [
    "suscetibilidade", "susceptibility", "risco", "risk_map",
    "modelo", "model", "zoneamento", "hazard_map",
]

DATE_COLUMNS = {"data", "date", "dt_ocorrencia", "data_ocorrencia", "timestamp", "data_evento"}
HAZARD_COLUMNS = {"tipo", "fenomeno", "hazard", "classe", "ocorrencia", "hazard_type", "natureza"}
LOCALITY_COLUMNS = {"bairro", "localidade", "municipio", "cidade", "rua", "endereco", "neighborhood"}
COORD_COLUMNS = {"latitude", "lat", "longitude", "lon", "lng", "x", "y", "coord_x", "coord_y"}


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_inventory_item(item):
    atype = item.get("asset_type", "")
    has_geom = item.get("has_geometry") == "true"
    internal = item.get("internal_path", "").lower()
    columns = item.get("columns_detected", "").lower()
    for ind in SUSCEPTIBILITY_INDICATORS:
        if ind in internal or ind in columns:
            return "SUSCEPTIBILITY_CONTEXT", False

    if atype == "static_map":
        return "STATIC_MAP_ONLY", False
    if atype == "document":
        return "DOCUMENT_ONLY", False

    if has_geom:
        return "OBSERVED_OCCURRENCE_POLYGONS_CANDIDATE", True

    if atype == "tabular":
        col_set = set(columns.split("|")) if columns else set()
        has_coords = bool(col_set & COORD_COLUMNS)
        if has_coords:
            return "OBSERVED_OCCURRENCE_POINTS_CANDIDATE", True
        return "DOCUMENT_ONLY", False

    if atype == "data_structured":
        return "DOCUMENT_ONLY", False

    return "REJECTED_GENERIC", False


def detect_fields(columns_str):
    if not columns_str:
        return {}, {}, {}, {}
    cols = set(c.strip().lower() for c in columns_str.split("|"))
    date_match = cols & DATE_COLUMNS
    hazard_match = cols & HAZARD_COLUMNS
    locality_match = cols & LOCALITY_COLUMNS
    coord_match = cols & COORD_COLUMNS
    return date_match, hazard_match, locality_match, coord_match


def main():
    parser = argparse.ArgumentParser(description="v1ui — Observed Geometry Extractor")
    parser.add_argument("--inventory", default="datasets/protocolo_c/v1ui_public_artifact_inventory.csv")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ui_observed_geometry_extraction_registry.csv")
    args = parser.parse_args()

    items = load_csv(args.inventory)
    rows = []
    seq = 0

    for item in items:
        gclass, can_be = classify_inventory_item(item)
        columns = item.get("columns_detected", "")
        date_f, hazard_f, locality_f, coord_f = detect_fields(columns)
        is_modeled = gclass == "SUSCEPTIBILITY_CONTEXT"
        is_observed = gclass in ("OBSERVED_OCCURRENCE_POINTS_CANDIDATE",
                                  "OBSERVED_OCCURRENCE_POLYGONS_CANDIDATE",
                                  "EVENT_FOOTPRINT_CANDIDATE")
        is_event_specific = item.get("event_term_detected") == "true"

        blockers = []
        if can_be and not date_f:
            blockers.append("no_date_field")
        if can_be and not hazard_f:
            blockers.append("no_hazard_field")

        rows.append({
            "extraction_id": f"EXT_{PROTOCOL_VERSION}_{seq:04d}",
            "event_id": item.get("event_id", ""),
            "artifact_id": item.get("artifact_id", ""),
            "source_id": item.get("source_id", ""),
            "geometry_candidate_class": gclass,
            "has_geometry": item.get("has_geometry", "false"),
            "geometry_type": item.get("geometry_type", ""),
            "crs": item.get("crs", ""),
            "feature_count": item.get("feature_count", ""),
            "has_date_field": str(bool(date_f)).lower(),
            "date_field": "|".join(date_f) if date_f else "",
            "has_hazard_field": str(bool(hazard_f)).lower(),
            "hazard_field": "|".join(hazard_f) if hazard_f else "",
            "has_locality_field": str(bool(locality_f)).lower(),
            "locality_field": "|".join(locality_f) if locality_f else "",
            "has_coordinate_fields": str(bool(coord_f)).lower(),
            "coordinate_fields": "|".join(coord_f) if coord_f else "",
            "is_observed_occurrence": str(is_observed).lower(),
            "is_modeled_product": str(is_modeled).lower(),
            "is_event_specific": str(is_event_specific).lower(),
            "can_be_observed_geometry_candidate": str(can_be).lower(),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocking_reason": "|".join(blockers),
            "required_next_action": "Supervisor review" if can_be and not blockers else "",
            "notes": "",
        })
        seq += 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EXTRACTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    candidates = sum(1 for r in rows if r["can_be_observed_geometry_candidate"] == "true")
    print(f"[Observed Geometry Extractor v1ui] {len(rows)} items | candidates={candidates}")
    print(f"  can_create_ground_reference=false (all)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
