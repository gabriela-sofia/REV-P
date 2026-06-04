#!/usr/bin/env python3
"""
v1ue — Station Candidate Builder

Builds an auditable registry of candidate official stations per city/event.
A station anchors temporal/hydrometeorological plausibility, NEVER flood geometry.
Coordinates are recorded ONLY when they come from the official source.
Never invent coordinates -> coordinate_status=MISSING.
"""

import argparse
import csv
import math
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None

PROTOCOL_VERSION = "v1ue"

STATION_COLUMNS = [
    "station_candidate_id", "source_id", "event_id", "city", "uf",
    "station_name", "station_code", "station_type", "latitude", "longitude",
    "coordinate_source", "coordinate_status", "distance_to_city_km",
    "distance_method", "is_official", "can_anchor_temporal_evidence",
    "can_anchor_spatial_evidence", "limitations", "acquisition_status", "notes",
]


def load_yaml(path: str) -> dict:
    if yaml is None:
        print("[ERROR] pyyaml not installed")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def main():
    parser = argparse.ArgumentParser(description="v1ue — Station Candidate Builder")
    parser.add_argument("--events", default="datasets/protocolo_c/event_candidate_registry.csv")
    parser.add_argument("--policy-config", default="configs/protocolo_c/v1ue_station_search_policy.yaml")
    parser.add_argument("--out", default="datasets/protocolo_c/v1ue_station_candidate_registry.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    policy = load_yaml(args.policy_config)
    events = load_csv(args.events)

    centroids = {}
    for c in policy.get("city_centroids", []):
        centroids[c["city"]] = c

    known_stations = policy.get("known_official_stations", [])

    rows = []
    seq = 0
    for event in events:
        city = event.get("city", "")
        event_id = event["event_id"]
        uf = event.get("uf", "")

        city_stations = [s for s in known_stations if s.get("city") == city]

        for st in city_stations:
            lat = st.get("latitude")
            lon = st.get("longitude")
            coord_source = st.get("coordinate_source", "NEEDS_OFFICIAL_RESOLUTION")

            if lat is not None and lon is not None and coord_source not in ("NEEDS_OFFICIAL_RESOLUTION",):
                coord_status = "FROM_OFFICIAL_SOURCE"
                lat_val = str(lat)
                lon_val = str(lon)
            else:
                coord_status = "MISSING"
                lat_val = ""
                lon_val = ""

            distance_km = ""
            distance_method = "N/A_coordinate_missing"
            if coord_status == "FROM_OFFICIAL_SOURCE" and city in centroids:
                cen = centroids[city]
                distance_km = f"{haversine_km(float(lat), float(lon), cen['latitude'], cen['longitude']):.2f}"
                distance_method = policy.get("distance_policy", {}).get("method", "haversine_city_centroid")

            rows.append({
                "station_candidate_id": f"STA_{PROTOCOL_VERSION}_{seq:04d}",
                "source_id": st.get("source_id", ""),
                "event_id": event_id,
                "city": city,
                "uf": uf,
                "station_name": st.get("station_name", ""),
                "station_code": st.get("station_code", ""),
                "station_type": st.get("station_type", ""),
                "latitude": lat_val,
                "longitude": lon_val,
                "coordinate_source": coord_source,
                "coordinate_status": coord_status,
                "distance_to_city_km": distance_km,
                "distance_method": distance_method,
                "is_official": "true",
                "can_anchor_temporal_evidence": "true",
                "can_anchor_spatial_evidence": "false",
                "limitations": "Station anchors time/plausibility only; NOT flood geometry",
                "acquisition_status": "IDENTIFIED_NEEDS_SERIES",
                "notes": st.get("note", "").strip(),
            })
            seq += 1

    if args.dry_run:
        print(f"[Station Candidate Builder v1ue] DRY RUN — would create {len(rows)} candidates")
        return

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    missing = sum(1 for r in rows if r["coordinate_status"] == "MISSING")
    print(f"[Station Candidate Builder v1ue] {len(rows)} station candidates")
    print(f"  coordinate_status=MISSING: {missing}")
    print(f"  can_anchor_spatial_evidence=false (all — stations are not flood geometry)")
    print(f"\nRegistry: {args.out}")


if __name__ == "__main__":
    main()
