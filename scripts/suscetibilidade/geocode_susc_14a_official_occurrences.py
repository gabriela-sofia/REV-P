"""
SUSC-14A FASE 5b - geocode official flood occurrences against official references.

Matches normalized flood occurrences (no lat/lon) to official reference features
(address points / street segments / neighborhood polygons) parsed from
official/traceable layers only. Matching ladder (FASE 5):
  1. exact street + neighborhood
  2. fuzzy street + neighborhood
  3. street only, if a single unambiguous reference
  4. neighborhood only -> weak neighborhood context (never patch-level)
  5. multiple distinct references -> blocked_ambiguous_address
  6. no official reference -> blocked_no_official_reference

Classification (review-only):
  official_address_point_match  -> moderate_official_occurrence_geocoded
  official_street_segment_match -> moderate_official_street_segment_match
  official_neighborhood_only    -> weak_official_neighborhood_context
  ambiguous / no_match          -> blocked_*

A street-segment / geocoded point is at most a MODERATE candidate; never strong.
No Google Maps, no generic geocoding, no invented coordinate, no municipal /
neighborhood centroid as a patch-level event.

Writes:
  - datasets/suscetibilidade/susc_14a_geocoded_official_occurrences_v1.csv
  - outputs_public/suscetibilidade/SUSC_14A_official_occurrence_geocoding_audit.csv
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv  # noqa: E402

OCC = ROOT / "datasets" / "suscetibilidade" / "susc_14a_normalized_official_occurrences_v1.csv"
FEATURES = ROOT / "datasets" / "suscetibilidade" / "susc_14a_official_geocoding_reference_features_v1.csv"
OUT = ROOT / "datasets" / "suscetibilidade" / "susc_14a_geocoded_official_occurrences_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14A_official_occurrence_geocoding_audit.csv"

FIELDS = [
    "geocoded_id", "occurrence_id", "region", "municipality", "event_date",
    "event_year", "raw_occurrence_text", "raw_address", "raw_neighborhood",
    "normalized_street", "normalized_neighborhood", "evidence_level",
    "georeferencing_status", "match_method", "matched_reference_feature_id",
    "matched_feature_type", "matched_reference_name", "lat", "lon",
    "similarity_score", "geocoding_confidence", "requires_manual_review",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
]
AUDIT_FIELDS = ["metric", "value", "review_only"]

FUZZY_THRESHOLD = 0.86
ADDRESS_TYPES = {"address_point", "flood_point"}
STREET_TYPES = {"street_segment"}


def _gov(row: dict) -> dict:
    row["requires_manual_review"] = "true"
    row["can_be_ground_truth"] = "false"
    row["can_be_used_as_ground_truth"] = "false"
    row["allowed_for_training"] = "false"
    row["review_only"] = "true"
    return row


def _level_for(status: str) -> str:
    return {
        "official_address_point_match": "moderate_official_occurrence_geocoded",
        "official_street_segment_match": "moderate_official_street_segment_match",
        "official_neighborhood_only": "weak_official_neighborhood_context",
        "blocked_ambiguous_address": "rejected_not_observed_event",
        "blocked_no_official_reference": "rejected_not_observed_event",
        "blocked_non_flood": "rejected_not_observed_event",
    }.get(status, "rejected_not_observed_event")


def _build_index(feats: list[dict]):
    by_hood: dict[str, list[dict]] = defaultdict(list)
    hood_centroids: dict[str, list] = defaultdict(list)
    for f in feats:
        hood = f.get("normalized_neighborhood", "")
        by_hood[hood].append(f)
        try:
            hood_centroids[hood].append((float(f["lon"]), float(f["lat"])))
        except (ValueError, KeyError):
            pass
    return by_hood, hood_centroids


def _match(occ: dict, by_hood, hood_centroids) -> dict:
    street = occ.get("normalized_street", "")
    hood = occ.get("normalized_neighborhood", "")
    candidates = by_hood.get(hood, [])

    # (1) exact street + neighborhood
    if street and hood and candidates:
        exact = [f for f in candidates if f.get("normalized_name", "") == street]
        if exact:
            return _result(exact[0], "official_address_point_match", "exact_street_neighborhood", 1.0, "high")
        # (2) fuzzy street + neighborhood
        scored = [(g14.similarity(street, f.get("normalized_name", "")), f) for f in candidates]
        scored = [(s, f) for s, f in scored if s >= FUZZY_THRESHOLD]
        if scored:
            scored.sort(key=lambda t: -t[0])
            best_s = scored[0][0]
            top = [f for s, f in scored if abs(s - best_s) < 1e-9]
            if len(top) == 1:
                ftype = top[0].get("feature_type", "")
                status = "official_street_segment_match" if ftype in STREET_TYPES else "official_address_point_match"
                return _result(top[0], status, "fuzzy_street_neighborhood", round(best_s, 3), "medium")
            return _blocked("blocked_ambiguous_address", "fuzzy_multiple_equal_matches")

    # (3) street only, if a single unambiguous reference across all hoods
    if street:
        glob = []
        for fs in by_hood.values():
            for f in fs:
                s = g14.similarity(street, f.get("normalized_name", ""))
                if s >= FUZZY_THRESHOLD:
                    glob.append((s, f))
        if len(glob) == 1:
            ftype = glob[0][1].get("feature_type", "")
            status = "official_street_segment_match" if ftype in STREET_TYPES else "official_address_point_match"
            return _result(glob[0][1], status, "street_only_unique", round(glob[0][0], 3), "low")
        if len(glob) > 1:
            return _blocked("blocked_ambiguous_address", "street_only_multiple_references")

    # (4) neighborhood only -> weak context (NOT patch-level), no event coordinate
    if hood and hood in hood_centroids and hood_centroids[hood]:
        return _blocked("official_neighborhood_only", "neighborhood_reference_present_no_street_match",
                        lat="", lon="")

    # (6) no official reference
    return _blocked("blocked_no_official_reference", "no_official_layer_for_address")


def _result(feat: dict, status: str, method: str, sim: float, conf: str) -> dict:
    return {
        "georeferencing_status": status, "match_method": method,
        "matched_reference_feature_id": feat.get("reference_feature_id", ""),
        "matched_feature_type": feat.get("feature_type", ""),
        "matched_reference_name": feat.get("name", "")[:120],
        "lat": feat.get("lat", ""), "lon": feat.get("lon", ""),
        "similarity_score": sim, "geocoding_confidence": conf,
    }


def _blocked(status: str, method: str, lat: str = "", lon: str = "") -> dict:
    return {
        "georeferencing_status": status, "match_method": method,
        "matched_reference_feature_id": "", "matched_feature_type": "",
        "matched_reference_name": "", "lat": lat, "lon": lon,
        "similarity_score": 0.0, "geocoding_confidence": "none",
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-14A Official Occurrence Geocoding")
    print("=" * 60)
    feats = read_csv(FEATURES) if FEATURES.exists() else []
    occs = read_csv(OCC) if OCC.exists() else []
    by_hood, hood_centroids = _build_index(feats)
    print(f"reference features: {len(feats)} | flood occurrences: {len(occs)}")

    rows: list[dict] = []
    for occ in occs:
        m = _match(occ, by_hood, hood_centroids)
        status = m["georeferencing_status"]
        level = _level_for(status)
        # neighborhood-only must never carry an event coordinate (no centroid as event)
        if status == "official_neighborhood_only":
            m["lat"], m["lon"] = "", ""
        row = _gov({
            "occurrence_id": occ.get("occurrence_id", ""), "region": occ.get("region", ""),
            "municipality": occ.get("municipality", ""), "event_date": occ.get("event_date", ""),
            "event_year": occ.get("event_year", ""),
            "raw_occurrence_text": occ.get("raw_occurrence_text", ""),
            "raw_address": occ.get("raw_address", ""), "raw_neighborhood": occ.get("raw_neighborhood", ""),
            "normalized_street": occ.get("normalized_street", ""),
            "normalized_neighborhood": occ.get("normalized_neighborhood", ""),
            "evidence_level": level, **m,
        })
        rows.append(row)

    for i, r in enumerate(rows):
        r["geocoded_id"] = f"S14AGEO_{i:06d}"
    rows = [{k: r.get(k, "") for k in FIELDS} for r in rows]
    write_csv(OUT, rows, FIELDS)

    by_status = Counter(r["georeferencing_status"] for r in rows)
    by_level = Counter(r["evidence_level"] for r in rows)
    geocoded = sum(1 for r in rows if r["georeferencing_status"] in
                   {"official_address_point_match", "official_street_segment_match"})
    audit = [
        {"metric": "n_occurrences", "value": len(rows)},
        {"metric": "n_geocoded_official", "value": geocoded},
        {"metric": "n_address_point_match", "value": by_status.get("official_address_point_match", 0)},
        {"metric": "n_street_segment_match", "value": by_status.get("official_street_segment_match", 0)},
        {"metric": "n_neighborhood_only", "value": by_status.get("official_neighborhood_only", 0)},
        {"metric": "n_blocked_no_reference", "value": by_status.get("blocked_no_official_reference", 0)},
        {"metric": "n_blocked_ambiguous", "value": by_status.get("blocked_ambiguous_address", 0)},
        {"metric": "by_georeferencing_status", "value": "|".join(f"{k}:{v}" for k, v in sorted(by_status.items()))},
        {"metric": "by_evidence_level", "value": "|".join(f"{k}:{v}" for k, v in sorted(by_level.items()))},
    ]
    for a in audit:
        a["review_only"] = "true"
    write_csv(AUDIT, audit, AUDIT_FIELDS)

    print(f"geocoded official: {geocoded} | by status: {dict(by_status)}")
    print("street/geocoded points are MODERATE candidates only; never strong.")
    print("review-only. No ground truth. No training. No invented coordinate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
