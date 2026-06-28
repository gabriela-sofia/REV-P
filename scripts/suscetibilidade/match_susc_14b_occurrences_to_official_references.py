"""SUSC-14B FASE 5 - match official flood occurrences to official references."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv  # noqa: E402

OCC = ROOT / "datasets" / "suscetibilidade" / "susc_14a_geocoded_official_occurrences_v1.csv"
CATALOG_14A = ROOT / "datasets" / "suscetibilidade" / "susc_14a_rescued_observed_event_catalog_v1.csv"
REFS = ROOT / "datasets" / "suscetibilidade" / "susc_14b_official_spatial_reference_features_v1.csv"
MATCHED = ROOT / "datasets" / "suscetibilidade" / "susc_14b_matched_official_occurrences_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14B_occurrence_reference_match_audit.csv"

FIELDS = [
    "matched_id", "occurrence_id", "region", "municipality", "event_date", "event_year",
    "raw_occurrence_text", "raw_address", "raw_neighborhood", "normalized_street",
    "normalized_neighborhood", "evidence_level", "georeferencing_status", "match_method",
    "matched_reference_feature_id", "matched_feature_type", "matched_reference_name",
    "matched_source_id", "lat", "lon", "bbox", "wkt", "geojson_ref", "match_score",
    "name_similarity", "neighborhood_match", "unique_candidate", "geometry_available",
    "match_confidence", "requires_manual_review", "can_be_ground_truth",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
]
ADDRESS_TYPES = {"address_point", "flood_point"}
STREET_TYPES = {"street_segment", "drainage_line", "river_line"}
HOOD_TYPES = {"neighborhood_polygon", "risk_area"}
FUZZY = 0.86


def _gov(row):
    row.update({
        "requires_manual_review": "true",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
    })
    return row


def _level(status):
    return {
        "official_address_point_match": "moderate_official_occurrence_geocoded",
        "official_street_segment_match": "moderate_official_street_segment_match",
        "official_neighborhood_only": "weak_official_neighborhood_context",
        "blocked_ambiguous_address": "rejected_not_observed_event",
        "blocked_no_official_reference": "rejected_not_observed_event",
    }.get(status, "rejected_not_observed_event")


def _geometry_available(ref):
    return bool(ref.get("lat") and ref.get("lon") or ref.get("bbox") or ref.get("wkt") or ref.get("geojson_ref"))


def _base_match(ref, status, method, sim, confidence, unique):
    return {
        "georeferencing_status": status,
        "match_method": method,
        "matched_reference_feature_id": ref.get("reference_feature_id", ""),
        "matched_feature_type": ref.get("feature_type", ""),
        "matched_reference_name": ref.get("name", ""),
        "matched_source_id": ref.get("source_id", ""),
        "lat": ref.get("lat", "") if status != "official_neighborhood_only" else "",
        "lon": ref.get("lon", "") if status != "official_neighborhood_only" else "",
        "bbox": ref.get("bbox", "") if status != "official_neighborhood_only" else "",
        "wkt": ref.get("wkt", "") if status != "official_neighborhood_only" else "",
        "geojson_ref": ref.get("geojson_ref", "") if status != "official_neighborhood_only" else "",
        "match_score": round(sim, 3),
        "name_similarity": round(sim, 3),
        "neighborhood_match": str(bool(ref.get("normalized_neighborhood"))).lower(),
        "unique_candidate": str(unique).lower(),
        "geometry_available": str(_geometry_available(ref) and status != "official_neighborhood_only").lower(),
        "match_confidence": confidence,
    }


def _blocked(status, method, sim=0.0):
    return {
        "georeferencing_status": status, "match_method": method,
        "matched_reference_feature_id": "", "matched_feature_type": "", "matched_reference_name": "",
        "matched_source_id": "", "lat": "", "lon": "", "bbox": "", "wkt": "", "geojson_ref": "",
        "match_score": round(sim, 3), "name_similarity": round(sim, 3),
        "neighborhood_match": "false", "unique_candidate": "false", "geometry_available": "false",
        "match_confidence": "none",
    }


def _build_indexes(refs):
    by_hood = defaultdict(list)
    by_name = defaultdict(list)
    for ref in refs:
        if ref.get("feature_type") not in ADDRESS_TYPES | STREET_TYPES | HOOD_TYPES:
            continue
        name = ref.get("normalized_name", "")
        hood = ref.get("normalized_neighborhood", "")
        if hood:
            by_hood[hood].append(ref)
        if name:
            by_name[name].append(ref)
    return by_hood, by_name


def _status_for(ref):
    if ref.get("feature_type") in ADDRESS_TYPES:
        return "official_address_point_match"
    if ref.get("feature_type") in STREET_TYPES:
        return "official_street_segment_match"
    return "official_neighborhood_only"


def _match_one(occ, by_hood, by_name):
    street = occ.get("normalized_street", "")
    hood = occ.get("normalized_neighborhood", "")
    candidates = by_hood.get(hood, [])
    if street and candidates:
        exact = [r for r in candidates if r.get("normalized_name") == street and r.get("feature_type") not in HOOD_TYPES]
        if len(exact) == 1:
            ref = exact[0]
            return _base_match(ref, _status_for(ref), "exact_street_neighborhood", 1.0, "high", True)
        if len(exact) > 1:
            return _blocked("blocked_ambiguous_address", "exact_street_neighborhood_multiple", 1.0)
        scored = [(g14.similarity(street, r.get("normalized_name", "")), r) for r in candidates if r.get("feature_type") not in HOOD_TYPES]
        scored = [(s, r) for s, r in scored if s >= FUZZY]
        if scored:
            scored.sort(key=lambda item: -item[0])
            best = scored[0][0]
            top = [r for s, r in scored if abs(s - best) < 1e-9]
            if len(top) == 1:
                ref = top[0]
                return _base_match(ref, _status_for(ref), "fuzzy_street_neighborhood", best, "medium", True)
            return _blocked("blocked_ambiguous_address", "fuzzy_street_neighborhood_multiple", best)
    if street:
        scored = []
        for name, refs in by_name.items():
            sim = g14.similarity(street, name)
            if sim >= FUZZY:
                for ref in refs:
                    if ref.get("feature_type") not in HOOD_TYPES:
                        scored.append((sim, ref))
        if len(scored) == 1:
            sim, ref = scored[0]
            return _base_match(ref, _status_for(ref), "street_only_unique", sim, "low", True)
        if len(scored) > 1:
            return _blocked("blocked_ambiguous_address", "street_only_multiple_candidates", max(s for s, _ in scored))
    hood_refs = [r for r in candidates if r.get("feature_type") in HOOD_TYPES or r.get("normalized_neighborhood") == hood]
    if hood_refs:
        return _base_match(hood_refs[0], "official_neighborhood_only", "neighborhood_reference_present_no_street_match", 0.5, "low", len(hood_refs) == 1)
    return _blocked("blocked_no_official_reference", "no_official_reference_for_address")


def main() -> int:
    print("=" * 60)
    print("SUSC-14B Occurrence -> Official Reference Matching")
    print("=" * 60)
    occs = read_csv(OCC) if OCC.exists() else []
    refs = read_csv(REFS) if REFS.exists() else []
    by_hood, by_name = _build_indexes(refs)
    out = []
    for idx, occ in enumerate(occs):
        match = _match_one(occ, by_hood, by_name)
        status = match["georeferencing_status"]
        row = _gov({
            "matched_id": f"S14BMATCH_{idx:06d}",
            "occurrence_id": occ.get("occurrence_id", ""),
            "region": occ.get("region", ""),
            "municipality": occ.get("municipality", ""),
            "event_date": occ.get("event_date", ""),
            "event_year": occ.get("event_year", ""),
            "raw_occurrence_text": occ.get("raw_occurrence_text", ""),
            "raw_address": occ.get("raw_address", ""),
            "raw_neighborhood": occ.get("raw_neighborhood", ""),
            "normalized_street": occ.get("normalized_street", ""),
            "normalized_neighborhood": occ.get("normalized_neighborhood", ""),
            "evidence_level": _level(status),
            **match,
        })
        out.append({k: row.get(k, "") for k in FIELDS})
    write_csv(MATCHED, out, FIELDS)
    status_counts = Counter(r["georeferencing_status"] for r in out)
    method_counts = Counter(r["match_method"] for r in out)
    level_counts = Counter(r["evidence_level"] for r in out)
    audit = [
        {"metric": "occurrences_input_count", "value": len(occs), "review_only": "true"},
        {"metric": "flood_occurrences_count", "value": len(occs), "review_only": "true"},
        {"metric": "official_reference_features_count", "value": len(refs), "review_only": "true"},
        {"metric": "street_segment_reference_count", "value": sum(1 for r in refs if r.get("feature_type") == "street_segment"), "review_only": "true"},
        {"metric": "address_point_reference_count", "value": sum(1 for r in refs if r.get("feature_type") == "address_point"), "review_only": "true"},
        {"metric": "neighborhood_polygon_reference_count", "value": sum(1 for r in refs if r.get("feature_type") == "neighborhood_polygon"), "review_only": "true"},
        {"metric": "exact_matches_count", "value": method_counts.get("exact_street_neighborhood", 0), "review_only": "true"},
        {"metric": "fuzzy_matches_count", "value": method_counts.get("fuzzy_street_neighborhood", 0) + method_counts.get("street_only_unique", 0), "review_only": "true"},
        {"metric": "ambiguous_matches_count", "value": status_counts.get("blocked_ambiguous_address", 0), "review_only": "true"},
        {"metric": "blocked_no_reference_count", "value": status_counts.get("blocked_no_official_reference", 0), "review_only": "true"},
        {"metric": "by_georeferencing_status", "value": json.dumps(dict(status_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "by_evidence_level", "value": json.dumps(dict(level_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "allowed_for_training_true", "value": sum(1 for r in out if r["allowed_for_training"] == "true"), "review_only": "true"},
        {"metric": "can_be_ground_truth_true", "value": sum(1 for r in out if r["can_be_ground_truth"] == "true"), "review_only": "true"},
    ]
    write_csv(AUDIT, audit, ["metric", "value", "review_only"])
    print(f"matched -> {MATCHED.relative_to(ROOT)} ({len(out)} rows; {dict(status_counts)})")
    print("review-only. No ground truth. No training. No generic geocoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
