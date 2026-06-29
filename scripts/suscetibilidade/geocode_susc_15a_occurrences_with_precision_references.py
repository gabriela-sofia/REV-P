"""SUSC-15A FASE 6 - official precision geocoder for flood occurrences."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import susc_14a_geocoding_common as g14  # noqa: E402
from susc_io import read_csv, write_csv  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
OCC = DAT / "susc_14a_geocoded_official_occurrences_v1.csv"
MATCH_14B = DAT / "susc_14b_matched_official_occurrences_v1.csv"
MATCH_14C = DAT / "susc_14c_matched_official_occurrences_v1.csv"
REFS = DAT / "susc_15a_precision_reference_features_v1.csv"
GEOCODED = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
AUDIT = OUT / "SUSC_15A_precision_geocoding_audit.csv"

FIELDS = [
    "event_id", "source_event_id", "region", "municipality", "event_date", "event_type",
    "raw_address", "normalized_address", "street_name", "street_number", "cross_street",
    "neighborhood", "source_name", "source_url_or_path", "precision_tier", "geocoding_method",
    "geometry_type", "lat", "lon", "bbox", "wkt", "geojson_ref", "uncertainty_m",
    "match_score", "candidate_count", "is_ambiguous", "calibration_eligible",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training",
    "review_only", "classification_reason",
]
TIER_BY_METHOD = {
    "official_event_geometry_direct": "T1_official_event_point",
    "official_address_point_match": "T2_official_address_point_or_parcel",
    "official_parcel_match": "T2_official_address_point_or_parcel",
    "official_intersection_match": "T3_official_intersection_point",
    "official_house_number_linear_reference": "T4_official_house_number_linear_reference",
    "official_street_segment_candidate": "T5_official_street_segment_candidate",
    "neighborhood_context_only": "T6_neighborhood_context_only",
}


def _number(text: str) -> str:
    low = g14.normalize_text(text)
    match = re.search(r"\b(?:n|no|num|numero)?\s*(\d{1,6}[a-z]?)\b", low)
    return match.group(1) if match else ""


def _cross(text: str) -> str:
    low = g14.normalize_text(text)
    parts = re.split(r"\b(?:esquina com|cruzamento com|proximo ao cruzamento| com )\b", low, maxsplit=1)
    return g14.normalize_street(parts[1]) if len(parts) > 1 else ""


def _geom(ref: dict) -> bool:
    return bool((ref.get("lat") and ref.get("lon")) or ref.get("bbox") or ref.get("wkt") or ref.get("geojson_ref"))


def _indexes(refs: list[dict]):
    by_region = defaultdict(list)
    by_region_hood = defaultdict(list)
    for ref in refs:
        by_region[ref.get("region", "")].append(ref)
        hood = ref.get("neighborhood", "")
        if hood:
            by_region_hood[(ref.get("region", ""), hood)].append(ref)
    return by_region, by_region_hood


def _score(street: str, hood: str, ref: dict) -> tuple[float, bool]:
    sim = g14.similarity(street, ref.get("street_name") or ref.get("normalized_name", ""))
    hood_match = bool(hood and ref.get("neighborhood") and hood == ref.get("neighborhood"))
    if hood and ref.get("neighborhood") and hood != ref.get("neighborhood"):
        sim -= 0.12
    return max(0.0, min(1.0, sim)), hood_match


def _best(street: str, hood: str, refs: list[dict], types: set[str], min_score: float) -> tuple[dict | None, float, list[tuple[float, dict]]]:
    scored = []
    for ref in refs:
        if ref.get("feature_type") not in types or not _geom(ref):
            continue
        score, hood_match = _score(street, hood, ref)
        if score >= min_score and (not hood or hood_match or not ref.get("neighborhood")):
            scored.append((score, ref))
    scored.sort(key=lambda item: (-item[0], item[1].get("reference_feature_id", "")))
    if not scored:
        return None, 0.0, []
    return scored[0][1], scored[0][0], scored


def _street_segment_candidates(street: str, refs: list[dict]) -> list[dict]:
    tokens = set(street.split())
    if not tokens:
        return []
    exact = [
        ref for ref in refs
        if ref.get("feature_type") in {"street_segment_without_number_range", "street_segment_with_number_range"}
        and (ref.get("street_name") or ref.get("normalized_name", "")) == street
    ]
    if exact:
        return exact[:200]
    first = next(iter(street.split()), "")
    last = street.split()[-1]
    out = []
    for ref in refs:
        if ref.get("feature_type") not in {"street_segment_without_number_range", "street_segment_with_number_range"}:
            continue
        name = ref.get("street_name") or ref.get("normalized_name", "")
        name_tokens = set(name.split())
        if not name_tokens:
            continue
        if first in name_tokens or last in name_tokens or len(tokens & name_tokens) >= 2:
            out.append(ref)
        if len(out) >= 300:
            break
    return out


def _eligible(method: str, uncertainty: str, ambiguous: bool) -> bool:
    if method in {"official_event_geometry_direct", "official_address_point_match", "official_parcel_match", "official_intersection_match"}:
        return not ambiguous
    if method == "official_house_number_linear_reference":
        try:
            return float(uncertainty) <= 100 and not ambiguous
        except ValueError:
            return False
    return False


def _mk(idx: int, occ: dict, method: str, ref: dict | None, score: float, candidates: int, reason: str) -> dict:
    raw = occ.get("raw_address", "")
    street = occ.get("normalized_street") or g14.normalize_street(raw)
    number = _number(raw)
    cross = _cross(raw)
    tier = TIER_BY_METHOD.get(method, "T7_rejected_or_non_flood")
    ambiguous = candidates > 1 and method not in {"official_street_segment_candidate", "neighborhood_context_only"}
    uncertainty = ""
    if method == "official_address_point_match":
        uncertainty = "30"
    elif method == "official_parcel_match":
        uncertainty = "75"
    elif method == "official_intersection_match":
        uncertainty = "50"
    elif method == "official_house_number_linear_reference":
        uncertainty = "100"
    elif method == "official_street_segment_candidate":
        uncertainty = "250"
    eligible = _eligible(method, uncertainty, ambiguous)
    return {
        "event_id": f"S15AEVT_{idx:06d}",
        "source_event_id": occ.get("occurrence_id", ""),
        "region": occ.get("region", ""),
        "municipality": occ.get("municipality", ""),
        "event_date": occ.get("event_date", ""),
        "event_type": "flood",
        "raw_address": raw,
        "normalized_address": g14.normalize_street(raw),
        "street_name": street,
        "street_number": number,
        "cross_street": cross,
        "neighborhood": occ.get("normalized_neighborhood", ""),
        "source_name": (ref or {}).get("source_id", ""),
        "source_url_or_path": (ref or {}).get("geojson_ref", ""),
        "precision_tier": tier,
        "geocoding_method": method,
        "geometry_type": (ref or {}).get("feature_type", ""),
        "lat": (ref or {}).get("lat", ""),
        "lon": (ref or {}).get("lon", ""),
        "bbox": (ref or {}).get("bbox", ""),
        "wkt": (ref or {}).get("wkt", ""),
        "geojson_ref": (ref or {}).get("geojson_ref", ""),
        "uncertainty_m": uncertainty,
        "match_score": round(score, 3),
        "candidate_count": candidates,
        "is_ambiguous": str(ambiguous).lower(),
        "calibration_eligible": str(eligible).lower(),
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
        "classification_reason": reason,
    }


def _match(idx: int, occ: dict, by_region, by_region_hood) -> dict:
    region = occ.get("region", "")
    hood = occ.get("normalized_neighborhood", "")
    street = occ.get("normalized_street") or g14.normalize_street(occ.get("raw_address", ""))
    number = _number(occ.get("raw_address", ""))
    cross = _cross(occ.get("raw_address", ""))
    scoped = by_region_hood.get((region, hood), []) or by_region.get(region, [])
    broad = by_region.get(region, [])
    if not g14.is_flood_text(occ.get("raw_occurrence_text", ""), occ.get("raw_address", "")):
        return _mk(idx, occ, "blocked_non_flood", None, 0.0, 0, "blocked_non_flood")
    ref, score, candidates = _best(street, hood, scoped, {"official_event_point", "official_event_polygon", "flood_point", "flood_polygon"}, 0.90)
    if ref:
        return _mk(idx, occ, "official_event_geometry_direct", ref, score, len(candidates), "direct official/technical event geometry")
    if street and number:
        ref, score, candidates = _best(street, hood, scoped, {"address_point"}, 0.90)
        if ref and len(candidates) == 1:
            return _mk(idx, occ, "official_address_point_match", ref, score, len(candidates), "street number matched official address point")
        ref, score, candidates = _best(street, hood, scoped, {"parcel_polygon"}, 0.90)
        if ref and len(candidates) == 1:
            return _mk(idx, occ, "official_parcel_match", ref, score, len(candidates), "street number matched official parcel")
    if street and cross:
        ref, score, candidates = _best(street, hood, scoped, {"intersection_point"}, 0.90)
        if ref and len(candidates) == 1:
            return _mk(idx, occ, "official_intersection_match", ref, score, len(candidates), "official intersection point")
    if street and number:
        ref, score, candidates = _best(street, hood, scoped, {"street_segment_with_number_range"}, 0.92)
        if ref and len(candidates) == 1:
            return _mk(idx, occ, "official_house_number_linear_reference", ref, score, len(candidates), "official linear reference with number range")
    if street:
        ref, score, candidates = _best(street, "", _street_segment_candidates(street, broad), {"street_segment_without_number_range", "street_segment_with_number_range"}, 0.84)
        if ref:
            return _mk(idx, occ, "official_street_segment_candidate", ref, score, len(candidates), "street segment candidate without precise number/intersection")
        return _mk(idx, occ, "blocked_missing_official_reference", None, 0.0, 0, "street present but no official precision reference matched")
    if hood:
        return _mk(idx, occ, "neighborhood_context_only", None, 0.5, 1, "neighborhood-only excluded from calibration")
    return _mk(idx, occ, "blocked_insufficient_address", None, 0.0, 0, "insufficient official address content")


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Geocoder")
    print("=" * 60)
    refs = read_csv(REFS) if REFS.exists() else []
    by_region, by_region_hood = _indexes(refs)
    rows = [_match(idx, occ, by_region, by_region_hood) for idx, occ in enumerate(read_csv(OCC) if OCC.exists() else [])]
    write_csv(GEOCODED, [{k: row.get(k, "") for k in FIELDS} for row in rows], FIELDS)
    tier_counts = Counter(r["precision_tier"] for r in rows)
    method_counts = Counter(r["geocoding_method"] for r in rows)
    audit = [
        {"metric": "precision_events_total", "value": len(rows), "review_only": "true"},
        {"metric": "calibration_eligible_events", "value": sum(1 for r in rows if r["calibration_eligible"] == "true"), "review_only": "true"},
        {"metric": "by_precision_tier", "value": json.dumps(dict(tier_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "by_geocoding_method", "value": json.dumps(dict(method_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "allowed_for_training_true", "value": 0, "review_only": "true"},
        {"metric": "can_be_ground_truth_true", "value": 0, "review_only": "true"},
    ]
    write_csv(AUDIT, audit, ["metric", "value", "review_only"])
    print(f"events: {len(rows)} | eligible {sum(1 for r in rows if r['calibration_eligible'] == 'true')} | {dict(tier_counts)}")
    print("review-only. No generic geocoding. No bairro centroid. No score v7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
