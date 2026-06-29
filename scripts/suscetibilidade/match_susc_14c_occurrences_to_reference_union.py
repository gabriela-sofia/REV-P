"""SUSC-14C FASE 4 - match occurrences to expanded official reference union."""

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

OCC = ROOT / "datasets" / "suscetibilidade" / "susc_14a_geocoded_official_occurrences_v1.csv"
REFS = ROOT / "datasets" / "suscetibilidade" / "susc_14c_official_spatial_reference_features_union_v1.csv"
MATCHED = ROOT / "datasets" / "suscetibilidade" / "susc_14c_matched_official_occurrences_v1.csv"
AUDIT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_occurrence_reference_match_audit.csv"

ADDRESS_TYPES = {"address_point", "flood_point"}
STREET_TYPES = {"street_segment", "drainage_line", "river_line"}
HOOD_TYPES = {"neighborhood_polygon", "risk_area"}
FUZZY = 0.84

FIELDS = [
    "matched_id", "occurrence_id", "region", "municipality", "event_date", "event_year",
    "raw_occurrence_text", "raw_address", "raw_neighborhood", "normalized_street",
    "normalized_neighborhood", "secondary_street", "evidence_level", "georeferencing_status",
    "match_method", "matched_reference_feature_id", "matched_feature_type", "matched_reference_name",
    "matched_source_id", "lat", "lon", "bbox", "wkt", "geojson_ref", "match_score",
    "name_similarity", "neighborhood_match", "unique_candidate", "geometry_available",
    "candidate_count", "second_best_score", "score_gap", "candidate_summary",
    "match_confidence", "requires_manual_review", "can_be_ground_truth",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
]


def _clean_street(text: str) -> str:
    text = re.sub(r"\b(cep|lote|quadra|qd|bloco|apto|casa|numero|num|nro|no)\b\s*[a-z0-9\-\/]*", " ", text, flags=re.I)
    text = re.sub(r"\b\d+[a-z]?\b", " ", text, flags=re.I)
    for marker in ("proximo a", "proximo de", "em frente", "ao lado", "esquina com", "cruzamento com"):
        text = g14.normalize_text(text).replace(marker, " ")
    return g14.normalize_street(text)


def _primary_secondary(raw: str) -> tuple[str, str]:
    norm = g14.normalize_text(raw)
    pieces = re.split(r"\b(?:esquina com|cruzamento com|proximo a|proximo de|em frente|ao lado)\b", norm, maxsplit=1)
    primary = _clean_street(pieces[0] if pieces else norm)
    secondary = _clean_street(pieces[1]) if len(pieces) > 1 else ""
    return primary, secondary


def _geometry_available(ref: dict) -> bool:
    return bool((ref.get("lat") and ref.get("lon")) or ref.get("bbox") or ref.get("wkt") or ref.get("geojson_ref"))


def _gov(row: dict) -> dict:
    row.update({
        "requires_manual_review": "true",
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
    })
    return row


def _level(status: str) -> str:
    return {
        "official_address_point_match": "moderate_official_occurrence_geocoded",
        "official_street_segment_match": "moderate_official_street_segment_match",
        "official_neighborhood_only": "weak_official_neighborhood_context",
    }.get(status, "rejected_not_observed_event")


def _status_for(ref: dict) -> str:
    if ref.get("feature_type") in ADDRESS_TYPES:
        return "official_address_point_match"
    if ref.get("feature_type") in STREET_TYPES:
        return "official_street_segment_match"
    return "official_neighborhood_only"


def _score(street: str, hood: str, ref: dict, region: str) -> tuple[float, bool]:
    name = ref.get("normalized_name", "")
    ref_hood = ref.get("normalized_neighborhood", "")
    sim = g14.similarity(street, name)
    hood_match = bool(hood and ref_hood and hood == ref_hood)
    if hood and ref_hood and hood != ref_hood:
        sim -= 0.12
    if region and ref.get("region") and region != ref.get("region"):
        sim -= 0.20
    if ref.get("feature_type") in ADDRESS_TYPES:
        sim += 0.025
    if ref.get("feature_type") in STREET_TYPES:
        sim += 0.010
    if ref.get("union_source") == "14c_additional":
        sim += 0.005
    return max(0.0, min(1.0, sim)), hood_match


def _candidate_summary(scored: list[tuple[float, bool, dict]]) -> str:
    payload = []
    for score, hood_match, ref in scored[:5]:
        payload.append({
            "score": round(score, 3),
            "feature_id": ref.get("reference_feature_id", ""),
            "type": ref.get("feature_type", ""),
            "name": ref.get("name", ""),
            "neighborhood_match": hood_match,
            "geometry_available": _geometry_available(ref),
        })
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _base(ref: dict, status: str, method: str, score: float, hood_match: bool, scored: list[tuple[float, bool, dict]], confidence: str) -> dict:
    second = scored[1][0] if len(scored) > 1 else 0.0
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
        "match_score": round(score, 3),
        "name_similarity": round(score, 3),
        "neighborhood_match": str(hood_match).lower(),
        "unique_candidate": str(len(scored) == 1).lower(),
        "geometry_available": str(_geometry_available(ref) and status != "official_neighborhood_only").lower(),
        "candidate_count": len(scored),
        "second_best_score": round(second, 3),
        "score_gap": round(score - second, 3),
        "candidate_summary": _candidate_summary(scored),
        "match_confidence": confidence,
    }


def _blocked(status: str, method: str, scored: list[tuple[float, bool, dict]] | None = None) -> dict:
    scored = scored or []
    best = scored[0][0] if scored else 0.0
    second = scored[1][0] if len(scored) > 1 else 0.0
    return {
        "georeferencing_status": status, "match_method": method,
        "matched_reference_feature_id": "", "matched_feature_type": "", "matched_reference_name": "",
        "matched_source_id": "", "lat": "", "lon": "", "bbox": "", "wkt": "", "geojson_ref": "",
        "match_score": round(best, 3), "name_similarity": round(best, 3),
        "neighborhood_match": str(scored[0][1]).lower() if scored else "false",
        "unique_candidate": "false", "geometry_available": str(_geometry_available(scored[0][2])).lower() if scored else "false",
        "candidate_count": len(scored), "second_best_score": round(second, 3),
        "score_gap": round(best - second, 3), "candidate_summary": _candidate_summary(scored),
        "match_confidence": "none",
    }


def _indexes(refs: list[dict]):
    by_region = defaultdict(list)
    by_hood = defaultdict(list)
    for ref in refs:
        if ref.get("feature_type") not in ADDRESS_TYPES | STREET_TYPES | HOOD_TYPES:
            continue
        by_region[ref.get("region", "")].append(ref)
        if ref.get("normalized_neighborhood", ""):
            by_hood[(ref.get("region", ""), ref.get("normalized_neighborhood", ""))].append(ref)
    return by_region, by_hood


def _match_one(occ: dict, by_region, by_hood) -> tuple[dict, str, str]:
    region = occ.get("region", "")
    hood = occ.get("normalized_neighborhood", "")
    primary, secondary = _primary_secondary(occ.get("raw_address") or occ.get("raw_occurrence_text", ""))
    street = primary or occ.get("normalized_street", "")
    scoped = by_hood.get((region, hood), []) or by_region.get(region, [])
    scored = []
    if street:
        for ref in scoped:
            if ref.get("feature_type") in HOOD_TYPES:
                continue
            score, hood_match = _score(street, hood, ref, region)
            if score >= FUZZY:
                scored.append((score, hood_match, ref))
        scored.sort(key=lambda item: (-item[0], item[2].get("feature_type") not in ADDRESS_TYPES, item[2].get("reference_feature_id", "")))
        if scored:
            best, hood_match, ref = scored[0]
            gap = best - (scored[1][0] if len(scored) > 1 else 0.0)
            if len(scored) == 1 or gap >= 0.08:
                method = "exact_street_neighborhood" if best >= 0.995 and hood_match else "fuzzy_street_neighborhood"
                return _base(ref, _status_for(ref), method, best, hood_match, scored, "medium" if best < 0.995 else "high"), street, secondary
            return _blocked("blocked_ambiguous_address", "multiple_close_official_candidates", scored), street, secondary
    hood_refs = [r for r in by_hood.get((region, hood), []) if r.get("feature_type") in HOOD_TYPES or r.get("normalized_neighborhood") == hood]
    if hood_refs:
        ref = hood_refs[0]
        return _base(ref, "official_neighborhood_only", "neighborhood_reference_present_no_street_match", 0.5, True, [(0.5, True, ref)], "low"), street, secondary
    return _blocked("blocked_no_official_reference", "no_official_reference_for_address"), street, secondary


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Occurrence -> Expanded Official Reference Matching")
    print("=" * 60)
    occs = read_csv(OCC) if OCC.exists() else []
    refs = read_csv(REFS) if REFS.exists() else []
    by_region, by_hood = _indexes(refs)
    out = []
    for idx, occ in enumerate(occs):
        match, street, secondary = _match_one(occ, by_region, by_hood)
        status = match["georeferencing_status"]
        row = _gov({
            "matched_id": f"S14CMATCH_{idx:06d}",
            "occurrence_id": occ.get("occurrence_id", ""),
            "region": occ.get("region", ""),
            "municipality": occ.get("municipality", ""),
            "event_date": occ.get("event_date", ""),
            "event_year": occ.get("event_year", ""),
            "raw_occurrence_text": occ.get("raw_occurrence_text", ""),
            "raw_address": occ.get("raw_address", ""),
            "raw_neighborhood": occ.get("raw_neighborhood", ""),
            "normalized_street": street or occ.get("normalized_street", ""),
            "normalized_neighborhood": occ.get("normalized_neighborhood", ""),
            "secondary_street": secondary,
            "evidence_level": _level(status),
            **match,
        })
        out.append({k: row.get(k, "") for k in FIELDS})
    write_csv(MATCHED, out, FIELDS)
    status_counts = Counter(r["georeferencing_status"] for r in out)
    method_counts = Counter(r["match_method"] for r in out)
    audit = [
        {"metric": "occurrences_input_count", "value": len(occs), "review_only": "true"},
        {"metric": "reference_features_total", "value": len(refs), "review_only": "true"},
        {"metric": "exact_matches_count", "value": method_counts.get("exact_street_neighborhood", 0), "review_only": "true"},
        {"metric": "fuzzy_matches_count", "value": method_counts.get("fuzzy_street_neighborhood", 0), "review_only": "true"},
        {"metric": "ambiguous_matches_count", "value": status_counts.get("blocked_ambiguous_address", 0), "review_only": "true"},
        {"metric": "blocked_no_reference_count", "value": status_counts.get("blocked_no_official_reference", 0), "review_only": "true"},
        {"metric": "by_georeferencing_status", "value": json.dumps(dict(status_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
    ]
    write_csv(AUDIT, audit, ["metric", "value", "review_only"])
    print(f"matched -> {MATCHED.relative_to(ROOT)} ({len(out)} rows; {dict(status_counts)})")
    print("review-only. No ground truth. No training. No generic geocoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
