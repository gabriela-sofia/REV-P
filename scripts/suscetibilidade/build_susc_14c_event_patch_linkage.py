"""SUSC-14C FASE 6 - event-patch linkage from resolved official matches."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import build_susc_13b_auto_event_patch_linkage as l13b  # noqa: E402
from susc_geometry import bbox_intersect, haversine_m, parse_bbox  # noqa: E402
from susc_io import read_csv, write_csv, write_json  # noqa: E402

MATCHED = ROOT / "datasets" / "suscetibilidade" / "susc_14c_resolved_ambiguous_occurrences_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_14c_event_patch_linkage_v1.csv"
SUMMARY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_event_patch_linkage_summary.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_event_patch_linkage_limitations.json"
GEOJSON = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_14C_event_patch_linkage_geojson.geojson"

FIELDS = [
    "linkage_id", "matched_id", "occurrence_id", "patch_id", "region", "event_date",
    "evidence_level", "georeferencing_status", "matched_feature_type", "spatial_relation",
    "distance_m", "patch_score_v6", "patch_score_class_v6", "linkage_confidence",
    "can_be_used_for_observational_evaluation", "can_be_ground_truth",
    "can_be_used_as_ground_truth", "allowed_for_training", "review_only",
    "requires_manual_review", "interpretation", "limitations",
]
OBSERVED = {"moderate_official_occurrence_geocoded", "moderate_official_street_segment_match"}
NEAR_BUFFER_DEG = 0.005


def _point(row: dict):
    try:
        if row.get("lat") and row.get("lon"):
            return float(row["lon"]), float(row["lat"])
    except ValueError:
        return None
    return None


def _mk(row, patch_id, relation, distance, score, score_class, interp, limit):
    confidence = "moderate" if relation in {
        "moderate_address_point_inside_patch",
        "moderate_street_segment_intersects_patch",
        "moderate_street_segment_near_patch",
    } else "very_weak"
    can_eval = (
        row.get("evidence_level") in OBSERVED
        and row.get("georeferencing_status") in {"official_address_point_match", "official_street_segment_match"}
        and relation in {"moderate_address_point_inside_patch", "moderate_street_segment_intersects_patch", "moderate_street_segment_near_patch"}
        and patch_id != "REGION_LEVEL_NO_PATCH_RESOLUTION"
    )
    return {
        "linkage_id": "",
        "matched_id": row.get("matched_id", ""),
        "occurrence_id": row.get("occurrence_id", ""),
        "patch_id": patch_id,
        "region": row.get("region", ""),
        "event_date": row.get("event_date", ""),
        "evidence_level": row.get("evidence_level", ""),
        "georeferencing_status": row.get("georeferencing_status", ""),
        "matched_feature_type": row.get("matched_feature_type", ""),
        "spatial_relation": relation,
        "distance_m": "" if distance is None else round(distance, 1),
        "patch_score_v6": score,
        "patch_score_class_v6": score_class,
        "linkage_confidence": confidence,
        "can_be_used_for_observational_evaluation": str(can_eval).lower(),
        "can_be_ground_truth": "false",
        "can_be_used_as_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
        "requires_manual_review": "true",
        "interpretation": interp,
        "limitations": limit,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-14C Event-Patch Linkage")
    print("=" * 60)
    patches = l13b._load_patches()
    scores = l13b._load_scores()
    links = []
    for row in read_csv(MATCHED) if MATCHED.exists() else []:
        region = (row.get("region") or "").lower()
        pt = _point(row)
        bbox = parse_bbox(row.get("bbox", ""))
        matched = False
        if row.get("georeferencing_status") == "official_neighborhood_only":
            links.append(_mk(row, "REGION_LEVEL_NO_PATCH_RESOLUTION", "weak_neighborhood_context", None, "", "",
                             "Bairro oficial mantido como contexto fraco.", "Bairro sozinho nao resolve patch-level."))
            continue
        if row.get("evidence_level") not in OBSERVED:
            links.append(_mk(row, "REGION_LEVEL_NO_PATCH_RESOLUTION", "insufficient_for_patch_link", None, "", "",
                             "Ocorrencia sem referencia geometrica oficial suficiente.", "Sem geometria oficial patch-level."))
            continue
        for pid, pbb, cen in patches.get(region, []):
            relation = None
            dist = None
            if pt:
                inside = pbb[0] <= pt[0] <= pbb[2] and pbb[1] <= pt[1] <= pbb[3]
                near = (pbb[0] - NEAR_BUFFER_DEG <= pt[0] <= pbb[2] + NEAR_BUFFER_DEG and
                        pbb[1] - NEAR_BUFFER_DEG <= pt[1] <= pbb[3] + NEAR_BUFFER_DEG)
                if inside:
                    relation = "moderate_address_point_inside_patch" if row.get("matched_feature_type") in {"address_point", "flood_point"} else "moderate_street_segment_intersects_patch"
                elif near:
                    relation = "moderate_street_segment_near_patch"
                if relation:
                    dist = haversine_m(pt[0], pt[1], cen[0], cen[1])
            elif bbox and bbox_intersect(bbox, pbb):
                relation = "moderate_street_segment_intersects_patch"
            if relation:
                score, score_class = scores.get(pid, ("", ""))
                links.append(_mk(row, pid, relation, dist, score, score_class,
                                 "Referencia oficial intersecta ou esta proxima ao patch.",
                                 "Vinculo moderado review-only; nao e ground truth."))
                matched = True
        if not matched:
            links.append(_mk(row, "REGION_LEVEL_NO_PATCH_RESOLUTION", "same_region_period_context", None, "", "",
                             "Referencia oficial nao intersecta patches da matriz.", "Contexto regional/temporal apenas."))
    for idx, row in enumerate(links):
        row["linkage_id"] = f"S14CLINK_{idx:06d}"
    write_csv(LINKAGE, [{k: row.get(k, "") for k in FIELDS} for row in links], FIELDS)
    rel_counts = Counter(r["spatial_relation"] for r in links)
    conf_counts = Counter(r["linkage_confidence"] for r in links)
    eval_links = sum(1 for r in links if r["can_be_used_for_observational_evaluation"] == "true")
    moderate_links = sum(1 for r in links if r["linkage_confidence"] == "moderate")
    write_csv(SUMMARY, [
        {"metric": "total_links", "value": len(links), "review_only": "true"},
        {"metric": "moderate_links", "value": moderate_links, "review_only": "true"},
        {"metric": "observational_evaluation_links", "value": eval_links, "review_only": "true"},
        {"metric": "links_by_spatial_relation", "value": json.dumps(dict(rel_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "links_by_confidence", "value": json.dumps(dict(conf_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "allowed_for_training_true", "value": 0, "review_only": "true"},
        {"metric": "can_be_ground_truth_true", "value": 0, "review_only": "true"},
    ], ["metric", "value", "review_only"])
    features = []
    for link in links:
        if link["patch_id"] == "REGION_LEVEL_NO_PATCH_RESOLUTION":
            continue
        for pid, _bb, cen in patches.get(link["region"], []):
            if pid == link["patch_id"]:
                features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(cen[0], 6), round(cen[1], 6)]}, "properties": link})
                break
    write_json(GEOJSON, {"type": "FeatureCollection", "name": "susc_14c_event_patch_linkage", "review_only": True,
                         "can_be_ground_truth": False, "can_be_used_as_ground_truth": False, "allowed_for_training": False,
                         "features": features})
    write_json(LIMITS, {"artifact": "SUSC-14C official reference expansion event-patch linkage",
                        "n_links": len(links), "moderate_links": moderate_links,
                        "observational_evaluation_links": eval_links,
                        "spatial_relation_counts": dict(rel_counts),
                        "linkage_confidence_counts": dict(conf_counts),
                        "score_v7_created": False, "can_be_ground_truth": False,
                        "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True,
                        "key_limitations": [
                            "Street/address official matches are at most moderate and review-only.",
                            "Neighborhood-only references remain weak/contextual and never patch-level.",
                            "No score v7, no ground truth, no supervised training.",
                        ]})
    print(f"links: {len(links)} | moderate {moderate_links} | eval {eval_links}")
    print("review-only. No ground truth. No training. No score v7.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
