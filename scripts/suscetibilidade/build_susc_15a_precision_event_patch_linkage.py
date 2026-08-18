"""SUSC-15A FASE 9 - precision event-patch linkage."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
import build_susc_13b_auto_event_patch_linkage as l13b  # noqa: E402
from susc_geometry import haversine_m  # noqa: E402
from susc_io import read_csv, write_csv, write_json  # noqa: E402

DAT = ROOT / "datasets" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "suscetibilidade"
CATALOG = DAT / "susc_15a_calibration_eligible_event_catalog_v1.csv"
GEOCODED = DAT / "susc_15a_precision_geocoded_occurrences_v1.csv"
LINKAGE = DAT / "susc_15a_precision_event_patch_linkage_v1.csv"
SUMMARY = OUT / "SUSC_15A_precision_event_patch_linkage_summary.csv"
LIMITS = OUT / "SUSC_15A_precision_event_patch_linkage_limitations.json"
GEOJSON = OUT / "SUSC_15A_precision_event_patch_linkage_geojson.geojson"

FIELDS = [
    "linkage_id", "event_id", "patch_id", "region", "event_date", "precision_tier",
    "spatial_relation", "distance_m", "patch_score_v6", "patch_score_class_v6",
    "linkage_confidence", "can_be_used_for_observational_evaluation",
    "can_be_ground_truth", "can_be_used_as_ground_truth", "allowed_for_training",
    "review_only", "requires_manual_review", "interpretation", "limitations",
]
NEAR_DEG = 0.001


def _point(row: dict):
    try:
        if row.get("lat") and row.get("lon"):
            return float(row["lon"]), float(row["lat"])
    except ValueError:
        return None
    return None


def _mk(row, patch_id, relation, distance, score, score_class, confidence, interp, limit):
    can_eval = row.get("calibration_eligible") == "true" and relation in {
        "official_polygon_intersects_patch", "official_point_inside_patch",
        "official_address_point_inside_patch", "official_intersection_inside_patch",
        "linear_reference_point_inside_patch", "uncertainty_buffer_intersects_patch",
    } and confidence in {"strong", "moderate"} and patch_id != "REGION_LEVEL_NO_PATCH_RESOLUTION"
    return {
        "linkage_id": "", "event_id": row.get("event_id", ""), "patch_id": patch_id,
        "region": row.get("region", ""), "event_date": row.get("event_date", ""),
        "precision_tier": row.get("precision_tier", ""), "spatial_relation": relation,
        "distance_m": "" if distance is None else round(distance, 1),
        "patch_score_v6": score, "patch_score_class_v6": score_class,
        "linkage_confidence": confidence,
        "can_be_used_for_observational_evaluation": str(can_eval).lower(),
        "can_be_ground_truth": "false", "can_be_used_as_ground_truth": "false",
        "allowed_for_training": "false", "review_only": "true",
        "requires_manual_review": "true", "interpretation": interp, "limitations": limit,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-15A Precision Event-Patch Linkage")
    print("=" * 60)
    patches = l13b._load_patches()
    scores = l13b._load_scores()
    catalog = {r["event_id"]: r for r in read_csv(CATALOG) if CATALOG.exists()}
    source = read_csv(GEOCODED) if GEOCODED.exists() else []
    links = []
    for row in source:
        if row.get("event_id") in catalog:
            row = {**row, **catalog[row["event_id"]]}
        else:
            links.append(_mk(row, "REGION_LEVEL_NO_PATCH_RESOLUTION", "insufficient_for_patch_link", None, "", "", "very_weak",
                             "Evento sem elegibilidade de calibracao.", "T5/T6/T7 ou geometria insuficiente."))
            continue
        pt = _point(row)
        region = row.get("region", "")
        matched = []
        if pt:
            for pid, bb, cen in patches.get(region, []):
                inside = bb[0] <= pt[0] <= bb[2] and bb[1] <= pt[1] <= bb[3]
                near = bb[0] - NEAR_DEG <= pt[0] <= bb[2] + NEAR_DEG and bb[1] - NEAR_DEG <= pt[1] <= bb[3] + NEAR_DEG
                if inside or near:
                    matched.append((pid, bb, cen, inside))
        if len(matched) > 1:
            links.append(_mk(row, "MULTIPLE_PATCH_CANDIDATES", "ambiguous_multiple_patches", None, "", "", "weak",
                             "Evento intersecta mais de um patch candidato.", "Ambiguidade espacial impede uso automatico."))
        elif len(matched) == 1:
            pid, _bb, cen, inside = matched[0]
            score, score_class = scores.get(pid, ("", ""))
            relation = "uncertainty_buffer_intersects_patch"
            if inside and row.get("precision_tier") == "T1_official_event_point":
                relation = "official_point_inside_patch"
            elif inside and row.get("precision_tier") == "T2_official_address_point_or_parcel":
                relation = "official_address_point_inside_patch"
            elif inside and row.get("precision_tier") == "T3_official_intersection_point":
                relation = "official_intersection_inside_patch"
            elif inside and row.get("precision_tier") == "T4_official_house_number_linear_reference":
                relation = "linear_reference_point_inside_patch"
            dist = haversine_m(pt[0], pt[1], cen[0], cen[1])
            links.append(_mk(row, pid, relation, dist, score, score_class, "moderate",
                             "Geometria oficial precisa associada ao patch.", "Vinculo review-only; nao e ground truth."))
        else:
            links.append(_mk(row, "REGION_LEVEL_NO_PATCH_RESOLUTION", "insufficient_for_patch_link", None, "", "", "very_weak",
                             "Geometria elegivel nao intersecta patch da matriz.", "Sem link patch-level."))
    for idx, row in enumerate(links):
        row["linkage_id"] = f"S15ALINK_{idx:06d}"
    write_csv(LINKAGE, [{k: row.get(k, "") for k in FIELDS} for row in links], FIELDS)
    rel_counts = Counter(r["spatial_relation"] for r in links)
    obs = [r for r in links if r["can_be_used_for_observational_evaluation"] == "true"]
    write_csv(SUMMARY, [
        {"metric": "precision_links_total", "value": len(links), "review_only": "true"},
        {"metric": "precision_patch_links", "value": len(obs), "review_only": "true"},
        {"metric": "unique_patches_observational", "value": len({r["patch_id"] for r in obs}), "review_only": "true"},
        {"metric": "links_by_spatial_relation", "value": json.dumps(dict(rel_counts), ensure_ascii=False, sort_keys=True), "review_only": "true"},
        {"metric": "allowed_for_training_true", "value": 0, "review_only": "true"},
        {"metric": "can_be_ground_truth_true", "value": 0, "review_only": "true"},
    ], ["metric", "value", "review_only"])
    features = []
    for link in obs:
        for pid, _bb, cen in patches.get(link["region"], []):
            if pid == link["patch_id"]:
                features.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(cen[0], 6), round(cen[1], 6)]}, "properties": link})
                break
    write_json(GEOJSON, {"type": "FeatureCollection", "name": "susc_15a_precision_event_patch_linkage",
                         "features": features, "score_v7_created": False, "can_be_ground_truth": False,
                         "can_be_used_as_ground_truth": False, "allowed_for_training": False, "review_only": True})
    write_json(LIMITS, {"artifact": "SUSC-15A precision event-patch linkage",
                        "n_links": len(links), "precision_patch_links": len(obs),
                        "unique_patches_observational": len({r["patch_id"] for r in obs}),
                        "spatial_relation_counts": dict(rel_counts), "score_v7_created": False,
                        "can_be_ground_truth": False, "can_be_used_as_ground_truth": False,
                        "allowed_for_training": False, "review_only": True,
                        "key_limitations": [
                            "Bairro-only is excluded from calibration.",
                            "Street segment without number/intersection is a weak candidate only.",
                            "No score v7, no ground truth, no supervised training.",
                        ]})
    print(f"links: {len(links)} | precision patch links {len(obs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
