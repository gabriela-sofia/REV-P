"""SUSC-12B event feature contrast dataset builder."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, sha256_file, rel  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_11c_event_patch_linkage_v1.csv"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
ACTION = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_06B_feature_action_matrix.csv"
SCORE_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_12b_observed_event_feature_contrast_schema_v1.json"

DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12b_event_feature_contrast_dataset_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_12b_event_feature_contrast_manifest_v1.json"

OBSERVED_LEVELS = {"observed_flood_polygon_strong", "observed_flood_point_strong"}
MODERATE_LEVELS = {"observed_flood_bbox_moderate", "official_occurrence_point_moderate"}
WEAK_LEVELS = {"risk_area_context", "alert_only", "administrative_record_only", "documentary_context_only"}

FIELDS = [
    "contrast_row_id",
    "patch_id",
    "region",
    "contrast_group",
    "event_ids",
    "evidence_levels",
    "spatial_relations",
    "linkage_confidences",
    "score_v6",
    "score_v6_class",
    "control_group_id",
    "control_selection_reason",
    "event_patch_source",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "limitations",
]


def _f(value: str | None) -> float:
    try:
        return float(value or 0.0)
    except ValueError:
        return 0.0


def _urban(row: dict) -> float:
    return _f(row.get("urban_prop"))


def _choose_control(event_patch: dict, candidates: list[dict], used: set[str]) -> dict | None:
    if not candidates:
        return None
    target = _urban(event_patch)
    ordered = sorted(candidates, key=lambda r: (abs(_urban(r) - target), r["patch_id"]))
    for row in ordered:
        if row["patch_id"] not in used:
            used.add(row["patch_id"])
            return row
    return ordered[0]


def _group_for(infos: list[dict]) -> str:
    levels = {r["evidence_level"] for r in infos}
    if levels & OBSERVED_LEVELS:
        return "observed_event_patch"
    if levels & MODERATE_LEVELS:
        return "moderate_observed_event_patch"
    if levels & WEAK_LEVELS:
        return "weak_context_patch"
    return "excluded_context_only"


def _row(i: int, patch_id: str, region: str, group: str, infos: list[dict], score: dict,
         control_group_id: str, reason: str, source: str, limitations: str) -> dict:
    return {
        "contrast_row_id": f"SUSC12B_{i:05d}",
        "patch_id": patch_id,
        "region": region,
        "contrast_group": group,
        "event_ids": ";".join(sorted({r.get("event_id", "") for r in infos if r.get("event_id")})),
        "evidence_levels": ";".join(sorted({r.get("evidence_level", "") for r in infos if r.get("evidence_level")})),
        "spatial_relations": ";".join(sorted({r.get("spatial_relation", "") for r in infos if r.get("spatial_relation")})),
        "linkage_confidences": ";".join(sorted({r.get("linkage_confidence", "") for r in infos if r.get("linkage_confidence")})),
        "score_v6": score.get("susc_score_v6_candidate", ""),
        "score_v6_class": score.get("susc_class_v6_candidate", ""),
        "control_group_id": control_group_id,
        "control_selection_reason": reason,
        "event_patch_source": source,
        "can_be_ground_truth": "false",
        "allowed_for_training": "false",
        "review_only": "true",
        "limitations": limitations,
    }


def main() -> int:
    print("=" * 60)
    print("SUSC-12B Event Feature Contrast Dataset")
    print("=" * 60)
    for path in (MATRIX, SCORE, LINKAGE, CARDS, ACTION, SCORE_MANIFEST, SCHEMA):
        if not path.exists():
            print(f"STOP: required input not found: {path}")
            return 1

    matrix = read_csv(MATRIX)
    matrix_by_patch = {r["patch_id"]: r for r in matrix}
    score_by_patch = {r["patch_id"]: r for r in read_csv(SCORE)}
    links = [
        r for r in read_csv(LINKAGE)
        if r.get("patch_id") and r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"
    ]
    by_patch: dict[str, list[dict]] = defaultdict(list)
    for link in links:
        by_patch[link["patch_id"]].append(link)

    rows = []
    used_control: set[str] = set()
    documented = set(by_patch)
    controls_by_region: dict[str, list[dict]] = defaultdict(list)
    for patch in matrix:
        if patch["patch_id"] not in documented:
            controls_by_region[patch.get("regiao", "")].append(patch)

    event_like_patches = []
    weak_context_count = 0
    for patch_id in sorted(by_patch):
        infos = by_patch[patch_id]
        region = infos[0].get("region") or matrix_by_patch.get(patch_id, {}).get("regiao", "")
        group = _group_for(infos)
        if group in {"observed_event_patch", "moderate_observed_event_patch"}:
            event_like_patches.append(patch_id)
        if group == "weak_context_patch":
            weak_context_count += 1
        rows.append(_row(
            len(rows), patch_id, region, group, infos, score_by_patch.get(patch_id, {}),
            "", "event_or_context_patch_from_SUSC_11C", "SUSC_11C_linkage",
            "Review-only; nao e ground truth nem alvo de treino.",
        ))

    for patch_id in event_like_patches:
        patch = matrix_by_patch.get(patch_id, {})
        region = patch.get("regiao", "")
        control = _choose_control(patch, controls_by_region.get(region, []), used_control)
        if not control:
            continue
        control_group_id = f"CTL_{patch_id}"
        # assign the group id to the event patch row and add the matched control
        for row in rows:
            if row["patch_id"] == patch_id:
                row["control_group_id"] = control_group_id
                row["control_selection_reason"] = "event_patch_same_region_basis"
        rows.append(_row(
            len(rows), control["patch_id"], region, "no_documented_event_control", [],
            score_by_patch.get(control["patch_id"], {}), control_group_id,
            "same_region_nearest_urban_prop_no_documented_event_control",
            "SUSC_12B_control_selection",
            "Controle sem evento documentado nao representa ausencia formal de evento.",
        ))

    write_csv(DATASET, rows, FIELDS)

    groups = Counter(r["contrast_group"] for r in rows)
    manifest = {
        "artifact": "SUSC-12B event feature contrast dataset",
        "dataset": rel(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "schema": rel(SCHEMA),
        "rows": len(rows),
        "groups": dict(groups),
        "event_like_patch_count": len(event_like_patches),
        "weak_context_patch_count": weak_context_count,
        "control_count": groups.get("no_documented_event_control", 0),
        "features_used_for_score_v6": read_json(SCORE_MANIFEST)["features_used"],
        "feature_cards": rel(CARDS),
        "feature_action_matrix": rel(ACTION),
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "observed_event_feature_contrast_review_only",
        "limitations": [
            "Eventos fortes/moderados sao escassos; controles nao representam ausencia formal.",
            "risk_area_context, alert_only e contextos documentais nao calibram como evento observado forte.",
            "Sem treino supervisionado, sem ground truth e sem modelo persistido.",
        ],
    }
    write_json(MANIFEST, manifest)

    print(f"dataset -> {rel(DATASET)} ({len(rows)} rows)")
    print("groups:", dict(groups))
    print("review-only. No ground truth. No supervised training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
