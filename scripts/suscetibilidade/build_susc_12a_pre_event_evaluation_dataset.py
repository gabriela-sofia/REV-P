"""SUSC-12A temporal pre-event evaluation dataset builder.

Builds a review-only event/control table for asking whether linked patches had
higher candidate susceptibility before the observed/contextual event. Temporal
gates fail closed: if a score uses temporal features whose reference date is not
proven pre-event, the row is marked uncertain or post-event risk.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
from susc_io import read_csv, read_json, write_csv, write_json, sha256_file, rel  # noqa: E402

MATRIX = ROOT / "datasets" / "suscetibilidade" / "susc_features_by_patch_v1.csv"
SCORE = ROOT / "datasets" / "suscetibilidade" / "susc_score_v6_candidate_by_patch_v1.csv"
LINKAGE = ROOT / "datasets" / "suscetibilidade" / "susc_11c_event_patch_linkage_v1.csv"
CARDS = ROOT / "outputs_public" / "suscetibilidade" / "feature_cards" / "susc_feature_cards_v1.json"
SCORE_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_score_v6_candidate_manifest_v1.json"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_12a_temporal_evaluation_schema_v1.json"

DATASET = ROOT / "datasets" / "suscetibilidade" / "susc_12a_pre_event_evaluation_dataset_v1.csv"
MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_12a_pre_event_evaluation_manifest_v1.json"

STATIC_GROUPS = {"topography", "topography_hydrology", "hydrology"}
TEMPORAL_GROUPS = {"precipitation", "spectral_index", "sar", "land_use", "interaction"}
OBSERVED_LEVELS = {
    "observed_flood_polygon_strong",
    "observed_flood_point_strong",
    "observed_flood_bbox_moderate",
    "official_occurrence_point_moderate",
}

FIELDS = [
    "dataset_row_id",
    "row_role",
    "patch_id",
    "event_id",
    "linkage_id",
    "region",
    "event_date",
    "feature_reference_date",
    "event_patch_status",
    "control_patch_status",
    "score_v6",
    "score_v6_class",
    "feature_temporal_status",
    "pre_event_valid",
    "temporal_leakage_risk",
    "control_group_id",
    "control_selection_reason",
    "used_feature_count",
    "static_feature_count",
    "pre_event_confirmed_feature_count",
    "temporal_uncertain_feature_count",
    "post_event_risk_feature_count",
    "missing_temporal_metadata_feature_count",
    "feature_temporal_status_counts",
    "can_be_ground_truth",
    "allowed_for_training",
    "review_only",
    "limitations",
]


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except ValueError:
        return None


def _reference_date(card: dict, patch_row: dict) -> str:
    temporal_ref = str(card.get("temporal_reference") or "")
    m = re.search(r"(\d{4}-\d{2}-\d{2})", temporal_ref)
    if m:
        return m.group(1)
    return (patch_row.get("reference_date") or "").strip()


def _feature_status(feature: str, card: dict, patch_row: dict, event_date: str) -> tuple[str, str, bool]:
    group = card.get("feature_group", "")
    ref = _reference_date(card, patch_row)
    if group in STATIC_GROUPS:
        return "static_feature", ref, True
    if group not in TEMPORAL_GROUPS:
        return "not_applicable", ref, False
    ed = _parse_date(event_date)
    rd = _parse_date(ref)
    if not ed:
        return "missing_temporal_metadata", ref, False
    if not rd:
        return "same_period_uncertain", ref, False
    if rd < ed:
        return "pre_event_confirmed", ref, True
    if rd == ed:
        return "same_period_uncertain", ref, False
    return "post_event_risk", ref, False


def _aggregate_temporal_status(counts: Counter) -> tuple[str, str, str]:
    if counts["post_event_risk"]:
        return "post_event_risk", "false", "high"
    if counts["missing_temporal_metadata"]:
        return "missing_temporal_metadata", "false", "temporal_uncertain"
    if counts["same_period_uncertain"]:
        return "same_period_uncertain", "false", "temporal_uncertain"
    if counts["pre_event_confirmed"]:
        return "pre_event_confirmed", "true", "low"
    if counts["static_feature"]:
        return "static_feature", "true", "low"
    return "not_applicable", "false", "temporal_uncertain"


def _event_patch_status(link: dict) -> str:
    if link.get("can_be_used_as_observed_evidence") == "true":
        return "observed_event_patch_review_only"
    if link.get("evidence_level") in OBSERVED_LEVELS:
        return "observed_context_link_review_only"
    return "contextual_patch_review_only"


def _score_float(row: dict) -> float:
    try:
        return float(row.get("susc_score_v6_candidate") or row.get("score_v6") or 0.0)
    except ValueError:
        return 0.0


def _urban(row: dict) -> float:
    try:
        return float(row.get("urban_prop") or 0.0)
    except ValueError:
        return 0.0


def _temporal_eval(patch_row: dict, cards_by_feature: dict, features_used: list[str], event_date: str) -> dict:
    counts = Counter()
    refs = []
    for feat in features_used:
        status, ref, _valid = _feature_status(feat, cards_by_feature.get(feat, {}), patch_row, event_date)
        counts[status] += 1
        if ref:
            refs.append(ref)
    aggregate, pre_event_valid, leakage = _aggregate_temporal_status(counts)
    return {
        "feature_reference_date": ";".join(sorted(set(refs))),
        "feature_temporal_status": aggregate,
        "pre_event_valid": pre_event_valid,
        "temporal_leakage_risk": leakage,
        "used_feature_count": str(len(features_used)),
        "static_feature_count": str(counts["static_feature"]),
        "pre_event_confirmed_feature_count": str(counts["pre_event_confirmed"]),
        "temporal_uncertain_feature_count": str(counts["same_period_uncertain"]),
        "post_event_risk_feature_count": str(counts["post_event_risk"]),
        "missing_temporal_metadata_feature_count": str(counts["missing_temporal_metadata"]),
        "feature_temporal_status_counts": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
    }


def _choose_control(event_patch: dict, candidates: list[dict], used_controls: set[str]) -> dict | None:
    if not candidates:
        return None
    target_urban = _urban(event_patch)
    ordered = sorted(candidates, key=lambda r: (abs(_urban(r) - target_urban), r["patch_id"]))
    for row in ordered:
        key = row["patch_id"]
        if key not in used_controls:
            used_controls.add(key)
            return row
    return ordered[0]


def main() -> int:
    print("=" * 60)
    print("SUSC-12A Pre-Event Evaluation Dataset")
    print("=" * 60)
    for path in (MATRIX, SCORE, LINKAGE, CARDS, SCORE_MANIFEST, SCHEMA):
        if not path.exists():
            print(f"STOP: required input not found: {path}")
            return 1

    matrix_rows = read_csv(MATRIX)
    matrix_by_patch = {r["patch_id"]: r for r in matrix_rows}
    score_rows = read_csv(SCORE)
    score_by_patch = {r["patch_id"]: r for r in score_rows}
    links = read_csv(LINKAGE)
    cards_by_feature = {c["feature_name"]: c for c in read_json(CARDS)["cards"]}
    features_used = read_json(SCORE_MANIFEST)["features_used"]

    event_links = [
        r for r in links
        if r.get("patch_id") and r["patch_id"] != "REGION_LEVEL_NO_PATCH_RESOLUTION"
    ]
    documented_patch_ids = {r["patch_id"] for r in event_links}
    controls_by_region = {}
    for row in matrix_rows:
        if row["patch_id"] in documented_patch_ids:
            continue
        controls_by_region.setdefault(row.get("regiao", ""), []).append(row)

    out = []
    used_controls: set[str] = set()
    for link in event_links:
        patch_id = link["patch_id"]
        patch = matrix_by_patch.get(patch_id, {})
        score = score_by_patch.get(patch_id, {})
        event_date = "2022-05-24" if link.get("temporal_relation") == "dated_event" else ""
        control_group_id = f"CTL_{link['linkage_id']}"
        temporal = _temporal_eval(patch, cards_by_feature, features_used, event_date)
        out.append({
            "dataset_row_id": f"SUSC12A_{len(out):05d}",
            "row_role": "event_patch",
            "patch_id": patch_id,
            "event_id": link["event_id"],
            "linkage_id": link["linkage_id"],
            "region": link["region"],
            "event_date": event_date,
            "event_patch_status": _event_patch_status(link),
            "control_patch_status": "not_control",
            "score_v6": score.get("susc_score_v6_candidate", link.get("patch_score_v6", "")),
            "score_v6_class": score.get("susc_class_v6_candidate", link.get("patch_score_class_v6", "")),
            "control_group_id": control_group_id,
            "control_selection_reason": "event_patch_same_region_basis",
            "can_be_ground_truth": "false",
            "allowed_for_training": "false",
            "review_only": "true",
            "limitations": "Evento/ligacao permanece review-only; nao e ground truth.",
            **temporal,
        })
        control = _choose_control(patch, controls_by_region.get(link["region"], []), used_controls)
        if control:
            cscore = score_by_patch.get(control["patch_id"], {})
            ctemporal = _temporal_eval(control, cards_by_feature, features_used, event_date)
            out.append({
                "dataset_row_id": f"SUSC12A_{len(out):05d}",
                "row_role": "control_patch",
                "patch_id": control["patch_id"],
                "event_id": link["event_id"],
                "linkage_id": link["linkage_id"],
                "region": link["region"],
                "event_date": event_date,
                "event_patch_status": "not_event_patch",
                "control_patch_status": "no_documented_event_control",
                "score_v6": cscore.get("susc_score_v6_candidate", ""),
                "score_v6_class": cscore.get("susc_class_v6_candidate", ""),
                "control_group_id": control_group_id,
                "control_selection_reason": "same_region_nearest_urban_prop_no_documented_event_control",
                "can_be_ground_truth": "false",
                "allowed_for_training": "false",
                "review_only": "true",
                "limitations": "Controle sem evento documentado nao representa ausencia formal de evento.",
                **ctemporal,
            })

    write_csv(DATASET, out, FIELDS)

    status_counts = Counter(r["feature_temporal_status"] for r in out)
    leakage_counts = Counter(r["temporal_leakage_risk"] for r in out)
    manifest = {
        "artifact": "SUSC-12A temporal pre-event evaluation dataset",
        "dataset": rel(DATASET),
        "dataset_sha256": sha256_file(DATASET),
        "schema": rel(SCHEMA),
        "rows": len(out),
        "event_patch_rows": sum(1 for r in out if r["row_role"] == "event_patch"),
        "control_patch_rows": sum(1 for r in out if r["row_role"] == "control_patch"),
        "events": len({r["event_id"] for r in out if r["row_role"] == "event_patch"}),
        "event_patches": len({r["patch_id"] for r in out if r["row_role"] == "event_patch"}),
        "controls": len({r["patch_id"] for r in out if r["row_role"] == "control_patch"}),
        "features_used": features_used,
        "feature_temporal_status_counts": dict(status_counts),
        "temporal_leakage_risk_counts": dict(leakage_counts),
        "low_sample_size": sum(1 for r in out if r["row_role"] == "event_patch") < 30,
        "can_be_ground_truth": False,
        "allowed_for_training": False,
        "review_only": True,
        "scientific_status": "temporal_pre_event_evaluation_review_only",
        "limitations": [
            "Patches de evento sao derivados de ligacoes SUSC-11C review-only.",
            "Controles sao no_documented_event_control e nao representam ausencia formal de evento.",
            "Features temporais sem data comprovadamente pre-evento ficam incertas ou post_event_risk.",
            "Nenhum modelo e treinado ou persistido.",
        ],
    }
    write_json(MANIFEST, manifest)

    print(f"dataset -> {rel(DATASET)} ({len(out)} rows)")
    print(f"event rows: {manifest['event_patch_rows']} | controls: {manifest['control_patch_rows']}")
    print("feature temporal status:", dict(status_counts))
    print("leakage risk:", dict(leakage_counts))
    print("review-only. No ground truth. No supervised training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
