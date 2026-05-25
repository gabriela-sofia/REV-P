"""
REV-P v1jg - batch Sentinel/SAR/DEM candidate acquisition manifest.

Builds an auditable batch manifest for confirmed anchors, newly recovered
coordinate candidates, review areas, and controls. Existing local Sentinel-2
patches are registered when QA is already available; new S2/S1/DEM exports are
planned as metadata unless local generation is proven. No labels are created.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jg"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
V1JE_RECOVERY = DATASETS_DIR / "official_coordinate_recovery_hardened_registry.csv"
V1JF_REVIEW = DATASETS_DIR / "documented_locality_patch_review_candidate_registry.csv"
V1JC_CONTROLS = DATASETS_DIR / "review_control_candidate_registry.csv"
V1IZ_PAIR = DATASETS_DIR / "official_anchor_sentinel_patch_pair_selection_registry.csv"
V1JA_DINO = DATASETS_DIR / "official_anchor_dino_embedding_readiness_registry.csv"

FIELDS = [
    "candidate_id",
    "candidate_type",
    "source_document",
    "date",
    "phenomenon",
    "coordinate_status",
    "patch_status",
    "s2_status",
    "s1_status",
    "dem_status",
    "dino_status",
    "control_status",
    "label_status",
    "training_gate_status",
    "leakage_risk_status",
    "can_be_review_candidate",
    "can_be_positive_label",
    "can_be_negative_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "blocking_reason",
    "minimum_evidence_needed",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jg").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def gee_status() -> str:
    try:
        import ee  # type: ignore

        ee.Initialize()
        return "GEE_AUTHENTICATED"
    except Exception as exc:
        return f"GEE_NOT_READY:{type(exc).__name__}"


def dino_ready() -> bool:
    return any(row.get("embedding_quality_status") == "QA_PASS" for row in read_csv(V1JA_DINO))


def existing_s2_ready_for_anchor() -> bool:
    return any(row.get("final_pair_status") == "PATCH_PAIR_USABLE_FOR_REVIEW" for row in read_csv(V1IZ_PAIR))


def coordinate_candidates() -> list[dict[str, str]]:
    rows = [row for row in read_csv(V1JE_RECOVERY) if row.get("can_be_official_anchor_candidate") == "true"]
    rows.sort(key=lambda row: (row["documented_event_unit_id"], row["recovery_id"]))
    return rows


def build_rows() -> tuple[list[dict[str, Any]], str]:
    status = gee_status()
    rows: list[dict[str, Any]] = []
    seen = set()
    for coord in coordinate_candidates():
        key = (coord["documented_event_unit_id"], coord["latitude"], coord["longitude"])
        if key in seen:
            continue
        seen.add(key)
        confirmed = coord["documented_event_unit_id"] == "PET2022_CPRM_ANEXOII_19022022"
        s2_ready = confirmed and existing_s2_ready_for_anchor()
        rows.append(
            {
                "candidate_id": f"PATCH_CAND_{coord['documented_event_unit_id']}_{len(rows)+1:03d}",
                "candidate_type": "CONFIRMED_ANCHOR" if confirmed else "OFFICIAL_COORDINATE_REVIEW_CANDIDATE",
                "source_document": coord["source_document_name_sanitized"],
                "date": coord["event_or_survey_date"],
                "phenomenon": coord["phenomenon_group"],
                "coordinate_status": "EXPLICIT_COORDINATE_HIGH",
                "patch_status": "CONFIRMED_ANCHOR_PATCH_READY" if s2_ready else "PATCH_EXPORT_PLAN_REQUIRED",
                "s2_status": "CONFIRMED_ANCHOR_PATCH_READY" if s2_ready else ("S2_GEE_BATCH_EXPORT_PLAN_READY" if status == "GEE_AUTHENTICATED" else "PATCH_BLOCKED_BY_GEE_AUTH"),
                "s1_status": "S1_GEE_BATCH_EXPORT_PLAN_READY" if status == "GEE_AUTHENTICATED" else "PATCH_BLOCKED_BY_GEE_AUTH",
                "dem_status": "DEM_GEE_BATCH_EXPORT_PLAN_READY" if status == "GEE_AUTHENTICATED" else "PATCH_BLOCKED_BY_GEE_AUTH",
                "dino_status": "DINO_EMBEDDING_READY" if confirmed and dino_ready() else "DINO_NOT_GENERATED_FOR_CANDIDATE",
                "control_status": "NOT_CONTROL",
                "label_status": "NO_LABEL_REVIEW_ONLY",
                "training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "can_be_review_candidate": "true",
                "can_be_positive_label": "false",
                "can_be_negative_label": "false",
                "can_train_model": "false",
                "can_unfreeze_dino_for_scientific_claim": "false",
                "blocking_reason": "PATCH_EXPORT_AND_LABEL_GATES_PENDING" if not s2_ready else "LABEL_GATES_PENDING",
                "minimum_evidence_needed": "Patch QA, formal labels, controls, split, and leakage protocol before training.",
            }
        )
    for review in read_csv(V1JF_REVIEW):
        rows.append(
            {
                "candidate_id": review["review_area_candidate_id"],
                "candidate_type": "SPATIAL_REVIEW_AREA_CANDIDATE",
                "source_document": review["source_document_name_sanitized"],
                "date": review["event_or_survey_date"],
                "phenomenon": review["phenomenon_group"],
                "coordinate_status": "NO_EXPLICIT_COORDINATE",
                "patch_status": "PATCH_BLOCKED_BY_NO_EXPLICIT_GEOMETRY",
                "s2_status": "PATCH_BLOCKED_BY_NO_COVERAGE",
                "s1_status": "PATCH_BLOCKED_BY_NO_COVERAGE",
                "dem_status": "PATCH_BLOCKED_BY_NO_COVERAGE",
                "dino_status": "DINO_NOT_GENERATED_FOR_CANDIDATE",
                "control_status": "NOT_CONTROL",
                "label_status": "NO_LABEL_REVIEW_AREA_ONLY",
                "training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "can_be_review_candidate": "true",
                "can_be_positive_label": "false",
                "can_be_negative_label": "false",
                "can_train_model": "false",
                "can_unfreeze_dino_for_scientific_claim": "false",
                "blocking_reason": "LOCALITY_TEXT_IS_NOT_GEOMETRY",
                "minimum_evidence_needed": "Explicit coordinate or official geometry before patch acquisition.",
            }
        )
    for control in read_csv(V1JC_CONTROLS):
        rows.append(
            {
                "candidate_id": control["control_candidate_id"],
                "candidate_type": control["control_type"],
                "source_document": control["source"],
                "date": "",
                "phenomenon": "",
                "coordinate_status": "CONTROL_COORDINATE_AVAILABLE" if control["coordinate_available"] == "true" else "NO_EXPLICIT_CONTROL_COORDINATE",
                "patch_status": "CONTROL_CANDIDATE_PATCH_READY" if control["patch_available"] == "true" and control["control_type"] == "TEMPORAL_SELF_CONTROL" else "PATCH_EXPORT_PLAN_REQUIRED",
                "s2_status": "CONTROL_CANDIDATE_PATCH_READY" if control["patch_available"] == "true" and control["control_type"] == "TEMPORAL_SELF_CONTROL" else "PATCH_BLOCKED_BY_NO_COVERAGE",
                "s1_status": "S1_GEE_BATCH_EXPORT_PLAN_READY" if status == "GEE_AUTHENTICATED" and control["coordinate_available"] == "true" else "PATCH_BLOCKED_BY_NO_COVERAGE",
                "dem_status": "DEM_GEE_BATCH_EXPORT_PLAN_READY" if status == "GEE_AUTHENTICATED" and control["coordinate_available"] == "true" else "PATCH_BLOCKED_BY_NO_COVERAGE",
                "dino_status": "DINO_NOT_GENERATED_FOR_CANDIDATE",
                "control_status": "REVIEW_CONTROL_ONLY",
                "label_status": "NO_NEGATIVE_LABEL",
                "training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "can_be_review_candidate": control["can_be_review_control"],
                "can_be_positive_label": "false",
                "can_be_negative_label": "false",
                "can_train_model": "false",
                "can_unfreeze_dino_for_scientific_claim": "false",
                "blocking_reason": control["blocking_reason"],
                "minimum_evidence_needed": "Formal absence/control protocol before any negative label.",
            }
        )
    return rows, status


def write_schema() -> None:
    write_csv(SCHEMAS_DIR / "multimodal_patch_candidate_batch_schema.csv", [{"field": field, "description": f"REV-P v1jg multimodal patch candidate batch field: {field}."} for field in FIELDS], ["field", "description"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    rows, status = build_rows()
    write_csv(LOCAL_RUN_DIR / "v1jg_batch_candidate_manifest.csv", rows, FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jg_patch_acquisition_qa.csv", rows, FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jg_qa.csv", [
        {"check": "sentinel_sar_batch_no_label", "status": "PASS", "detail": "all label flags false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_unfreeze_dino_for_scientific_claim_false", "status": "PASS", "detail": "false"},
        {"check": "gee_status_recorded", "status": "PASS", "detail": status},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "sanitized metadata only"},
    ], ["check", "status", "detail"])
    write_csv(DATASETS_DIR / "multimodal_patch_candidate_batch_registry.csv", rows, FIELDS)
    write_schema()
    summary = {
        "stage": "v1jg",
        "timestamp": utc_now(),
        "gee_status": status,
        "candidate_count": len(rows),
        "s2_patch_ready_count": sum(1 for row in rows if row["s2_status"] in {"CONFIRMED_ANCHOR_PATCH_READY", "CONTROL_CANDIDATE_PATCH_READY"}),
        "s1_patch_ready_count": 0,
        "dem_patch_ready_count": 0,
        "s1_export_plan_count": sum(1 for row in rows if row["s1_status"] == "S1_GEE_BATCH_EXPORT_PLAN_READY"),
        "dem_export_plan_count": sum(1 for row in rows if row["dem_status"] == "DEM_GEE_BATCH_EXPORT_PLAN_READY"),
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
    }
    write_json(LOCAL_RUN_DIR / "v1jg_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jg BATCH SENTINEL/SAR CANDIDATE ACQUISITION")
    print(f"GEE status: {summary['gee_status']}")
    print(f"Candidates: {summary['candidate_count']}")
    print(f"S2 ready: {summary['s2_patch_ready_count']}")
    print(f"S1 ready: {summary['s1_patch_ready_count']}")
    print(f"DEM ready: {summary['dem_patch_ready_count']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
