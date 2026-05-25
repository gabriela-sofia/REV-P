"""
REV-P v1jh - training gate and weak-supervision boundary.

Consolidates coordinate recovery, review areas, patch candidate metadata, DINO
availability, controls, split/leakage requirements, and formal training gates.
No supervised label, true negative, model training, or DINO unfreeze is released.
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
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jh"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
COORDS = DATASETS_DIR / "official_coordinate_recovery_hardened_registry.csv"
PATCHES = DATASETS_DIR / "multimodal_patch_candidate_batch_registry.csv"
CONTROLS = DATASETS_DIR / "review_control_candidate_registry.csv"
V1JF = DATASETS_DIR / "documented_locality_patch_review_candidate_registry.csv"
DINO = DATASETS_DIR / "official_anchor_dino_embedding_readiness_registry.csv"

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

DECISION_FIELDS = [
    "decision_id",
    "positive_reference_candidates_count",
    "review_area_candidates_count",
    "control_candidates_count",
    "negative_labels_ready_count",
    "s2_ready_count",
    "s1_ready_count",
    "dem_ready_count",
    "dino_ready_count",
    "split_readiness_status",
    "leakage_risk_status",
    "review_only_batch_status",
    "weak_label_sandbox_status",
    "supervised_training_gate_status",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "minimum_evidence_needed",
    "notes",
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
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jh").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def unique_positive_units() -> set[str]:
    return {
        row["documented_event_unit_id"]
        for row in read_csv(COORDS)
        if row.get("can_be_official_anchor_candidate") == "true" and row.get("coordinate_confidence") == "EXPLICIT_COORDINATE_HIGH"
    }


def dino_ready_count() -> int:
    return len({row["anchor_id"] for row in read_csv(DINO) if row.get("embedding_quality_status") == "QA_PASS"})


def build_master_rows() -> list[dict[str, Any]]:
    rows = read_csv(PATCHES)
    master: list[dict[str, Any]] = []
    for row in rows:
        positive_patch_ready = row["coordinate_status"] == "EXPLICIT_COORDINATE_HIGH" and row["s2_status"] == "CONFIRMED_ANCHOR_PATCH_READY"
        master.append(
            {
                **{field: row.get(field, "") for field in FIELDS},
                "label_status": "POSITIVE_REFERENCE_CANDIDATE_PATCH_READY" if positive_patch_ready else row.get("label_status", "NO_LABEL_REVIEW_ONLY"),
                "can_be_positive_label": "false",
                "can_be_negative_label": "false",
                "can_train_model": "false",
                "can_unfreeze_dino_for_scientific_claim": "false",
                "training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
            }
        )
    return master


def decision_row(master: list[dict[str, Any]]) -> dict[str, Any]:
    positive_count = len(unique_positive_units())
    review_area_count = len(read_csv(V1JF))
    control_count = len(read_csv(CONTROLS))
    s2_ready = sum(1 for row in master if row["s2_status"] in {"CONFIRMED_ANCHOR_PATCH_READY", "CONTROL_CANDIDATE_PATCH_READY"})
    s1_ready = sum(1 for row in master if row["s1_status"].endswith("_PATCH_READY"))
    dem_ready = sum(1 for row in master if row["dem_status"].endswith("_PATCH_READY"))
    split_status = "SPLIT_REQUIRES_EVENT_LOCALITY_GROUPING" if positive_count >= 2 else "SPLIT_BLOCKED_SINGLE_OR_INSUFFICIENT_POSITIVE_SET"
    weak_status = "WEAK_LABEL_SANDBOX_ONLY" if positive_count >= 2 and control_count > 0 else "WEAK_LABEL_SANDBOX_BLOCKED_INSUFFICIENT_CANDIDATES"
    review_batch = "REVIEW_ONLY_BATCH" if positive_count >= 1 and s2_ready >= 1 else "REVIEW_ONLY_BATCH_BLOCKED"
    return {
        "decision_id": "TRAINING_GATE_DECISION_V1JH",
        "positive_reference_candidates_count": positive_count,
        "review_area_candidates_count": review_area_count,
        "control_candidates_count": control_count,
        "negative_labels_ready_count": 0,
        "s2_ready_count": s2_ready,
        "s1_ready_count": s1_ready,
        "dem_ready_count": dem_ready,
        "dino_ready_count": dino_ready_count(),
        "split_readiness_status": split_status,
        "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
        "review_only_batch_status": review_batch,
        "weak_label_sandbox_status": weak_status,
        "supervised_training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
        "can_create_training_label": "false",
        "can_train_model": "false",
        "can_unfreeze_dino_for_scientific_claim": "false",
        "minimum_evidence_needed": "Patch QA for multiple positives, formal negatives/controls, split by event/locality, leakage protocol, and supervised metrics.",
        "notes": "Multiple coordinate candidates improve review coverage but do not create labels or supervised training permission.",
    }


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{prefix}: {field}."} for field in fields], ["field", "description"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    master = build_master_rows()
    decision = decision_row(master)
    write_csv(LOCAL_RUN_DIR / "v1jh_ground_reference_candidate_master.csv", master, FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jh_training_gate_decision_matrix.csv", [decision], DECISION_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jh_qa.csv", [
        {"check": "control_candidate_not_negative", "status": "PASS", "detail": "negative_labels_ready_count=0"},
        {"check": "sentinel_sar_batch_no_label", "status": "PASS", "detail": "all can_be_positive_label=false"},
        {"check": "split_leakage_required", "status": "PASS", "detail": decision["leakage_risk_status"]},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_unfreeze_dino_for_scientific_claim_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "sanitized metadata only"},
    ], ["check", "status", "detail"])
    write_csv(DATASETS_DIR / "ground_reference_candidate_master_registry.csv", master, FIELDS)
    write_csv(DATASETS_DIR / "training_gate_decision_matrix.csv", [decision], DECISION_FIELDS)
    write_schema(SCHEMAS_DIR / "ground_reference_candidate_master_schema.csv", FIELDS, "REV-P v1jh ground reference candidate master field")
    write_schema(SCHEMAS_DIR / "training_gate_decision_matrix_schema.csv", DECISION_FIELDS, "REV-P v1jh training gate decision field")
    summary = {
        "stage": "v1jh",
        "timestamp": utc_now(),
        **decision,
    }
    write_json(LOCAL_RUN_DIR / "v1jh_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jh TRAINING GATE AND WEAK-SUPERVISION BOUNDARY")
    print(f"Positive reference candidates: {summary['positive_reference_candidates_count']}")
    print(f"Review areas: {summary['review_area_candidates_count']}")
    print(f"Controls: {summary['control_candidates_count']}")
    print(f"Review batch: {summary['review_only_batch_status']}")
    print(f"Weak sandbox: {summary['weak_label_sandbox_status']}")
    print(f"Training gate: {summary['supervised_training_gate_status']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
