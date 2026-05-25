"""
REV-P v1jl - FORMAL_NEGATIVE_AND_CONTROL_EVIDENCE_PROTOCOL.

Audits whether existing control candidates and context evidence can support a
formal negative label, a strong review-only control, a weak contextual control,
or must be blocked as an invalid negative assumption. This stage does not create
labels, training, DINO unfreeze, or heavy outputs.
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
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jl"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

CONTROL_EXPANSION = DATASETS_DIR / "control_candidate_expansion_registry.csv"
REVIEW_CONTROLS = DATASETS_DIR / "review_control_candidate_registry.csv"
SPLIT_PROTOCOL = DATASETS_DIR / "split_leakage_protocol_registry.csv"
SANDBOX_BOUNDARY = DATASETS_DIR / "sandbox_training_boundary_registry.csv"
GROUND_MASTER = DATASETS_DIR / "ground_reference_candidate_master_registry.csv"
PATCH_REGISTRY = DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv"
ANCHOR_REGISTRY = DATASETS_DIR / "official_multi_anchor_registry.csv"
TRAINING_GATE = DATASETS_DIR / "multi_anchor_training_gate_matrix.csv"
EXTERNAL_EVIDENCE = DATASETS_DIR / "external_evidence_registry.csv"
PATCH_TAXONOMY = DATASETS_DIR / "patch_corpus_taxonomy_registry.csv"

EVIDENCE_FIELDS = [
    "candidate_id",
    "candidate_type",
    "source_layer",
    "region",
    "coordinate_available",
    "patch_available",
    "official_absence_evidence",
    "official_stability_evidence",
    "distance_to_nearest_positive_anchor_m",
    "same_event_or_locality_risk",
    "control_strength_status",
    "negative_label_status",
    "can_be_negative_label",
    "can_be_review_control",
    "can_create_training_label",
    "can_train_model",
    "leakage_risk_status",
    "blocking_reason",
    "minimum_evidence_needed",
    "notes",
]

READINESS_FIELDS = [
    "matrix_id",
    "formal_negative_ready_count",
    "strong_control_candidate_count",
    "review_control_only_count",
    "invalid_negative_assumption_count",
    "insufficient_evidence_count",
    "positive_reference_candidate_count",
    "formal_positive_label_ready_count",
    "formal_negative_label_ready_count",
    "can_create_training_label",
    "negative_label_status",
    "minimum_evidence_needed",
    "notes",
]

GATE_FIELDS = [
    "gate_id",
    "formal_positive_labels_ready",
    "formal_negative_labels_ready",
    "review_only_controls_ready",
    "split_protocol_status",
    "leakage_risk_status",
    "supervised_training_boundary_status",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "primary_blocker",
    "minimum_evidence_needed",
    "notes",
]

PRIVATE_FRAGMENTS = ["C:\\Users\\gabriela", "Documents\\REV-P", "Documents/REV-P"]


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
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{prefix}: {field}."} for field in fields], ["field", "description"])


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jl").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def boolish(value: str) -> bool:
    return str(value).strip().lower() == "true"


def base_minimum_evidence() -> str:
    return "official absence or stability evidence for area/date/phenomenon; explicit coordinate or geometry; patch QA; split/leakage protocol; human review decision"


def classify_control(row: dict[str, str]) -> dict[str, str]:
    control_type = row.get("control_type", "")
    absence_evidence = "false"
    stability_evidence = "false"
    coordinate_available = "true" if row.get("nearest_anchor_id", "").startswith("ANCHOR_") or row.get("coordinate_available") == "true" else "false"
    patch_available = "true" if boolish(row.get("s2_available", "")) or boolish(row.get("patch_available", "")) else "false"
    distance = row.get("distance_to_nearest_anchor_m", "")
    risk = "HIGH" if control_type == "TEMPORAL_SELF_CONTROL" else "REQUIRES_PROTOCOL"
    can_review = "true"
    status = "REVIEW_CONTROL_ONLY"
    negative_status = "NO_NEGATIVE_LABEL"
    blocker = "EXPLICIT_ABSENCE_OR_STABILITY_EVIDENCE_REQUIRED"
    notes = "Candidate can support review context only; it does not satisfy formal negative evidence gates."

    if control_type == "TEMPORAL_SELF_CONTROL":
        status = "STRONG_CONTROL_CANDIDATE"
        negative_status = "NO_NEGATIVE_LABEL_TEMPORAL_BASELINE"
        blocker = "PRE_EVENT_SAME_ANCHOR_IS_NOT_INDEPENDENT_NEGATIVE"
        notes = "Strong within-anchor baseline for review, but not an independent negative and not separable across split sides."
    elif control_type == "INVALID_NEGATIVE_LABEL" or boolish(row.get("absence_claim_made", "")):
        status = "INVALID_NEGATIVE_ABSENCE_ASSUMPTION"
        negative_status = "NEGATIVE_LABEL_BLOCKED"
        can_review = "false"
        risk = "LEAKAGE_RISK_HIGH"
        blocker = "ABSENCE_OF_RECORD_IS_NOT_ABSENCE_OF_EVENT"
        notes = "Explicit guardrail row blocking absence-of-record from becoming a negative label."
    elif control_type == "CROSS_REGION_CONTEXT_CANDIDATE":
        status = "REVIEW_CONTROL_ONLY"
        negative_status = "NO_NEGATIVE_LABEL_CROSS_REGION_CONTEXT"
        blocker = "CROSS_REGION_CONTEXT_IS_NOT_EVENT_ABSENCE_EVIDENCE"
        notes = "Cross-region material is structural context, not absence evidence for the Petrópolis event process."
    elif control_type in {"EXISTING_PATCH_BACKGROUND_CANDIDATE", "SPATIAL_BACKGROUND_REVIEW_CANDIDATE", "SPATIAL_CONTEXT_CONTROL_CANDIDATE"}:
        status = "INSUFFICIENT_EVIDENCE"
        negative_status = "NO_NEGATIVE_LABEL_INSUFFICIENT_EVIDENCE"
        blocker = "NO_EXPLICIT_ABSENCE_STABILITY_EVIDENCE_OR_COMPLETE_BUFFER_AUDIT"
        notes = "Background or spatial context lacks explicit absence/stability evidence and cannot be a formal negative."

    return {
        "candidate_id": row.get("control_candidate_id", ""),
        "candidate_type": control_type,
        "source_layer": row.get("source_layer") or row.get("source") or "control_candidate_registry",
        "region": row.get("region", ""),
        "coordinate_available": coordinate_available,
        "patch_available": patch_available,
        "official_absence_evidence": absence_evidence,
        "official_stability_evidence": stability_evidence,
        "distance_to_nearest_positive_anchor_m": distance,
        "same_event_or_locality_risk": risk,
        "control_strength_status": status,
        "negative_label_status": negative_status,
        "can_be_negative_label": "false",
        "can_be_review_control": can_review,
        "can_create_training_label": "false",
        "can_train_model": "false",
        "leakage_risk_status": row.get("leakage_risk_status") or "LEAKAGE_PROTOCOL_REQUIRED",
        "blocking_reason": blocker,
        "minimum_evidence_needed": base_minimum_evidence(),
        "notes": notes,
    }


def external_context_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": row.get("evidence_id", ""),
                "candidate_type": "EXTERNAL_CONTEXT_EVIDENCE_LAYER",
                "source_layer": "external_evidence_registry.csv",
                "region": row.get("region", ""),
                "coordinate_available": "false",
                "patch_available": "false",
                "official_absence_evidence": "false",
                "official_stability_evidence": "false",
                "distance_to_nearest_positive_anchor_m": "",
                "same_event_or_locality_risk": "NOT_PATCH_TEMPORAL_EVIDENCE",
                "control_strength_status": "REVIEW_CONTROL_ONLY" if row.get("evidence_tier") in {"STRONG", "CONTEXTUAL"} else "INSUFFICIENT_EVIDENCE",
                "negative_label_status": "NO_NEGATIVE_LABEL_CONTEXT_LAYER",
                "can_be_negative_label": "false",
                "can_be_review_control": "true",
                "can_create_training_label": "false",
                "can_train_model": "false",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "blocking_reason": "CONTEXT_LAYER_HAS_NO_EXPLICIT_ABSENCE_STABILITY_FOR_PATCH_DATE_PHENOMENON",
                "minimum_evidence_needed": base_minimum_evidence(),
                "notes": "External evidence can support interpretation but does not document absence/stability for a matched control patch.",
            }
        )
    return out


def taxonomy_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        out.append(
            {
                "candidate_id": row.get("taxonomy_id", ""),
                "candidate_type": "PATCH_CORPUS_TAXONOMY_LAYER",
                "source_layer": "patch_corpus_taxonomy_registry.csv",
                "region": "MULTI_REGION",
                "coordinate_available": "false",
                "patch_available": "true" if row.get("count_total") else "false",
                "official_absence_evidence": "false",
                "official_stability_evidence": "false",
                "distance_to_nearest_positive_anchor_m": "",
                "same_event_or_locality_risk": "TAXONOMY_NOT_EVENT_ABSENCE_EVIDENCE",
                "control_strength_status": "REVIEW_CONTROL_ONLY",
                "negative_label_status": "NO_NEGATIVE_LABEL_TAXONOMY_LAYER",
                "can_be_negative_label": "false",
                "can_be_review_control": "true",
                "can_create_training_label": "false",
                "can_train_model": "false",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "blocking_reason": "CORPUS_COUNTS_DO_NOT_ESTABLISH_ABSENCE_OR_STABILITY",
                "minimum_evidence_needed": base_minimum_evidence(),
                "notes": "Corpus taxonomy is useful for scope and counts, not for negative labels.",
            }
        )
    return out


def build_evidence_registry() -> list[dict[str, str]]:
    controls = read_csv(CONTROL_EXPANSION) or read_csv(REVIEW_CONTROLS)
    rows = [classify_control(row) for row in controls]
    rows.extend(external_context_rows(read_csv(EXTERNAL_EVIDENCE)))
    rows.extend(taxonomy_rows(read_csv(PATCH_TAXONOMY)))
    return rows


def readiness_matrix(evidence_rows: list[dict[str, str]]) -> dict[str, str]:
    counts = {status: sum(1 for row in evidence_rows if row["control_strength_status"] == status) for status in ["FORMAL_NEGATIVE_READY", "STRONG_CONTROL_CANDIDATE", "REVIEW_CONTROL_ONLY", "INVALID_NEGATIVE_ABSENCE_ASSUMPTION", "INSUFFICIENT_EVIDENCE"]}
    anchors = read_csv(ANCHOR_REGISTRY)
    formal_negative_count = counts["FORMAL_NEGATIVE_READY"]
    return {
        "matrix_id": "V1JL_NEGATIVE_LABEL_READINESS",
        "formal_negative_ready_count": str(formal_negative_count),
        "strong_control_candidate_count": str(counts["STRONG_CONTROL_CANDIDATE"]),
        "review_control_only_count": str(counts["REVIEW_CONTROL_ONLY"]),
        "invalid_negative_assumption_count": str(counts["INVALID_NEGATIVE_ABSENCE_ASSUMPTION"]),
        "insufficient_evidence_count": str(counts["INSUFFICIENT_EVIDENCE"]),
        "positive_reference_candidate_count": str(len(anchors)),
        "formal_positive_label_ready_count": "0",
        "formal_negative_label_ready_count": str(formal_negative_count),
        "can_create_training_label": "false",
        "negative_label_status": "NO_FORMAL_NEGATIVES_READY" if formal_negative_count == 0 else "FORMAL_NEGATIVE_REVIEW_REQUIRED",
        "minimum_evidence_needed": base_minimum_evidence(),
        "notes": "No candidate has explicit official absence/stability evidence for a matched area, date window, and phenomenon.",
    }


def supervised_gate(readiness: dict[str, str]) -> dict[str, str]:
    split_rows = read_csv(SPLIT_PROTOCOL)
    split_status = ";".join(sorted({row.get("split_readiness_status", "") for row in split_rows if row.get("split_readiness_status")})) or "SPLIT_PROTOCOL_NOT_FOUND"
    leakage_status = ";".join(sorted({row.get("leakage_risk_status", "") for row in split_rows if row.get("leakage_risk_status")})) or "LEAKAGE_PROTOCOL_REQUIRED"
    formal_negatives = int(readiness["formal_negative_ready_count"])
    formal_positives = int(readiness["formal_positive_label_ready_count"])
    if formal_negatives == 0:
        boundary = "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES"
        blocker = "NO_FORMAL_NEGATIVES_READY"
    elif formal_positives == 0:
        boundary = "SUPERVISED_TRAINING_BLOCKED_LABEL_GATES"
        blocker = "NO_FORMAL_POSITIVE_LABELS_READY"
    else:
        boundary = "REVIEW_ONLY_PLUS_CONTROLS_READY"
        blocker = "SPLIT_AND_LEAKAGE_GATES_STILL_REQUIRED"
    return {
        "gate_id": "V1JL_SUPERVISED_TRAINING_MINIMUM_GATE",
        "formal_positive_labels_ready": str(formal_positives),
        "formal_negative_labels_ready": str(formal_negatives),
        "review_only_controls_ready": readiness["strong_control_candidate_count"],
        "split_protocol_status": split_status,
        "leakage_risk_status": leakage_status,
        "supervised_training_boundary_status": boundary,
        "can_create_training_label": "false",
        "can_train_model": "false",
        "can_unfreeze_dino_for_scientific_claim": "false",
        "primary_blocker": blocker,
        "minimum_evidence_needed": "formal positive labels; formal negative labels; split/leakage protocol closed; independent validation metrics",
        "notes": "Strong review controls can support review-only analysis, but they do not unlock supervised training.",
    }


def public_text_has_private_path(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(fragment in text for fragment in PRIVATE_FRAGMENTS)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    evidence = build_evidence_registry()
    readiness = readiness_matrix(evidence)
    gate = supervised_gate(readiness)

    write_csv(LOCAL_RUN_DIR / "v1jl_formal_negative_control_evidence_audit.csv", evidence, EVIDENCE_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jl_negative_label_readiness_matrix.csv", [readiness], READINESS_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jl_supervised_training_minimum_gate_matrix.csv", [gate], GATE_FIELDS)

    write_csv(DATASETS_DIR / "formal_negative_control_evidence_registry.csv", evidence, EVIDENCE_FIELDS)
    write_csv(DATASETS_DIR / "negative_label_readiness_matrix.csv", [readiness], READINESS_FIELDS)
    write_csv(DATASETS_DIR / "supervised_training_minimum_gate_matrix.csv", [gate], GATE_FIELDS)
    write_schema(SCHEMAS_DIR / "formal_negative_control_evidence_schema.csv", EVIDENCE_FIELDS, "REV-P v1jl formal negative/control evidence field")
    write_schema(SCHEMAS_DIR / "negative_label_readiness_schema.csv", READINESS_FIELDS, "REV-P v1jl negative label readiness field")
    write_schema(SCHEMAS_DIR / "supervised_training_minimum_gate_schema.csv", GATE_FIELDS, "REV-P v1jl supervised training minimum gate field")

    qa_rows = [
        {"check": "absence_of_record_not_negative", "status": "PASS" if all(row["can_be_negative_label"] == "false" for row in evidence if "ABSENCE" in row["blocking_reason"] or row["control_strength_status"] == "INVALID_NEGATIVE_ABSENCE_ASSUMPTION") else "FAIL", "detail": readiness["invalid_negative_assumption_count"]},
        {"check": "temporal_self_not_independent_negative", "status": "PASS" if all(row["can_be_negative_label"] == "false" for row in evidence if row["candidate_type"] == "TEMPORAL_SELF_CONTROL") else "FAIL", "detail": readiness["strong_control_candidate_count"]},
        {"check": "cross_region_not_negative", "status": "PASS" if all(row["can_be_negative_label"] == "false" for row in evidence if row["candidate_type"] == "CROSS_REGION_CONTEXT_CANDIDATE") else "FAIL", "detail": "checked"},
        {"check": "formal_negative_requires_explicit_evidence", "status": "PASS" if int(readiness["formal_negative_ready_count"]) == 0 else "FAIL", "detail": readiness["formal_negative_ready_count"]},
        {"check": "can_train_model_false_without_negatives", "status": "PASS" if gate["can_train_model"] == "false" and gate["supervised_training_boundary_status"] == "SUPERVISED_TRAINING_BLOCKED_NO_NEGATIVES" else "FAIL", "detail": gate["supervised_training_boundary_status"]},
        {"check": "can_create_training_label_false", "status": "PASS" if readiness["can_create_training_label"] == "false" else "FAIL", "detail": readiness["can_create_training_label"]},
        {"check": "can_unfreeze_false", "status": "PASS" if gate["can_unfreeze_dino_for_scientific_claim"] == "false" else "FAIL", "detail": gate["can_unfreeze_dino_for_scientific_claim"]},
    ]
    public_files = [
        DATASETS_DIR / "formal_negative_control_evidence_registry.csv",
        DATASETS_DIR / "negative_label_readiness_matrix.csv",
        DATASETS_DIR / "supervised_training_minimum_gate_matrix.csv",
    ]
    qa_rows.append({"check": "no_private_path_in_public_outputs", "status": "PASS" if not any(public_text_has_private_path(path) for path in public_files) else "FAIL", "detail": "public registries checked"})
    write_csv(LOCAL_RUN_DIR / "v1jl_qa.csv", qa_rows, ["check", "status", "detail"])

    summary = {
        "stage": "v1jl",
        "timestamp": utc_now(),
        "audited_candidate_count": len(evidence),
        "formal_negative_ready_count": int(readiness["formal_negative_ready_count"]),
        "strong_control_candidate_count": int(readiness["strong_control_candidate_count"]),
        "review_control_only_count": int(readiness["review_control_only_count"]),
        "invalid_negative_assumption_count": int(readiness["invalid_negative_assumption_count"]),
        "insufficient_evidence_count": int(readiness["insufficient_evidence_count"]),
        "supervised_training_boundary_status": gate["supervised_training_boundary_status"],
        "project_status": "REVIEW_ONLY_PLUS_CONTROLS_READY",
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
    }
    write_json(LOCAL_RUN_DIR / "v1jl_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jl FORMAL NEGATIVE AND CONTROL EVIDENCE PROTOCOL")
    print(f"Audited candidates: {summary['audited_candidate_count']}")
    print(f"Formal negatives ready: {summary['formal_negative_ready_count']}")
    print(f"Strong review controls: {summary['strong_control_candidate_count']}")
    print(f"Review-only controls: {summary['review_control_only_count']}")
    print(f"Invalid negative assumptions: {summary['invalid_negative_assumption_count']}")
    print(f"Insufficient evidence: {summary['insufficient_evidence_count']}")
    print(f"Training boundary: {summary['supervised_training_boundary_status']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
