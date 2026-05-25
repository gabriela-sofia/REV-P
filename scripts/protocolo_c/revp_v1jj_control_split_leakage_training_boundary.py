"""
REV-P v1jj - CONTROL_SPLIT_LEAKAGE_AND_SANDBOX_TRAINING_BOUNDARY.

Builds a formal control-candidate, split/leakage, and sandbox boundary for the
multi-anchor v1ji batch. This stage is metadata-only: it does not create labels,
formal negatives, model training, DINO unfreeze, or heavy outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1jj"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

ANCHORS_PATH = DATASETS_DIR / "official_multi_anchor_registry.csv"
PATCH_PATH = DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv"
DINO_PATH = DATASETS_DIR / "multi_anchor_dino_review_embedding_registry.csv"
CONTROLS_PATH = DATASETS_DIR / "review_control_candidate_registry.csv"
GATE_PATH = DATASETS_DIR / "multi_anchor_training_gate_matrix.csv"

CONTROL_FIELDS = [
    "control_candidate_id",
    "control_type",
    "source_layer",
    "region",
    "nearest_anchor_id",
    "distance_to_nearest_anchor_m",
    "buffer_status",
    "s2_available",
    "s1_available",
    "dem_available",
    "dino_available",
    "absence_claim_made",
    "can_be_review_control",
    "can_be_negative_label",
    "can_create_training_label",
    "leakage_risk_status",
    "blocking_reason",
    "notes",
]

SPLIT_FIELDS = [
    "protocol_id",
    "split_unit",
    "spatial_buffer_rule",
    "temporal_rule",
    "same_anchor_pair_rule",
    "control_rule",
    "cross_region_rule",
    "leakage_risk_status",
    "split_readiness_status",
    "can_train_model",
    "blocking_reason",
    "notes",
]

SANDBOX_FIELDS = [
    "boundary_id",
    "review_only_batch_ready",
    "weak_label_sandbox_allowed_local_only",
    "one_class_prototype_sandbox_allowed",
    "supervised_training_ready",
    "can_unfreeze_dino_for_scientific_claim",
    "can_create_training_label",
    "can_train_model",
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


def write_schema(path: Path, fields: list[str], description_prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{description_prefix}: {field}."} for field in fields], ["field", "description"])


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1jj").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")[:120]


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lam = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lam / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def by_anchor(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["anchor_id"]: row for row in rows if row.get("anchor_id")}


def audit_dino_counts(dino_rows: list[dict[str, str]], patch_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    qa_pass = [row for row in dino_rows if row.get("dino_status") == "DINO_QA_PASS"]
    dims = sorted({row.get("embedding_dim", "") for row in qa_pass if row.get("embedding_dim")})
    pair_ready = [row for row in patch_rows if row.get("s2_pre_status") == "QA_PASS" and row.get("s2_post_status") == "QA_PASS"]
    return [
        {
            "audit_id": "V1JJ_DINO_BATCH_COUNT_NORMALIZATION",
            "source_registry": "multi_anchor_dino_review_embedding_registry.csv",
            "registry_record_count": len(dino_rows),
            "pre_embedding_count": len(qa_pass),
            "post_embedding_count": len(qa_pass),
            "pair_diagnostic_count": len(qa_pass),
            "s2_pair_qa_pass_count": len(pair_ready),
            "embedding_dim": ";".join(dims) if dims else "",
            "qa_status": "QA_PASS" if len(qa_pass) == len(pair_ready) and dims == ["768"] else "QA_REVIEW_REQUIRED",
            "notes": "v1ji stores one diagnostic row per pre/post pair; each row implies one pre embedding and one post embedding used for comparison.",
        }
    ]


def availability_for(anchor_id: str, patch_map: dict[str, dict[str, str]], dino_map: dict[str, dict[str, str]]) -> dict[str, str]:
    patch = patch_map.get(anchor_id, {})
    return {
        "s2_available": bool_text(patch.get("s2_pre_status") == "QA_PASS" and patch.get("s2_post_status") == "QA_PASS"),
        "s1_available": bool_text(patch.get("s1_pre_status") == "QA_PASS" and patch.get("s1_post_status") == "QA_PASS"),
        "dem_available": bool_text(patch.get("dem_status") == "QA_PASS"),
        "dino_available": bool_text(dino_map.get(anchor_id, {}).get("dino_status") == "DINO_QA_PASS"),
    }


def build_control_candidates(
    anchors: list[dict[str, str]], patch_rows: list[dict[str, str]], dino_rows: list[dict[str, str]], previous_controls: list[dict[str, str]]
) -> list[dict[str, Any]]:
    patch_map = by_anchor(patch_rows)
    dino_map = by_anchor(dino_rows)
    controls: list[dict[str, Any]] = []

    for anchor in anchors:
        aid = anchor["anchor_id"]
        controls.append(
            {
                "control_candidate_id": f"V1JJ_TEMPORAL_SELF_CONTROL_{safe_id(aid)}",
                "control_type": "TEMPORAL_SELF_CONTROL",
                "source_layer": "multi_anchor_multimodal_patch_registry.csv",
                "region": "PET",
                "nearest_anchor_id": aid,
                "distance_to_nearest_anchor_m": "0.000",
                "buffer_status": "SAME_ANCHOR_NOT_INDEPENDENT",
                **availability_for(aid, patch_map, dino_map),
                "absence_claim_made": "false",
                "can_be_review_control": "true",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "leakage_risk_status": "LEAKAGE_RISK_HIGH",
                "blocking_reason": "TEMPORAL_SELF_CONTROL_IS_NOT_NEGATIVE_OR_INDEPENDENT",
                "notes": "Pre-event context from the same anchor can support review but cannot be split as an independent negative sample.",
            }
        )

    existing_controls = [row for row in previous_controls if row.get("control_type") == "EXISTING_PATCH_BACKGROUND_CANDIDATE"]
    for row in existing_controls:
        controls.append(
            {
                "control_candidate_id": f"V1JJ_{row['control_candidate_id']}",
                "control_type": "EXISTING_PATCH_BACKGROUND_CANDIDATE",
                "source_layer": row.get("source", "review_control_candidate_registry.csv"),
                "region": row.get("region", "PET"),
                "nearest_anchor_id": "NOT_ASSESSED_NO_EXPLICIT_COORDINATE",
                "distance_to_nearest_anchor_m": "",
                "buffer_status": "BUFFER_NOT_ASSESSED_COORDINATE_REQUIRED",
                "s2_available": row.get("patch_available", "false"),
                "s1_available": "false",
                "dem_available": "false",
                "dino_available": row.get("patch_available", "false"),
                "absence_claim_made": "false",
                "can_be_review_control": "true",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "blocking_reason": "BACKGROUND_CANDIDATE_HAS_NO_ABSENCE_PROTOCOL",
                "notes": "Existing sanitized patch can be reviewed as context only; no absence claim is made.",
            }
        )

    controls.extend(
        [
            {
                "control_candidate_id": "V1JJ_SPATIAL_BACKGROUND_REVIEW_CANDIDATE_PET_BUFFERED",
                "control_type": "SPATIAL_BACKGROUND_REVIEW_CANDIDATE",
                "source_layer": "future_spatial_sampling_protocol",
                "region": "PET",
                "nearest_anchor_id": "REQUIRES_EXPLICIT_SAMPLE_POINT",
                "distance_to_nearest_anchor_m": "",
                "buffer_status": "BUFFER_RULE_DEFINED_NOT_SAMPLED",
                "s2_available": "false",
                "s1_available": "false",
                "dem_available": "false",
                "dino_available": "false",
                "absence_claim_made": "false",
                "can_be_review_control": "true",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "blocking_reason": "SPATIAL_BACKGROUND_REQUIRES_EXPLICIT_POINT_AND_ABSENCE_PROTOCOL",
                "notes": "Spatial background review needs a sampled point outside the buffer and later absence evidence before any negative label discussion.",
            },
            {
                "control_candidate_id": "V1JJ_CROSS_REGION_CONTEXT_RECIFE",
                "control_type": "CROSS_REGION_CONTEXT_CANDIDATE",
                "source_layer": "patch_corpus_or_sentinel_manifest_context",
                "region": "REC",
                "nearest_anchor_id": "NOT_COMPARABLE_TO_PET_OFFICIAL_ANCHORS",
                "distance_to_nearest_anchor_m": "",
                "buffer_status": "CROSS_REGION_CONTEXT_ONLY",
                "s2_available": "true",
                "s1_available": "false",
                "dem_available": "false",
                "dino_available": "true",
                "absence_claim_made": "false",
                "can_be_review_control": "true",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "leakage_risk_status": "CONTEXT_ONLY_NOT_EVENT_VALIDATION",
                "blocking_reason": "CROSS_REGION_CONTEXT_IS_NOT_FORMAL_NEGATIVE",
                "notes": "Cross-region material can support structural context only and is not event validation.",
            },
            {
                "control_candidate_id": "V1JJ_CROSS_REGION_CONTEXT_CURITIBA",
                "control_type": "CROSS_REGION_CONTEXT_CANDIDATE",
                "source_layer": "patch_corpus_or_sentinel_manifest_context",
                "region": "CUR",
                "nearest_anchor_id": "NOT_COMPARABLE_TO_PET_OFFICIAL_ANCHORS",
                "distance_to_nearest_anchor_m": "",
                "buffer_status": "CROSS_REGION_CONTEXT_ONLY",
                "s2_available": "true",
                "s1_available": "false",
                "dem_available": "false",
                "dino_available": "true",
                "absence_claim_made": "false",
                "can_be_review_control": "true",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "leakage_risk_status": "CONTEXT_ONLY_NOT_EVENT_VALIDATION",
                "blocking_reason": "CROSS_REGION_CONTEXT_IS_NOT_FORMAL_NEGATIVE",
                "notes": "Cross-region material can support structural context only and is not event validation.",
            },
            {
                "control_candidate_id": "V1JJ_INVALID_NEGATIVE_LABEL_ABSENCE_OF_RECORD",
                "control_type": "INVALID_NEGATIVE_LABEL",
                "source_layer": "label_governance_rule",
                "region": "ALL",
                "nearest_anchor_id": "NOT_APPLICABLE",
                "distance_to_nearest_anchor_m": "",
                "buffer_status": "INVALID_WITHOUT_ABSENCE_PROTOCOL",
                "s2_available": "false",
                "s1_available": "false",
                "dem_available": "false",
                "dino_available": "false",
                "absence_claim_made": "true",
                "can_be_review_control": "false",
                "can_be_negative_label": "false",
                "can_create_training_label": "false",
                "leakage_risk_status": "LEAKAGE_RISK_HIGH",
                "blocking_reason": "ABSENCE_OF_RECORD_IS_NOT_ABSENCE_OF_EVENT",
                "notes": "This row explicitly blocks promoting no-record context to a formal negative label.",
            },
        ]
    )
    return controls


def distance_audit(anchors: list[dict[str, str]], controls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        lat = float(anchor["latitude"])
        lon = float(anchor["longitude"])
        nearest = None
        nearest_dist = None
        for other in anchors:
            if other["anchor_id"] == anchor["anchor_id"]:
                continue
            distance = haversine_m(lat, lon, float(other["latitude"]), float(other["longitude"]))
            if nearest_dist is None or distance < nearest_dist:
                nearest = other["anchor_id"]
                nearest_dist = distance
        rows.append(
            {
                "audit_id": f"V1JJ_ANCHOR_DISTANCE_{safe_id(anchor['anchor_id'])}",
                "candidate_id": anchor["anchor_id"],
                "candidate_type": "OFFICIAL_ANCHOR",
                "nearest_anchor_id": nearest or "",
                "distance_to_nearest_anchor_m": f"{nearest_dist:.3f}" if nearest_dist is not None else "",
                "buffer_status": "ANCHOR_CLUSTER_REQUIRES_SPLIT_BY_DOCUMENT_OR_LOCALITY",
                "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
                "notes": "Nearby official anchors require split rules by unit, locality, and date before training can be considered.",
            }
        )
    for control in controls:
        rows.append(
            {
                "audit_id": f"V1JJ_CONTROL_DISTANCE_{safe_id(control['control_candidate_id'])}",
                "candidate_id": control["control_candidate_id"],
                "candidate_type": control["control_type"],
                "nearest_anchor_id": control["nearest_anchor_id"],
                "distance_to_nearest_anchor_m": control["distance_to_nearest_anchor_m"],
                "buffer_status": control["buffer_status"],
                "leakage_risk_status": control["leakage_risk_status"],
                "notes": control["notes"],
            }
        )
    return rows


def split_protocol() -> list[dict[str, str]]:
    common = {
        "spatial_buffer_rule": "Minimum buffer must be enforced between independent train/test samples; exact distance requires later protocol calibration.",
        "temporal_rule": "Split must preserve event/date integrity and cannot treat same-event pre/post as independent sides.",
        "same_anchor_pair_rule": "Pre and post patches from the same anchor must stay together as one paired unit.",
        "control_rule": "Candidate controls are review controls only and cannot become negatives without a later absence protocol.",
        "cross_region_rule": "Cross-region material is context or robustness review only, not event validation.",
        "can_train_model": "false",
    }
    return [
        {
            "protocol_id": "V1JJ_SPLIT_BY_DOCUMENTED_EVENT_UNIT",
            "split_unit": "DOCUMENTED_EVENT_UNIT",
            **common,
            "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
            "split_readiness_status": "SPLIT_NOT_READY_INSUFFICIENT_LABELS",
            "blocking_reason": "NO_FORMAL_NEGATIVES_AND_NO_LABEL_PROTOCOL",
            "notes": "Documented event unit is the primary split unit once labels exist.",
        },
        {
            "protocol_id": "V1JJ_SPLIT_BY_LOCALITY",
            "split_unit": "LOCALITY",
            **common,
            "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
            "split_readiness_status": "SPLIT_DESIGN_READY_FOR_REVIEW_ONLY",
            "blocking_reason": "LOCALITY_SPLIT_DESIGN_EXISTS_BUT_TRAINING_LABELS_DO_NOT",
            "notes": "Locality grouping is ready as a review design constraint, not as a training split.",
        },
        {
            "protocol_id": "V1JJ_SPLIT_BY_EVENT_DATE",
            "split_unit": "EVENT_DATE",
            **common,
            "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
            "split_readiness_status": "SPLIT_DESIGN_READY_FOR_REVIEW_ONLY",
            "blocking_reason": "DATE_SPLIT_DESIGN_EXISTS_BUT_FORMAL_NEGATIVES_DO_NOT",
            "notes": "Event/date grouping prevents temporal leakage between paired observations.",
        },
    ]


def sandbox_boundary(anchors: list[dict[str, str]], patch_rows: list[dict[str, str]], dino_rows: list[dict[str, str]], gate_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    s2_ready = sum(1 for row in patch_rows if row.get("s2_pre_status") == "QA_PASS" and row.get("s2_post_status") == "QA_PASS")
    dino_ready = sum(1 for row in dino_rows if row.get("dino_status") == "DINO_QA_PASS")
    review_ready = bool(anchors and s2_ready and dino_ready)
    gate = gate_rows[0] if gate_rows else {}
    minimum = (
        "multiple positive labels with formal label protocol; formal negatives or controls with absence evidence; "
        "split by event/locality; calibrated spatial buffer; leakage audit; held-out metrics"
    )
    return [
        {
            "boundary_id": "V1JJ_SANDBOX_TRAINING_BOUNDARY",
            "review_only_batch_ready": bool_text(review_ready),
            "weak_label_sandbox_allowed_local_only": bool_text(review_ready),
            "one_class_prototype_sandbox_allowed": bool_text(review_ready),
            "supervised_training_ready": "false",
            "can_unfreeze_dino_for_scientific_claim": "false",
            "can_create_training_label": "false",
            "can_train_model": "false",
            "minimum_evidence_needed": minimum,
            "notes": f"Sandbox status is INVALID_FOR_SCIENTIFIC_CLAIM; previous gate remains {gate.get('training_gate_status', 'SUPERVISED_TRAINING_BLOCKED')}. No weights are saved by this stage.",
        }
    ]


def public_text_has_private_path(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(fragment in text for fragment in PRIVATE_FRAGMENTS)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    anchors = read_csv(ANCHORS_PATH)
    patch_rows = read_csv(PATCH_PATH)
    dino_rows = read_csv(DINO_PATH)
    previous_controls = read_csv(CONTROLS_PATH)
    gate_rows = read_csv(GATE_PATH)

    dino_audit = audit_dino_counts(dino_rows, patch_rows)
    controls = build_control_candidates(anchors, patch_rows, dino_rows, previous_controls)
    distances = distance_audit(anchors, controls)
    split_rows = split_protocol()
    sandbox_rows = sandbox_boundary(anchors, patch_rows, dino_rows, gate_rows)

    write_csv(
        LOCAL_RUN_DIR / "v1jj_dino_batch_count_audit.csv",
        dino_audit,
        ["audit_id", "source_registry", "registry_record_count", "pre_embedding_count", "post_embedding_count", "pair_diagnostic_count", "s2_pair_qa_pass_count", "embedding_dim", "qa_status", "notes"],
    )
    write_csv(LOCAL_RUN_DIR / "v1jj_control_candidate_expansion.csv", controls, CONTROL_FIELDS)
    write_csv(
        LOCAL_RUN_DIR / "v1jj_anchor_control_distance_audit.csv",
        distances,
        ["audit_id", "candidate_id", "candidate_type", "nearest_anchor_id", "distance_to_nearest_anchor_m", "buffer_status", "leakage_risk_status", "notes"],
    )
    write_csv(LOCAL_RUN_DIR / "v1jj_split_leakage_protocol.csv", split_rows, SPLIT_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1jj_sandbox_training_boundary.csv", sandbox_rows, SANDBOX_FIELDS)

    write_csv(DATASETS_DIR / "control_candidate_expansion_registry.csv", controls, CONTROL_FIELDS)
    write_csv(DATASETS_DIR / "split_leakage_protocol_registry.csv", split_rows, SPLIT_FIELDS)
    write_csv(DATASETS_DIR / "sandbox_training_boundary_registry.csv", sandbox_rows, SANDBOX_FIELDS)
    write_schema(SCHEMAS_DIR / "control_candidate_expansion_schema.csv", CONTROL_FIELDS, "REV-P v1jj control candidate field")
    write_schema(SCHEMAS_DIR / "split_leakage_protocol_schema.csv", SPLIT_FIELDS, "REV-P v1jj split and leakage field")
    write_schema(SCHEMAS_DIR / "sandbox_training_boundary_schema.csv", SANDBOX_FIELDS, "REV-P v1jj sandbox boundary field")

    qa_rows = [
        {"check": "dino_count_normalized", "status": "PASS" if dino_audit[0]["qa_status"] == "QA_PASS" else "FAIL", "detail": f"pairs={dino_audit[0]['pair_diagnostic_count']}"},
        {"check": "controls_not_negative", "status": "PASS" if all(row["can_be_negative_label"] == "false" for row in controls) else "FAIL", "detail": str(len(controls))},
        {"check": "absence_not_label", "status": "PASS" if all(row["can_create_training_label"] == "false" for row in controls) else "FAIL", "detail": "all false"},
        {"check": "same_anchor_pair_rule_present", "status": "PASS" if all("same anchor" in row["same_anchor_pair_rule"].lower() for row in split_rows) else "FAIL", "detail": str(len(split_rows))},
        {"check": "training_blocked", "status": "PASS" if sandbox_rows[0]["can_train_model"] == "false" else "FAIL", "detail": sandbox_rows[0]["supervised_training_ready"]},
        {"check": "unfreeze_blocked", "status": "PASS" if sandbox_rows[0]["can_unfreeze_dino_for_scientific_claim"] == "false" else "FAIL", "detail": "false"},
        {"check": "sandbox_invalid_for_claim", "status": "PASS" if "INVALID_FOR_SCIENTIFIC_CLAIM" in sandbox_rows[0]["notes"] else "FAIL", "detail": sandbox_rows[0]["weak_label_sandbox_allowed_local_only"]},
    ]
    public_files = [
        DATASETS_DIR / "control_candidate_expansion_registry.csv",
        DATASETS_DIR / "split_leakage_protocol_registry.csv",
        DATASETS_DIR / "sandbox_training_boundary_registry.csv",
    ]
    qa_rows.append({"check": "no_private_path_in_public_outputs", "status": "PASS" if not any(public_text_has_private_path(path) for path in public_files) else "FAIL", "detail": "public metadata checked"})
    write_csv(LOCAL_RUN_DIR / "v1jj_qa.csv", qa_rows, ["check", "status", "detail"])

    type_counts = Counter(row["control_type"] for row in controls)
    summary = {
        "stage": "v1jj",
        "timestamp": utc_now(),
        "dino_pre_embedding_count": dino_audit[0]["pre_embedding_count"],
        "dino_post_embedding_count": dino_audit[0]["post_embedding_count"],
        "dino_pair_diagnostic_count": dino_audit[0]["pair_diagnostic_count"],
        "embedding_dim": dino_audit[0]["embedding_dim"],
        "control_candidate_count": len(controls),
        "control_candidate_type_counts": dict(sorted(type_counts.items())),
        "negative_labels_ready_count": 0,
        "split_readiness_status": "SPLIT_NOT_READY_INSUFFICIENT_LABELS",
        "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
        "review_only_batch_ready": sandbox_rows[0]["review_only_batch_ready"] == "true",
        "weak_label_sandbox_allowed_local_only": sandbox_rows[0]["weak_label_sandbox_allowed_local_only"] == "true",
        "one_class_prototype_sandbox_allowed": sandbox_rows[0]["one_class_prototype_sandbox_allowed"] == "true",
        "sandbox_status": "INVALID_FOR_SCIENTIFIC_CLAIM",
        "supervised_training_ready": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
    }
    write_json(LOCAL_RUN_DIR / "v1jj_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--read-v1ji-batch", action="store_true")
    parser.add_argument("--build-control-candidates", action="store_true")
    parser.add_argument("--design-split-leakage-protocol", action="store_true")
    parser.add_argument("--evaluate-sandbox-boundary", action="store_true")
    parser.add_argument("--emit-training-boundary", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1jj CONTROL SPLIT LEAKAGE AND SANDBOX TRAINING BOUNDARY")
    print(f"DINO pre embeddings: {summary['dino_pre_embedding_count']}")
    print(f"DINO post embeddings: {summary['dino_post_embedding_count']}")
    print(f"DINO pair diagnostics: {summary['dino_pair_diagnostic_count']}")
    print(f"Control candidates: {summary['control_candidate_count']}")
    print(f"Negative formal labels: {summary['negative_labels_ready_count']}")
    print(f"Split status: {summary['split_readiness_status']}")
    print(f"Sandbox status: {summary['sandbox_status']}")
    print(f"Training allowed: {summary['can_train_model']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
