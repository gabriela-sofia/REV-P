#!/usr/bin/env python3
"""v2ai Human Review & Adjudication Package.

Builds review assignments, empty decision templates, adjudication scaffolding,
uncertainty registries, outcome placeholders, and promotion blockers from the
v2ah review-only queue. It never simulates completed human review or creates an
operational ground reference, label, overlay, prediction, or Protocol B state.
"""

import argparse
import csv
import hashlib
import os
import re
from collections import Counter

PROTOCOL_VERSION = "v2ai"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"

PENDING_REVIEW = "PENDING_HUMAN_REVIEW"
WAITING_ADJUDICATION = "WAITING_FOR_HUMAN_REVIEW"

REQUIRED_V2AH = [
    "v2ah_ground_truth_search_stop_gate.csv",
    "v2ah_candidate_reference_review_queue.csv",
    "v2ah_candidate_evidence_dossier_index.csv",
    "v2ah_reopen_conditions_registry.csv",
    "v2ah_safe_tcc_export_registry.csv",
    "v2ah_guardrail_regression.csv",
    "v2ah_next_actions_registry.csv",
]

FORBIDDEN_TRUE_FIELDS = {
    "ground_truth", "ground_reference", "label", "training", "overlay",
    "prediction", "protocol_b_reopen", "sentinel_date_inferred",
    "crosswalk_inferred", "human_review_completed", "adjudication_completed",
    "can_create_ground_reference", "can_create_training_label",
    "promotion_allowed", "can_promote_after_adjudication",
}
FORBIDDEN_STATUSES = {
    "GROUND_TRUTH_VALIDATED", "GROUND_REFERENCE_TRUE", "LABEL_POSITIVE",
    "LABEL_NEGATIVE", "TRAINING_READY", "PROTOCOL_B_OPEN",
    "OPERATIONAL_VALIDATION", "GROUND_REFERENCE", "GROUND_TRUTH",
    "TRAINING_LABEL", "PATCH_POSITIVE", "PATCH_NEGATIVE",
    "OPERATIONAL_VALIDATED", "FLOOD_DETECTED",
}
FORBIDDEN_NEXT_ACTION_TOKENS = re.compile(
    r"(TRAIN|MODEL|OVERLAY|LABEL|GROUND_TRUTH|PROTOCOL_B|PREDICTION|INFER_SENTINEL_DATE)",
    re.I,
)
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")

ASSIGNMENT_COLUMNS = [
    "assignment_id", "package_id", "event_id", "region", "patch_id",
    "hazard_type", "review_round", "reviewer_slot", "reviewer_role",
    "assignment_status", "source_queue_rank", "dominant_blocker",
    "allowed_use", "forbidden_use",
]
DECISION_TEMPLATE_COLUMNS = [
    "assignment_id", "package_id", "reviewer_slot", "decision_status",
    "phenomenon_observed", "spatial_confidence", "temporal_confidence",
    "source_confidence", "evidence_sufficiency", "uncertainty_level",
    "needs_adjudication", "reviewer_notes", "decision_timestamp",
]
ADJUDICATION_COLUMNS = [
    "adjudication_id", "package_id", "adjudication_status",
    "trigger_condition", "reviewer_a_status", "reviewer_b_status",
    "conflict_type", "required_adjudicator_action",
    "can_promote_after_adjudication", "still_requires_external_evidence",
]
UNCERTAINTY_COLUMNS = [
    "package_id", "event_id", "region", "uncertainty_spatial",
    "uncertainty_temporal", "uncertainty_phenomenon", "uncertainty_source",
    "uncertainty_crosswalk", "dominant_uncertainty", "uncertainty_level",
    "uncertainty_reason", "effect_on_allowed_use",
]
OUTCOME_COLUMNS = [
    "package_id", "event_id", "region", "patch_id", "review_outcome_status",
    "human_review_completed", "adjudication_completed",
    "can_create_ground_reference", "can_create_training_label",
    "allowed_use", "forbidden_use", "notes",
]
PROMOTION_BLOCKER_COLUMNS = [
    "package_id", "blocker_observed_geometry",
    "blocker_occurrence_coordinates", "blocker_sentinel_date_crosswalk",
    "blocker_operational_ground_reference", "blocker_human_review",
    "blocker_adjudication", "blocker_phenomenon_conflict",
    "promotion_status", "promotion_allowed", "promotion_reason",
]
GUARDRAIL_COLUMNS = [
    "guardrail_check_id", "artifact_path", "check_type", "violation_count",
    "status", "severity", "notes",
]
NEXT_ACTION_COLUMNS = [
    "rank", "next_action", "score", "allowed", "blocked_operational_use",
    "required_input", "recommended_script_or_artifact", "notes",
]
COMPLETION_COLUMNS = [
    "completion_id", "metric", "value", "status", "notes",
]

V2AI_DATASETS = [
    "v2ai_review_assignment_registry.csv",
    "v2ai_reviewer_decision_template.csv",
    "v2ai_adjudication_queue.csv",
    "v2ai_uncertainty_registry.csv",
    "v2ai_review_outcome_registry.csv",
    "v2ai_safe_promotion_blockers.csv",
    "v2ai_guardrail_regression.csv",
    "v2ai_next_actions_registry.csv",
    "v2ai_completion_report.csv",
]


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path, lines):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_dataset(name):
    return f"datasets/protocolo_c/{name}"


def clean(value):
    return str(value or "").strip()


def is_true(value):
    return clean(value).lower() == "true"


def bool_closed(value):
    return "true" if is_true(value) else "false"


def assert_min_schema(rows, required, artifact):
    if not rows:
        raise FileNotFoundError(f"Required artifact is missing or empty: {artifact}")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{artifact} missing required columns: {','.join(missing)}")
    return True


def assert_v2ah_ready():
    missing = [name for name in REQUIRED_V2AH if not os.path.exists(dataset_path(name))]
    if missing:
        raise FileNotFoundError("v2ah is required before v2ai; missing: " + ",".join(missing))
    stop = load_csv(dataset_path("v2ah_ground_truth_search_stop_gate.csv"))
    assert_min_schema(stop, ["ground_truth_search_status"], "v2ah_ground_truth_search_stop_gate.csv")
    if stop[0]["ground_truth_search_status"] != "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE":
        raise ValueError("v2ah stop gate is not fail-closed.")
    queue = load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"))
    assert_min_schema(queue, ["package_id", "event_id", "region", "patch_id", "dominant_blocker"], "v2ah_candidate_reference_review_queue.csv")
    return True


def queue_rows():
    assert_v2ah_ready()
    return load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"))


def assignment_rows():
    path = dataset_path("v2ai_review_assignment_registry.csv")
    if os.path.exists(path):
        return load_csv(path)
    return run_review_assignment_builder(parse_args([]))


def build_blocker_signature(row):
    parts = [
        clean(row.get("dominant_blocker")),
        clean(row.get("candidate_status")),
        clean(row.get("evidence_strength")),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def assert_no_operational_promotion(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                violations.append((idx, key, "forbidden_true"))
            if value_s in FORBIDDEN_STATUSES:
                violations.append((idx, key, "forbidden_status"))
            if ABSOLUTE_PATH_RE.search(value_s):
                violations.append((idx, key, "absolute_path"))
            if "local" + "_" + "only" in value_s and any(token in key_l for token in ("path", "source", "artifact")):
                violations.append((idx, key, "non_versionable_path_marker"))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]} type={r[2]}" for r in violations[:5])
        raise ValueError(f"Operational promotion violation: {sample}")
    return True


def assert_no_fake_human_review(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in {"human_review_completed", "adjudication_completed"} and is_true(value_s):
                violations.append((idx, key))
            if value_s in {"REVIEW_COMPLETED", "ADJUDICATION_COMPLETED", "HUMAN_REVIEW_DONE"}:
                violations.append((idx, key))
            if key_l == "decision_timestamp" and value_s:
                violations.append((idx, key))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]}" for r in violations[:5])
        raise ValueError(f"Fake human review/adjudication detected: {sample}")
    return True


def run_review_assignment_builder(args=None):
    queue = sorted(queue_rows(), key=lambda r: (int(clean(r.get("review_priority_rank")) or "0"), clean(r.get("package_id"))))
    rows = []
    for row in queue:
        for slot in ("reviewer_a", "reviewer_b"):
            rows.append({
                "assignment_id": f"ASN_v2ai_{len(rows):05d}",
                "package_id": clean(row.get("package_id")),
                "event_id": clean(row.get("event_id")),
                "region": clean(row.get("region")),
                "patch_id": clean(row.get("patch_id")),
                "hazard_type": clean(row.get("hazard_type")),
                "review_round": "round_1",
                "reviewer_slot": slot,
                "reviewer_role": "independent_human_reviewer_slot",
                "assignment_status": "ASSIGNED_SLOT_PENDING_HUMAN_REVIEW",
                "source_queue_rank": clean(row.get("review_priority_rank")),
                "dominant_blocker": clean(row.get("dominant_blocker")),
                "allowed_use": "future_human_review_assignment_only",
                "forbidden_use": "real_reviewer_identity|ground_reference|label|training|overlay|prediction|protocol_b_reopen",
            })
    assert_no_operational_promotion(rows)
    assert_no_fake_human_review(rows)
    write_csv(dataset_path("v2ai_review_assignment_registry.csv"), ASSIGNMENT_COLUMNS, rows)
    return rows


def run_reviewer_decision_template_builder(args=None):
    rows = []
    for assignment in assignment_rows():
        rows.append({
            "assignment_id": assignment["assignment_id"],
            "package_id": assignment["package_id"],
            "reviewer_slot": assignment["reviewer_slot"],
            "decision_status": PENDING_REVIEW,
            "phenomenon_observed": "UNREVIEWED",
            "spatial_confidence": "UNREVIEWED",
            "temporal_confidence": "UNREVIEWED",
            "source_confidence": "UNREVIEWED",
            "evidence_sufficiency": "UNREVIEWED",
            "uncertainty_level": "UNKNOWN_UNTIL_REVIEW",
            "needs_adjudication": "UNKNOWN_UNTIL_REVIEW",
            "reviewer_notes": "",
            "decision_timestamp": "",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_human_review(rows)
    write_csv(dataset_path("v2ai_reviewer_decision_template.csv"), DECISION_TEMPLATE_COLUMNS, rows)
    return rows


def run_adjudication_queue_builder(args=None):
    packages = sorted({r["package_id"] for r in assignment_rows()})
    rows = []
    for package_id in packages:
        rows.append({
            "adjudication_id": f"ADJ_v2ai_{len(rows):05d}",
            "package_id": package_id,
            "adjudication_status": WAITING_ADJUDICATION,
            "trigger_condition": "reviewer_disagreement_or_uncertainty_after_future_review",
            "reviewer_a_status": PENDING_REVIEW,
            "reviewer_b_status": PENDING_REVIEW,
            "conflict_type": "UNKNOWN_UNTIL_REVIEW",
            "required_adjudicator_action": "wait_for_two_independent_reviews",
            "can_promote_after_adjudication": "false",
            "still_requires_external_evidence": "true",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_human_review(rows)
    write_csv(dataset_path("v2ai_adjudication_queue.csv"), ADJUDICATION_COLUMNS, rows)
    return rows


def uncertainty_from_blocker(blocker):
    b = clean(blocker).lower()
    if "geometry" in b:
        return "spatial"
    if "coordinate" in b:
        return "spatial"
    if "date" in b or "crosswalk" in b:
        return "temporal_crosswalk"
    if "phenomenon" in b:
        return "phenomenon"
    return "source"


def run_uncertainty_registry_builder(args=None):
    rows = []
    for row in queue_rows():
        blocker = clean(row.get("dominant_blocker"))
        dom = uncertainty_from_blocker(blocker)
        rows.append({
            "package_id": clean(row.get("package_id")),
            "event_id": clean(row.get("event_id")),
            "region": clean(row.get("region")),
            "uncertainty_spatial": "HIGH" if dom == "spatial" else "UNRESOLVED",
            "uncertainty_temporal": "HIGH" if dom == "temporal_crosswalk" else "UNRESOLVED",
            "uncertainty_phenomenon": "HIGH" if dom == "phenomenon" else "UNREVIEWED",
            "uncertainty_source": "UNRESOLVED",
            "uncertainty_crosswalk": "HIGH" if "crosswalk" in blocker.lower() else "UNRESOLVED",
            "dominant_uncertainty": dom,
            "uncertainty_level": "HIGH_UNRESOLVED",
            "uncertainty_reason": f"{blocker}|signature={build_blocker_signature(row)}",
            "effect_on_allowed_use": "review_only_no_automatic_resolution",
        })
    assert_no_operational_promotion(rows)
    write_csv(dataset_path("v2ai_uncertainty_registry.csv"), UNCERTAINTY_COLUMNS, rows)
    return rows


def run_review_outcome_registry_builder(args=None):
    rows = []
    for row in queue_rows():
        rows.append({
            "package_id": clean(row.get("package_id")),
            "event_id": clean(row.get("event_id")),
            "region": clean(row.get("region")),
            "patch_id": clean(row.get("patch_id")),
            "review_outcome_status": PENDING_REVIEW,
            "human_review_completed": "false",
            "adjudication_completed": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "allowed_use": "review_outcome_placeholder_only",
            "forbidden_use": "ground_reference|label|training|overlay|prediction|protocol_b_reopen",
            "notes": "Outcome registry is a future-review scaffold; no human decision is simulated.",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_human_review(rows)
    write_csv(dataset_path("v2ai_review_outcome_registry.csv"), OUTCOME_COLUMNS, rows)
    return rows


def blocker_flags(row):
    blocker = clean(row.get("dominant_blocker")).lower()
    return {
        "blocker_observed_geometry": "true" if "geometry" in blocker else "false",
        "blocker_occurrence_coordinates": "true" if "coordinate" in blocker else "false",
        "blocker_sentinel_date_crosswalk": "true" if "crosswalk" in blocker or "date" in blocker else "false",
        "blocker_operational_ground_reference": "true",
        "blocker_human_review": "true",
        "blocker_adjudication": "true",
        "blocker_phenomenon_conflict": "true" if "phenomenon" in blocker else "false",
    }


def run_safe_promotion_blockers(args=None):
    rows = []
    for row in queue_rows():
        flags = blocker_flags(row)
        rows.append({
            "package_id": clean(row.get("package_id")),
            **flags,
            "promotion_status": "PROMOTION_BLOCKED_PENDING_REAL_REVIEW_AND_EXTERNAL_EVIDENCE",
            "promotion_allowed": "false",
            "promotion_reason": "human review and adjudication are not complete, and external operational evidence remains required",
        })
    assert_no_operational_promotion(rows)
    assert_no_fake_human_review(rows)
    write_csv(dataset_path("v2ai_safe_promotion_blockers.csv"), PROMOTION_BLOCKER_COLUMNS, rows)
    return rows


def scan_artifact(path):
    rows = load_csv(path)
    checks = Counter()
    for row in rows:
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                checks["forbidden_true_flag"] += 1
            if value_s in FORBIDDEN_STATUSES:
                checks["forbidden_promotion_status"] += 1
            if key_l == "decision_timestamp" and value_s:
                checks["fake_human_review_timestamp"] += 1
            if ABSOLUTE_PATH_RE.search(value_s):
                checks["absolute_path"] += 1
            if "local" + "_" + "only" in value_s and any(token in key_l for token in ("path", "source", "artifact")):
                checks["non_versionable_path_marker"] += 1
    return checks


def run_guardrail_regression(args=None):
    names = []
    if os.path.exists(DATASET_DIR):
        for name in os.listdir(DATASET_DIR):
            if name.endswith(".csv") and (name.startswith("v2ai_") or name.startswith("v2ah_")):
                names.append(name)
    rows = []
    check_types = [
        "forbidden_true_flag", "forbidden_promotion_status",
        "fake_human_review_timestamp", "absolute_path",
        "non_versionable_path_marker",
    ]
    for name in sorted(set(names)):
        checks = scan_artifact(dataset_path(name))
        for ctype in check_types:
            count = checks.get(ctype, 0)
            rows.append({
                "guardrail_check_id": f"GR_v2ai_{len(rows):05d}",
                "artifact_path": rel_dataset(name),
                "check_type": ctype,
                "violation_count": str(count),
                "status": "PASS" if count == 0 else "FAIL",
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed scan over v2ai and v2ah outputs.",
            })
    write_csv(dataset_path("v2ai_guardrail_regression.csv"), GUARDRAIL_COLUMNS, rows)
    return rows


def run_next_action_ranker(args=None):
    options = [
        ("HUMAN_REVIEW_EXECUTION_OR_SAFE_TCC_EXPORT", 100, "v2ai_review_assignment_registry.csv|v2ai_reviewer_decision_template.csv", "v2ai_next_actions_registry.csv"),
        ("SAFE_TCC_PROTOCOL_C_WRITEUP", 85, "v2ah_safe_tcc_export_registry.csv|v2ai_completion_report.csv", "docs/metodologia_cientifica/protocolo_c_v2ai_completion_report.md"),
        ("REVIEW_GUIDE_FOR_HUMAN_ADJUDICATION", 80, "v2ai_adjudication_queue.csv|v2ai_uncertainty_registry.csv", "v2ai_adjudication_queue.csv"),
        ("WAIT_FOR_NEW_QUALIFIED_SOURCE", 65, "new qualified external evidence", "v2ai_safe_promotion_blockers.csv"),
        ("TRAINING_OVERLAY_LABEL_PROTOCOL_B", 0, "blocked by v2ai guardrails", "none"),
    ]
    rows = []
    for rank, (action, score, required, artifact) in enumerate(sorted(options, key=lambda x: (-x[1], x[0])), 1):
        rows.append({
            "rank": str(rank),
            "next_action": action,
            "score": str(score),
            "allowed": "false" if score == 0 or FORBIDDEN_NEXT_ACTION_TOKENS.search(action) else "true",
            "blocked_operational_use": "true",
            "required_input": required,
            "recommended_script_or_artifact": artifact,
            "notes": "Safe next actions cannot create training, labels, overlay, ground truth, Protocol B, or date inference.",
        })
    write_csv(dataset_path("v2ai_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, rows)
    return rows


def count_failures(name):
    return sum(1 for r in load_csv(dataset_path(name)) if clean(r.get("status")) == "FAIL")


def run_completion_report(args=None):
    assert_v2ah_ready()
    queue = load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"))
    assignments = load_csv(dataset_path("v2ai_review_assignment_registry.csv"))
    templates = load_csv(dataset_path("v2ai_reviewer_decision_template.csv"))
    adjudication = load_csv(dataset_path("v2ai_adjudication_queue.csv"))
    blockers = load_csv(dataset_path("v2ai_safe_promotion_blockers.csv"))
    actions = load_csv(dataset_path("v2ai_next_actions_registry.csv"))
    generated = [name for name in V2AI_DATASETS if os.path.exists(dataset_path(name))]
    rows = [
        {"completion_id": "CR_v2ai_000", "metric": "inputs_read", "value": "|".join(rel_dataset(n) for n in REQUIRED_V2AH), "status": "RECORDED", "notes": "v2ah artifacts required and present."},
        {"completion_id": "CR_v2ai_001", "metric": "outputs_created", "value": str(len(generated)), "status": "RECORDED", "notes": "|".join(rel_dataset(n) for n in generated)},
        {"completion_id": "CR_v2ai_002", "metric": "candidates", "value": str(len(queue)), "status": "REVIEW_ONLY", "notes": "Read from v2ah review queue."},
        {"completion_id": "CR_v2ai_003", "metric": "assignments", "value": str(len(assignments)), "status": "PENDING_HUMAN_REVIEW", "notes": "Two reviewer slots per candidate."},
        {"completion_id": "CR_v2ai_004", "metric": "decision_templates", "value": str(len(templates)), "status": "PENDING_HUMAN_REVIEW", "notes": "No reviewer decision simulated."},
        {"completion_id": "CR_v2ai_005", "metric": "promotion_blockers", "value": str(len(blockers)), "status": "PROMOTION_BLOCKED", "notes": "promotion_allowed=false for every package."},
        {"completion_id": "CR_v2ai_006", "metric": "human_review_status", "value": PENDING_REVIEW, "status": "NOT_COMPLETED", "notes": "No real human review recorded."},
        {"completion_id": "CR_v2ai_007", "metric": "adjudication_status", "value": WAITING_ADJUDICATION, "status": "NOT_COMPLETED", "notes": f"{len(adjudication)} packages waiting."},
        {"completion_id": "CR_v2ai_008", "metric": "guardrail_failures", "value": str(count_failures("v2ai_guardrail_regression.csv")), "status": "PASS" if count_failures("v2ai_guardrail_regression.csv") == 0 else "FAIL", "notes": "v2ai/v2ah guardrail regression."},
        {"completion_id": "CR_v2ai_009", "metric": "next_action_rank_1", "value": actions[0]["next_action"] if actions else "", "status": "SAFE_NEXT_ACTION", "notes": "No operational promotion."},
        {"completion_id": "CR_v2ai_010", "metric": "decision_final", "value": "human_review_package_prepared_no_operational_promotion", "status": "NO_OPERATIONAL_PROMOTION", "notes": "No label, training, overlay, prediction, ground truth, or Protocol B."},
    ]
    write_csv(dataset_path("v2ai_completion_report.csv"), COMPLETION_COLUMNS, rows)
    write_text(doc_path("protocolo_c_v2ai_completion_report.md"), [
        "# Protocolo C v2ai completion report",
        "",
        f"Inputs read: {len(REQUIRED_V2AH)} v2ah artifacts.",
        f"Candidates: {len(queue)}.",
        f"Assignments: {len(assignments)}.",
        f"Decision templates: {len(templates)}.",
        f"Promotion blockers: {len(blockers)}.",
        f"Human review status: {PENDING_REVIEW}.",
        f"Adjudication status: {WAITING_ADJUDICATION}.",
        f"Guardrail failures: {count_failures('v2ai_guardrail_regression.csv')}.",
        f"Next action rank 1: {actions[0]['next_action'] if actions else ''}.",
        "Final decision: human review package prepared with no operational promotion.",
    ])
    return rows


def run_all(args=None):
    run_review_assignment_builder(args)
    run_reviewer_decision_template_builder(args)
    run_adjudication_queue_builder(args)
    run_uncertainty_registry_builder(args)
    run_review_outcome_registry_builder(args)
    run_safe_promotion_blockers(args)
    run_guardrail_regression(args)
    run_next_action_ranker(args)
    return run_completion_report(args)
