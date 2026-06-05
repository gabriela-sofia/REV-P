#!/usr/bin/env python3
"""v2ah Candidate Reference Review Workbench / stop gate.

Consolidates v2 candidate packages into a review-only workbench and formally
stops operational ground-reference search until a new qualified source exists.
The stage is additive and fail-closed: it does not mutate v2ac-v2ag outputs,
apply dates, run overlay, create labels, or reopen Protocol B.
"""

import argparse
import csv
import hashlib
import os
import re
from collections import Counter, defaultdict

PROTOCOL_VERSION = "v2ah"
DATASET_DIR = os.environ.get("DATASET_DIR", "datasets/protocolo_c")
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"

STOP_STATUS = "GROUND_TRUTH_SEARCH_STOPPED_UNTIL_NEW_QUALIFIED_SOURCE"
REQUIRED_ACTION = "maintain_candidate_review_only_layer"

DOMINANT_BLOCKERS = [
    "no observed geometry",
    "no occurrence coordinates",
    "no explicit Sentinel date crosswalk",
    "no operational ground reference",
    "no patch-bound truth",
]

FORBIDDEN_TRUE_FIELDS = {
    "ground_truth", "ground_reference", "label", "training", "overlay",
    "prediction", "protocol_b_reopen", "sentinel_date_inferred",
    "crosswalk_inferred", "ground_truth_operational",
    "can_create_ground_reference", "can_create_training_label",
    "can_reopen_protocol_b", "can_apply_overlay", "can_infer_sentinel_date",
    "applied_to_package",
}
FORBIDDEN_PROMOTION_VALUES = {
    "GROUND_REFERENCE", "GROUND_TRUTH", "TRAINING_LABEL", "PATCH_POSITIVE",
    "PATCH_NEGATIVE", "OPERATIONAL_VALIDATED", "OBSERVED_FLOOD_LABEL",
    "FLOOD_DETECTED", "EVENT_VALIDATED_BY_SENTINEL", "PATCH_DATE_INFERRED",
    "CROSSWALK_INFERRED", "SENTINEL_DATE_APPLIED_TO_PATCH",
    "LABEL_CREATED", "MODEL_TRAINING_READY", "PROTOCOL_B_REOPENED",
}
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")


STOP_GATE_COLUMNS = [
    "stop_gate_id", "ground_truth_search_status", "source_versions",
    "dominant_blockers", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "can_apply_overlay",
    "can_infer_sentinel_date", "required_action", "gate_basis",
    "guardrail_status", "notes",
]
QUEUE_COLUMNS = [
    "review_queue_id", "package_id", "event_id", "region", "patch_id",
    "hazard_type", "candidate_status", "evidence_strength",
    "dominant_blocker", "review_priority_score", "review_priority_rank",
    "review_action", "allowed_use", "forbidden_use",
]
DOSSIER_COLUMNS = [
    "dossier_id", "package_id", "event_id", "region", "patch_id",
    "existing_evidence", "source_artifacts", "source_artifact_hashes",
    "blockers", "use_status", "non_promotion_reason",
    "next_evidence_needed",
]
REOPEN_COLUMNS = [
    "condition_id", "condition_name", "required_evidence",
    "minimum_source_class", "can_reopen_if_met",
    "still_forbidden_without_human_review",
]
SAMPLING_COLUMNS = [
    "sample_id", "package_id", "event_id", "region", "patch_id",
    "phenomenon_type", "package_status", "dominant_blocker",
    "evidence_strength", "stratum_id", "sample_rank_within_stratum",
    "sample_seed", "sample_use", "forbidden_use",
]
TCC_COLUMNS = [
    "export_id", "export_scope", "metric_name", "metric_value",
    "claim_allowed", "claim_forbidden", "safe_wording", "unsafe_wording",
    "allowed_use", "forbidden_use",
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

V2AH_DATASETS = [
    "v2ah_ground_truth_search_stop_gate.csv",
    "v2ah_candidate_reference_review_queue.csv",
    "v2ah_candidate_evidence_dossier_index.csv",
    "v2ah_reopen_conditions_registry.csv",
    "v2ah_stratified_review_sampling.csv",
    "v2ah_safe_tcc_export_registry.csv",
    "v2ah_guardrail_regression.csv",
    "v2ah_next_actions_registry.csv",
    "v2ah_completion_report.csv",
]

V2_INPUTS = [
    "v2ac_event_patch_v2_package_registry.csv",
    "v2ad_qa_gate_summary.csv",
    "v2ae_canonical_event_patch_registry.csv",
    "v2af_qa_gate_orchestration.csv",
    "v2ag_sentinel_date_linkability_audit.csv",
    "v2ag_unlinkable_date_guard_update.csv",
    "v2ag_next_programming_target_ranker.csv",
]


def parse_args(argv=None):
    return argparse.ArgumentParser().parse_args(argv)


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


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


def bool_closed(value):
    return "true" if str(value).strip().lower() == "true" else "false"


def is_true(value):
    return str(value).strip().lower() == "true"


def clean(value):
    return str(value or "").strip()


def collect_artifacts(prefixes=("v2ac", "v2ad", "v2ae", "v2af", "v2ag")):
    rows = []
    if not os.path.exists(DATASET_DIR):
        return rows
    for name in sorted(os.listdir(DATASET_DIR)):
        if not name.endswith(".csv"):
            continue
        if not any(name.startswith(prefix + "_") for prefix in prefixes):
            continue
        path = dataset_path(name)
        rows.append({
            "artifact_path": rel_dataset(name),
            "artifact_hash": sha256_file(path),
            "row_count": str(len(load_csv(path))),
        })
    return rows


def source_hashes(names):
    hashes = []
    for name in names:
        path = dataset_path(name)
        if os.path.exists(path):
            hashes.append(f"{name}:{sha256_file(path)[:16]}")
    return "|".join(hashes)


def package_rows():
    rows = load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))
    if rows:
        return rows
    return load_csv(dataset_path("v2ae_canonical_event_patch_registry.csv"))


def linkability_by_patch():
    return {clean(r.get("patch_id")): r for r in load_csv(dataset_path("v2ag_sentinel_date_linkability_audit.csv"))}


def guard_by_patch():
    return {clean(r.get("patch_id")): r for r in load_csv(dataset_path("v2ag_unlinkable_date_guard_update.csv"))}


def dominant_blocker(row):
    text = "|".join([
        clean(row.get("blocker")),
        clean(row.get("temporal_blocker")),
        clean(row.get("geometry_status")),
        clean(row.get("coordinate_status")),
        clean(row.get("date_linkability_status")),
        clean(row.get("ground_reference_status")),
    ]).lower()
    if "geometry" in text:
        return "no observed geometry"
    if "coordinate" in text:
        return "no occurrence coordinates"
    if "crosswalk" in text or "unlinkable" in text or "date" in text:
        return "no explicit Sentinel date crosswalk"
    if "ground" in text:
        return "no operational ground reference"
    return "no patch-bound truth"


def evidence_strength(row):
    evidence = clean(row.get("evidence_status"))
    dino = clean(row.get("dino_review_support_status") or row.get("crosswalk_status"))
    if "DINO" in dino.upper():
        return "STRUCTURAL_SUPPORT_REVIEW_ONLY"
    if "DOCUMENT" in evidence.upper():
        return "DOCUMENTARY_CONTEXT_ONLY"
    if evidence:
        return evidence
    return "EVIDENCE_LIMITED_REVIEW_ONLY"


def package_identifier(row, idx):
    return clean(row.get("canonical_package_id") or row.get("event_patch_candidate_id") or f"PKG_v2ah_{idx:05d}")


def review_score(row):
    score = 0
    if clean(row.get("patch_id")):
        score += 20
    if "DINO" in clean(row.get("dino_review_support_status") or row.get("crosswalk_status")).upper():
        score += 15
    if clean(row.get("event_id")):
        score += 10
    if dominant_blocker(row) == "no explicit Sentinel date crosswalk":
        score += 10
    if clean(row.get("geometry_status")) == "GEOMETRY_STILL_MISSING":
        score += 5
    return score


def assert_no_forbidden_promotion(rows):
    violations = []
    for idx, row in enumerate(rows):
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                violations.append((idx, key, value_s, "forbidden_true"))
            if value_s in FORBIDDEN_PROMOTION_VALUES:
                violations.append((idx, key, value_s, "forbidden_status"))
            if ABSOLUTE_PATH_RE.search(value_s):
                violations.append((idx, key, value_s, "absolute_path"))
            if "local" + "_" + "only" in value_s:
                violations.append((idx, key, value_s, "non_versionable_path_marker"))
    if violations:
        sample = "; ".join(f"row={r[0]} field={r[1]} type={r[3]}" for r in violations[:5])
        raise ValueError(f"Forbidden promotion or path marker found: {sample}")
    return True


def run_ground_truth_search_stop_gate(args=None):
    artifacts = collect_artifacts()
    rows = [{
        "stop_gate_id": "STOP_v2ah_000",
        "ground_truth_search_status": STOP_STATUS,
        "source_versions": "v2ac|v2ad|v2ae|v2af|v2ag",
        "dominant_blockers": "|".join(DOMINANT_BLOCKERS),
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "can_reopen_protocol_b": "false",
        "can_apply_overlay": "false",
        "can_infer_sentinel_date": "false",
        "required_action": REQUIRED_ACTION,
        "gate_basis": f"{len(artifacts)} prior versionable artifacts reviewed",
        "guardrail_status": "FAIL_CLOSED_REVIEW_ONLY",
        "notes": "Operational ground-reference search remains stopped until a new qualified source is available.",
    }]
    assert_no_forbidden_promotion(rows)
    write_csv(dataset_path("v2ah_ground_truth_search_stop_gate.csv"), STOP_GATE_COLUMNS, rows)
    return rows


def run_candidate_reference_review_queue(args=None):
    rows = []
    packages = package_rows()
    guards = guard_by_patch()
    for idx, pkg in enumerate(packages):
        patch_id = clean(pkg.get("patch_id"))
        guard = guards.get(patch_id, {})
        blocker = dominant_blocker(pkg)
        status = "BLOCKED_REFERENCE_CANDIDATE" if blocker in {"no observed geometry", "no occurrence coordinates"} else "REVIEW_ONLY_CANDIDATE"
        rows.append({
            "review_queue_id": f"RQ_v2ah_{idx:05d}",
            "package_id": package_identifier(pkg, idx),
            "event_id": clean(pkg.get("event_id")),
            "region": clean(pkg.get("event_region") or pkg.get("region")),
            "patch_id": patch_id,
            "hazard_type": clean(pkg.get("phenomenon_status")) or "HAZARD_CONTEXT_REVIEW_ONLY",
            "candidate_status": status,
            "evidence_strength": evidence_strength(pkg),
            "dominant_blocker": blocker,
            "review_priority_score": str(review_score(pkg)),
            "review_priority_rank": "0",
            "review_action": clean(guard.get("future_allowed_action")) or "review_candidate_context_only",
            "allowed_use": "review_queue_and_tcc_context_only",
            "forbidden_use": "ground_reference|label|training|overlay|prediction|protocol_b_reopen",
        })
    rows.sort(key=lambda r: (-int(r["review_priority_score"]), r["event_id"], r["patch_id"], r["package_id"]))
    for rank, row in enumerate(rows, 1):
        row["review_priority_rank"] = str(rank)
    assert_no_forbidden_promotion(rows)
    write_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"), QUEUE_COLUMNS, rows)
    return rows


def run_candidate_evidence_dossier_index(args=None):
    packages = package_rows()
    links = linkability_by_patch()
    source_names = [name for name in V2_INPUTS if os.path.exists(dataset_path(name))]
    hashes = source_hashes(source_names)
    rows = []
    for idx, pkg in enumerate(packages):
        patch_id = clean(pkg.get("patch_id"))
        link = links.get(patch_id, {})
        blockers = "|".join(sorted({dominant_blocker(pkg), clean(link.get("linkability_status")) or "DATE_REMAINS_UNLINKABLE"}))
        rows.append({
            "dossier_id": f"DOS_v2ah_{idx:05d}",
            "package_id": package_identifier(pkg, idx),
            "event_id": clean(pkg.get("event_id")),
            "region": clean(pkg.get("event_region") or pkg.get("region")),
            "patch_id": patch_id,
            "existing_evidence": "|".join(filter(None, [clean(pkg.get("evidence_status")), clean(pkg.get("dino_review_support_status")), clean(pkg.get("crosswalk_status"))])),
            "source_artifacts": "|".join(rel_dataset(name) for name in source_names),
            "source_artifact_hashes": hashes,
            "blockers": blockers,
            "use_status": "REVIEW_ONLY_NOT_OPERATIONAL",
            "non_promotion_reason": "no explicit observed geometry, coordinate, Sentinel date crosswalk, or patch-bound truth",
            "next_evidence_needed": "qualified institutional source with explicit geometry or coordinate and patch/date/asset crosswalk plus review",
        })
    assert_no_forbidden_promotion(rows)
    write_csv(dataset_path("v2ah_candidate_evidence_dossier_index.csv"), DOSSIER_COLUMNS, rows)
    return rows


def run_reopen_conditions_registry(args=None):
    specs = [
        ("explicit_observed_geometry", "observed polygon or geometry tied to the event and patch", "institutional geospatial product"),
        ("verifiable_coordinate_or_polygon", "coordinate or polygon with source lineage and event linkage", "institutional source or validated operational product"),
        ("compatible_temporal_window", "event-compatible timestamp or acquisition window", "source timestamp with product lineage"),
        ("explicit_patch_date_asset_crosswalk", "same-row or documented patch to date or asset crosswalk", "versionable registry or manifest"),
        ("institutional_or_operational_source", "institutional source or operational product with stable identifier", "public institutional source"),
        ("concordant_human_review", "documented reviewer agreement after evidence package review", "human review record"),
        ("no_phenomenon_conflict", "evidence shows no flood, mass-movement, or context conflict", "adjudicated evidence matrix"),
    ]
    rows = []
    for idx, (name, evidence, source_class) in enumerate(specs):
        rows.append({
            "condition_id": f"RC_v2ah_{idx:03d}",
            "condition_name": name,
            "required_evidence": evidence,
            "minimum_source_class": source_class,
            "can_reopen_if_met": "review_gate_only",
            "still_forbidden_without_human_review": "true",
        })
    write_csv(dataset_path("v2ah_reopen_conditions_registry.csv"), REOPEN_COLUMNS, rows)
    return rows


def run_stratified_review_sampler(args=None):
    queue_path = dataset_path("v2ah_candidate_reference_review_queue.csv")
    queue = load_csv(queue_path) if os.path.exists(queue_path) else run_candidate_reference_review_queue(args)
    grouped = defaultdict(list)
    for row in queue:
        key = "|".join([
            row["region"], row["event_id"], row["hazard_type"],
            row["candidate_status"], row["dominant_blocker"], row["evidence_strength"],
        ])
        grouped[key].append(row)
    rows = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda r: (r["patch_id"], r["package_id"]))
        for rank, item in enumerate(group[:2], 1):
            rows.append({
                "sample_id": f"SMP_v2ah_{len(rows):05d}",
                "package_id": item["package_id"],
                "event_id": item["event_id"],
                "region": item["region"],
                "patch_id": item["patch_id"],
                "phenomenon_type": item["hazard_type"],
                "package_status": item["candidate_status"],
                "dominant_blocker": item["dominant_blocker"],
                "evidence_strength": item["evidence_strength"],
                "stratum_id": hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                "sample_rank_within_stratum": str(rank),
                "sample_seed": "20260604",
                "sample_use": "human_review_sampling_only_not_training_balance",
                "forbidden_use": "supervised_balance|label|training|overlay|prediction",
            })
    assert_no_forbidden_promotion(rows)
    write_csv(dataset_path("v2ah_stratified_review_sampling.csv"), SAMPLING_COLUMNS, rows)
    return rows


def run_safe_tcc_export_builder(args=None):
    queue = load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv")) or run_candidate_reference_review_queue(args)
    stop = load_csv(dataset_path("v2ah_ground_truth_search_stop_gate.csv")) or run_ground_truth_search_stop_gate(args)
    status_counts = Counter(r["candidate_status"] for r in queue)
    blocker_counts = Counter(r["dominant_blocker"] for r in queue)
    rows = [
        {
            "export_id": "TCC_v2ah_000",
            "export_scope": "stop_gate",
            "metric_name": "ground_truth_search_status",
            "metric_value": stop[0]["ground_truth_search_status"],
            "claim_allowed": "Search for operational patch-level reference is stopped until new qualified source.",
            "claim_forbidden": "Operational validation or label exists.",
            "safe_wording": "The current evidence supports a review-only candidate layer.",
            "unsafe_wording": "ground truth validado",
            "allowed_use": "methods_status_summary",
            "forbidden_use": "operational_claim",
        },
        {
            "export_id": "TCC_v2ah_001",
            "export_scope": "review_queue",
            "metric_name": "review_candidate_count",
            "metric_value": str(len(queue)),
            "claim_allowed": "Candidate packages are available for review-only inspection.",
            "claim_forbidden": "Candidates are labels or classes.",
            "safe_wording": "review-only candidate packages",
            "unsafe_wording": "label operacional|classe positiva",
            "allowed_use": "counts_and_limitations",
            "forbidden_use": "training_dataset_description",
        },
        {
            "export_id": "TCC_v2ah_002",
            "export_scope": "guardrails",
            "metric_name": "blocked_operational_actions",
            "metric_value": "overlay|prediction|training|protocol_b_reopen",
            "claim_allowed": "Overlay, prediction, training, and Protocol B reopening remain blocked.",
            "claim_forbidden": "Flood detection or observed flood validation was performed.",
            "safe_wording": "No operational overlay or prediction is produced.",
            "unsafe_wording": "deteccao de enchente|predicao|validacao de inundacao observada",
            "allowed_use": "limitations",
            "forbidden_use": "results_as_detection",
        },
    ]
    for status, count in sorted(status_counts.items()):
        rows.append({
            "export_id": f"TCC_v2ah_{len(rows):03d}",
            "export_scope": "candidate_status",
            "metric_name": status,
            "metric_value": str(count),
            "claim_allowed": "Status count is descriptive and review-only.",
            "claim_forbidden": "Status count creates ground reference.",
            "safe_wording": f"{count} packages have status {status}.",
            "unsafe_wording": "ground truth validado",
            "allowed_use": "descriptive_count",
            "forbidden_use": "promotion_or_label",
        })
    for blocker, count in sorted(blocker_counts.items()):
        rows.append({
            "export_id": f"TCC_v2ah_{len(rows):03d}",
            "export_scope": "blocker",
            "metric_name": blocker,
            "metric_value": str(count),
            "claim_allowed": "Blocker count explains why promotion is unavailable.",
            "claim_forbidden": "Blocker count proves a positive or negative class.",
            "safe_wording": f"{count} packages remain blocked by {blocker}.",
            "unsafe_wording": "classe positiva",
            "allowed_use": "blocker_summary",
            "forbidden_use": "class_assignment",
        })
    write_csv(dataset_path("v2ah_safe_tcc_export_registry.csv"), TCC_COLUMNS, rows)
    return rows


def scan_artifact_for_guardrails(path):
    rows = load_csv(path)
    checks = Counter()
    for row in rows:
        for key, value in row.items():
            key_l = clean(key).lower()
            value_s = clean(value)
            if key_l in FORBIDDEN_TRUE_FIELDS and is_true(value_s):
                checks["forbidden_true_flag"] += 1
            if value_s in FORBIDDEN_PROMOTION_VALUES:
                checks["forbidden_promotion_status"] += 1
            if ABSOLUTE_PATH_RE.search(value_s):
                checks["absolute_path"] += 1
            if "local" + "_" + "only" in value_s and any(token in key_l for token in ("path", "source", "artifact")):
                checks["non_versionable_path_marker"] += 1
    return checks


def run_guardrail_regression(args=None):
    names = []
    for prefix in ("v2ah", "v2af", "v2ag"):
        if os.path.exists(DATASET_DIR):
            names.extend(n for n in os.listdir(DATASET_DIR) if n.startswith(prefix + "_") and n.endswith(".csv"))
    rows = []
    for name in sorted(set(names)):
        path = dataset_path(name)
        checks = scan_artifact_for_guardrails(path)
        for ctype in ("forbidden_true_flag", "forbidden_promotion_status", "absolute_path", "non_versionable_path_marker"):
            count = checks.get(ctype, 0)
            rows.append({
                "guardrail_check_id": f"GR_v2ah_{len(rows):05d}",
                "artifact_path": rel_dataset(name),
                "check_type": ctype,
                "violation_count": str(count),
                "status": "PASS" if count == 0 else "FAIL",
                "severity": "none" if count == 0 else "blocking",
                "notes": "Fail-closed scan over v2ah and recent QA/discovery outputs.",
            })
    write_csv(dataset_path("v2ah_guardrail_regression.csv"), GUARDRAIL_COLUMNS, rows)
    return rows


def run_next_action_ranker(args=None):
    queue = load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv")) or run_candidate_reference_review_queue(args)
    stop = load_csv(dataset_path("v2ah_ground_truth_search_stop_gate.csv")) or run_ground_truth_search_stop_gate(args)
    stopped = stop and stop[0]["ground_truth_search_status"] == STOP_STATUS
    options = [
        ("HUMAN_REVIEW_ADJUDICATION_PACKAGE", 95 if queue else 55, "review queue and dossier index", "v2ah_candidate_reference_review_queue.csv"),
        ("SAFE_TCC_EVIDENCE_EXPORT", 90 if stopped else 40, "safe claim registry", "v2ah_safe_tcc_export_registry.csv"),
        ("REOPEN_CONDITION_MONITORING", 60, "new qualified source", "v2ah_reopen_conditions_registry.csv"),
        ("REGISTRY_MAINTENANCE_ONLY", 35, "existing metadata", "v2ah_completion_report.csv"),
        ("OPERATIONAL_MODELING_TRAINING_OVERLAY", 0, "blocked by stop gate", "none"),
    ]
    rows = []
    for rank, (action, score, required, artifact) in enumerate(sorted(options, key=lambda x: (-x[1], x[0])), 1):
        rows.append({
            "rank": str(rank),
            "next_action": action,
            "score": str(score),
            "allowed": "false" if score == 0 else "true",
            "blocked_operational_use": "true",
            "required_input": required,
            "recommended_script_or_artifact": artifact,
            "notes": "No next action may create labels, training data, overlay, prediction, or Protocol B reopening.",
        })
    write_csv(dataset_path("v2ah_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, rows)
    return rows


def run_completion_report(args=None):
    generated = [name for name in V2AH_DATASETS if os.path.exists(dataset_path(name))]
    read_artifacts = collect_artifacts()
    queue = load_csv(dataset_path("v2ah_candidate_reference_review_queue.csv"))
    stop = load_csv(dataset_path("v2ah_ground_truth_search_stop_gate.csv"))
    guards = load_csv(dataset_path("v2ah_guardrail_regression.csv"))
    failures = sum(1 for r in guards if r.get("status") == "FAIL")
    rows = [
        {"completion_id": "CR_v2ah_000", "metric": "files_read", "value": str(len(read_artifacts)), "status": "RECORDED", "notes": "Prior v2 artifacts inspected by stage utilities."},
        {"completion_id": "CR_v2ah_001", "metric": "files_generated", "value": str(len(generated)), "status": "RECORDED", "notes": "|".join(f"datasets/protocolo_c/{n}" for n in generated)},
        {"completion_id": "CR_v2ah_002", "metric": "review_queue_rows", "value": str(len(queue)), "status": "REVIEW_ONLY", "notes": "Candidates are not labels or references."},
        {"completion_id": "CR_v2ah_003", "metric": "stop_gate", "value": stop[0]["ground_truth_search_status"] if stop else "", "status": "FAIL_CLOSED", "notes": "Search remains stopped until a new qualified source."},
        {"completion_id": "CR_v2ah_004", "metric": "guardrail_failures", "value": str(failures), "status": "PASS" if failures == 0 else "FAIL", "notes": "Guardrail regression over v2ah plus v2af/v2ag."},
        {"completion_id": "CR_v2ah_005", "metric": "decision_final", "value": "maintain_candidate_review_only_layer", "status": "NO_OPERATIONAL_PROMOTION", "notes": "No overlay, label, training, prediction, or Protocol B reopening."},
    ]
    write_csv(dataset_path("v2ah_completion_report.csv"), COMPLETION_COLUMNS, rows)
    write_text(doc_path("protocolo_c_v2ah_completion_report.md"), [
        "# Protocolo C v2ah completion report",
        "",
        f"Files read: {len(read_artifacts)}.",
        f"Files generated: {len(generated)}.",
        f"Review queue rows: {len(queue)}.",
        f"Stop gate: {stop[0]['ground_truth_search_status'] if stop else ''}.",
        f"Guardrail failures: {failures}.",
        "Final decision: maintain candidate review-only layer.",
        "No operational overlay, label, training, prediction, or Protocol B reopening is produced.",
    ])
    return rows


def run_all(args=None):
    run_ground_truth_search_stop_gate(args)
    run_candidate_reference_review_queue(args)
    run_candidate_evidence_dossier_index(args)
    run_reopen_conditions_registry(args)
    run_stratified_review_sampler(args)
    run_safe_tcc_export_builder(args)
    run_guardrail_regression(args)
    run_next_action_ranker(args)
    return run_completion_report(args)
