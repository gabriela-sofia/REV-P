#!/usr/bin/env python3
"""v2af Event-patch package v2 QA automation.

A single, command-runnable QA orchestrator that validates the v2 state before
any new stage: it builds an input manifest, audits artifact freshness, validates
expected counts, runs guardrail / canonical-registry / event-patch regressions,
consolidates one approval gate, and emits a failure report. It reads prior
outputs read-only; it never modifies canonical registries or v2ac packages,
seeks new sources, infers dates/crosswalks/coordinates, executes overlay, or
creates ground truth, ground reference or labels.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v2af"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/event_patch_v2_qa_automation/staging/v2af"
REPORTS_DIR = "local_only/protocolo_c/event_patch_v2_qa_automation/reports/v2af"

MAX_STATUS = "EVENT_PATCH_V2_QA_AUTOMATION_NON_OPERATIONAL"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "qa_automation_only", "crosswalk_inferred",
    "sentinel_date_inferred", "raw_data_versioned",
]
GUARDRAIL_MUST_BE_FALSE = {
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "patch_bound_truth",
    "operational_validation", "crosswalk_inferred", "sentinel_date_inferred",
    "raw_data_versioned",
}
FORBIDDEN_STATUS_TOKENS = [
    "GROUND_REFERENCE", "GROUND_TRUTH", "TRAINING_LABEL", "PATCH_POSITIVE",
    "PATCH_NEGATIVE", "OPERATIONAL_VALIDATED", "OBSERVED_FLOOD_LABEL",
    "FLOOD_DETECTED", "EVENT_VALIDATED_BY_SENTINEL", "PATCH_DATE_INFERRED",
    "CROSSWALK_INFERRED",
]
FORBIDDEN_STATUS_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_STATUS_TOKENS) + r")\b")
TOOL_NAME_RE = re.compile(r"\b(claude|codex|llm|assistant|chatgpt|openai|anthropic|copilot|gemini)\b", re.I)
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")

EXPECTED_REGIONS = ["REC", "PET", "CUR"]
EXPECTED_EVENTS = ["REC_2022_05_24_30", "PET_2022_02_15", "PET_2024_03_21_28", "CUR_2022_01_15"]
EXPECTED_PACKAGES = 172
EXPECTED_READINESS_V2 = 2580

# Input artifacts the automation consumes: (relative_path, source_version,
# required, expected_row_count or None).
INPUT_ARTIFACTS = [
    ("datasets/protocolo_c/v2ae_canonical_region_registry.csv", "v2ae", True, 3),
    ("datasets/protocolo_c/v2ae_canonical_event_registry.csv", "v2ae", True, 4),
    ("datasets/protocolo_c/v2ae_canonical_event_patch_registry.csv", "v2ae", True, EXPECTED_PACKAGES),
    ("datasets/protocolo_c/v2ae_multiregion_blocker_consolidation.csv", "v2ae", True, None),
    ("datasets/protocolo_c/v2ae_multiregion_readiness_consolidation.csv", "v2ae", True, None),
    ("datasets/protocolo_c/v2ae_region_reopen_condition_registry.csv", "v2ae", True, 3),
    ("datasets/protocolo_c/v2ae_safe_use_policy_registry.csv", "v2ae", True, None),
    ("datasets/protocolo_c/v2ae_registry_consistency_qa.csv", "v2ae", True, None),
    ("datasets/protocolo_c/v2ae_next_programming_target_ranker.csv", "v2ae", False, None),
    ("datasets/protocolo_c/v2ad_package_contract_qa.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ad_namespace_crosswalk_qa.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ad_temporal_safety_qa.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ad_guardrail_qa.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ad_readiness_consistency_qa.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ad_migration_integrity_qa.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ad_negative_fixture_qa.csv", "v2ad", True, 10),
    ("datasets/protocolo_c/v2ad_qa_gate_summary.csv", "v2ad", True, None),
    ("datasets/protocolo_c/v2ac_event_patch_v2_package_registry.csv", "v2ac", True, EXPECTED_PACKAGES),
    ("datasets/protocolo_c/v2ac_v2_readiness_matrix.csv", "v2ac", True, EXPECTED_READINESS_V2),
    ("datasets/protocolo_c/v2ac_schema_contract_validation.csv", "v2ac", True, EXPECTED_PACKAGES),
    ("datasets/protocolo_c/v2ac_migration_diff_audit.csv", "v2ac", True, EXPECTED_PACKAGES),
    ("datasets/protocolo_c/v2ab_unlinkable_date_guard_registry.csv", "v2ab", False, None),
    ("datasets/protocolo_c/v2aa_multiregion_temporal_blocker_reduction.csv", "v2aa", False, None),
]

# Artifacts swept by the guardrail regression.
GUARDRAIL_SWEEP_GLOBS = ["v2ac_*.csv", "v2ad_*.csv", "v2ae_*.csv", "v2af_*.csv"]

# Column definitions ------------------------------------------------------
MANIFEST_INPUT_COLUMNS = [
    "input_id", "artifact_path", "source_version", "artifact_type", "required",
    "expected_row_count", "actual_row_count", "sha256", "existence_status",
    "freshness_status", "notes",
]
FRESHNESS_COLUMNS = [
    "freshness_id", "artifact_path", "source_version", "required",
    "exists", "non_empty", "header_readable", "schema_minimal_present",
    "hash_computable", "freshness_status", "notes",
]
COUNT_COLUMNS = [
    "count_check_id", "check_name", "expected_count", "actual_count", "status",
    "severity", "notes",
]
REGRESSION_COLUMNS = [
    "regression_id", "check_group", "check_name", "expected", "observed",
    "status", "severity", "notes",
]
GUARDRAIL_REG_COLUMNS = [
    "guardrail_regression_id", "artifact", "check_type", "violation_count",
    "status", "severity", "notes",
]
GATE_COLUMNS = [
    "gate_id", "qa_component", "total_checks", "passed_checks", "failed_checks",
    "expected_blockers", "gate_status", "required_action", "notes",
]
FAILURE_COLUMNS = [
    "failure_id", "artifact", "check", "severity", "recommended_fix",
    "blocking_status", "notes",
]
RANKER_COLUMNS = [
    "rank", "next_target", "programming_value", "ground_truth_value",
    "blocker_reduction_value", "expected_effort", "overclaim_risk",
    "recommended_version", "recommended_action", "notes",
]
BLOCKER_MATRIX_COLUMNS = [
    "blocker_id", "region", "event_id", "blocker", "status", *GUARDRAIL_COLUMNS,
    "notes",
]
NEXT_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V2AF_ARTIFACTS = [
    "configs/protocolo_c/v2af_qa_input_manifest_policy.yaml",
    "configs/protocolo_c/v2af_expected_count_policy.yaml",
    "configs/protocolo_c/v2af_guardrail_regression_policy.yaml",
    "configs/protocolo_c/v2af_registry_regression_policy.yaml",
    "configs/protocolo_c/v2af_qa_gate_policy.yaml",
    "configs/protocolo_c/v2af_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2af_qa_input_manifest.csv",
    "datasets/protocolo_c/v2af_artifact_freshness_audit.csv",
    "datasets/protocolo_c/v2af_expected_count_validation.csv",
    "datasets/protocolo_c/v2af_guardrail_regression.csv",
    "datasets/protocolo_c/v2af_canonical_registry_regression.csv",
    "datasets/protocolo_c/v2af_event_patch_v2_regression.csv",
    "datasets/protocolo_c/v2af_qa_gate_orchestration.csv",
    "datasets/protocolo_c/v2af_failure_report.csv",
    "datasets/protocolo_c/v2af_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2af_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2af_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v2af_event_patch_v2_qa_automation.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2af_event_patch_v2_qa_automation.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2af.md",
]


# Helpers -----------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def dataset_path(name):
    return os.path.join(DATASET_DIR, name)


def config_path(name):
    return os.path.join(CONFIG_DIR, name)


def doc_path(name):
    return os.path.join(DOCS_DIR, name)


def artifact_path(path):
    base = os.path.basename(path)
    if path.startswith("datasets/protocolo_c/"):
        return dataset_path(base)
    if path.startswith("configs/protocolo_c/"):
        return config_path(base)
    if path.startswith("docs/metodologia_cientifica/"):
        return doc_path(base)
    return path


def _resolve_input(rel_path):
    """Resolve an INPUT_ARTIFACTS relative path against the (mockable) dirs."""
    return artifact_path(rel_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def guardrails():
    return {
        "ground_truth_operational": "false",
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "can_reopen_protocol_b": "false",
        "dino_usage": "SUPPORT_ONLY",
        "no_overlay_executed": "true",
        "no_coordinates_invented": "true",
        "patch_bound_truth": "false",
        "operational_validation": "false",
        "qa_automation_only": "true",
        "crosswalk_inferred": "false",
        "sentinel_date_inferred": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v2af_qa_input_manifest_policy.yaml": [
            "use_absolute_path: false",
            "scan_local_only: false",
            "required_inputs_must_exist: true",
            "max_status: EVENT_PATCH_V2_QA_AUTOMATION_NON_OPERATIONAL",
        ],
        "v2af_expected_count_policy.yaml": [
            "canonical_regions: 3",
            "canonical_events: 4",
            "canonical_event_patch_packages: 172",
            "v2ac_packages: 172",
            "v2_readiness_rows: 2580",
            "allow_extra_only_if_justified: false",
        ],
        "v2af_guardrail_regression_policy.yaml": [
            "sweep_globs: [v2ac_*.csv, v2ad_*.csv, v2ae_*.csv, v2af_*.csv]",
            "forbidden_true_blocked: true",
            "forbidden_status_blocked: true",
            "overlay_ground_reference_training_must_be_blocked: true",
        ],
        "v2af_registry_regression_policy.yaml": [
            "regions: [REC, PET, CUR]",
            "events: [REC_2022_05_24_30, PET_2022_02_15, PET_2024_03_21_28, CUR_2022_01_15]",
            "region_status_must_not_change: true",
            "qa_gate_must_be_preserved: true",
        ],
        "v2af_qa_gate_policy.yaml": [
            "gate_states: [QA_AUTOMATION_PASS, QA_AUTOMATION_PASS_WITH_EXPECTED_BLOCKERS, QA_AUTOMATION_FAIL]",
            "expected_blockers: [no_observed_geometry, no_occurrence_coordinates, unlinkable_or_missing_sentinel_date, no_explicit_anchor_crosswalk, no_ground_reference]",
        ],
        "v2af_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def _row_count(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return sum(1 for _ in reader)


# 1. QA Input Manifest Builder --------------------------------------------

def run_qa_input_manifest_builder(args=None):
    write_policy_configs()
    rows = []
    for rel, version, required, expected in INPUT_ARTIFACTS:
        real = _resolve_input(rel)
        exists = os.path.exists(real)
        actual = _row_count(real) if exists else None
        sha = sha256_file(real)[:16] if exists else ""
        if not exists:
            existence = "MISSING" if required else "OPTIONAL_MISSING"
            freshness = "MISSING_ARTIFACT" if required else "OPTIONAL_MISSING"
        elif actual == 0:
            existence = "PRESENT"
            freshness = "EMPTY_ARTIFACT"
        else:
            existence = "PRESENT"
            freshness = "FRESH_ENOUGH_FOR_QA"
        rows.append({
            "input_id": f"IN_v2af_{len(rows):04d}",
            "artifact_path": rel,
            "source_version": version,
            "artifact_type": os.path.splitext(rel)[1].lstrip(".") or "text",
            "required": "true" if required else "false",
            "expected_row_count": str(expected) if expected is not None else "",
            "actual_row_count": str(actual) if actual is not None else "",
            "sha256": sha,
            "existence_status": existence,
            "freshness_status": freshness,
            "notes": "Relative repo path only; local_only never referenced.",
        })
    out = dataset_path("v2af_qa_input_manifest.csv")
    write_csv(out, MANIFEST_INPUT_COLUMNS, rows)
    missing = sum(1 for r in rows if r["existence_status"] == "MISSING")
    print(f"[v2af input manifest] inputs={len(rows)} missing_required={missing} -> {out}")
    return rows


# 2. Artifact Freshness Auditor -------------------------------------------

def _min_schema_ok(rel, header):
    """Lightweight per-artifact minimal-schema check."""
    base = os.path.basename(rel)
    needed = {
        "v2ae_canonical_region_registry.csv": {"region", "canonical_region_status"},
        "v2ae_canonical_event_registry.csv": {"event_id", "canonical_event_status"},
        "v2ae_canonical_event_patch_registry.csv": {"event_patch_candidate_id", "patch_namespace"},
        "v2ad_qa_gate_summary.csv": {"qa_group", "gate_status"},
        "v2ac_event_patch_v2_package_registry.csv": {"event_patch_candidate_id", "crosswalk_status"},
    }.get(base)
    if not needed:
        return True
    return needed <= set(header)


def run_artifact_freshness_auditor(args=None):
    rows = []
    for rel, version, required, _ in INPUT_ARTIFACTS:
        real = _resolve_input(rel)
        exists = os.path.exists(real)
        non_empty = header_ok = schema_ok = hash_ok = False
        header = []
        if exists:
            try:
                with open(real, "r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header = next(reader, [])
                    header_ok = bool(header)
                    non_empty = any(True for _ in reader)
                schema_ok = _min_schema_ok(rel, header)
                sha256_file(real)
                hash_ok = True
            except (OSError, csv.Error, StopIteration):
                header_ok = False
        if not exists:
            status = "OPTIONAL_MISSING" if not required else "MISSING_ARTIFACT"
        elif not header_ok:
            status = "SCHEMA_UNREADABLE"
        elif not non_empty:
            status = "EMPTY_ARTIFACT"
        elif not schema_ok:
            status = "SCHEMA_UNREADABLE"
        else:
            status = "FRESH_ENOUGH_FOR_QA"
        rows.append({
            "freshness_id": f"FR_v2af_{len(rows):04d}",
            "artifact_path": rel,
            "source_version": version,
            "required": "true" if required else "false",
            "exists": "true" if exists else "false",
            "non_empty": "true" if non_empty else "false",
            "header_readable": "true" if header_ok else "false",
            "schema_minimal_present": "true" if schema_ok else "false",
            "hash_computable": "true" if hash_ok else "false",
            "freshness_status": status,
            "notes": "Freshness is a technical QA check; modified time is never used as scientific evidence.",
        })
    out = dataset_path("v2af_artifact_freshness_audit.csv")
    write_csv(out, FRESHNESS_COLUMNS, rows)
    bad = sum(1 for r in rows if r["freshness_status"] in {"MISSING_ARTIFACT", "EMPTY_ARTIFACT", "SCHEMA_UNREADABLE"})
    print(f"[v2af freshness] artifacts={len(rows)} problems={bad} -> {out}")
    return rows


# 3. Expected Count Validator ---------------------------------------------

def _count_distinct(path, column):
    return len({r.get(column) for r in load_csv(path) if r.get(column)})


def run_expected_count_validator(args=None):
    rows = []

    def check(name, expected, actual, severity="critical"):
        rows.append({
            "count_check_id": f"CNT_v2af_{len(rows):04d}",
            "check_name": name,
            "expected_count": str(expected),
            "actual_count": str(actual),
            "status": "PASS" if expected == actual else "FAIL",
            "severity": severity,
            "notes": "Loss of a region/event/package fails the gate." if expected != actual else "",
        })

    regions = _count_distinct(dataset_path("v2ae_canonical_region_registry.csv"), "region")
    events = _count_distinct(dataset_path("v2ae_canonical_event_registry.csv"), "event_id")
    canon_pkgs = _row_count(dataset_path("v2ae_canonical_event_patch_registry.csv")) or 0
    v2ac_pkgs = _row_count(dataset_path("v2ac_event_patch_v2_package_registry.csv")) or 0
    readiness = _row_count(dataset_path("v2ac_v2_readiness_matrix.csv")) or 0
    gate_exists = 1 if load_csv(dataset_path("v2ad_qa_gate_summary.csv")) else 0
    consistency_exists = 1 if load_csv(dataset_path("v2ae_registry_consistency_qa.csv")) else 0

    check("canonical_regions", 3, regions)
    check("canonical_events", 4, events)
    check("canonical_event_patch_packages", EXPECTED_PACKAGES, canon_pkgs)
    check("v2ac_packages", EXPECTED_PACKAGES, v2ac_pkgs)
    check("v2_readiness_rows", EXPECTED_READINESS_V2, readiness, "high")
    check("v2ad_gate_exists", 1, gate_exists)
    check("v2ae_consistency_qa_exists", 1, consistency_exists)

    out = dataset_path("v2af_expected_count_validation.csv")
    write_csv(out, COUNT_COLUMNS, rows)
    print(f"[v2af expected count] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 4. Guardrail Regression Runner ------------------------------------------

def _sweep_files():
    out = []
    if not os.path.isdir(DATASET_DIR):
        return out
    for name in sorted(os.listdir(DATASET_DIR)):
        if not name.endswith(".csv"):
            continue
        if any(_glob_match(name, g) for g in GUARDRAIL_SWEEP_GLOBS):
            out.append(os.path.join(DATASET_DIR, name))
    return out


def _glob_match(name, glob):
    prefix = glob.split("*")[0]
    return name.startswith(prefix) and name.endswith(".csv")


def _scan_guardrails(path):
    counts = {
        "forbidden_true_value": 0, "forbidden_status": 0, "absolute_path": 0,
        "local_only_leak": 0, "tool_name_leak": 0, "overlay_or_gr_or_training_released": 0,
    }
    for rec in load_csv(path):
        for key, value in rec.items():
            value = value or ""
            if key in GUARDRAIL_MUST_BE_FALSE and value.strip().lower() == "true":
                counts["forbidden_true_value"] += 1
            if FORBIDDEN_STATUS_RE.search(value):
                counts["forbidden_status"] += 1
            if "local_only/" in value or "local_only\\" in value:
                counts["local_only_leak"] += 1
            if ABSOLUTE_PATH_RE.search(value):
                counts["absolute_path"] += 1
            if TOOL_NAME_RE.search(value):
                counts["tool_name_leak"] += 1
        for field in ("overlay_status", "ground_reference_status", "training_label_status"):
            if field in rec and rec.get(field) not in ("BLOCKED", "", None):
                counts["overlay_or_gr_or_training_released"] += 1
    return counts


def run_guardrail_regression_runner(args=None, files=None):
    files = files if files is not None else _sweep_files()
    rows = []
    for path in files:
        counts = _scan_guardrails(path)
        for check_type, n in counts.items():
            status = "PASS" if n == 0 else "FAIL"
            rows.append({
                "guardrail_regression_id": f"GR_v2af_{len(rows):05d}",
                "artifact": os.path.basename(path),
                "check_type": check_type,
                "violation_count": str(n),
                "status": status,
                "severity": "info" if status == "PASS" else "critical",
                "notes": "Clean" if status == "PASS" else "Guardrail regression failure.",
            })
    out = dataset_path("v2af_guardrail_regression.csv")
    write_csv(out, GUARDRAIL_REG_COLUMNS, rows)
    print(f"[v2af guardrail regression] rows={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 5. Canonical Registry Regression Runner ---------------------------------

EXPECTED_REGION_STATUS = {
    "REC": "REGION_HARDENED_CONTEXTUAL_COORDINATE_NON_OPERATIONAL",
    "PET": "REGION_HARDENED_DOCUMENT_ONLY_NO_GEODATA",
    "CUR": "REGION_HARDENED_CONTEXT_ONLY_HOLD",
}


def _reg(rows, group, name, expected, observed, status, severity="critical", notes=""):
    rows.append({
        "regression_id": f"{group}_{len(rows):04d}", "check_group": group,
        "check_name": name, "expected": expected, "observed": observed,
        "status": status, "severity": severity, "notes": notes,
    })


def run_canonical_registry_regression_runner(args=None):
    regions = load_csv(dataset_path("v2ae_canonical_region_registry.csv"))
    events = load_csv(dataset_path("v2ae_canonical_event_registry.csv"))
    packages = load_csv(dataset_path("v2ae_canonical_event_patch_registry.csv"))
    safe = load_csv(dataset_path("v2ae_safe_use_policy_registry.csv"))
    reopen = load_csv(dataset_path("v2ae_region_reopen_condition_registry.csv"))
    rows = []
    region_set = {r.get("region") for r in regions}
    _reg(rows, "registry", "three_regions", "REC|PET|CUR", "|".join(sorted(region_set)),
         "PASS" if set(EXPECTED_REGIONS) <= region_set else "FAIL")
    event_set = {e.get("event_id") for e in events}
    _reg(rows, "registry", "four_events", "|".join(EXPECTED_EVENTS),
         "missing:" + "|".join(e for e in EXPECTED_EVENTS if e not in event_set) if not set(EXPECTED_EVENTS) <= event_set else "all_present",
         "PASS" if set(EXPECTED_EVENTS) <= event_set else "FAIL")
    _reg(rows, "registry", "packages_172", str(EXPECTED_PACKAGES), str(len(packages)),
         "PASS" if len(packages) == EXPECTED_PACKAGES else "FAIL")
    # Region statuses unchanged.
    status_by_region = {r.get("region"): r.get("canonical_region_status") for r in regions}
    for region, expected_status in EXPECTED_REGION_STATUS.items():
        observed = status_by_region.get(region, "MISSING")
        _reg(rows, "registry", f"region_status_{region}", expected_status, observed,
             "PASS" if observed == expected_status else "FAIL")
    # QA gate preserved.
    gate = _qa_gate_status()
    _reg(rows, "registry", "qa_gate_preserved", "QA_PASS_WITH_EXPECTED_BLOCKERS", gate,
         "PASS" if gate.startswith("QA_PASS") else "FAIL", "high")
    _reg(rows, "registry", "safe_use_policy_present", "non_empty", str(len(safe)),
         "PASS" if safe else "FAIL")
    _reg(rows, "registry", "reopen_conditions_present", "3", str(len(reopen)),
         "PASS" if len(reopen) == 3 else "FAIL")
    # Overlay/gr/training blocked across canonical packages.
    released = sum(1 for p in packages for f in ("overlay_status", "ground_reference_status", "training_label_status") if p.get(f) != "BLOCKED")
    _reg(rows, "registry", "overlay_gr_training_blocked", "0_released", str(released),
         "PASS" if released == 0 else "FAIL")
    out = dataset_path("v2af_canonical_registry_regression.csv")
    write_csv(out, REGRESSION_COLUMNS, rows)
    print(f"[v2af registry regression] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


def _qa_gate_status():
    gate = load_csv(dataset_path("v2ad_qa_gate_summary.csv"))
    overall = next((r for r in gate if r.get("qa_group") == "OVERALL"), {})
    return overall.get("gate_status", "QA_GATE_UNKNOWN")


# 6. Event-Patch V2 Regression Runner -------------------------------------

def run_event_patch_v2_regression_runner(args=None):
    packages = load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))
    rows = []
    total = len(packages)
    valid = sum(1 for p in packages if p.get("package_validation_status") == "PACKAGE_V2_SCHEMA_VALID_WITH_TEMPORAL_BLOCKER")
    missing_patch = sum(1 for p in packages if not (p.get("patch_id") or "").strip())
    dino_xw = sum(1 for p in packages if p.get("crosswalk_status", "").startswith("EXPLICIT_DINO"))
    anchor_xw = sum(1 for p in packages if (p.get("anchor_patch_id") or "").strip() or (p.get("refpatch_id") or "").strip())
    unlinkable = sum(1 for p in packages if p.get("sentinel_date_status") == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE")
    missing_date = sum(1 for p in packages if p.get("sentinel_date_status") == "SENTINEL_DATE_MISSING_WITH_BLOCKER")
    applied_unlinkable = sum(1 for p in packages if p.get("date_linkability_status") == "UNLINKABLE_NAMESPACE" and (p.get("sentinel_scene_date") or "").strip())
    released = sum(1 for p in packages for f in ("overlay_status", "ground_reference_status", "training_label_status") if p.get(f) != "BLOCKED")

    _reg(rows, "event_patch", "packages_172", str(EXPECTED_PACKAGES), str(total),
         "PASS" if total == EXPECTED_PACKAGES else "FAIL")
    _reg(rows, "event_patch", "valid_non_operational", "171", str(valid),
         "PASS" if valid == 171 else "FAIL", "high")
    _reg(rows, "event_patch", "missing_patch_expected_blocker", "1", str(missing_patch),
         "PASS" if missing_patch == 1 else "FAIL", "expected" if missing_patch == 1 else "high")
    _reg(rows, "event_patch", "dino_crosswalk_explicit", "171", str(dino_xw),
         "PASS" if dino_xw == 171 else "FAIL", "high")
    _reg(rows, "event_patch", "no_anchor_crosswalk", "0", str(anchor_xw),
         "PASS" if anchor_xw == 0 else "FAIL")
    _reg(rows, "event_patch", "unlinkable_date_count", "171", str(unlinkable),
         "PASS" if unlinkable == 171 else "FAIL", "high")
    _reg(rows, "event_patch", "missing_date_count", "1", str(missing_date),
         "PASS" if missing_date == 1 else "FAIL", "high")
    _reg(rows, "event_patch", "no_unlinkable_date_applied", "0", str(applied_unlinkable),
         "PASS" if applied_unlinkable == 0 else "FAIL")
    _reg(rows, "event_patch", "overlay_gr_training_blocked", "0_released", str(released),
         "PASS" if released == 0 else "FAIL")
    out = dataset_path("v2af_event_patch_v2_regression.csv")
    write_csv(out, REGRESSION_COLUMNS, rows)
    print(f"[v2af event-patch regression] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 7. QA Gate Orchestrator -------------------------------------------------

GATE_COMPONENTS = [
    ("input_manifest", "v2af_qa_input_manifest.csv", "existence"),
    ("artifact_freshness", "v2af_artifact_freshness_audit.csv", "freshness"),
    ("expected_counts", "v2af_expected_count_validation.csv", "status"),
    ("guardrail_regression", "v2af_guardrail_regression.csv", "status"),
    ("canonical_registry_regression", "v2af_canonical_registry_regression.csv", "status"),
    ("event_patch_v2_regression", "v2af_event_patch_v2_regression.csv", "status"),
]


def run_qa_gate_orchestrator(args=None):
    rows = []
    overall_fail = False
    overall_expected = False
    for component, filename, mode in GATE_COMPONENTS:
        data = load_csv(dataset_path(filename))
        total = len(data)
        failed = expected = 0
        if mode == "existence":
            failed = sum(1 for r in data if r.get("existence_status") == "MISSING")
            passed = total - failed
        elif mode == "freshness":
            failed = sum(1 for r in data if r.get("freshness_status") in {"MISSING_ARTIFACT", "EMPTY_ARTIFACT", "SCHEMA_UNREADABLE"})
            passed = total - failed
        else:
            failed = sum(1 for r in data if r.get("status") == "FAIL")
            expected = sum(1 for r in data if r.get("status") == "EXPECTED_BLOCKER" or r.get("severity") == "expected")
            passed = total - failed - expected
        if failed:
            gate = "QA_AUTOMATION_FAIL"
            overall_fail = True
            action = "fix_failures_before_next_stage"
        elif expected:
            gate = "QA_AUTOMATION_PASS_WITH_EXPECTED_BLOCKERS"
            overall_expected = True
            action = "none_expected_blockers_documented"
        else:
            gate = "QA_AUTOMATION_PASS"
            action = "none"
        rows.append({
            "gate_id": f"GATE_v2af_{len(rows):04d}", "qa_component": component,
            "total_checks": str(total), "passed_checks": str(passed),
            "failed_checks": str(failed), "expected_blockers": str(expected),
            "gate_status": gate, "required_action": action, "notes": "Per-component QA gate.",
        })
    overall = "QA_AUTOMATION_FAIL" if overall_fail else ("QA_AUTOMATION_PASS_WITH_EXPECTED_BLOCKERS" if overall_expected else "QA_AUTOMATION_PASS")
    rows.append({
        "gate_id": "GATE_v2af_OVERALL", "qa_component": "OVERALL",
        "total_checks": str(sum(int(r["total_checks"]) for r in rows)),
        "passed_checks": str(sum(int(r["passed_checks"]) for r in rows)),
        "failed_checks": str(sum(int(r["failed_checks"]) for r in rows)),
        "expected_blockers": str(sum(int(r["expected_blockers"]) for r in rows)),
        "gate_status": overall,
        "required_action": "next_stage_may_start" if overall != "QA_AUTOMATION_FAIL" else "next_stage_blocked",
        "notes": "Expected blockers: no observed geometry, no occurrence coordinates, unlinkable/missing Sentinel date, no explicit anchor crosswalk, no ground reference.",
    })
    out = dataset_path("v2af_qa_gate_orchestration.csv")
    write_csv(out, GATE_COLUMNS, rows)
    print(f"[v2af gate] overall={overall} -> {out}")
    return rows


def _overall_gate():
    rows = load_csv(dataset_path("v2af_qa_gate_orchestration.csv"))
    overall = next((r for r in rows if r.get("qa_component") == "OVERALL"), {})
    return overall.get("gate_status", "")


# 8. Failure Report Builder -----------------------------------------------

FAILURE_SOURCES = [
    ("v2af_expected_count_validation.csv", "check_name", "status"),
    ("v2af_guardrail_regression.csv", "check_type", "status"),
    ("v2af_canonical_registry_regression.csv", "check_name", "status"),
    ("v2af_event_patch_v2_regression.csv", "check_name", "status"),
]
FIX_HINTS = {
    "guardrail_regression": "Remove forbidden values/statuses or restore BLOCKED guards before versioning.",
    "expected_count_validation": "Investigate the lost/extra region/event/package and restore the canonical count.",
    "canonical_registry_regression": "Restore the canonical region/event status from the hardened v2ae registry.",
    "event_patch_v2_regression": "Restore the v2ac package invariants (counts, crosswalk, unlinkable date).",
}


def run_failure_report_builder(args=None):
    rows = []
    for filename, name_col, status_col in FAILURE_SOURCES:
        for rec in load_csv(dataset_path(filename)):
            if rec.get(status_col) == "FAIL":
                group = filename.replace("v2af_", "").replace(".csv", "")
                rows.append({
                    "failure_id": f"FAIL_v2af_{len(rows):04d}",
                    "artifact": rec.get("artifact", filename),
                    "check": rec.get(name_col, ""),
                    "severity": rec.get("severity", "critical"),
                    "recommended_fix": FIX_HINTS.get(group, "Investigate the failing check."),
                    "blocking_status": "BLOCKING",
                    "notes": "Detected by the v2af QA automation.",
                })
    if not rows:
        rows.append({
            "failure_id": "FAIL_v2af_0000",
            "artifact": "NONE",
            "check": "NO_FAILURES_DETECTED",
            "severity": "info",
            "recommended_fix": "none",
            "blocking_status": "NON_BLOCKING",
            "notes": "All QA automation components passed (expected blockers remain).",
        })
    out = dataset_path("v2af_failure_report.csv")
    write_csv(out, FAILURE_COLUMNS, rows)
    print(f"[v2af failure report] rows={len(rows)} -> {out}")
    return rows


# 9. Next Programming Target Ranker ----------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}

TARGET_VERSION = {
    "SENTINEL_DATE_CROSSWALK_DISCOVERY": "v2ag — Sentinel Date Crosswalk Discovery",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2ag — DINO Review Support Completion",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2ag — Public Source Recheck Hold",
    "MULTI_REGION_REGISTRY_MAINTENANCE": "v2ag — Multi-Region Registry Maintenance",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2ag — Stop Ground Truth Search Until New Source",
}


def _ranker_metrics():
    gate = _overall_gate()
    qa_pass = gate.startswith("QA_AUTOMATION_PASS")
    packages = load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))
    total = len(packages) or 1
    no_anchor = sum(1 for p in packages if "NO_ANCHOR_CROSSWALK" in p.get("crosswalk_status", "") or p.get("crosswalk_status") in {"NO_EXPLICIT_CROSSWALK", "NO_CROSSWALK_PATCH_ID_MISSING"})
    return {"qa_pass": qa_pass, "no_anchor_rate": no_anchor / total}


def _candidate_targets(m):
    # Ground-truth search is exhausted across all three regions; the dominant
    # residual blocker is the absence of an explicit numeric<->anchor crosswalk.
    # Crosswalk discovery is uncertain (medium overclaim risk); a documented stop
    # is low value. Score decides.
    return [
        {
            "next_target": "SENTINEL_DATE_CROSSWALK_DISCOVERY",
            "programming_value": round(45 * m["no_anchor_rate"]),
            "ground_truth_value": 0,
            "blocker_reduction_value": round(55 * m["no_anchor_rate"]),
            "expected_effort": "MEDIUM",
            "overclaim_risk": "MEDIUM",
            "notes": "Attempt to discover an explicit key linking the numeric and anchor namespaces; the dominant residual blocker, but uncertain and only useful if a real key exists.",
        },
        {
            "next_target": "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE",
            "programming_value": 35 if m["qa_pass"] else 10,
            "ground_truth_value": 0,
            "blocker_reduction_value": 30,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "QA automation is green and ground-truth search is exhausted in all three regions; formally stop until a qualifying new public source appears.",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_MAINTENANCE",
            "programming_value": 40,
            "ground_truth_value": 0,
            "blocker_reduction_value": 25,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Maintain the hardened canonical registries and the automated QA gate.",
        },
        {
            "next_target": "DINO_REVIEW_SUPPORT_COMPLETION",
            "programming_value": 10,
            "ground_truth_value": 0,
            "blocker_reduction_value": 10,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "DINO review support already attached for nearly all candidates.",
        },
        {
            "next_target": "PUBLIC_SOURCE_RECHECK_HOLD",
            "programming_value": 20,
            "ground_truth_value": 0,
            "blocker_reduction_value": 20,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Periodic recheck hold for new public occurrence-level sources.",
        },
    ]


def _score(t):
    base = 0.5 * t["programming_value"] + 0.5 * t["blocker_reduction_value"]
    return base - EFFORT_PENALTY.get(t["expected_effort"], 5) - OVERCLAIM_PENALTY.get(t["overclaim_risk"], 10)


def run_next_programming_target_ranker(args=None):
    metrics = _ranker_metrics()
    targets = _candidate_targets(metrics)
    targets.sort(key=_score, reverse=True)
    rows = []
    for idx, t in enumerate(targets, start=1):
        rows.append({
            "rank": str(idx),
            "next_target": t["next_target"],
            "programming_value": str(t["programming_value"]),
            "ground_truth_value": str(t["ground_truth_value"]),
            "blocker_reduction_value": str(t["blocker_reduction_value"]),
            "expected_effort": t["expected_effort"],
            "overclaim_risk": t["overclaim_risk"],
            "recommended_version": TARGET_VERSION.get(t["next_target"], ""),
            "recommended_action": "SELECTED_NEXT_TARGET" if idx == 1 else "RANKED_ALTERNATIVE",
            "notes": t["notes"],
        })
    out = dataset_path("v2af_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v2af ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
    return rows


# 10. Completion Report ----------------------------------------------------

def run_ground_reference_blocker_matrix(args=None):
    region_event = {"REC": "REC_2022_05_24_30", "PET": "PET_2022_02_15", "CUR": "CUR_2022_01_15"}
    blockers = [
        "no_observed_geometry", "no_occurrence_coordinates", "unlinkable_sentinel_date",
        "no_explicit_anchor_crosswalk", "no_overlay", "no_ground_reference",
        "no_training_label", "patch_truth_forbidden",
    ]
    rows = []
    for region, event_id in region_event.items():
        for blocker in blockers:
            rows.append({
                "blocker_id": f"GB_v2af_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "blocker": blocker,
                "status": "BLOCKED",
                **guardrails(),
                "notes": "QA automation does not unblock geometry, overlay, ground reference or labels.",
            })
    out = dataset_path("v2af_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_MATRIX_COLUMNS, rows)
    print(f"[v2af gr blockers] rows={len(rows)} -> {out}")
    return rows


def _fails(rows, col="status"):
    return sum(1 for r in rows if r.get(col) == "FAIL")


def run_completion_report(args=None):
    write_policy_configs()
    manifest = load_csv(dataset_path("v2af_qa_input_manifest.csv")) or run_qa_input_manifest_builder(args)
    freshness = load_csv(dataset_path("v2af_artifact_freshness_audit.csv")) or run_artifact_freshness_auditor(args)
    counts = load_csv(dataset_path("v2af_expected_count_validation.csv")) or run_expected_count_validator(args)
    guardrail = load_csv(dataset_path("v2af_guardrail_regression.csv")) or run_guardrail_regression_runner(args)
    registry = load_csv(dataset_path("v2af_canonical_registry_regression.csv")) or run_canonical_registry_regression_runner(args)
    eventpatch = load_csv(dataset_path("v2af_event_patch_v2_regression.csv")) or run_event_patch_v2_regression_runner(args)
    gate = load_csv(dataset_path("v2af_qa_gate_orchestration.csv")) or run_qa_gate_orchestrator(args)
    failures = load_csv(dataset_path("v2af_failure_report.csv")) or run_failure_report_builder(args)
    ranker = load_csv(dataset_path("v2af_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    blockers = run_ground_reference_blocker_matrix(args)

    overall = next((r for r in gate if r.get("qa_component") == "OVERALL"), {})
    missing_artifacts = sum(1 for r in manifest if r.get("existence_status") == "MISSING")
    no_failures = len(failures) == 1 and failures[0].get("check") == "NO_FAILURES_DETECTED"
    next_target = ranker[0].get("next_target", "") if ranker else ""
    next_version = ranker[0].get("recommended_version", "") if ranker else ""

    write_csv(dataset_path("v2af_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v2af_0000",
        "event_id": "MULTI_REGION",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v2af score-based next-programming-target ranker after QA automation.",
        "target": "EVENT_PATCH_PACKAGE_V2",
        "status": "RECOMMENDED_NEXT_STEP" if overall.get("gate_status") != "QA_AUTOMATION_FAIL" else "BLOCKED_BY_QA_FAILURE",
        "notes": "No overlay, labels, ground truth, ground reference, inferred date or inferred crosswalk.",
    }])

    lines = [
        "# Protocolo C v2af - Event-Patch Package V2 QA Automation",
        "",
        f"- input artifacts validated: `{len(manifest)}` (missing required: `{missing_artifacts}`)",
        f"- freshness problems: `{sum(1 for r in freshness if r.get('freshness_status') in {'MISSING_ARTIFACT', 'EMPTY_ARTIFACT', 'SCHEMA_UNREADABLE'})}`",
        f"- expected count checks: `{len(counts)}` (fails: `{_fails(counts)}`)",
        f"- guardrail regression checks: `{len(guardrail)}` (fails: `{_fails(guardrail)}`)",
        f"- canonical registry regression checks: `{len(registry)}` (fails: `{_fails(registry)}`)",
        f"- event-patch v2 regression checks: `{len(eventpatch)}` (fails: `{_fails(eventpatch)}`)",
        f"- overall QA automation gate: `{overall.get('gate_status', '')}`",
        f"- failures detected: `{'none' if no_failures else len(failures)}`",
        f"- selected next target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v2af is a read-only QA automation orchestrator. It modified no prior output, sought no new source, inferred no date/crosswalk/coordinate, executed no overlay, and created no ground truth, ground reference or label.",
    ]
    write_text(doc_path("protocolo_c_v2af_event_patch_v2_qa_automation.md"), lines)

    report = lines + [
        "",
        "## Input artifacts and freshness",
        f"{len(manifest)} input artifacts were validated; {missing_artifacts} required artifacts were missing. Freshness audit uses technical checks only (existence, non-empty, readable header, minimal schema, computable hash) and never uses modified time as scientific evidence.",
        "",
        "## Expected counts",
        f"Expected-count validation: {_fails(counts)} failures. The canonical 3 regions, 4 events, 172 packages and 2580 v2 readiness rows are verified, and a loss fails the gate.",
        "",
        "## Guardrail regression",
        f"{_fails(guardrail)} guardrail failures across the swept v2ac/v2ad/v2ae/v2af outputs (forbidden true values/statuses, absolute paths, local_only leaks, tool-name leaks, overlay/ground reference/training release).",
        "",
        "## Canonical registry regression",
        f"{_fails(registry)} failures: regions, events, package count, region statuses, QA gate, safe-use policy and reopen conditions are preserved unchanged.",
        "",
        "## Event-patch v2 regression",
        f"{_fails(eventpatch)} failures: 172 packages, 171 valid non-operational, 1 expected missing-patch blocker, 171 explicit DINO crosswalks, 0 anchor crosswalk, 171 unlinkable dates, 1 missing date, no unlinkable date applied, overlay/ground reference/training blocked.",
        "",
        "## QA automation gate and failures",
        f"Overall gate: `{overall.get('gate_status', '')}`. " + ("No failures detected." if no_failures else f"{len(failures)} failures recorded in the failure report."),
        "",
        "## Expected blockers",
        "no observed geometry, no occurrence coordinates, unlinkable/missing Sentinel date, no explicit anchor crosswalk, no ground reference.",
        "",
        "## Why there is still no overlay",
        "No overlay was executed and overlay readiness stays BLOCKED; automation verifies state but establishes no observed occurrence geometry.",
        "",
        "## Why there is still no ground reference",
        "No region or package has a linkable observed occurrence geometry; without it there is no basis for ground reference, and none was created.",
        "",
        "## Why there is still no label",
        "Labels require observed occurrence truth, which does not exist; creating one would be an unsupported overclaim.",
        "",
        "## Next programming step",
        f"The score-based ranker selected `{next_target}` (`{next_version}`).",
    ]
    write_text(doc_path("protocolo_c_relatorio_v2af_event_patch_v2_qa_automation.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v2af.md"), [
        "# Status atual - Protocolo C v2af",
        "",
        f"QA automation status: `{MAX_STATUS}`.",
        f"Overall QA automation gate: `{overall.get('gate_status', '')}`.",
        f"Failures detected: `{'none' if no_failures else len(failures)}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Overlay, ground reference, training labels, ground truth, inferred Sentinel dates and inferred crosswalks remain blocked.",
    ])

    manifest_rows = []
    for idx, artifact in enumerate(V2AF_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest_rows.append({
            "artifact_id": f"MAN_v2af_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v2af QA automation artifact; no raw data, no private path, no inferred date or crosswalk.",
        })
    write_csv(dataset_path("v2af_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest_rows)
    for folder in (STAGING_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v2af completion] gate={overall.get('gate_status', '')} failures={'none' if no_failures else len(failures)} next={next_target}")
    return {"gate": overall.get("gate_status", ""), "no_failures": no_failures, "missing_artifacts": missing_artifacts, "next_target": next_target, "next_version": next_version}


def run_all(args=None):
    args = args or parse_args([])
    run_qa_input_manifest_builder(args)
    run_artifact_freshness_auditor(args)
    run_expected_count_validator(args)
    run_guardrail_regression_runner(args)
    run_canonical_registry_regression_runner(args)
    run_event_patch_v2_regression_runner(args)
    run_qa_gate_orchestrator(args)
    run_failure_report_builder(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
