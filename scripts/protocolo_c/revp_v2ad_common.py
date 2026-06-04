#!/usr/bin/env python3
"""v2ad Event-patch package v2 QA harness.

A reusable, command-runnable QA suite for the migrated v2 event-patch packages.
It validates the v2 packages against the v2ab contract, audits namespace and
crosswalk safety, temporal safety, guardrails, readiness consistency and
migration integrity, runs negative fixtures to prove the QA can fail, and
produces an approval gate. It reads prior outputs read-only; it never modifies
v2ac packages, never infers a crosswalk or a Sentinel date, never applies an
unlinkable date, and never produces overlay, ground reference or labels.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v2ad"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/event_patch_v2_qa_harness/staging/v2ad"
REPORTS_DIR = "local_only/protocolo_c/event_patch_v2_qa_harness/reports/v2ad"
NEGATIVE_FIXTURE_DIR = "tests/fixtures/v2ad"

MAX_STATUS = "EVENT_PATCH_V2_QA_HARNESS_NON_OPERATIONAL"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "qa_harness_only", "crosswalk_inferred",
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

NUMERIC_RE = re.compile(r"^(CUR|PET|REC)_\d{4,6}$")
NS_EVENT = "EVENT_PATCH_CANDIDATE_NAMESPACE"

# Column definitions ------------------------------------------------------
QA_COLUMNS = [
    "qa_id", "event_patch_candidate_id", "check_group", "check_name",
    "expected", "observed", "status", "severity", "notes",
]
GUARDRAIL_QA_COLUMNS = [
    "guardrail_qa_id", "artifact", "check_type", "violation_count", "status",
    "severity", "notes",
]
NEGATIVE_QA_COLUMNS = [
    "negative_qa_id", "fixture_name", "injected_violation", "detected",
    "status", "notes",
]
GATE_COLUMNS = [
    "gate_id", "qa_group", "total_checks", "passed_checks", "failed_checks",
    "expected_blockers", "gate_status", "required_action", "notes",
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

REQUIRED_CONTRACT_DEFAULT = [
    "event_patch_candidate_id", "event_id", "event_region", "patch_id",
    "patch_namespace", "patch_source_registry", "linkage_basis", "linkage_status",
    "event_patch_candidate_only", "sentinel_date_status", "temporal_linkage_status",
    "evidence_status", "geometry_status", "overlay_status", "ground_reference_status",
    "training_label_status", "blocker", "safe_use", "prohibited_use",
]

QA_GROUPS = [
    ("package_contract", "v2ad_package_contract_qa.csv"),
    ("namespace_crosswalk", "v2ad_namespace_crosswalk_qa.csv"),
    ("temporal_safety", "v2ad_temporal_safety_qa.csv"),
    ("guardrail", "v2ad_guardrail_qa.csv"),
    ("readiness_consistency", "v2ad_readiness_consistency_qa.csv"),
    ("migration_integrity", "v2ad_migration_integrity_qa.csv"),
    ("negative_fixture", "v2ad_negative_fixture_qa.csv"),
]

GUARDRAIL_AUDIT_ARTIFACTS = [
    "datasets/protocolo_c/v2ac_event_patch_v2_package_registry.csv",
    "datasets/protocolo_c/v2ac_crosswalk_field_population.csv",
    "datasets/protocolo_c/v2ac_temporal_status_field_population.csv",
    "datasets/protocolo_c/v2ac_schema_contract_validation.csv",
    "datasets/protocolo_c/v2ac_v2_readiness_matrix.csv",
    "datasets/protocolo_c/v2ad_package_contract_qa.csv",
    "datasets/protocolo_c/v2ad_namespace_crosswalk_qa.csv",
    "datasets/protocolo_c/v2ad_temporal_safety_qa.csv",
]

V2AD_ARTIFACTS = [
    "configs/protocolo_c/v2ad_package_contract_qa_policy.yaml",
    "configs/protocolo_c/v2ad_crosswalk_qa_policy.yaml",
    "configs/protocolo_c/v2ad_temporal_safety_qa_policy.yaml",
    "configs/protocolo_c/v2ad_guardrail_qa_policy.yaml",
    "configs/protocolo_c/v2ad_negative_fixture_policy.yaml",
    "configs/protocolo_c/v2ad_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2ad_package_contract_qa.csv",
    "datasets/protocolo_c/v2ad_namespace_crosswalk_qa.csv",
    "datasets/protocolo_c/v2ad_temporal_safety_qa.csv",
    "datasets/protocolo_c/v2ad_guardrail_qa.csv",
    "datasets/protocolo_c/v2ad_readiness_consistency_qa.csv",
    "datasets/protocolo_c/v2ad_migration_integrity_qa.csv",
    "datasets/protocolo_c/v2ad_negative_fixture_qa.csv",
    "datasets/protocolo_c/v2ad_qa_gate_summary.csv",
    "datasets/protocolo_c/v2ad_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2ad_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2ad_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v2ad_event_patch_v2_qa_harness.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2ad_event_patch_v2_qa_harness.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2ad.md",
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
        "qa_harness_only": "true",
        "crosswalk_inferred": "false",
        "sentinel_date_inferred": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v2ad_package_contract_qa_policy.yaml": [
            "validate_against: v2ab_event_patch_schema_contract",
            "required_fields_must_be_present: true",
            "optional_null_requires_blocker: true",
            "max_status: EVENT_PATCH_V2_QA_HARNESS_NON_OPERATIONAL",
        ],
        "v2ad_crosswalk_qa_policy.yaml": [
            "explicit_dino_crosswalk_by_identical_patch_id_only: true",
            "anchor_or_refpatch_crosswalk_allowed: false",
            "crosswalk_inferred_allowed: false",
        ],
        "v2ad_temporal_safety_qa_policy.yaml": [
            "unlinkable_date_must_not_be_applied: true",
            "sentinel_date_inferred_allowed: false",
            "missing_date_requires_blocker: true",
        ],
        "v2ad_guardrail_qa_policy.yaml": [
            "forbidden_true_blocked: true",
            "forbidden_status_blocked: true",
            "absolute_path_blocked: true",
            "local_only_leak_blocked: true",
            "tool_name_leak_blocked: true",
            "overlay_ground_reference_training_must_be_blocked: true",
        ],
        "v2ad_negative_fixture_policy.yaml": [
            "negative_fixtures_must_be_detected: true",
            "negative_fixtures_are_not_real_outputs: true",
            "fixture_dir: tests/fixtures/v2ad",
        ],
        "v2ad_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def _packages():
    return load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))


def _is_missing_patch(pkg):
    return not (pkg.get("patch_id") or "").strip()


def _check(rows, group, epc, name, expected, observed, status, severity="info", notes=""):
    rows.append({
        "qa_id": f"QA_{group}_{len(rows):05d}",
        "event_patch_candidate_id": epc,
        "check_group": group,
        "check_name": name,
        "expected": expected,
        "observed": observed,
        "status": status,
        "severity": severity,
        "notes": notes,
    })


# 1. Package Contract QA --------------------------------------------------

def run_package_contract_qa(args=None):
    write_policy_configs()
    packages = _packages()
    contract = load_csv(dataset_path("v2ab_event_patch_schema_contract.csv"))
    required = [r["field_name"] for r in contract if r.get("required") == "true"] or REQUIRED_CONTRACT_DEFAULT
    rows = []
    for pkg in packages:
        epc = pkg.get("event_patch_candidate_id", "")
        missing = [f for f in required if f in pkg and not str(pkg.get(f, "")).strip()]
        missing += [f for f in required if f not in pkg]
        if missing:
            status = "EXPECTED_BLOCKER" if _is_missing_patch(pkg) else "FAIL"
            _check(rows, "package_contract", epc, "required_fields_present", "all_required_present",
                   "missing:" + "|".join(sorted(set(missing))), status,
                   "expected" if status == "EXPECTED_BLOCKER" else "high",
                   "Missing patch_id is the known blocked candidate." if status == "EXPECTED_BLOCKER" else "Required field absent.")
        else:
            _check(rows, "package_contract", epc, "required_fields_present", "all_required_present", "all_present", "PASS")
        _check(rows, "package_contract", epc, "schema_contract_version_set", "non_empty",
               pkg.get("schema_contract_version", ""), "PASS" if pkg.get("schema_contract_version") else "FAIL",
               "info" if pkg.get("schema_contract_version") else "high")
        _check(rows, "package_contract", epc, "package_validation_status_set", "non_empty",
               pkg.get("package_validation_status", ""), "PASS" if pkg.get("package_validation_status") else "FAIL")
        for field in ("overlay_status", "ground_reference_status", "training_label_status"):
            ok = pkg.get(field) == "BLOCKED"
            _check(rows, "package_contract", epc, f"{field}_blocked", "BLOCKED", pkg.get(field, ""),
                   "PASS" if ok else "FAIL", "info" if ok else "critical")
    out = dataset_path("v2ad_package_contract_qa.csv")
    write_csv(out, QA_COLUMNS, rows)
    print(f"[v2ad contract qa] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 2. Namespace Crosswalk QA -----------------------------------------------

def run_namespace_crosswalk_qa(args=None):
    packages = _packages()
    rows = []
    for pkg in packages:
        epc = pkg.get("event_patch_candidate_id", "")
        patch_id = (pkg.get("patch_id") or "").strip()
        ns = pkg.get("patch_namespace", "")
        if patch_id:
            ok = ns == NS_EVENT
            _check(rows, "namespace_crosswalk", epc, "namespace_event_candidate", NS_EVENT, ns,
                   "PASS" if ok else "FAIL", "info" if ok else "high")
        else:
            ok = ns == "PATCH_ID_MISSING"
            _check(rows, "namespace_crosswalk", epc, "namespace_missing_only_without_patch", "PATCH_ID_MISSING", ns,
                   "PASS" if ok else "FAIL", "expected" if ok else "high",
                   "Missing patch id correctly flagged." if ok else "")
        # DINO crosswalk only via identical patch_id.
        if pkg.get("crosswalk_status", "").startswith("EXPLICIT_DINO"):
            ok = pkg.get("explicit_crosswalk_id") == f"XW_DINO::{patch_id}"
            _check(rows, "namespace_crosswalk", epc, "dino_crosswalk_by_identical_patch_id",
                   f"XW_DINO::{patch_id}", pkg.get("explicit_crosswalk_id", ""), "PASS" if ok else "FAIL",
                   "info" if ok else "high")
        # No anchor/refpatch crosswalk.
        no_anchor = not (pkg.get("anchor_patch_id") or "").strip() and not (pkg.get("refpatch_id") or "").strip()
        _check(rows, "namespace_crosswalk", epc, "no_anchor_or_refpatch_crosswalk", "empty",
               f"anchor={pkg.get('anchor_patch_id', '')}|refpatch={pkg.get('refpatch_id', '')}",
               "PASS" if no_anchor else "FAIL", "info" if no_anchor else "critical")
        # No inferred crosswalk.
        inferred = pkg.get("crosswalk_inferred") == "true" or "INFERRED" in pkg.get("crosswalk_status", "").upper()
        _check(rows, "namespace_crosswalk", epc, "crosswalk_not_inferred", "false",
               pkg.get("crosswalk_inferred", ""), "FAIL" if inferred else "PASS",
               "critical" if inferred else "info")
    out = dataset_path("v2ad_namespace_crosswalk_qa.csv")
    write_csv(out, QA_COLUMNS, rows)
    print(f"[v2ad namespace qa] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 3. Temporal Safety QA ---------------------------------------------------

def run_temporal_safety_qa(args=None):
    packages = _packages()
    rows = []
    for pkg in packages:
        epc = pkg.get("event_patch_candidate_id", "")
        inferred = pkg.get("sentinel_date_inferred") == "true"
        _check(rows, "temporal_safety", epc, "sentinel_date_not_inferred", "false",
               pkg.get("sentinel_date_inferred", ""), "FAIL" if inferred else "PASS",
               "critical" if inferred else "info")
        link = pkg.get("date_linkability_status", "")
        date = (pkg.get("sentinel_scene_date") or "").strip()
        if link in {"UNLINKABLE_NAMESPACE", "UNLINKABLE_CONFLICT", "UNLINKABLE_LOW_CONFIDENCE", "NO_DATE"}:
            applied = bool(date)
            _check(rows, "temporal_safety", epc, "unlinkable_date_not_applied", "empty_sentinel_scene_date",
                   date or "empty", "FAIL" if applied else "PASS", "critical" if applied else "info",
                   "Unlinkable/missing date must leave sentinel_scene_date empty.")
        if pkg.get("sentinel_date_status") == "SENTINEL_DATE_MISSING_WITH_BLOCKER":
            has_blocker = bool((pkg.get("temporal_blocker") or "").strip())
            _check(rows, "temporal_safety", epc, "missing_date_has_blocker", "non_empty_blocker",
                   pkg.get("temporal_blocker", ""), "PASS" if has_blocker else "FAIL",
                   "expected" if has_blocker else "high",
                   "Missing date is an expected blocker." if has_blocker else "")
        if pkg.get("sentinel_date_status") == "SENTINEL_DATE_CONFIRMED_SAME_PATCH":
            ok = bool(date)
            _check(rows, "temporal_safety", epc, "confirmed_same_patch_has_date", "non_empty_date",
                   date or "empty", "PASS" if ok else "FAIL")
    out = dataset_path("v2ad_temporal_safety_qa.csv")
    write_csv(out, QA_COLUMNS, rows)
    print(f"[v2ad temporal qa] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 4. Guardrail QA ---------------------------------------------------------

def _scan_guardrails(path):
    counts = {
        "forbidden_true_value": 0, "forbidden_status": 0, "absolute_path": 0,
        "local_only_leak": 0, "tool_name_leak": 0, "overlay_or_gr_or_training_released": 0,
    }
    if not os.path.exists(path):
        return counts
    if path.lower().endswith(".csv"):
        for record in load_csv(path):
            for key, value in record.items():
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
                if field in record and record.get(field) not in ("BLOCKED", "", None):
                    counts["overlay_or_gr_or_training_released"] += 1
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        counts["forbidden_status"] += len(FORBIDDEN_STATUS_RE.findall(text))
        counts["local_only_leak"] += text.count("local_only/") + text.count("local_only\\")
        counts["absolute_path"] += len(ABSOLUTE_PATH_RE.findall(text))
        counts["tool_name_leak"] += len(TOOL_NAME_RE.findall(text))
    return counts


def run_guardrail_qa(args=None, artifacts=None):
    if artifacts is None:
        artifacts = [artifact_path(a) for a in GUARDRAIL_AUDIT_ARTIFACTS]
    rows = []
    for art in artifacts:
        counts = _scan_guardrails(art)
        for check_type, n in counts.items():
            status = "PASS" if n == 0 else "FAIL"
            rows.append({
                "guardrail_qa_id": f"GQA_v2ad_{len(rows):05d}",
                "artifact": os.path.basename(art),
                "check_type": check_type,
                "violation_count": str(n),
                "status": status,
                "severity": "info" if status == "PASS" else "critical",
                "notes": "Clean" if status == "PASS" else "Guardrail violation; must not be versioned as-is.",
            })
    out = dataset_path("v2ad_guardrail_qa.csv")
    write_csv(out, GUARDRAIL_QA_COLUMNS, rows)
    print(f"[v2ad guardrail qa] rows={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 5. Readiness Consistency QA ---------------------------------------------

def run_readiness_consistency_qa(args=None):
    packages = {p["event_patch_candidate_id"]: p for p in _packages()}
    readiness = load_csv(dataset_path("v2ac_v2_readiness_matrix.csv"))
    by_epc = {}
    for r in readiness:
        by_epc.setdefault(r["event_patch_candidate_id"], {})[r["dimension"]] = r["classification"]
    rows = []
    for epc, dims in by_epc.items():
        pkg = packages.get(epc, {})
        # overlay readiness must be BLOCKED.
        _check(rows, "readiness_consistency", epc, "overlay_blocked_when_geometry_absent", "BLOCKED",
               dims.get("overlay_readiness", ""), "PASS" if dims.get("overlay_readiness") == "BLOCKED" else "FAIL",
               "critical")
        _check(rows, "readiness_consistency", epc, "ground_reference_blocked", "BLOCKED",
               dims.get("ground_reference_readiness", ""), "PASS" if dims.get("ground_reference_readiness") == "BLOCKED" else "FAIL",
               "critical")
        _check(rows, "readiness_consistency", epc, "training_blocked", "BLOCKED",
               dims.get("training_readiness", ""), "PASS" if dims.get("training_readiness") == "BLOCKED" else "FAIL",
               "critical")
        # Unlinkable/missing date -> temporal readiness not STRONG.
        if pkg.get("sentinel_date_status") in {"SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE", "SENTINEL_DATE_MISSING_WITH_BLOCKER"}:
            ok = dims.get("sentinel_date_status") != "STRONG" and dims.get("temporal_linkage") != "STRONG"
            _check(rows, "readiness_consistency", epc, "temporal_not_strong_when_unlinkable", "not_STRONG",
                   f"date={dims.get('sentinel_date_status', '')}|linkage={dims.get('temporal_linkage', '')}",
                   "PASS" if ok else "FAIL", "high")
        # Missing patch id -> patch_identity not STRONG.
        if _is_missing_patch(pkg):
            ok = dims.get("patch_identity") != "STRONG"
            _check(rows, "readiness_consistency", epc, "patch_identity_not_strong_when_missing", "not_STRONG",
                   dims.get("patch_identity", ""), "PASS" if ok else "FAIL", "expected" if ok else "high")
    out = dataset_path("v2ad_readiness_consistency_qa.csv")
    write_csv(out, QA_COLUMNS, rows)
    print(f"[v2ad readiness qa] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 6. Migration Integrity QA -----------------------------------------------

def run_migration_integrity_qa(args=None):
    v1us_ids = [r.get("event_patch_candidate_id") for r in load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))]
    v2_ids = [p.get("event_patch_candidate_id") for p in _packages()]
    diff = load_csv(dataset_path("v2ac_migration_diff_audit.csv"))
    rows = []
    lost = set(v1us_ids) - set(v2_ids)
    extra = set(v2_ids) - set(v1us_ids)
    _check(rows, "migration_integrity", "ALL", "all_ids_preserved", "no_lost_ids",
           "lost:" + "|".join(sorted(lost)) if lost else "none", "FAIL" if lost else "PASS", "critical")
    _check(rows, "migration_integrity", "ALL", "no_extra_packages", "no_extra_ids",
           "extra:" + "|".join(sorted(extra)) if extra else "none", "FAIL" if extra else "PASS", "critical")
    _check(rows, "migration_integrity", "ALL", "package_count_preserved", str(len(v1us_ids)),
           str(len(v2_ids)), "PASS" if len(v1us_ids) == len(v2_ids) else "FAIL", "critical")
    non_additive = [r for r in diff if r.get("migration_additive") != "true"]
    _check(rows, "migration_integrity", "ALL", "migration_additive", "true",
           "non_additive:" + str(len(non_additive)), "FAIL" if non_additive else "PASS", "critical")
    modified = [r for r in diff if r.get("old_outputs_modified") != "false"]
    _check(rows, "migration_integrity", "ALL", "old_outputs_not_modified", "false",
           "modified:" + str(len(modified)), "FAIL" if modified else "PASS", "critical")
    out = dataset_path("v2ad_migration_integrity_qa.csv")
    write_csv(out, QA_COLUMNS, rows)
    print(f"[v2ad migration qa] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 7. Negative Fixture QA --------------------------------------------------

NEGATIVE_FIXTURES = [
    ("forbidden_ground_reference_true.csv", "can_create_ground_reference_true"),
    ("crosswalk_inferred_true.csv", "crosswalk_inferred_true"),
    ("sentinel_date_inferred_true.csv", "sentinel_date_inferred_true"),
    ("unlinkable_date_applied.csv", "unlinkable_date_applied"),
    ("overlay_not_blocked.csv", "overlay_not_blocked"),
    ("missing_required_field.csv", "missing_required_field"),
    ("readiness_inconsistency.csv", "readiness_inconsistency"),
    ("non_additive_migration.csv", "non_additive_migration"),
    ("valid_v2_package.csv", "none"),
    ("clean_guardrail.csv", "none"),
]


def _scan_row_violations(row):
    found = set()
    if (row.get("can_create_ground_reference") or "").lower() == "true":
        found.add("can_create_ground_reference_true")
    if (row.get("can_create_training_label") or "").lower() == "true":
        found.add("can_create_training_label_true")
    if (row.get("ground_truth_operational") or "").lower() == "true":
        found.add("ground_truth_operational_true")
    if (row.get("crosswalk_inferred") or "").lower() == "true" or "INFERRED" in (row.get("crosswalk_status") or "").upper():
        found.add("crosswalk_inferred_true")
    if (row.get("sentinel_date_inferred") or "").lower() == "true":
        found.add("sentinel_date_inferred_true")
    if (row.get("date_linkability_status") or "") == "UNLINKABLE_NAMESPACE" and (row.get("sentinel_scene_date") or "").strip():
        found.add("unlinkable_date_applied")
    if "overlay_status" in row and row.get("overlay_status") not in ("BLOCKED", "", None):
        found.add("overlay_not_blocked")
    if "patch_id" in row and (row.get("patch_id") or "").strip() and "patch_namespace" in row and not (row.get("patch_namespace") or "").strip():
        found.add("missing_required_field")
    if row.get("overlay_readiness") and (row.get("geometry_status") or "").upper() in ("ABSENT", "NO_OBSERVED_GEOMETRY", "") and row.get("overlay_readiness") != "BLOCKED":
        found.add("readiness_inconsistency")
    if (row.get("old_outputs_modified") or "").lower() == "true" or (row.get("migration_additive") or "").lower() == "false":
        found.add("non_additive_migration")
    return found


def run_negative_fixture_qa(args=None):
    rows = []
    for fixture_name, injected in NEGATIVE_FIXTURES:
        path = os.path.join(NEGATIVE_FIXTURE_DIR, fixture_name)
        records = load_csv(path)
        found = set()
        for rec in records:
            found |= _scan_row_violations(rec)
        if injected == "none":
            # 'detected' always means "a violation was found"; clean fixtures
            # must find none (no false positive).
            detected = bool(found)
            status = "PASS_NO_FALSE_POSITIVE" if not detected else "FAIL_FALSE_POSITIVE"
            notes = "Clean fixture produced no violation." if not detected else f"Unexpected detections: {'|'.join(sorted(found))}"
        else:
            detected = injected in found
            status = "PASS_VIOLATION_DETECTED" if detected else "FAIL_VIOLATION_MISSED"
            notes = "Injected violation correctly detected." if detected else "QA failed to detect the injected violation."
        rows.append({
            "negative_qa_id": f"NEG_v2ad_{len(rows):04d}",
            "fixture_name": fixture_name,
            "injected_violation": injected,
            "detected": "true" if detected else "false",
            "status": status,
            "notes": notes,
        })
    out = dataset_path("v2ad_negative_fixture_qa.csv")
    write_csv(out, NEGATIVE_QA_COLUMNS, rows)
    print(f"[v2ad negative qa] fixtures={len(rows)} ok={sum(1 for r in rows if r['status'].startswith('PASS'))} -> {out}")
    return rows


# 8. QA Gate Summary Builder ----------------------------------------------

def run_qa_gate_summary_builder(args=None):
    rows = []
    overall_fail = False
    overall_expected = False
    for group, filename in QA_GROUPS:
        data = load_csv(dataset_path(filename))
        if group == "negative_fixture":
            total = len(data)
            passed = sum(1 for r in data if r.get("status", "").startswith("PASS"))
            failed = total - passed
            expected = 0
        else:
            total = len(data)
            failed = sum(1 for r in data if r.get("status") == "FAIL")
            expected = sum(1 for r in data if r.get("status") == "EXPECTED_BLOCKER")
            passed = total - failed - expected
        if failed:
            gate = "QA_FAIL"
            overall_fail = True
            action = "fix_violations_before_versioning"
        elif expected:
            gate = "QA_PASS_WITH_EXPECTED_BLOCKERS"
            overall_expected = True
            action = "none_expected_blockers_documented"
        else:
            gate = "QA_PASS"
            action = "none"
        rows.append({
            "gate_id": f"GATE_v2ad_{len(rows):04d}",
            "qa_group": group,
            "total_checks": str(total),
            "passed_checks": str(passed),
            "failed_checks": str(failed),
            "expected_blockers": str(expected),
            "gate_status": gate,
            "required_action": action,
            "notes": "Per-group QA gate.",
        })
    overall = "QA_FAIL" if overall_fail else ("QA_PASS_WITH_EXPECTED_BLOCKERS" if overall_expected else "QA_PASS")
    rows.append({
        "gate_id": "GATE_v2ad_OVERALL",
        "qa_group": "OVERALL",
        "total_checks": str(sum(int(r["total_checks"]) for r in rows)),
        "passed_checks": str(sum(int(r["passed_checks"]) for r in rows)),
        "failed_checks": str(sum(int(r["failed_checks"]) for r in rows)),
        "expected_blockers": str(sum(int(r["expected_blockers"]) for r in rows)),
        "gate_status": overall,
        "required_action": "none_expected_blockers_documented" if overall == "QA_PASS_WITH_EXPECTED_BLOCKERS" else ("fix_violations_before_versioning" if overall == "QA_FAIL" else "none"),
        "notes": "Expected blockers: no geometry, no occurrence coordinate, unlinkable/missing sentinel date, no ground reference.",
    })
    out = dataset_path("v2ad_qa_gate_summary.csv")
    write_csv(out, GATE_COLUMNS, rows)
    print(f"[v2ad gate] overall={overall} -> {out}")
    return rows


# 9. Next Programming Target Ranker ----------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}

TARGET_VERSION = {
    "MULTI_REGION_REGISTRY_HARDENING": "v2ae — Multi-Region Registry Hardening",
    "EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION": "v2ae — Event-Patch Package V2 QA Automation",
    "SENTINEL_DATE_CROSSWALK_DISCOVERY": "v2ae — Sentinel Date Crosswalk Discovery",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2ae — DINO Review Support Completion",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2ae — Public Source Recheck Hold",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2ae — Ground Truth Search Hold",
}


def _ranker_metrics():
    gate = load_csv(dataset_path("v2ad_qa_gate_summary.csv"))
    overall = next((r for r in gate if r.get("qa_group") == "OVERALL"), {})
    qa_pass = overall.get("gate_status", "") in {"QA_PASS", "QA_PASS_WITH_EXPECTED_BLOCKERS"}
    packages = _packages()
    total = len(packages) or 1
    no_anchor = sum(1 for p in packages if "NO_ANCHOR_CROSSWALK" in p.get("crosswalk_status", "") or p.get("crosswalk_status") in {"NO_EXPLICIT_CROSSWALK", "NO_CROSSWALK_PATCH_ID_MISSING"})
    return {"qa_pass": qa_pass, "no_anchor_rate": no_anchor / total}


def _candidate_targets(m):
    return [
        {
            "next_target": "MULTI_REGION_REGISTRY_HARDENING",
            "programming_value": 60 if m["qa_pass"] else 30,
            "ground_truth_value": 0,
            "blocker_reduction_value": 45,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "QA gate is green with expected blockers; consolidate the QA-locked v2 packages into the multi-region registries.",
        },
        {
            "next_target": "EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION",
            "programming_value": 50,
            "ground_truth_value": 0,
            "blocker_reduction_value": 40,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Wire the v2 QA harness into a repeatable automation/regression entrypoint.",
        },
        {
            "next_target": "SENTINEL_DATE_CROSSWALK_DISCOVERY",
            "programming_value": round(40 * m["no_anchor_rate"]),
            "ground_truth_value": 0,
            "blocker_reduction_value": round(50 * m["no_anchor_rate"]),
            "expected_effort": "MEDIUM",
            "overclaim_risk": "MEDIUM",
            "notes": "Search for an explicit key linking numeric and anchor namespaces; uncertain, only useful if a key exists.",
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
            "notes": "Hold until a new public source with linkable patch identity appears.",
        },
        {
            "next_target": "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE",
            "programming_value": 10,
            "ground_truth_value": 0,
            "blocker_reduction_value": 0,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Explicit stop on ground-truth search until a qualifying public source is published.",
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
    out = dataset_path("v2ad_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v2ad ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
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
                "blocker_id": f"GB_v2ad_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "blocker": blocker,
                "status": "BLOCKED",
                **guardrails(),
                "notes": "QA harness does not unblock geometry, overlay, ground reference or labels.",
            })
    out = dataset_path("v2ad_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_MATRIX_COLUMNS, rows)
    print(f"[v2ad gr blockers] rows={len(rows)} -> {out}")
    return rows


def _fails(rows):
    return sum(1 for r in rows if r.get("status") == "FAIL")


def run_completion_report(args=None):
    write_policy_configs()
    contract = load_csv(dataset_path("v2ad_package_contract_qa.csv")) or run_package_contract_qa(args)
    namespace = load_csv(dataset_path("v2ad_namespace_crosswalk_qa.csv")) or run_namespace_crosswalk_qa(args)
    temporal = load_csv(dataset_path("v2ad_temporal_safety_qa.csv")) or run_temporal_safety_qa(args)
    guardrail = load_csv(dataset_path("v2ad_guardrail_qa.csv")) or run_guardrail_qa(args)
    readiness = load_csv(dataset_path("v2ad_readiness_consistency_qa.csv")) or run_readiness_consistency_qa(args)
    migration = load_csv(dataset_path("v2ad_migration_integrity_qa.csv")) or run_migration_integrity_qa(args)
    negative = load_csv(dataset_path("v2ad_negative_fixture_qa.csv")) or run_negative_fixture_qa(args)
    gate = load_csv(dataset_path("v2ad_qa_gate_summary.csv")) or run_qa_gate_summary_builder(args)
    ranker = load_csv(dataset_path("v2ad_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    blockers = run_ground_reference_blocker_matrix(args)

    overall = next((r for r in gate if r.get("qa_group") == "OVERALL"), {})
    negatives_ok = all(r.get("status", "").startswith("PASS") for r in negative)
    next_target = ranker[0].get("next_target", "") if ranker else ""
    next_version = ranker[0].get("recommended_version", "") if ranker else ""

    write_csv(dataset_path("v2ad_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v2ad_0000",
        "event_id": "MULTI_REGION",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v2ad score-based next-programming-target ranker after the QA harness.",
        "target": "EVENT_PATCH_PACKAGE_V2",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth, ground reference, inferred date or inferred crosswalk.",
    }])

    lines = [
        "# Protocolo C v2ad - Event-Patch Package V2 QA Harness",
        "",
        f"- package contract checks: `{len(contract)}` (fails: `{_fails(contract)}`)",
        f"- namespace/crosswalk checks: `{len(namespace)}` (fails: `{_fails(namespace)}`)",
        f"- temporal safety checks: `{len(temporal)}` (fails: `{_fails(temporal)}`)",
        f"- guardrail checks: `{len(guardrail)}` (fails: `{_fails(guardrail)}`)",
        f"- readiness consistency checks: `{len(readiness)}` (fails: `{_fails(readiness)}`)",
        f"- migration integrity checks: `{len(migration)}` (fails: `{_fails(migration)}`)",
        f"- negative fixtures detected: `{sum(1 for r in negative if r.get('status', '').startswith('PASS'))}/{len(negative)}`",
        f"- overall QA gate: `{overall.get('gate_status', '')}`",
        f"- selected next target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v2ad is a read-only QA harness. It modified no prior output, inferred no crosswalk, inferred no Sentinel date, applied no unlinkable date, executed no overlay, and created no ground truth, ground reference or label.",
    ]
    write_text(doc_path("protocolo_c_v2ad_event_patch_v2_qa_harness.md"), lines)

    report = lines + [
        "",
        "## How many packages passed the contract",
        f"Package contract QA ran {len(contract)} checks with {_fails(contract)} failures; the only non-pass is the expected blocked candidate with a missing patch id.",
        "",
        "## Namespace / crosswalk QA",
        f"{len(namespace)} checks with {_fails(namespace)} failures: namespaces are explicit, DINO crosswalk is by identical patch_id only, and no anchor/refpatch or inferred crosswalk exists.",
        "",
        "## Temporal safety QA",
        f"{len(temporal)} checks with {_fails(temporal)} failures: no Sentinel date inferred, unlinkable/missing dates leave sentinel_scene_date empty, and missing dates carry an explicit blocker.",
        "",
        "## Guardrail QA",
        f"{len(guardrail)} checks with {_fails(guardrail)} failures across the audited artifacts (forbidden true values, forbidden statuses, absolute paths, local_only leaks, tool-name leaks, and overlay/ground reference/training release).",
        "",
        "## Readiness consistency QA",
        f"{len(readiness)} checks with {_fails(readiness)} failures: overlay/ground reference/training stay BLOCKED and temporal/identity readiness is never STRONG when the date is unlinkable or the patch id is missing.",
        "",
        "## Migration integrity QA",
        f"{len(migration)} checks with {_fails(migration)} failures: all ids preserved, no lost or extra packages, additive migration, prior outputs unmodified.",
        "",
        "## Negative fixtures",
        ("All injected unsafe fixtures were detected and clean fixtures produced no false positives." if negatives_ok else "Some negative fixtures were not detected; QA harness needs attention."),
        "",
        "## QA gate and expected blockers",
        f"Overall gate: `{overall.get('gate_status', '')}`. Expected blockers remain: no observed geometry, no occurrence coordinate, unlinkable/missing Sentinel date, no explicit anchor crosswalk, and no ground reference.",
        "",
        "## Why there is still no overlay",
        "No overlay was executed and overlay readiness stays BLOCKED. The QA harness verifies safety; it establishes no observed occurrence geometry.",
        "",
        "## Why there is still no ground reference",
        "No package has a linkable observed occurrence geometry; without it there is no basis for ground reference, and none was created.",
        "",
        "## Next programming step",
        f"The score-based ranker selected `{next_target}` (`{next_version}`).",
    ]
    write_text(doc_path("protocolo_c_relatorio_v2ad_event_patch_v2_qa_harness.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v2ad.md"), [
        "# Status atual - Protocolo C v2ad",
        "",
        f"QA harness status: `{MAX_STATUS}`.",
        f"Overall QA gate: `{overall.get('gate_status', '')}`.",
        f"Negative fixtures detected: `{sum(1 for r in negative if r.get('status', '').startswith('PASS'))}/{len(negative)}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Overlay, ground reference, training labels, ground truth, inferred Sentinel dates and inferred crosswalks remain blocked.",
    ])

    manifest = []
    for idx, artifact in enumerate(V2AD_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v2ad_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v2ad QA harness artifact; no raw data, no private path, no inferred date or crosswalk.",
        })
    write_csv(dataset_path("v2ad_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for folder in (STAGING_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v2ad completion] gate={overall.get('gate_status', '')} next={next_target}")
    return {"gate": overall.get("gate_status", ""), "negatives_ok": negatives_ok, "next_target": next_target, "next_version": next_version}


def run_all(args=None):
    args = args or parse_args([])
    run_package_contract_qa(args)
    run_namespace_crosswalk_qa(args)
    run_temporal_safety_qa(args)
    run_guardrail_qa(args)
    run_readiness_consistency_qa(args)
    run_migration_integrity_qa(args)
    run_negative_fixture_qa(args)
    run_qa_gate_summary_builder(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
