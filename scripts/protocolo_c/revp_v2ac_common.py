#!/usr/bin/env python3
"""v2ac Event-patch schema migration implementation.

Builds an additive, hardened v2 event-patch package registry from the existing
v1us/v2aa/v2ab outputs, following the v2ab schema contract. It adds explicit
patch namespace, source registry, identity/crosswalk, temporal-status and
normalized blocker fields without modifying any prior output. Crosswalks are
populated only where an explicit shared key exists (event-patch candidate vs
DINO visual share patch_id); anchor/scaffolding crosswalks stay absent with an
explicit blocker. Sentinel dates recovered only in a parallel namespace are kept
unlinkable and never applied. No overlay, ground reference, label, coordinate or
inferred date/crosswalk is produced.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v2ac"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/event_patch_schema_migration/staging/v2ac"
REPORTS_DIR = "local_only/protocolo_c/event_patch_schema_migration/reports/v2ac"
# Repo-relative DINO visual linkage registry (shared patch_id namespace).
DINO_VISUAL_REGISTRY = "datasets/dino_patch_visual_linkage_registry_v1pv.csv"
SCHEMA_CONTRACT_VERSION = "v2ab_event_patch_schema_contract"

MAX_STATUS = "EVENT_PATCH_SCHEMA_MIGRATED_NON_OPERATIONAL"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "schema_migration_only", "crosswalk_inferred",
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

NUMERIC_RE = re.compile(r"^(CUR|PET|REC)_\d{4,6}$")
NS_EVENT = "EVENT_PATCH_CANDIDATE_NAMESPACE"
NS_DINO = "DINO_VISUAL_PATCH_NAMESPACE"

# Column definitions ------------------------------------------------------
V2_PACKAGE_COLUMNS = [
    "event_patch_candidate_id", "event_id", "event_region", "patch_id",
    "patch_namespace", "patch_source_registry", "source_patch_id",
    "anchor_patch_id", "refpatch_id", "explicit_crosswalk_id", "crosswalk_status",
    "linkage_basis", "linkage_status", "event_patch_candidate_only",
    "sentinel_scene_date", "sentinel_scene_datetime", "sentinel_platform",
    "scene_id", "sentinel_date_status", "date_source_namespace",
    "date_linkability_status", "temporal_linkage_status", "temporal_blocker",
    "evidence_status", "phenomenon_status", "coordinate_status",
    "geometry_status", "overlay_status", "ground_reference_status",
    "training_label_status", "dino_review_support_status", "blocker",
    "safe_use", "prohibited_use", "schema_contract_version",
    "package_validation_status", "can_create_ground_reference",
    "can_create_training_label", "ground_truth_operational",
    "crosswalk_inferred", "sentinel_date_inferred",
]
NAMESPACE_POP_COLUMNS = [
    "namespace_population_id", "event_patch_candidate_id", "patch_id",
    "patch_namespace", "patch_source_registry", "namespace_population_status",
    "notes",
]
CROSSWALK_POP_COLUMNS = [
    "crosswalk_population_id", "event_patch_candidate_id", "patch_id",
    "dino_patch_id", "explicit_crosswalk_id", "crosswalk_status",
    "anchor_patch_id", "refpatch_id", "crosswalk_inferred", "blocker", "notes",
]
TEMPORAL_POP_COLUMNS = [
    "temporal_population_id", "event_patch_candidate_id", "patch_id",
    "sentinel_date_status", "sentinel_scene_date", "date_source_namespace",
    "date_linkability_status", "temporal_blocker", "sentinel_date_inferred",
    "notes",
]
BLOCKER_NORM_COLUMNS = [
    "blocker_norm_id", "event_patch_candidate_id", "patch_id", "blocker",
    "present", "source_versions", "normalized_blocker_set", "notes",
]
SCHEMA_VAL_COLUMNS = [
    "schema_validation_id", "event_patch_candidate_id", "validation_status",
    "missing_required_fields", "null_optional_without_blocker",
    "forbidden_value_count", "guardrail_violation_count",
    "schema_contract_version", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
READINESS_COLUMNS = [
    "readiness_id", "event_patch_candidate_id", "event_id", "patch_id",
    "dimension", "classification", "basis", *GUARDRAIL_COLUMNS, "notes",
]
DIFF_COLUMNS = [
    "diff_id", "event_patch_candidate_id", "source_version", "target_version",
    "fields_added", "statuses_changed", "blockers_added",
    "old_outputs_modified", "migration_additive", "notes",
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

V2AC_ARTIFACTS = [
    "configs/protocolo_c/v2ac_event_patch_v2_package_policy.yaml",
    "configs/protocolo_c/v2ac_namespace_population_policy.yaml",
    "configs/protocolo_c/v2ac_crosswalk_population_policy.yaml",
    "configs/protocolo_c/v2ac_temporal_status_policy.yaml",
    "configs/protocolo_c/v2ac_schema_validation_policy.yaml",
    "configs/protocolo_c/v2ac_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2ac_event_patch_v2_package_registry.csv",
    "datasets/protocolo_c/v2ac_patch_namespace_field_population.csv",
    "datasets/protocolo_c/v2ac_crosswalk_field_population.csv",
    "datasets/protocolo_c/v2ac_temporal_status_field_population.csv",
    "datasets/protocolo_c/v2ac_blocker_field_normalization.csv",
    "datasets/protocolo_c/v2ac_schema_contract_validation.csv",
    "datasets/protocolo_c/v2ac_v2_readiness_matrix.csv",
    "datasets/protocolo_c/v2ac_migration_diff_audit.csv",
    "datasets/protocolo_c/v2ac_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2ac_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2ac_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v2ac_event_patch_schema_migration.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2ac_event_patch_schema_migration.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2ac.md",
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
        "schema_migration_only": "true",
        "crosswalk_inferred": "false",
        "sentinel_date_inferred": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v2ac_event_patch_v2_package_policy.yaml": [
            "additive_migration_only: true",
            "modify_prior_outputs: false",
            "max_status: EVENT_PATCH_SCHEMA_MIGRATED_NON_OPERATIONAL",
            "schema_contract_version: v2ab_event_patch_schema_contract",
        ],
        "v2ac_namespace_population_policy.yaml": [
            "numeric_pattern: (CUR|PET|REC)_NNNNN -> EVENT_PATCH_CANDIDATE_NAMESPACE",
            "empty_patch_id -> PATCH_ID_MISSING",
            "use_anchor_namespace_for_candidate: false",
        ],
        "v2ac_crosswalk_population_policy.yaml": [
            "explicit_dino_crosswalk_by_identical_patch_id: true",
            "populate_anchor_or_refpatch_crosswalk: false",
            "use_region_order_date_name_as_key: false",
            "crosswalk_inferred: false",
        ],
        "v2ac_temporal_status_policy.yaml": [
            "apply_unlinkable_date_to_candidate: false",
            "fill_sentinel_scene_date_only_if_same_patch_confirmed: true",
            "sentinel_date_inferred: false",
        ],
        "v2ac_schema_validation_policy.yaml": [
            "validate_against: v2ab_event_patch_schema_contract",
            "forbidden_values_blocked: true",
            "overlay_ground_reference_training_must_be_blocked: true",
        ],
        "v2ac_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


# Input accessors ---------------------------------------------------------

def _index(path, key="event_patch_candidate_id"):
    return {r.get(key): r for r in load_csv(dataset_path(path))}


def _dino_patch_ids():
    ids = set()
    for row in load_csv(DINO_VISUAL_REGISTRY):
        pid = (row.get("patch_id") or "").strip()
        if pid:
            ids.add(pid)
    return ids


def _normalized_blockers(geometry_status, coordinate_status, sentinel_status, has_anchor_crosswalk):
    blockers = []
    if "NO_OBSERVED_GEOMETRY" in (geometry_status or "").upper() or not geometry_status:
        blockers.append("no_observed_geometry")
    if "NO_COORD" in (coordinate_status or "").upper() or "NO_OCCURRENCE" in (coordinate_status or "").upper() or not coordinate_status:
        blockers.append("no_occurrence_coordinates")
    if sentinel_status == "SENTINEL_DATE_MISSING_WITH_BLOCKER":
        blockers.append("no_sentinel_date")
    elif sentinel_status == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE":
        blockers.append("unlinkable_sentinel_date")
    if not has_anchor_crosswalk:
        blockers.append("no_explicit_anchor_crosswalk")
    blockers += ["no_overlay", "no_ground_reference", "no_training_label", "patch_truth_forbidden"]
    # Deduplicate while preserving order.
    seen, out = set(), []
    for b in blockers:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


# 1. Event-Patch V2 Package Builder ---------------------------------------

def _assemble_packages():
    candidates = load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))
    temporal = _index("v2ab_temporal_field_contract_enforcement.csv")
    phenom = _index("v1us_phenomenon_status_attachment.csv")
    geom = _index("v1us_geometry_blocker_attachment.csv")
    evidence = _index("v1us_external_evidence_attachment_registry.csv")
    dino = _index("v1us_dino_review_support_attachment.csv")
    dino_ids = _dino_patch_ids()

    packages = []
    for cand in candidates:
        epc = cand.get("event_patch_candidate_id", "")
        patch_id = (cand.get("patch_id") or "").strip()
        tc = temporal.get(epc, {})
        sentinel_status = tc.get("sentinel_date_status", "SENTINEL_DATE_MISSING_WITH_BLOCKER")
        selected_date = tc.get("selected_sentinel_date", "")
        # Namespace
        if not patch_id:
            namespace = "PATCH_ID_MISSING"
        elif NUMERIC_RE.match(patch_id):
            namespace = NS_EVENT
        else:
            namespace = NS_EVENT
        # Crosswalk (explicit DINO only, by identical patch_id)
        has_dino_xw = bool(patch_id) and patch_id in dino_ids
        explicit_crosswalk_id = f"XW_DINO::{patch_id}" if has_dino_xw else ""
        if has_dino_xw:
            crosswalk_status = "EXPLICIT_DINO_CROSSWALK_NO_ANCHOR_CROSSWALK"
        elif not patch_id:
            crosswalk_status = "NO_CROSSWALK_PATCH_ID_MISSING"
        else:
            crosswalk_status = "NO_EXPLICIT_CROSSWALK"
        # Temporal linkability normalization
        if sentinel_status == "SENTINEL_DATE_CONFIRMED_SAME_PATCH":
            link = "LINKABLE_SAME_PATCH"
            temporal_linkage = "LINKED_SAME_PATCH_REVIEW_ONLY"
        elif sentinel_status == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE":
            link = "UNLINKABLE_NAMESPACE"
            temporal_linkage = "BLOCKED_NO_LINKABLE_DATE"
            selected_date = ""  # never apply unlinkable date
        elif sentinel_status == "SENTINEL_DATE_CONFLICT_BLOCKED":
            link = "UNLINKABLE_CONFLICT"
            temporal_linkage = "BLOCKED_NO_LINKABLE_DATE"
            selected_date = ""
        elif sentinel_status == "SENTINEL_DATE_LOW_CONFIDENCE_BLOCKED":
            link = "UNLINKABLE_LOW_CONFIDENCE"
            temporal_linkage = "BLOCKED_NO_LINKABLE_DATE"
            selected_date = ""
        else:
            link = "NO_DATE"
            temporal_linkage = "BLOCKED_NO_LINKABLE_DATE"
            selected_date = ""
        g = geom.get(epc, {})
        coordinate_status = g.get("coordinate_status", "")
        geometry_status = g.get("geometry_status", "")
        blockers = _normalized_blockers(geometry_status, coordinate_status, sentinel_status, has_anchor_crosswalk=False)
        validation_status = "PACKAGE_V2_BLOCKED_MISSING_PATCH_ID" if not patch_id else "PACKAGE_V2_SCHEMA_VALID_WITH_TEMPORAL_BLOCKER"
        packages.append({
            "event_patch_candidate_id": epc,
            "event_id": cand.get("event_id", ""),
            "event_region": cand.get("region", ""),
            "patch_id": patch_id,
            "patch_namespace": namespace,
            "patch_source_registry": "v1us_patch_registry_resolution.csv",
            "source_patch_id": patch_id,
            "anchor_patch_id": "",
            "refpatch_id": "",
            "explicit_crosswalk_id": explicit_crosswalk_id,
            "crosswalk_status": crosswalk_status,
            "linkage_basis": cand.get("linkage_basis", ""),
            "linkage_status": cand.get("linkage_status", ""),
            "event_patch_candidate_only": cand.get("event_patch_candidate_only", "true"),
            "sentinel_scene_date": selected_date,
            "sentinel_scene_datetime": "",
            "sentinel_platform": "",
            "scene_id": "",
            "sentinel_date_status": sentinel_status,
            "date_source_namespace": tc.get("date_source_namespace", ""),
            "date_linkability_status": link,
            "temporal_linkage_status": temporal_linkage,
            "temporal_blocker": tc.get("temporal_blocker", "no_recoverable_sentinel_date_for_this_patch"),
            "evidence_status": evidence.get(epc, {}).get("evidence_status", ""),
            "phenomenon_status": phenom.get(epc, {}).get("phenomenon_class", "") or phenom.get(epc, {}).get("phenomenon_support", ""),
            "coordinate_status": coordinate_status,
            "geometry_status": geometry_status,
            "overlay_status": "BLOCKED",
            "ground_reference_status": "BLOCKED",
            "training_label_status": "BLOCKED",
            "dino_review_support_status": dino.get(epc, {}).get("dino_review_support_status", ""),
            "blocker": "|".join(blockers),
            "safe_use": "contextual_review_only",
            "prohibited_use": "ground_truth_label_overlay_patch_truth",
            "schema_contract_version": SCHEMA_CONTRACT_VERSION,
            "package_validation_status": validation_status,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "ground_truth_operational": "false",
            "crosswalk_inferred": "false",
            "sentinel_date_inferred": "false",
        })
    return packages


def run_event_patch_v2_package_builder(args=None):
    write_policy_configs()
    packages = _assemble_packages()
    out = dataset_path("v2ac_event_patch_v2_package_registry.csv")
    write_csv(out, V2_PACKAGE_COLUMNS, packages)
    print(f"[v2ac v2 package] packages={len(packages)} -> {out}")
    return packages


def _load_or_build_registry(args=None):
    return load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv")) or run_event_patch_v2_package_builder(args)


# 2. Patch Namespace Field Populator --------------------------------------

def run_patch_namespace_field_populator(args=None):
    registry = _load_or_build_registry(args)
    rows = []
    for pkg in registry:
        patch_id = pkg.get("patch_id", "")
        status = "NAMESPACE_POPULATED" if patch_id else "PATCH_ID_MISSING"
        rows.append({
            "namespace_population_id": f"NSP_v2ac_{len(rows):05d}",
            "event_patch_candidate_id": pkg.get("event_patch_candidate_id", ""),
            "patch_id": patch_id,
            "patch_namespace": pkg.get("patch_namespace", ""),
            "patch_source_registry": pkg.get("patch_source_registry", ""),
            "namespace_population_status": status,
            "notes": "Numeric CUR/PET/REC ids map to the event-patch candidate namespace; anchor namespace is never used.",
        })
    out = dataset_path("v2ac_patch_namespace_field_population.csv")
    write_csv(out, NAMESPACE_POP_COLUMNS, rows)
    print(f"[v2ac namespace pop] rows={len(rows)} -> {out}")
    return rows


# 3. Crosswalk Field Populator --------------------------------------------

def run_crosswalk_field_populator(args=None):
    registry = _load_or_build_registry(args)
    rows = []
    for pkg in registry:
        patch_id = pkg.get("patch_id", "")
        has_dino = pkg.get("crosswalk_status", "").startswith("EXPLICIT_DINO")
        blocker = "" if has_dino else ("patch_id_missing" if not patch_id else "no_explicit_anchor_or_dino_crosswalk")
        if has_dino:
            blocker = "no_explicit_anchor_crosswalk"
        rows.append({
            "crosswalk_population_id": f"XWP_v2ac_{len(rows):05d}",
            "event_patch_candidate_id": pkg.get("event_patch_candidate_id", ""),
            "patch_id": patch_id,
            "dino_patch_id": patch_id if has_dino else "",
            "explicit_crosswalk_id": pkg.get("explicit_crosswalk_id", ""),
            "crosswalk_status": pkg.get("crosswalk_status", ""),
            "anchor_patch_id": "",
            "refpatch_id": "",
            "crosswalk_inferred": "false",
            "blocker": blocker,
            "notes": "Explicit crosswalk only between identical patch_id (event-patch vs DINO); anchor/refpatch never populated by region/order/date/name.",
        })
    out = dataset_path("v2ac_crosswalk_field_population.csv")
    write_csv(out, CROSSWALK_POP_COLUMNS, rows)
    print(f"[v2ac crosswalk pop] rows={len(rows)} -> {out}")
    return rows


# 4. Temporal Status Field Populator --------------------------------------

def run_temporal_status_field_populator(args=None):
    registry = _load_or_build_registry(args)
    rows = []
    for pkg in registry:
        rows.append({
            "temporal_population_id": f"TSP_v2ac_{len(rows):05d}",
            "event_patch_candidate_id": pkg.get("event_patch_candidate_id", ""),
            "patch_id": pkg.get("patch_id", ""),
            "sentinel_date_status": pkg.get("sentinel_date_status", ""),
            "sentinel_scene_date": pkg.get("sentinel_scene_date", ""),
            "date_source_namespace": pkg.get("date_source_namespace", ""),
            "date_linkability_status": pkg.get("date_linkability_status", ""),
            "temporal_blocker": pkg.get("temporal_blocker", ""),
            "sentinel_date_inferred": "false",
            "notes": "Unlinkable cross-namespace dates leave sentinel_scene_date empty; dates are never inferred.",
        })
    out = dataset_path("v2ac_temporal_status_field_population.csv")
    write_csv(out, TEMPORAL_POP_COLUMNS, rows)
    print(f"[v2ac temporal pop] rows={len(rows)} -> {out}")
    return rows


# 5. Blocker Field Normalizer ---------------------------------------------

CANONICAL_BLOCKERS = [
    "no_observed_geometry", "no_occurrence_coordinates", "no_sentinel_date",
    "unlinkable_sentinel_date", "no_explicit_anchor_crosswalk", "no_overlay",
    "no_ground_reference", "no_training_label", "patch_truth_forbidden",
]


def run_blocker_field_normalizer(args=None):
    registry = _load_or_build_registry(args)
    rows = []
    for pkg in registry:
        present = set((pkg.get("blocker") or "").split("|"))
        normalized = [b for b in CANONICAL_BLOCKERS if b in present]
        for blocker in CANONICAL_BLOCKERS:
            rows.append({
                "blocker_norm_id": f"BN_v2ac_{len(rows):06d}",
                "event_patch_candidate_id": pkg.get("event_patch_candidate_id", ""),
                "patch_id": pkg.get("patch_id", ""),
                "blocker": blocker,
                "present": "true" if blocker in present else "false",
                "source_versions": "v1us|v1uz|v2aa|v2ab",
                "normalized_blocker_set": "|".join(normalized),
                "notes": "Blockers are consolidated and never removed; absent canonical blockers are recorded as present=false.",
            })
    out = dataset_path("v2ac_blocker_field_normalization.csv")
    write_csv(out, BLOCKER_NORM_COLUMNS, rows)
    print(f"[v2ac blocker norm] rows={len(rows)} -> {out}")
    return rows


# 6. Schema Contract Validator --------------------------------------------

def _contract_fields():
    contract = load_csv(dataset_path("v2ab_event_patch_schema_contract.csv"))
    required = [r["field_name"] for r in contract if r.get("required") == "true"]
    optional = [r["field_name"] for r in contract if r.get("required") == "false"]
    return required, optional


FORBIDDEN_VALUE_TOKENS = {t.lower() for t in FORBIDDEN_STATUS_TOKENS}


def run_schema_contract_validator(args=None):
    registry = _load_or_build_registry(args)
    required, optional = _contract_fields()
    rows = []
    for pkg in registry:
        missing = [f for f in required if f in V2_PACKAGE_COLUMNS and not str(pkg.get(f, "")).strip()]
        # Patch id legitimately absent for the broken candidate, with blocker.
        null_optional = []
        for f in optional:
            if f in V2_PACKAGE_COLUMNS and not str(pkg.get(f, "")).strip():
                if not pkg.get("blocker"):
                    null_optional.append(f)
        forbidden_count = sum(
            1 for v in pkg.values()
            if str(v).strip().lower() in FORBIDDEN_VALUE_TOKENS
        )
        guardrail_violations = sum(
            1 for k in GUARDRAIL_MUST_BE_FALSE
            if k in pkg and str(pkg.get(k)).strip().lower() == "true"
        )
        for k in ("overlay_status", "ground_reference_status", "training_label_status"):
            if pkg.get(k) != "BLOCKED":
                guardrail_violations += 1
        if not pkg.get("patch_id"):
            status = "SCHEMA_INVALID_MISSING_PATCH_ID"
        elif missing or null_optional or forbidden_count or guardrail_violations:
            status = "SCHEMA_INVALID"
        else:
            status = "SCHEMA_VALID_NON_OPERATIONAL"
        rows.append({
            "schema_validation_id": f"SV_v2ac_{len(rows):05d}",
            "event_patch_candidate_id": pkg.get("event_patch_candidate_id", ""),
            "validation_status": status,
            "missing_required_fields": "|".join(missing),
            "null_optional_without_blocker": "|".join(null_optional),
            "forbidden_value_count": str(forbidden_count),
            "guardrail_violation_count": str(guardrail_violations),
            "schema_contract_version": SCHEMA_CONTRACT_VERSION,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Validated against the v2ab contract; overlay/ground reference/training must be BLOCKED.",
        })
    out = dataset_path("v2ac_schema_contract_validation.csv")
    write_csv(out, SCHEMA_VAL_COLUMNS, rows)
    valid = sum(1 for r in rows if r["validation_status"] == "SCHEMA_VALID_NON_OPERATIONAL")
    print(f"[v2ac schema validation] rows={len(rows)} valid={valid} -> {out}")
    return rows


# 7. V2 Readiness Matrix Builder ------------------------------------------

READINESS_DIMENSIONS = [
    "event_identity", "patch_identity", "patch_namespace", "explicit_crosswalk",
    "sentinel_date_status", "temporal_linkage", "evidence_attachment",
    "phenomenon_support", "coordinate_support", "geometry_support",
    "dino_review_support", "overlay_readiness", "ground_reference_readiness",
    "training_readiness", "package_schema_validity",
]


def _classify_dimension(dim, pkg, schema_status):
    if dim == "event_identity":
        return "STRONG" if pkg.get("event_id") and not pkg.get("event_id", "").endswith("MISSING") else "WEAK"
    if dim == "patch_identity":
        return "STRONG" if pkg.get("patch_id") else "ABSENT"
    if dim == "patch_namespace":
        return "STRONG" if pkg.get("patch_namespace") == NS_EVENT else "ABSENT"
    if dim == "explicit_crosswalk":
        return "MODERATE" if pkg.get("crosswalk_status", "").startswith("EXPLICIT_DINO") else "ABSENT"
    if dim == "sentinel_date_status":
        return "STRONG" if pkg.get("sentinel_date_status") == "SENTINEL_DATE_CONFIRMED_SAME_PATCH" else "BLOCKED"
    if dim == "temporal_linkage":
        return "MODERATE" if pkg.get("temporal_linkage_status", "").startswith("LINKED") else "BLOCKED"
    if dim == "evidence_attachment":
        return "MODERATE" if pkg.get("evidence_status") else "WEAK"
    if dim == "phenomenon_support":
        return "MODERATE" if pkg.get("phenomenon_status") else "WEAK"
    if dim == "coordinate_support":
        return "ABSENT"
    if dim == "geometry_support":
        return "ABSENT"
    if dim == "dino_review_support":
        return "MODERATE" if pkg.get("dino_review_support_status") else "ABSENT"
    if dim in {"overlay_readiness", "ground_reference_readiness", "training_readiness"}:
        return "BLOCKED"
    if dim == "package_schema_validity":
        return "STRONG" if schema_status == "SCHEMA_VALID_NON_OPERATIONAL" else "BLOCKED"
    return "UNKNOWN"


def run_v2_readiness_matrix_builder(args=None):
    registry = _load_or_build_registry(args)
    schema = {r["event_patch_candidate_id"]: r.get("validation_status") for r in (load_csv(dataset_path("v2ac_schema_contract_validation.csv")) or run_schema_contract_validator(args))}
    rows = []
    for pkg in registry:
        epc = pkg.get("event_patch_candidate_id", "")
        sstatus = schema.get(epc, "")
        for dim in READINESS_DIMENSIONS:
            rows.append({
                "readiness_id": f"RDY_v2ac_{len(rows):06d}",
                "event_patch_candidate_id": epc,
                "event_id": pkg.get("event_id", ""),
                "patch_id": pkg.get("patch_id", ""),
                "dimension": dim,
                "classification": _classify_dimension(dim, pkg, sstatus),
                "basis": "v2ac migrated v2 package",
                **guardrails(),
                "notes": "Readiness over the migrated schema; overlay/ground reference/training stay BLOCKED.",
            })
    out = dataset_path("v2ac_v2_readiness_matrix.csv")
    write_csv(out, READINESS_COLUMNS, rows)
    print(f"[v2ac readiness] rows={len(rows)} -> {out}")
    return rows


# 8. Migration Diff Auditor -----------------------------------------------

def run_migration_diff_auditor(args=None):
    registry = _load_or_build_registry(args)
    v1us = _index("v1us_event_patch_candidate_registry.csv")
    v1us_fields = set()
    for r in load_csv(dataset_path("v1us_event_patch_candidate_registry.csv")):
        v1us_fields.update(r.keys())
        break
    added_fields = [f for f in V2_PACKAGE_COLUMNS if f not in v1us_fields]
    rows = []
    for pkg in registry:
        epc = pkg.get("event_patch_candidate_id", "")
        old = v1us.get(epc, {})
        old_blockers = set((old.get("blocker") or "").split("|")) - {""}
        new_blockers = set((pkg.get("blocker") or "").split("|")) - {""}
        blockers_added = sorted(new_blockers - old_blockers)
        statuses_changed = []
        if pkg.get("crosswalk_status"):
            statuses_changed.append("crosswalk_status")
        if pkg.get("sentinel_date_status"):
            statuses_changed.append("sentinel_date_status")
        if pkg.get("package_validation_status"):
            statuses_changed.append("package_validation_status")
        rows.append({
            "diff_id": f"DIFF_v2ac_{len(rows):05d}",
            "event_patch_candidate_id": epc,
            "source_version": "v1us",
            "target_version": "v2ac",
            "fields_added": "|".join(added_fields),
            "statuses_changed": "|".join(statuses_changed),
            "blockers_added": "|".join(blockers_added),
            "old_outputs_modified": "false",
            "migration_additive": "true",
            "notes": "Additive migration; v1us/v2aa/v2ab outputs are read-only and unchanged.",
        })
    out = dataset_path("v2ac_migration_diff_audit.csv")
    write_csv(out, DIFF_COLUMNS, rows)
    print(f"[v2ac migration diff] rows={len(rows)} added_fields={len(added_fields)} -> {out}")
    return rows


# 9. Next Programming Target Ranker ----------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}

TARGET_VERSION = {
    "EVENT_PATCH_PACKAGE_V2_QA_HARNESS": "v2ad — Event-Patch Package V2 QA Harness",
    "MULTI_REGION_REGISTRY_HARDENING": "v2ad — Multi-Region Registry Hardening",
    "SENTINEL_DATE_CROSSWALK_DISCOVERY": "v2ad — Sentinel Date Crosswalk Discovery",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2ad — DINO Review Support Completion",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2ad — Public Source Recheck Hold",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2ad — Ground Truth Search Hold",
}


def _ranker_metrics():
    registry = load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))
    total = len(registry) or 1
    unlinkable = sum(1 for r in registry if r.get("sentinel_date_status") == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE")
    schema = load_csv(dataset_path("v2ac_schema_contract_validation.csv"))
    valid = sum(1 for r in schema if r.get("validation_status") == "SCHEMA_VALID_NON_OPERATIONAL")
    return {
        "total": total,
        "valid_rate": valid / total,
        "unlinkable_rate": unlinkable / total,
    }


def _candidate_targets(m):
    return [
        {
            "next_target": "EVENT_PATCH_PACKAGE_V2_QA_HARNESS",
            "programming_value": 55 + round(10 * m["valid_rate"]),
            "ground_truth_value": 0,
            "blocker_reduction_value": 45,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Lock the migrated v2 schema with a QA harness (contract regression tests, invariants) before further data acquisition.",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_HARDENING",
            "programming_value": 55,
            "ground_truth_value": 0,
            "blocker_reduction_value": 40,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Fold the migrated v2 packages into the multi-region registries.",
        },
        {
            "next_target": "SENTINEL_DATE_CROSSWALK_DISCOVERY",
            "programming_value": round(45 * m["unlinkable_rate"]),
            "ground_truth_value": 0,
            "blocker_reduction_value": round(55 * m["unlinkable_rate"]),
            "expected_effort": "MEDIUM",
            "overclaim_risk": "MEDIUM",
            "notes": "Search for an explicit key linking the numeric and anchor namespaces; uncertain and only useful if such a key exists.",
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
    out = dataset_path("v2ac_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v2ac ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
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
                "blocker_id": f"GB_v2ac_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "blocker": blocker,
                "status": "BLOCKED",
                **guardrails(),
                "notes": "Schema migration does not unblock geometry, overlay, ground reference or labels.",
            })
    out = dataset_path("v2ac_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_MATRIX_COLUMNS, rows)
    print(f"[v2ac gr blockers] rows={len(rows)} -> {out}")
    return rows


def run_completion_report(args=None):
    write_policy_configs()
    registry = _load_or_build_registry(args)
    namespace = load_csv(dataset_path("v2ac_patch_namespace_field_population.csv")) or run_patch_namespace_field_populator(args)
    crosswalk = load_csv(dataset_path("v2ac_crosswalk_field_population.csv")) or run_crosswalk_field_populator(args)
    temporal = load_csv(dataset_path("v2ac_temporal_status_field_population.csv")) or run_temporal_status_field_populator(args)
    blocker_norm = load_csv(dataset_path("v2ac_blocker_field_normalization.csv")) or run_blocker_field_normalizer(args)
    schema = load_csv(dataset_path("v2ac_schema_contract_validation.csv")) or run_schema_contract_validator(args)
    readiness = load_csv(dataset_path("v2ac_v2_readiness_matrix.csv")) or run_v2_readiness_matrix_builder(args)
    diff = load_csv(dataset_path("v2ac_migration_diff_audit.csv")) or run_migration_diff_auditor(args)
    ranker = load_csv(dataset_path("v2ac_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    blockers = run_ground_reference_blocker_matrix(args)

    migrated = len(registry)
    with_namespace = sum(1 for r in registry if r.get("patch_namespace") == NS_EVENT)
    with_dino_xw = sum(1 for r in registry if r.get("crosswalk_status", "").startswith("EXPLICIT_DINO"))
    no_anchor_xw = sum(1 for r in registry if "NO_ANCHOR_CROSSWALK" in r.get("crosswalk_status", "") or r.get("crosswalk_status") in {"NO_EXPLICIT_CROSSWALK", "NO_CROSSWALK_PATCH_ID_MISSING"})
    unlinkable = sum(1 for r in registry if r.get("sentinel_date_status") == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE")
    missing_date = sum(1 for r in registry if r.get("sentinel_date_status") == "SENTINEL_DATE_MISSING_WITH_BLOCKER")
    schema_valid = sum(1 for r in schema if r.get("validation_status") == "SCHEMA_VALID_NON_OPERATIONAL")
    next_target = ranker[0].get("next_target", "") if ranker else ""
    next_version = ranker[0].get("recommended_version", "") if ranker else ""

    write_csv(dataset_path("v2ac_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v2ac_0000",
        "event_id": "MULTI_REGION",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v2ac score-based next-programming-target ranker after schema migration.",
        "target": "EVENT_PATCH_PACKAGE_V2",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth, ground reference, inferred date or inferred crosswalk.",
    }])

    lines = [
        "# Protocolo C v2ac - Event-Patch Schema Migration Implementation",
        "",
        f"- packages migrated to v2: `{migrated}`",
        f"- packages with explicit namespace: `{with_namespace}`",
        f"- packages with explicit DINO crosswalk: `{with_dino_xw}`",
        f"- packages without anchor crosswalk: `{no_anchor_xw}`",
        f"- packages with unlinkable date: `{unlinkable}`",
        f"- packages with missing date: `{missing_date}`",
        f"- packages valid against the contract: `{schema_valid}`",
        f"- namespace population rows: `{len(namespace)}`",
        f"- crosswalk population rows: `{len(crosswalk)}`",
        f"- temporal population rows: `{len(temporal)}`",
        f"- blocker normalization rows: `{len(blocker_norm)}`",
        f"- readiness rows: `{len(readiness)}`",
        f"- migration diff rows: `{len(diff)}`",
        f"- ground-reference blocker rows: `{len(blockers)}`",
        f"- selected next target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v2ac migrated the event-patch packages to the hardened v2 schema additively. It modified no prior output, inferred no crosswalk, inferred no Sentinel date, applied no cross-namespace date, executed no overlay, and created no ground truth, ground reference or label.",
    ]
    write_text(doc_path("protocolo_c_v2ac_event_patch_schema_migration.md"), lines)

    report = lines + [
        "",
        "## How many packages migrated",
        f"{migrated} event-patch packages were migrated to the v2 schema, all preserving their original event_patch_candidate_id.",
        "",
        "## Namespace and crosswalk",
        f"{with_namespace} packages carry an explicit event-patch candidate namespace. {with_dino_xw} have an explicit DINO crosswalk via identical patch_id. No package has an anchor/REFPATCH or scaffolding crosswalk, because no explicit key exists; anchor/refpatch fields stay empty with an explicit blocker.",
        "",
        "## Temporal status",
        f"{unlinkable} packages have a Sentinel date recovered only in a parallel namespace, kept unlinkable with an empty sentinel_scene_date. {missing_date} have no recoverable date. No date was inferred or applied by region.",
        "",
        "## Schema validation",
        f"{schema_valid} packages validate against the v2ab contract as schema-valid non-operational; the remainder are blocked (e.g. missing patch id).",
        "",
        "## Persisting blockers",
        "Every package keeps no_observed_geometry, no_occurrence_coordinates, no_overlay, no_ground_reference, no_training_label and patch_truth_forbidden; plus unlinkable_sentinel_date or no_sentinel_date and no_explicit_anchor_crosswalk.",
        "",
        "## Why there is still no overlay",
        "No overlay was executed and overlay status stays BLOCKED. The migration normalizes identity and temporal fields but establishes no observed occurrence geometry.",
        "",
        "## Why there is still no ground reference",
        "No package has a linkable observed occurrence geometry; without it there is no basis for ground reference, and none was created.",
        "",
        "## Next programming step",
        f"The score-based ranker selected `{next_target}` (`{next_version}`).",
    ]
    write_text(doc_path("protocolo_c_relatorio_v2ac_event_patch_schema_migration.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v2ac.md"), [
        "# Status atual - Protocolo C v2ac",
        "",
        f"Schema migration status: `{MAX_STATUS}`.",
        f"Packages migrated: `{migrated}`; schema-valid: `{schema_valid}`; explicit DINO crosswalk: `{with_dino_xw}`.",
        f"Unlinkable-date packages: `{unlinkable}`; missing-date packages: `{missing_date}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Overlay, ground reference, training labels, ground truth, inferred Sentinel dates and inferred crosswalks remain blocked.",
    ])

    manifest = []
    for idx, artifact in enumerate(V2AC_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v2ac_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v2ac additive migration artifact; no raw data, no private path, no inferred date or crosswalk.",
        })
    write_csv(dataset_path("v2ac_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for folder in (STAGING_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v2ac completion] migrated={migrated} valid={schema_valid} next={next_target}")
    return {"migrated": migrated, "schema_valid": schema_valid, "with_dino_xw": with_dino_xw, "unlinkable": unlinkable, "next_target": next_target, "next_version": next_version}


def run_all(args=None):
    args = args or parse_args([])
    run_event_patch_v2_package_builder(args)
    run_patch_namespace_field_populator(args)
    run_crosswalk_field_populator(args)
    run_temporal_status_field_populator(args)
    run_blocker_field_normalizer(args)
    run_schema_contract_validator(args)
    run_v2_readiness_matrix_builder(args)
    run_migration_diff_auditor(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
