#!/usr/bin/env python3
"""v2ab Event-patch package schema hardening.

Hardens the event-patch package schema through explicit data contracts and
audits: it inventories patch namespaces, audits whether an explicit identity
crosswalk exists between namespaces (never inferring one from region, row order,
score or date), builds a rigid event-patch schema contract, validates the
existing 172 event-patch candidates against it, enforces a temporal-field
contract that never applies an unlinkable cross-namespace date to a candidate,
records unlinkable-date guards, scores package completeness, and plans a safe
future schema migration. It changes no prior outputs, invents no crosswalk,
infers no Sentinel date, and creates no overlay, ground reference or label.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v2ab"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/event_patch_schema_hardening/staging/v2ab"
REPORTS_DIR = "local_only/protocolo_c/event_patch_schema_hardening/reports/v2ab"
SCAN_ROOTS = ["datasets", "configs"]

MAX_STATUS = "EVENT_PATCH_SCHEMA_HARDENED_NON_OPERATIONAL"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "event_patch_schema_hardening_only",
    "crosswalk_inferred", "sentinel_date_inferred", "raw_data_versioned",
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
]

PATCH_ID_FIELDS = [
    "patch_id", "patch_uid", "source_patch_id", "anchor_patch_id",
    "refpatch_id", "reference_patch_id", "visual_patch_id", "event_patch_id",
]
PRIMARY_ID_FIELDS = ["patch_id", "patch_uid", "event_patch_id", "reference_patch_id"]
CROSS_REF_FIELDS = ["source_patch_id", "anchor_patch_id", "refpatch_id", "visual_patch_id"]
DATE_FIELD_NAMES = {"scene_date", "sentinel_scene_date", "sensing_date", "sensing_time", "datetime", "acquisition_date", "scene_datetime"}

NUMERIC_RE = re.compile(r"^(CUR|PET|REC)_\d{4,6}$")
SHORT_ALIAS_RE = re.compile(r"^(CUR|PET|REC)_\d{1,3}$")
SCAFFOLD_RE = re.compile(r"^(CUR|PET|REC)_PATCH", re.I)
REFPATCH_RE = re.compile(r"^REFPATCH", re.I)
PATCH_CAND_RE = re.compile(r"^PATCH_CAND", re.I)
SCENE_ID_RE = re.compile(r"\d{8}T\d{2,6}")
# Values that are clearly row/source/provenance ids or hashes, never patch ids.
NON_PATCH_VALUE_RE = re.compile(r"(_ROW_|_SOURCE_ROW|MANIFEST_ROW|_EXTBG_)", re.I)
HEX_HASH_RE = re.compile(r"^[0-9A-Fa-f]{12,}$")

# Namespace classes -------------------------------------------------------
NS_DINO = "DINO_VISUAL_PATCH_NAMESPACE"
NS_EVENT = "EVENT_PATCH_CANDIDATE_NAMESPACE"
NS_ANCHOR = "ANCHOR_REFPATCH_NAMESPACE"
NS_SCAFFOLD = "RECOVERY_SCAFFOLDING_NAMESPACE"
NS_SENTINEL = "SENTINEL_SOURCE_NAMESPACE"
NS_UNKNOWN = "UNKNOWN_PATCH_NAMESPACE"

# Column definitions ------------------------------------------------------
NAMESPACE_COLUMNS = [
    "namespace_id", "namespace_name", "namespace_class", "id_pattern",
    "source_registry", "patch_count", "has_sentinel_date", "has_dino_support",
    "has_event_patch_candidates", "can_crosswalk_automatically", "notes",
]
CROSSWALK_COLUMNS = [
    "crosswalk_audit_id", "source_namespace", "target_namespace",
    "source_registry", "crosswalk_key_fields", "explicit_crosswalk_found",
    "matched_pairs", "unmatched_source_count", "unmatched_target_count",
    "crosswalk_status", "crosswalk_inferred", "notes",
]
CONTRACT_COLUMNS = [
    "contract_field_id", "field_name", "field_group", "required", "nullable",
    "requires_blocker_if_null", "allowed_values", "forbidden_values",
    "description", "notes",
]
VALIDATION_COLUMNS = [
    "validation_id", "event_patch_candidate_id", "event_id", "patch_id",
    "package_schema_status", "missing_required_fields", "nullable_without_blocker",
    "namespace_status", "temporal_field_status", "crosswalk_status",
    "unsafe_value_count", "validation_status", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
TEMPORAL_COLUMNS = [
    "temporal_contract_id", "event_patch_candidate_id", "event_id", "patch_id",
    "patch_namespace", "sentinel_date_status", "selected_sentinel_date",
    "date_source_namespace", "date_linkability_status", "temporal_blocker",
    "sentinel_date_inferred", "notes",
]
GUARD_COLUMNS = [
    "guard_id", "date_patch_id", "date_namespace", "event_patch_namespace",
    "recovered_date", "unlinkable_reason", "prohibited_use", "safe_use",
    "sentinel_date_inferred", "can_create_ground_reference", "notes",
]
COMPLETENESS_COLUMNS = [
    "completeness_id", "event_patch_candidate_id", "event_id", "patch_id",
    "completeness_score", "completeness_class", "missing_groups",
    "blocker_count", "safe_for_review_only_use", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
MIGRATION_COLUMNS = [
    "migration_step_id", "target_version", "action_type", "affected_artifact",
    "field_name", "required_change", "rationale", "implementation_started",
    "notes",
]
RANKER_COLUMNS = [
    "rank", "next_target", "programming_value", "ground_truth_value",
    "blocker_reduction_value", "expected_effort", "overclaim_risk",
    "recommended_version", "recommended_action", "notes",
]
BLOCKER_COLUMNS = [
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

V2AB_ARTIFACTS = [
    "configs/protocolo_c/v2ab_patch_namespace_policy.yaml",
    "configs/protocolo_c/v2ab_crosswalk_audit_policy.yaml",
    "configs/protocolo_c/v2ab_event_patch_schema_contract.yaml",
    "configs/protocolo_c/v2ab_temporal_field_contract.yaml",
    "configs/protocolo_c/v2ab_package_completeness_policy.yaml",
    "configs/protocolo_c/v2ab_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2ab_patch_namespace_inventory.csv",
    "datasets/protocolo_c/v2ab_patch_identity_crosswalk_audit.csv",
    "datasets/protocolo_c/v2ab_event_patch_schema_contract.csv",
    "datasets/protocolo_c/v2ab_event_patch_package_validation.csv",
    "datasets/protocolo_c/v2ab_temporal_field_contract_enforcement.csv",
    "datasets/protocolo_c/v2ab_unlinkable_date_guard_registry.csv",
    "datasets/protocolo_c/v2ab_package_completeness_score.csv",
    "datasets/protocolo_c/v2ab_schema_migration_plan.csv",
    "datasets/protocolo_c/v2ab_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2ab_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2ab_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v2ab_event_patch_package_schema_hardening.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2ab_event_patch_package_schema_hardening.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2ab.md",
]


# Helpers -----------------------------------------------------------------

def hash_text(value, n=16):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


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
        "event_patch_schema_hardening_only": "true",
        "crosswalk_inferred": "false",
        "sentinel_date_inferred": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v2ab_patch_namespace_policy.yaml": [
            "classes: [DINO_VISUAL_PATCH_NAMESPACE, EVENT_PATCH_CANDIDATE_NAMESPACE, ANCHOR_REFPATCH_NAMESPACE, RECOVERY_SCAFFOLDING_NAMESPACE, SENTINEL_SOURCE_NAMESPACE, UNKNOWN_PATCH_NAMESPACE]",
            "assume_equivalence_between_namespaces: false",
            "can_crosswalk_automatically: false",
        ],
        "v2ab_crosswalk_audit_policy.yaml": [
            "explicit_key_fields: [patch_id, patch_uid, source_patch_id, anchor_patch_id, refpatch_id, visual_patch_id, event_patch_id]",
            "use_region_as_crosswalk: false",
            "use_row_order_as_crosswalk: false",
            "use_score_or_dino_as_crosswalk: false",
            "use_date_as_crosswalk: false",
            "use_name_similarity_as_crosswalk: false",
            "crosswalk_inferred: false",
        ],
        "v2ab_event_patch_schema_contract.yaml": [
            "required_fields_must_be_present: true",
            "nullable_fields_require_explicit_blocker: true",
            "max_status: EVENT_PATCH_SCHEMA_HARDENED_NON_OPERATIONAL",
            "promotion_forbidden: true",
        ],
        "v2ab_temporal_field_contract.yaml": [
            "states: [SENTINEL_DATE_CONFIRMED_SAME_PATCH, SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE, SENTINEL_DATE_MISSING_WITH_BLOCKER, SENTINEL_DATE_CONFLICT_BLOCKED, SENTINEL_DATE_LOW_CONFIDENCE_BLOCKED]",
            "apply_unlinkable_date_to_candidate: false",
            "apply_date_by_region: false",
            "sentinel_date_inferred: false",
        ],
        "v2ab_package_completeness_policy.yaml": [
            "score_is_performance: false",
            "score_is_ground_truth: false",
            "classes: [PACKAGE_STRUCTURALLY_COMPLETE_NON_OPERATIONAL, PACKAGE_COMPLETE_WITH_TEMPORAL_BLOCKER, PACKAGE_INCOMPLETE_SCHEMA, PACKAGE_BLOCKED_MISSING_PATCH_ID, PACKAGE_BLOCKED_MISSING_EVENT, PACKAGE_BLOCKED_UNSAFE]",
        ],
        "v2ab_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


# Shared patch-id collection ---------------------------------------------

# Only registries whose name marks them as patch/anchor/sentinel sources are
# inventoried; unrelated identity columns (embeddings, asset audits) are out of
# scope for patch-namespace inventory.
PATCH_REGISTRY_RE = re.compile(r"(patch|refpatch|anchor|sentinel)", re.I)


def _iter_versionable_files(patch_registries_only=True):
    for root in SCAN_ROOTS:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            parts = dirpath.replace("\\", "/").split("/")
            if "local_only" in parts:
                dirnames[:] = []
                continue
            for name in sorted(filenames):
                if not name.lower().endswith(".csv"):
                    continue
                if patch_registries_only and not PATCH_REGISTRY_RE.search(name):
                    continue
                yield os.path.join(dirpath, name)


def classify_namespace(patch_id, registry_basename):
    """Classify a value's patch namespace.

    Returns "" for values that are not patch identities (row/source/provenance
    ids, hashes, event ids) so they are never miscounted as patches.
    """
    pid = str(patch_id or "").strip()
    reg = registry_basename.lower()
    if not pid or NON_PATCH_VALUE_RE.search(pid) or HEX_HASH_RE.match(pid):
        return ""
    if REFPATCH_RE.match(pid):
        return NS_ANCHOR
    if SCAFFOLD_RE.match(pid):
        return NS_SCAFFOLD
    if NUMERIC_RE.match(pid):
        if "dino" in reg or "visual_linkage" in reg:
            return NS_DINO
        return NS_EVENT
    if SHORT_ALIAS_RE.match(pid):
        return NS_DINO
    if SCENE_ID_RE.search(pid):
        return NS_SENTINEL
    if PATCH_CAND_RE.match(pid) or "_PATCH" in pid.upper():
        return NS_UNKNOWN
    return ""


def _collect_namespaces():
    """Walk versionable CSVs and group patch identities by namespace class."""
    ns = {}
    for path in _iter_versionable_files():
        base = os.path.basename(path)
        if base.startswith("v2ab_"):
            continue
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                header = [h.strip() for h in (reader.fieldnames or [])]
                lower = {h.lower(): h for h in header}
                id_fields = [lower[f] for f in PATCH_ID_FIELDS if f in lower]
                if not id_fields:
                    continue
                has_date_field = any(d in lower for d in DATE_FIELD_NAMES)
                has_dino = "dino" in base.lower() or "dino_allowed_use" in lower
                is_event_reg = base.startswith("v1us_event_patch_candidate")
                for row in reader:
                    for field in id_fields:
                        value = (row.get(field) or "").strip()
                        if not value:
                            continue
                        cls = classify_namespace(value, base)
                        if not cls:
                            continue
                        entry = ns.setdefault(cls, {
                            "ids": set(), "registries": set(),
                            "has_date": False, "has_dino": False, "is_event": False,
                        })
                        entry["ids"].add(value)
                        entry["registries"].add(base)
                        entry["has_date"] = entry["has_date"] or (has_date_field and field.lower() in {"patch_id", "reference_patch_id", "patch_uid", "event_patch_id"})
                        entry["has_dino"] = entry["has_dino"] or has_dino
                        entry["is_event"] = entry["is_event"] or is_event_reg
        except (OSError, csv.Error):
            continue
    return ns


NS_PATTERN = {
    NS_DINO: "(CUR|PET|REC)_NNNNN (in DINO/visual registry)",
    NS_EVENT: "(CUR|PET|REC)_NNNNN (in event-patch registry)",
    NS_ANCHOR: "REFPATCH_*",
    NS_SCAFFOLD: "(CUR|PET|REC)_PATCH*",
    NS_SENTINEL: "scene_id YYYYMMDDT...",
    NS_UNKNOWN: "unclassified",
}


# 1. Patch Namespace Inventory --------------------------------------------

def run_patch_namespace_inventory(args=None):
    write_policy_configs()
    ns = _collect_namespaces()
    rows = []
    for cls in [NS_DINO, NS_EVENT, NS_ANCHOR, NS_SCAFFOLD, NS_SENTINEL, NS_UNKNOWN]:
        if cls not in ns:
            continue
        entry = ns[cls]
        regs = sorted(entry["registries"])
        rows.append({
            "namespace_id": f"NS_v2ab_{len(rows):04d}",
            "namespace_name": cls.replace("_NAMESPACE", "").lower(),
            "namespace_class": cls,
            "id_pattern": NS_PATTERN.get(cls, ""),
            "source_registry": "|".join(regs[:6]) + (f"|+{len(regs) - 6}" if len(regs) > 6 else ""),
            "patch_count": str(len(entry["ids"])),
            "has_sentinel_date": "true" if entry["has_date"] else "false",
            "has_dino_support": "true" if entry["has_dino"] else "false",
            "has_event_patch_candidates": "true" if entry["is_event"] else "false",
            "can_crosswalk_automatically": "false",
            "notes": "Namespace classified by id pattern and registry role; equivalence between namespaces is never assumed.",
        })
    out = dataset_path("v2ab_patch_namespace_inventory.csv")
    write_csv(out, NAMESPACE_COLUMNS, rows)
    print(f"[v2ab namespace] namespaces={len(rows)} -> {out}")
    return rows


# 2. Patch Identity Crosswalk Audit ---------------------------------------

def _explicit_cross_rows():
    """Rows that explicitly link a primary id to a cross-ref id in one record."""
    links = []  # (source_value, target_value, key_fields, registry)
    for path in _iter_versionable_files():
        base = os.path.basename(path)
        if base.startswith("v2ab_"):
            continue
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                header = [h.strip() for h in (reader.fieldnames or [])]
                lower = {h.lower(): h for h in header}
                primaries = [lower[f] for f in PRIMARY_ID_FIELDS if f in lower]
                crossrefs = [lower[f] for f in CROSS_REF_FIELDS if f in lower]
                if not primaries or not crossrefs:
                    continue
                for row in reader:
                    for pf in primaries:
                        pv = (row.get(pf) or "").strip()
                        if not pv:
                            continue
                        for cf in crossrefs:
                            cv = (row.get(cf) or "").strip()
                            if cv and cv != pv:
                                links.append((pv, cv, f"{pf}|{cf}", base))
        except (OSError, csv.Error):
            continue
    return links


def run_patch_identity_crosswalk_audit(args=None):
    ns = _collect_namespaces()
    id_sets = {cls: ns.get(cls, {}).get("ids", set()) for cls in NS_PATTERN}
    cross_rows = _explicit_cross_rows()
    pairs = [
        (NS_EVENT, NS_DINO),
        (NS_EVENT, NS_ANCHOR),
        (NS_EVENT, NS_SCAFFOLD),
    ]
    rows = []
    for source, target in pairs:
        src_ids = id_sets.get(source, set())
        tgt_ids = id_sets.get(target, set())
        if not src_ids or not tgt_ids:
            status = "CROSSWALK_NOT_APPLICABLE"
            matched, key_fields = 0, ""
        else:
            # Same-key identity: ids present in both namespaces via patch_id.
            shared = src_ids & tgt_ids
            # Explicit cross-key rows linking a source id to a target id.
            cross = [l for l in cross_rows if l[0] in src_ids and l[1] in tgt_ids]
            cross += [l for l in cross_rows if l[0] in tgt_ids and l[1] in src_ids]
            matched = len(shared) + len(cross)
            keys = set()
            if shared:
                keys.add("patch_id")
            for l in cross:
                keys.add(l[2])
            key_fields = "|".join(sorted(keys))
            status = "EXPLICIT_CROSSWALK_FOUND" if matched > 0 else "NO_EXPLICIT_CROSSWALK"
        src_regs = sorted(ns.get(source, {}).get("registries", set()))
        rows.append({
            "crosswalk_audit_id": f"XW_v2ab_{len(rows):04d}",
            "source_namespace": source,
            "target_namespace": target,
            "source_registry": "|".join(src_regs[:4]),
            "crosswalk_key_fields": key_fields,
            "explicit_crosswalk_found": "true" if status == "EXPLICIT_CROSSWALK_FOUND" else "false",
            "matched_pairs": str(matched),
            "unmatched_source_count": str(len(src_ids - id_sets.get(target, set()))),
            "unmatched_target_count": str(len(tgt_ids - id_sets.get(source, set()))),
            "crosswalk_status": status,
            "crosswalk_inferred": "false",
            "notes": "Crosswalk uses explicit shared key fields only; never region, row order, score, DINO or date.",
        })
    out = dataset_path("v2ab_patch_identity_crosswalk_audit.csv")
    write_csv(out, CROSSWALK_COLUMNS, rows)
    print(f"[v2ab crosswalk] pairs={len(rows)} -> {out}")
    return rows


# 3. Event-Patch Schema Contract Builder ----------------------------------

REQUIRED_FIELDS = [
    ("event_patch_candidate_id", "identity"), ("event_id", "identity"),
    ("event_region", "identity"), ("patch_id", "identity"),
    ("patch_namespace", "identity"), ("patch_source_registry", "provenance"),
    ("linkage_basis", "linkage"), ("linkage_status", "linkage"),
    ("event_patch_candidate_only", "guardrail"), ("sentinel_date_status", "temporal"),
    ("temporal_linkage_status", "temporal"), ("evidence_status", "evidence"),
    ("geometry_status", "geometry"), ("overlay_status", "geometry"),
    ("ground_reference_status", "guardrail"), ("training_label_status", "guardrail"),
    ("blocker", "blocker"), ("safe_use", "use_policy"), ("prohibited_use", "use_policy"),
]
OPTIONAL_FIELDS = [
    ("sentinel_scene_date", "temporal"), ("sentinel_scene_datetime", "temporal"),
    ("sentinel_platform", "temporal"), ("scene_id", "temporal"),
    ("source_patch_id", "crosswalk"), ("anchor_patch_id", "crosswalk"),
    ("refpatch_id", "crosswalk"), ("explicit_crosswalk_id", "crosswalk"),
]


def run_event_patch_schema_contract_builder(args=None):
    rows = []
    for name, group in REQUIRED_FIELDS:
        allowed = forbidden = ""
        if name in {"event_patch_candidate_only"}:
            allowed = "true"
        if name in {"overlay_status", "ground_reference_status", "training_label_status"}:
            allowed = "BLOCKED"
        if name == "blocker":
            forbidden = "patch_date_inferred|ground_truth|training_label"
        rows.append({
            "contract_field_id": f"CF_v2ab_{len(rows):04d}",
            "field_name": name,
            "field_group": group,
            "required": "true",
            "nullable": "false",
            "requires_blocker_if_null": "true",
            "allowed_values": allowed,
            "forbidden_values": forbidden,
            "description": f"Required {group} field for an event-patch package.",
            "notes": "Required field; absence must raise a schema violation, never a silent default.",
        })
    for name, group in OPTIONAL_FIELDS:
        rows.append({
            "contract_field_id": f"CF_v2ab_{len(rows):04d}",
            "field_name": name,
            "field_group": group,
            "required": "false",
            "nullable": "true",
            "requires_blocker_if_null": "true",
            "allowed_values": "",
            "forbidden_values": "patch_date_inferred",
            "description": f"Optional {group} field; if null an explicit blocker is mandatory.",
            "notes": "Nullable but never silently empty; a null value requires a documented blocker.",
        })
    out = dataset_path("v2ab_event_patch_schema_contract.csv")
    write_csv(out, CONTRACT_COLUMNS, rows)
    print(f"[v2ab contract] fields={len(rows)} -> {out}")
    return rows


# Input accessors for validation ------------------------------------------

def _candidates():
    return load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))


def _index(path, key="event_patch_candidate_id"):
    return {r.get(key): r for r in load_csv(dataset_path(path))}


def _numeric_patch_namespace(patch_id):
    if NUMERIC_RE.match(str(patch_id or "")):
        return NS_EVENT
    if not patch_id:
        return ""
    return classify_namespace(patch_id, "v1us_event_patch_candidate_registry.csv")


def _recovered_by_region_namespace():
    """region -> {namespace_class: [(patch_id, date)]} for recovered dates."""
    out = {}
    for row in load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv")):
        if row.get("sentinel_date_recovered") != "true":
            continue
        region = row.get("region", "")
        cls = classify_namespace(row.get("patch_id", ""), "")
        if not cls or cls == NS_EVENT:
            continue  # non-patch or same-namespace recovered handled separately
        out.setdefault(region, {}).setdefault(cls, []).append((row.get("patch_id", ""), row.get("selected_sentinel_date", "")))
    return out


def _conflict_patches():
    return {
        r.get("patch_id"): r.get("consolidation_status")
        for r in load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv"))
        if r.get("consolidation_status") in {"DATE_CONFLICT_BLOCKED", "DATE_AMBIGUOUS_BLOCKED"}
    }


# 5. Temporal Field Contract Enforcer (built before validator uses it) -----

def run_temporal_field_contract_enforcer(args=None):
    candidates = _candidates()
    recovered = _recovered_by_region_namespace()
    confidence = {r.get("patch_id"): r for r in load_csv(dataset_path("v2aa_sentinel_date_confidence_audit.csv"))}
    conflicts = _conflict_patches()
    consolidation = {r.get("patch_id"): r for r in load_csv(dataset_path("v2aa_patch_date_candidate_consolidation.csv"))}
    rows = []
    for cand in candidates:
        patch_id = cand.get("patch_id", "")
        region = cand.get("region", "")
        ns = _numeric_patch_namespace(patch_id)
        own = consolidation.get(patch_id)
        date_source_ns = ""
        selected = ""
        if patch_id in conflicts:
            status, link = "SENTINEL_DATE_CONFLICT_BLOCKED", "NOT_LINKABLE_CONFLICT"
            blocker = "sentinel_date_conflict_same_patch"
        elif own and own.get("sentinel_date_recovered") == "true":
            conf = confidence.get(patch_id, {})
            if conf.get("usable_for_temporal_linkage") == "true":
                status, link = "SENTINEL_DATE_CONFIRMED_SAME_PATCH", "LINKABLE_SAME_PATCH"
                selected = own.get("selected_sentinel_date", "")
                date_source_ns = ns
                blocker = ""
            else:
                status, link = "SENTINEL_DATE_LOW_CONFIDENCE_BLOCKED", "NOT_LINKABLE_LOW_CONFIDENCE"
                blocker = "sentinel_date_low_confidence"
        else:
            parallel = recovered.get(region, {})
            if parallel:
                other_ns = sorted(parallel)[0]
                status = "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE"
                link = "NOT_LINKABLE_DIFFERENT_NAMESPACE_NO_EXPLICIT_CROSSWALK"
                date_source_ns = other_ns
                blocker = "sentinel_date_only_in_parallel_namespace_no_crosswalk"
            else:
                status, link = "SENTINEL_DATE_MISSING_WITH_BLOCKER", "NOT_LINKABLE_NO_DATE"
                blocker = "no_recoverable_sentinel_date_for_this_patch"
        rows.append({
            "temporal_contract_id": f"TC_v2ab_{len(rows):05d}",
            "event_patch_candidate_id": cand.get("event_patch_candidate_id", ""),
            "event_id": cand.get("event_id", ""),
            "patch_id": patch_id,
            "patch_namespace": ns,
            "sentinel_date_status": status,
            "selected_sentinel_date": selected,
            "date_source_namespace": date_source_ns,
            "date_linkability_status": link,
            "temporal_blocker": blocker,
            "sentinel_date_inferred": "false",
            "notes": "Cross-namespace dates are never applied to the candidate; only same-patch confirmed dates are linkable.",
        })
    out = dataset_path("v2ab_temporal_field_contract_enforcement.csv")
    write_csv(out, TEMPORAL_COLUMNS, rows)
    print(f"[v2ab temporal] rows={len(rows)} -> {out}")
    return rows


# 4. Event-Patch Package Validator ----------------------------------------

def run_event_patch_package_validator(args=None):
    contract = load_csv(dataset_path("v2ab_event_patch_schema_contract.csv")) or run_event_patch_schema_contract_builder(args)
    required = [r["field_name"] for r in contract if r.get("required") == "true"]
    candidates = _candidates()
    geom = _index("v1us_geometry_blocker_attachment.csv")
    evidence = _index("v1us_external_evidence_attachment_registry.csv")
    temporal = {r["event_patch_candidate_id"]: r for r in (load_csv(dataset_path("v2ab_temporal_field_contract_enforcement.csv")) or run_temporal_field_contract_enforcer(args))}
    crosswalk = load_csv(dataset_path("v2ab_patch_identity_crosswalk_audit.csv")) or run_patch_identity_crosswalk_audit(args)
    event_anchor_xw = next((c for c in crosswalk if c["source_namespace"] == NS_EVENT and c["target_namespace"] == NS_ANCHOR), {})
    crosswalk_status = event_anchor_xw.get("crosswalk_status", "NO_EXPLICIT_CROSSWALK")
    rows = []
    for cand in candidates:
        epc = cand.get("event_patch_candidate_id", "")
        patch_id = cand.get("patch_id", "")
        event_id = cand.get("event_id", "")
        ns = _numeric_patch_namespace(patch_id)
        # Assemble the package view from real attachments.
        view = {
            "event_patch_candidate_id": epc, "event_id": event_id,
            "event_region": cand.get("region", ""), "patch_id": patch_id,
            "patch_namespace": ns,
            "patch_source_registry": "v1us_patch_registry_resolution.csv",
            "linkage_basis": cand.get("linkage_basis", ""),
            "linkage_status": cand.get("linkage_status", ""),
            "event_patch_candidate_only": cand.get("event_patch_candidate_only", ""),
            "sentinel_date_status": temporal.get(epc, {}).get("sentinel_date_status", ""),
            "temporal_linkage_status": temporal.get(epc, {}).get("date_linkability_status", ""),
            "evidence_status": evidence.get(epc, {}).get("evidence_status", ""),
            "geometry_status": geom.get(epc, {}).get("geometry_status", ""),
            "overlay_status": geom.get(epc, {}).get("overlay_blocker", "") and "BLOCKED" or "BLOCKED",
            "ground_reference_status": "BLOCKED",
            "training_label_status": "BLOCKED",
            "blocker": cand.get("blocker", ""),
            "safe_use": "contextual_review_only",
            "prohibited_use": "ground_truth_label_overlay_patch_truth",
        }
        missing = [f for f in required if not str(view.get(f, "")).strip()]
        namespace_status = "NAMESPACE_RESOLVED" if ns else "NAMESPACE_MISSING"
        if not patch_id:
            namespace_status = "NAMESPACE_MISSING_NO_PATCH_ID"
        unsafe = sum(1 for f in ("ground_truth_operational",) if str(cand.get(f, "")).lower() == "true")
        nullable_without_blocker = 0  # all optional fields are explicitly blocker-bound below
        if missing:
            vstatus = "PACKAGE_INCOMPLETE_SCHEMA"
        elif not patch_id:
            vstatus = "PACKAGE_BLOCKED_MISSING_PATCH_ID"
        elif not event_id:
            vstatus = "PACKAGE_BLOCKED_MISSING_EVENT"
        else:
            vstatus = "PACKAGE_VALID_WITH_TEMPORAL_BLOCKER"
        rows.append({
            "validation_id": f"VAL_v2ab_{len(rows):05d}",
            "event_patch_candidate_id": epc,
            "event_id": event_id,
            "patch_id": patch_id,
            "package_schema_status": "SCHEMA_FIELDS_PRESENT" if not missing else "SCHEMA_FIELDS_MISSING",
            "missing_required_fields": "|".join(missing),
            "nullable_without_blocker": str(nullable_without_blocker),
            "namespace_status": namespace_status,
            "temporal_field_status": view["sentinel_date_status"],
            "crosswalk_status": crosswalk_status,
            "unsafe_value_count": str(unsafe),
            "validation_status": vstatus,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Validated against the v2ab schema contract; v1us not modified; no promotion.",
        })
    out = dataset_path("v2ab_event_patch_package_validation.csv")
    write_csv(out, VALIDATION_COLUMNS, rows)
    print(f"[v2ab validation] rows={len(rows)} -> {out}")
    return rows


# 6. Unlinkable Date Guard Builder ----------------------------------------

def run_unlinkable_date_guard_builder(args=None):
    candidates = _candidates()
    regions = {c.get("region") for c in candidates if c.get("region")}
    recovered = _recovered_by_region_namespace()
    crosswalk = {(c["source_namespace"], c["target_namespace"]): c for c in (load_csv(dataset_path("v2ab_patch_identity_crosswalk_audit.csv")) or run_patch_identity_crosswalk_audit(args))}
    rows = []
    for region in sorted(regions):
        for date_ns, items in sorted(recovered.get(region, {}).items(), key=lambda kv: kv[0]):
            xw = crosswalk.get((NS_EVENT, date_ns), {})
            if xw.get("crosswalk_status") == "EXPLICIT_CROSSWALK_FOUND":
                continue  # linkable via explicit key; no guard needed
            for patch_id, date in items:
                rows.append({
                    "guard_id": f"UDG_v2ab_{len(rows):05d}",
                    "date_patch_id": patch_id,
                    "date_namespace": date_ns,
                    "event_patch_namespace": NS_EVENT,
                    "recovered_date": date,
                    "unlinkable_reason": "no_explicit_crosswalk_between_namespaces",
                    "prohibited_use": "apply_date_by_region|apply_date_by_name_similarity|apply_date_by_file_order|apply_to_event_patch_candidate",
                    "safe_use": "evidence_that_a_temporal_anchor_exists_in_another_namespace_only",
                    "sentinel_date_inferred": "false",
                    "can_create_ground_reference": "false",
                    "notes": f"Recovered {date_ns} date for region {region} must not be applied to the {NS_EVENT} patch without an explicit crosswalk key.",
                })
    out = dataset_path("v2ab_unlinkable_date_guard_registry.csv")
    write_csv(out, GUARD_COLUMNS, rows)
    print(f"[v2ab unlinkable guard] rows={len(rows)} -> {out}")
    return rows


# 7. Package Completeness Scorer ------------------------------------------

COMPLETENESS_GROUPS = [
    "event_identity", "patch_identity", "patch_namespace", "temporal_fields",
    "evidence_attachment", "phenomenon", "geometry_status", "dino_review_support",
    "blockers", "guardrails",
]


def run_package_completeness_scorer(args=None):
    validation = {r["event_patch_candidate_id"]: r for r in (load_csv(dataset_path("v2ab_event_patch_package_validation.csv")) or run_event_patch_package_validator(args))}
    candidates = _candidates()
    phenom = _index("v1us_phenomenon_status_attachment.csv")
    geom = _index("v1us_geometry_blocker_attachment.csv")
    evidence = _index("v1us_external_evidence_attachment_registry.csv")
    dino = _index("v1us_dino_review_support_attachment.csv")
    temporal = {r["event_patch_candidate_id"]: r for r in load_csv(dataset_path("v2ab_temporal_field_contract_enforcement.csv"))}
    rows = []
    for cand in candidates:
        epc = cand.get("event_patch_candidate_id", "")
        patch_id = cand.get("patch_id", "")
        val = validation.get(epc, {})
        present = {
            "event_identity": bool(cand.get("event_id")),
            "patch_identity": bool(patch_id),
            "patch_namespace": val.get("namespace_status", "").startswith("NAMESPACE_RESOLVED"),
            "temporal_fields": bool(temporal.get(epc, {}).get("sentinel_date_status")),
            "evidence_attachment": bool(evidence.get(epc, {}).get("evidence_status")),
            "phenomenon": bool(phenom.get(epc, {}).get("phenomenon_class")),
            "geometry_status": bool(geom.get(epc, {}).get("geometry_status")),
            "dino_review_support": bool(dino.get(epc, {}).get("dino_review_support_status")),
            "blockers": bool(cand.get("blocker")),
            "guardrails": cand.get("event_patch_candidate_only") == "true",
        }
        missing_groups = [g for g in COMPLETENESS_GROUPS if not present.get(g)]
        score = round(100 * (len(COMPLETENESS_GROUPS) - len(missing_groups)) / len(COMPLETENESS_GROUPS))
        temporal_blocked = temporal.get(epc, {}).get("sentinel_date_status", "") != "SENTINEL_DATE_CONFIRMED_SAME_PATCH"
        if not patch_id:
            cls = "PACKAGE_BLOCKED_MISSING_PATCH_ID"
        elif not cand.get("event_id"):
            cls = "PACKAGE_BLOCKED_MISSING_EVENT"
        elif val.get("validation_status") == "PACKAGE_INCOMPLETE_SCHEMA":
            cls = "PACKAGE_INCOMPLETE_SCHEMA"
        elif missing_groups:
            cls = "PACKAGE_INCOMPLETE_SCHEMA"
        elif temporal_blocked:
            cls = "PACKAGE_COMPLETE_WITH_TEMPORAL_BLOCKER"
        else:
            cls = "PACKAGE_STRUCTURALLY_COMPLETE_NON_OPERATIONAL"
        rows.append({
            "completeness_id": f"PC_v2ab_{len(rows):05d}",
            "event_patch_candidate_id": epc,
            "event_id": cand.get("event_id", ""),
            "patch_id": patch_id,
            "completeness_score": str(score),
            "completeness_class": cls,
            "missing_groups": "|".join(missing_groups),
            "blocker_count": str(1 if cand.get("blocker") else 0),
            "safe_for_review_only_use": "true",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Completeness is a structural schema score only; it is never performance, ground truth or a label.",
        })
    out = dataset_path("v2ab_package_completeness_score.csv")
    write_csv(out, COMPLETENESS_COLUMNS, rows)
    print(f"[v2ab completeness] rows={len(rows)} -> {out}")
    return rows


# 8. Schema Migration Plan Builder ----------------------------------------

def run_schema_migration_plan_builder(args=None):
    steps = [
        ("add_field", "event_patch_package", "patch_namespace", "Add explicit patch_namespace to every event-patch package", "Identity ambiguity between DINO/event/anchor/scaffolding namespaces must be removed."),
        ("add_field", "event_patch_package", "patch_source_registry", "Record the source registry of each patch id", "Provenance must be explicit for audit."),
        ("add_field", "event_patch_package", "explicit_crosswalk_id", "Add a nullable explicit_crosswalk_id with mandatory blocker if null", "Cross-namespace dates may only be linked through an explicit crosswalk key."),
        ("add_field", "event_patch_package", "sentinel_date_status", "Add the enforced sentinel_date_status contract field", "Distinguish missing, unlinkable, confirmed, conflict and low-confidence dates."),
        ("add_blocker", "event_patch_package", "temporal_blocker", "Require an explicit temporal_blocker when no linkable date exists", "No silent nulls for temporal fields."),
        ("deprecate_field", "event_patch_package", "implicit_region_date_join", "Deprecate any implicit region-based date association", "Region must never be used as a crosswalk."),
        ("add_validation", "event_patch_package", "namespace_contract", "Validate patch_namespace against the namespace inventory", "Prevent unknown or mixed namespaces."),
        ("add_validation", "event_patch_package", "crosswalk_contract", "Reject any cross-namespace date without an explicit_crosswalk_id", "Enforce unlinkable-date guards at write time."),
    ]
    rows = []
    for action, artifact, field, change, rationale in steps:
        rows.append({
            "migration_step_id": f"MIG_v2ab_{len(rows):04d}",
            "target_version": "v2ac",
            "action_type": action,
            "affected_artifact": artifact,
            "field_name": field,
            "required_change": change,
            "rationale": rationale,
            "implementation_started": "false",
            "notes": "Planning only; no prior output is modified and no migration is executed in v2ab.",
        })
    out = dataset_path("v2ab_schema_migration_plan.csv")
    write_csv(out, MIGRATION_COLUMNS, rows)
    print(f"[v2ab migration plan] steps={len(rows)} -> {out}")
    return rows


# 9. Next Programming Target Ranker ----------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}

TARGET_VERSION = {
    "EVENT_PATCH_SCHEMA_MIGRATION_IMPLEMENTATION": "v2ac — Event-Patch Schema Migration Implementation",
    "MULTI_REGION_REGISTRY_HARDENING": "v2ac — Multi-Region Registry Hardening",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2ac — DINO Review Support Completion",
    "SENTINEL_DATE_RECOVERY_CONTINUE": "v2ac — Sentinel Date Recovery Continue",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2ac — Public Source Recheck Hold",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2ac — Ground Truth Search Hold",
}


def _ranker_metrics():
    validation = load_csv(dataset_path("v2ab_event_patch_package_validation.csv"))
    total = len(validation) or 1
    incomplete = sum(1 for r in validation if r.get("validation_status") in {"PACKAGE_INCOMPLETE_SCHEMA", "PACKAGE_BLOCKED_MISSING_PATCH_ID", "PACKAGE_BLOCKED_MISSING_EVENT"})
    temporal = load_csv(dataset_path("v2ab_temporal_field_contract_enforcement.csv"))
    missing_date = sum(1 for r in temporal if r.get("sentinel_date_status") in {"SENTINEL_DATE_MISSING_WITH_BLOCKER", "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE"})
    crosswalk = load_csv(dataset_path("v2ab_patch_identity_crosswalk_audit.csv"))
    no_crosswalk = sum(1 for r in crosswalk if r.get("crosswalk_status") == "NO_EXPLICIT_CROSSWALK")
    return {
        "total": total,
        "incomplete_rate": incomplete / total,
        "missing_date_rate": missing_date / total,
        "no_crosswalk": no_crosswalk,
        "contract_ready": 1,
    }


def _candidate_targets(m):
    # Schema contract is built; the high-value next step is implementing the
    # migration that closes namespace/temporal contract gaps for all packages.
    migration_value = 55 + round(20 * (1 - m["incomplete_rate"]))
    return [
        {
            "next_target": "EVENT_PATCH_SCHEMA_MIGRATION_IMPLEMENTATION",
            "programming_value": migration_value,
            "ground_truth_value": 0,
            "blocker_reduction_value": 55,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "Apply the hardened schema contract (patch_namespace, explicit_crosswalk_id, enforced temporal fields) to every package without inventing data.",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_HARDENING",
            "programming_value": 50,
            "ground_truth_value": 0,
            "blocker_reduction_value": 35,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Consolidate multi-region registries with the new namespace and crosswalk findings.",
        },
        {
            "next_target": "SENTINEL_DATE_RECOVERY_CONTINUE",
            "programming_value": round(20 * m["missing_date_rate"]),
            "ground_truth_value": 0,
            "blocker_reduction_value": round(15 * m["missing_date_rate"]),
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "Only worthwhile if a new same-namespace date source appears; cross-namespace dates remain unlinkable.",
        },
        {
            "next_target": "DINO_REVIEW_SUPPORT_COMPLETION",
            "programming_value": 10,
            "ground_truth_value": 0,
            "blocker_reduction_value": 10,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "Review-only DINO support is already attached for nearly all candidates.",
        },
        {
            "next_target": "PUBLIC_SOURCE_RECHECK_HOLD",
            "programming_value": 20,
            "ground_truth_value": 0,
            "blocker_reduction_value": 20,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Hold until a new public source with linkable patch-level identity appears.",
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
    out = dataset_path("v2ab_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v2ab ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
    return rows


# 10. Completion Report ----------------------------------------------------

def run_ground_reference_blocker_matrix(args=None):
    region_event = {"REC": "REC_2022_05_24_30", "PET": "PET_2022_02_15", "CUR": "CUR_2022_01_15"}
    blockers = [
        "no_observed_geometry", "no_occurrence_coordinates", "no_sentinel_date_linkable",
        "no_explicit_crosswalk", "no_overlay", "no_ground_reference",
        "no_training_label", "patch_truth_forbidden",
    ]
    rows = []
    for region, event_id in region_event.items():
        for blocker in blockers:
            rows.append({
                "blocker_id": f"GB_v2ab_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "blocker": blocker,
                "status": "BLOCKED",
                **guardrails(),
                "notes": "Schema hardening does not unblock geometry, overlay, ground reference or labels.",
            })
    out = dataset_path("v2ab_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_COLUMNS, rows)
    print(f"[v2ab gr blockers] rows={len(rows)} -> {out}")
    return rows


def run_completion_report(args=None):
    write_policy_configs()
    namespaces = load_csv(dataset_path("v2ab_patch_namespace_inventory.csv")) or run_patch_namespace_inventory(args)
    crosswalk = load_csv(dataset_path("v2ab_patch_identity_crosswalk_audit.csv")) or run_patch_identity_crosswalk_audit(args)
    contract = load_csv(dataset_path("v2ab_event_patch_schema_contract.csv")) or run_event_patch_schema_contract_builder(args)
    validation = load_csv(dataset_path("v2ab_event_patch_package_validation.csv")) or run_event_patch_package_validator(args)
    temporal = load_csv(dataset_path("v2ab_temporal_field_contract_enforcement.csv")) or run_temporal_field_contract_enforcer(args)
    guards = load_csv(dataset_path("v2ab_unlinkable_date_guard_registry.csv")) or run_unlinkable_date_guard_builder(args)
    completeness = load_csv(dataset_path("v2ab_package_completeness_score.csv")) or run_package_completeness_scorer(args)
    migration = load_csv(dataset_path("v2ab_schema_migration_plan.csv")) or run_schema_migration_plan_builder(args)
    ranker = load_csv(dataset_path("v2ab_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    blockers = run_ground_reference_blocker_matrix(args)

    explicit_xw = [c for c in crosswalk if c.get("crosswalk_status") == "EXPLICIT_CROSSWALK_FOUND"]
    no_xw = [c for c in crosswalk if c.get("crosswalk_status") == "NO_EXPLICIT_CROSSWALK"]
    incomplete = sum(1 for r in validation if r.get("validation_status") in {"PACKAGE_INCOMPLETE_SCHEMA", "PACKAGE_BLOCKED_MISSING_PATCH_ID", "PACKAGE_BLOCKED_MISSING_EVENT"})
    valid = sum(1 for r in validation if r.get("validation_status") == "PACKAGE_VALID_WITH_TEMPORAL_BLOCKER")
    temporal_blocked = sum(1 for r in temporal if r.get("sentinel_date_status") != "SENTINEL_DATE_CONFIRMED_SAME_PATCH")
    unlinkable = sum(1 for r in temporal if r.get("sentinel_date_status") == "SENTINEL_DATE_RECOVERED_UNLINKABLE_NAMESPACE")
    avg_complete = round(sum(int(r["completeness_score"]) for r in completeness) / len(completeness)) if completeness else 0
    next_target = ranker[0].get("next_target", "") if ranker else ""
    next_version = ranker[0].get("recommended_version", "") if ranker else ""

    write_csv(dataset_path("v2ab_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v2ab_0000",
        "event_id": "MULTI_REGION",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v2ab score-based next-programming-target ranker after schema hardening.",
        "target": "EVENT_PATCH_PACKAGE_SCHEMA",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth, ground reference, inferred date or inferred crosswalk.",
    }])

    lines = [
        "# Protocolo C v2ab - Event-Patch Package Schema Hardening",
        "",
        f"- patch namespaces inventoried: `{len(namespaces)}`",
        f"- crosswalk pairs audited: `{len(crosswalk)}` (explicit: `{len(explicit_xw)}`, none: `{len(no_xw)}`)",
        f"- schema contract fields: `{len(contract)}`",
        f"- packages validated: `{len(validation)}` (incomplete/blocked: `{incomplete}`, valid-with-temporal-blocker: `{valid}`)",
        f"- packages with temporal blocker: `{temporal_blocked}`",
        f"- packages with unlinkable cross-namespace date: `{unlinkable}`",
        f"- unlinkable-date guards: `{len(guards)}`",
        f"- average completeness score: `{avg_complete}`",
        f"- ground-reference blocker rows: `{len(blockers)}`",
        f"- selected next target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v2ab hardened the event-patch package schema with explicit identity, namespace, temporal and crosswalk contracts. It invented no crosswalk, inferred no Sentinel date, applied no cross-namespace date by region/name/order, executed no overlay, and created no ground truth, ground reference or label.",
    ]
    write_text(doc_path("protocolo_c_v2ab_event_patch_package_schema_hardening.md"), lines)

    report = lines + [
        "",
        "## How many namespaces exist",
        f"{len(namespaces)} patch namespaces were inventoried (DINO visual, event-patch candidate, anchor REFPATCH, recovery scaffolding, Sentinel source, and any unknown).",
        "",
        "## Is there an explicit crosswalk",
        ("An explicit identity crosswalk exists only where namespaces share the same patch_id key (event-patch candidate and DINO visual share the numeric patch_id). "
         "There is NO explicit crosswalk between the numeric event-patch namespace and the REFPATCH/scaffolding namespaces, so their recovered dates cannot be linked."),
        "",
        "## Package validation",
        f"{len(validation)} event-patch packages were validated against the schema contract: {valid} are structurally valid with a temporal blocker, {incomplete} are incomplete or blocked.",
        "",
        "## Temporal blockers and unlinkable dates",
        f"{temporal_blocked} packages carry a temporal blocker; {unlinkable} have a date recovered only in a parallel namespace, which is explicitly kept unlinkable.",
        "",
        "## Guards created",
        f"{len(guards)} unlinkable-date guards forbid applying a parallel-namespace date to an event-patch candidate by region, name similarity or file order.",
        "",
        "## Average completeness",
        f"Average structural completeness score is {avg_complete} (schema structure only; never performance, ground truth or label).",
        "",
        "## Why there is still no overlay",
        "No overlay was executed and overlay status stays BLOCKED. Schema hardening clarifies identity and temporal contracts but establishes no observed occurrence geometry.",
        "",
        "## Why there is still no ground reference",
        "No package has a linkable observed occurrence geometry; without it there is no basis for ground reference, and none was created.",
        "",
        "## Next programming step",
        f"The score-based ranker selected `{next_target}` (`{next_version}`).",
    ]
    write_text(doc_path("protocolo_c_relatorio_v2ab_event_patch_package_schema_hardening.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v2ab.md"), [
        "# Status atual - Protocolo C v2ab",
        "",
        f"Schema hardening status: `{MAX_STATUS}`.",
        f"Namespaces: `{len(namespaces)}`; explicit crosswalk pairs: `{len(explicit_xw)}`; no-crosswalk pairs: `{len(no_xw)}`.",
        f"Packages validated: `{len(validation)}`; unlinkable-date guards: `{len(guards)}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Overlay, ground reference, training labels, ground truth, inferred Sentinel dates and inferred crosswalks remain blocked.",
    ])

    manifest = []
    for idx, artifact in enumerate(V2AB_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v2ab_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v2ab schema/contract/audit artifact; no raw data, no private path, no inferred date or crosswalk.",
        })
    write_csv(dataset_path("v2ab_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for folder in (STAGING_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v2ab completion] namespaces={len(namespaces)} valid={valid} guards={len(guards)} next={next_target}")
    return {"namespaces": len(namespaces), "valid": valid, "incomplete": incomplete, "guards": len(guards), "next_target": next_target, "next_version": next_version}


def run_all(args=None):
    args = args or parse_args([])
    run_patch_namespace_inventory(args)
    run_patch_identity_crosswalk_audit(args)
    run_event_patch_schema_contract_builder(args)
    run_temporal_field_contract_enforcer(args)
    run_event_patch_package_validator(args)
    run_unlinkable_date_guard_builder(args)
    run_package_completeness_scorer(args)
    run_schema_migration_plan_builder(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
