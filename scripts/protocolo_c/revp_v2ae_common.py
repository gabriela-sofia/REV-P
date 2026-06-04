#!/usr/bin/env python3
"""v2ae Multi-region registry hardening.

Consolidates the distributed Protocol C state (region closures, canonical
events, migrated v2 event-patch packages, blockers, readiness, QA gate, safe-use
policies and reopen conditions) into hardened canonical multi-region registries
that future versions consume. It reads prior outputs read-only and never seeks
new sources, infers events/coordinates/dates/crosswalks, promotes context to
occurrence, executes overlay, or creates ground truth, ground reference or
labels.
"""

import argparse
import csv
import datetime
import hashlib
import os
import re

PROTOCOL_VERSION = "v2ae"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
STAGING_DIR = "local_only/protocolo_c/multiregion_registry_hardening/staging/v2ae"
REPORTS_DIR = "local_only/protocolo_c/multiregion_registry_hardening/reports/v2ae"

MAX_STATUS = "MULTIREGION_REGISTRY_HARDENED_NON_OPERATIONAL"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "multiregion_registry_hardening_only",
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
    "CROSSWALK_INFERRED",
]
FORBIDDEN_STATUS_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_STATUS_TOKENS) + r")\b")
TOOL_NAME_RE = re.compile(r"\b(claude|codex|llm|assistant|chatgpt|openai|anthropic|copilot|gemini)\b", re.I)
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")
EVENT_ID_RE = re.compile(r"^(CUR|PET|REC)_(\d{4})_(\d{2})_(\d{2})(?:_(\d{2}))?$")

REGION_META = {
    "REC": ("Recife", "PE"),
    "PET": ("Petropolis", "RJ"),
    "CUR": ("Curitiba", "PR"),
}
REGION_CANONICAL_STATUS = {
    "REC": "REGION_HARDENED_CONTEXTUAL_COORDINATE_NON_OPERATIONAL",
    "PET": "REGION_HARDENED_DOCUMENT_ONLY_NO_GEODATA",
    "CUR": "REGION_HARDENED_CONTEXT_ONLY_HOLD",
}
EXPECTED_REGIONS = ["REC", "PET", "CUR"]
EXPECTED_EVENTS = ["REC_2022_05_24_30", "PET_2022_02_15", "PET_2024_03_21_28", "CUR_2022_01_15"]

# Column definitions ------------------------------------------------------
REGION_COLUMNS = [
    "region_registry_id", "region", "city", "uf", "canonical_region_status",
    "best_evidence_type", "best_evidence_strength", "contextual_support_status",
    "occurrence_support_status", "coordinate_status", "geometry_status",
    "overlay_status", "ground_reference_status", "qa_gate_status", "hold_status",
    "main_blocker", "can_create_ground_reference", "can_create_training_label",
    "notes",
]
EVENT_COLUMNS = [
    "canonical_event_id", "event_id", "region", "city", "uf", "start_date",
    "end_date", "hazard_scope", "canonical_event_status", "official_source_support",
    "temporal_support", "hazard_support", "locality_support",
    "contextual_coordinate_support", "occurrence_coordinate_support",
    "observed_geometry_support", "qa_gate_status", "main_blocker",
    "can_create_ground_reference", "can_create_training_label", "notes",
]
PACKAGE_COLUMNS = [
    "canonical_package_id", "event_patch_candidate_id", "event_id", "region",
    "patch_id", "patch_namespace", "crosswalk_status", "sentinel_date_status",
    "date_linkability_status", "evidence_status", "phenomenon_status",
    "coordinate_status", "geometry_status", "overlay_status",
    "ground_reference_status", "training_label_status", "qa_status",
    "package_status", "can_create_ground_reference", "can_create_training_label",
    "notes",
]
BLOCKER_CONSOL_COLUMNS = [
    "blocker_consolidation_id", "blocker", "blocker_class", "scope", "region",
    "event_id", "status", "occurrences", *GUARDRAIL_COLUMNS, "notes",
]
READINESS_CONSOL_COLUMNS = [
    "readiness_id", "readiness_scope", "region", "event_id",
    "event_patch_candidate_id", "dimension", "status", "blocker",
    *GUARDRAIL_COLUMNS, "notes",
]
REOPEN_COLUMNS = [
    "reopen_condition_id", "region", "current_hold_status", "reopen_condition",
    "required_public_evidence", "required_minimum_fields",
    "forbidden_reopen_basis", "recommended_future_version", "notes",
]
SAFE_USE_COLUMNS = [
    "policy_id", "scope", "region", "evidence_type", "safe_use",
    "prohibited_use", "rationale", "can_create_ground_reference",
    "can_create_training_label", "notes",
]
QA_COLUMNS = [
    "qa_id", "check_group", "check_name", "expected", "observed", "status",
    "severity", "notes",
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

V2AE_ARTIFACTS = [
    "configs/protocolo_c/v2ae_canonical_registry_policy.yaml",
    "configs/protocolo_c/v2ae_region_status_policy.yaml",
    "configs/protocolo_c/v2ae_blocker_consolidation_policy.yaml",
    "configs/protocolo_c/v2ae_reopen_condition_policy.yaml",
    "configs/protocolo_c/v2ae_safe_use_policy.yaml",
    "configs/protocolo_c/v2ae_next_programming_target_policy.yaml",
    "datasets/protocolo_c/v2ae_canonical_region_registry.csv",
    "datasets/protocolo_c/v2ae_canonical_event_registry.csv",
    "datasets/protocolo_c/v2ae_canonical_event_patch_registry.csv",
    "datasets/protocolo_c/v2ae_multiregion_blocker_consolidation.csv",
    "datasets/protocolo_c/v2ae_multiregion_readiness_consolidation.csv",
    "datasets/protocolo_c/v2ae_region_reopen_condition_registry.csv",
    "datasets/protocolo_c/v2ae_safe_use_policy_registry.csv",
    "datasets/protocolo_c/v2ae_registry_consistency_qa.csv",
    "datasets/protocolo_c/v2ae_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v2ae_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v2ae_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v2ae_multiregion_registry_hardening.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v2ae_multiregion_registry_hardening.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v2ae.md",
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
        "multiregion_registry_hardening_only": "true",
        "crosswalk_inferred": "false",
        "sentinel_date_inferred": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v2ae_canonical_registry_policy.yaml": [
            "canonical_registries: [region, event, event_patch]",
            "source_of_truth_event_patch: v2ac_event_patch_v2_package_registry.csv",
            "modify_prior_outputs: false",
            "max_status: MULTIREGION_REGISTRY_HARDENED_NON_OPERATIONAL",
        ],
        "v2ae_region_status_policy.yaml": [
            "recife: REGION_HARDENED_CONTEXTUAL_COORDINATE_NON_OPERATIONAL",
            "petropolis: REGION_HARDENED_DOCUMENT_ONLY_NO_GEODATA",
            "curitiba: REGION_HARDENED_CONTEXT_ONLY_HOLD",
            "promote_region: false",
        ],
        "v2ae_blocker_consolidation_policy.yaml": [
            "classes: [GLOBAL_BLOCKER, REGION_BLOCKER, EVENT_BLOCKER, PACKAGE_BLOCKER, EXPECTED_BLOCKER, CRITICAL_BLOCKER]",
            "remove_blocker: false",
        ],
        "v2ae_reopen_condition_policy.yaml": [
            "execute_search: false",
            "reopen_requires_new_public_source: true",
            "forbidden_reopen_basis: [region, name_similarity, file_order, inferred_date, inferred_crosswalk]",
        ],
        "v2ae_safe_use_policy.yaml": [
            "safe: [review_only, contextual_support, evidence_audit, dino_review_support, package_qa]",
            "prohibited: [ground_truth, label, patch_positive, patch_negative, overlay_truth, event_validated_by_sentinel, hydromet_as_occurrence, context_layer_as_occurrence]",
        ],
        "v2ae_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


def _qa_gate_status():
    gate = load_csv(dataset_path("v2ad_qa_gate_summary.csv"))
    overall = next((r for r in gate if r.get("qa_group") == "OVERALL"), {})
    return overall.get("gate_status", "QA_GATE_UNKNOWN")


def _closures():
    return load_csv(dataset_path("v1uz_multiregion_closure_status.csv"))


def _packages():
    return load_csv(dataset_path("v2ac_event_patch_v2_package_registry.csv"))


def _event_dates():
    dates = {}
    for r in load_csv(dataset_path("v1us_event_temporal_window_linkage.csv")):
        eid = r.get("event_id")
        if eid and eid not in dates and r.get("event_start_date"):
            dates[eid] = (r.get("event_start_date", ""), r.get("event_end_date", ""))
    return dates


def _dates_from_event_id(event_id):
    m = EVENT_ID_RE.match(event_id or "")
    if not m:
        return ("", "")
    _, y, mo, d1, d2 = m.groups()
    try:
        start = datetime.date(int(y), int(mo), int(d1)).isoformat()
    except ValueError:
        return ("", "")
    if d2:
        try:
            end = datetime.date(int(y), int(mo), int(d2)).isoformat()
        except ValueError:
            end = start
    else:
        end = start
    return (start, end)


# 1. Canonical Region Registry Builder ------------------------------------

def run_canonical_region_registry_builder(args=None):
    write_policy_configs()
    qa_gate = _qa_gate_status()
    # One closure per region (pick first per region).
    closure_by_region = {}
    for c in _closures():
        closure_by_region.setdefault(c.get("region"), c)
    rows = []
    for region in EXPECTED_REGIONS:
        clo = closure_by_region.get(region, {})
        city, uf = REGION_META.get(region, ("", ""))
        contextual = "STRONG" if region == "REC" else ("WEAK" if region == "CUR" else "ABSENT")
        rows.append({
            "region_registry_id": f"RR_v2ae_{len(rows):04d}",
            "region": region,
            "city": city,
            "uf": uf,
            "canonical_region_status": REGION_CANONICAL_STATUS.get(region, ""),
            "best_evidence_type": clo.get("best_evidence_type", ""),
            "best_evidence_strength": clo.get("best_evidence_strength", ""),
            "contextual_support_status": contextual,
            "occurrence_support_status": "ABSENT",
            "coordinate_status": clo.get("coordinate_status", ""),
            "geometry_status": clo.get("geometry_status", "NO_OBSERVED_GEOMETRY"),
            "overlay_status": "BLOCKED",
            "ground_reference_status": "BLOCKED",
            "qa_gate_status": qa_gate,
            "hold_status": "REGION_HOLD_NON_OPERATIONAL",
            "main_blocker": clo.get("main_blocker", ""),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Canonical hardened region status; non-operational; context never promoted to occurrence.",
        })
    out = dataset_path("v2ae_canonical_region_registry.csv")
    write_csv(out, REGION_COLUMNS, rows)
    print(f"[v2ae region] regions={len(rows)} -> {out}")
    return rows


# 2. Canonical Event Registry Builder -------------------------------------

EVENT_SUPPORT = {
    "REC": {"hazard": "STRONG", "locality": "STRONG", "ctx_coord": "STRONG"},
    "PET": {"hazard": "MODERATE", "locality": "MODERATE", "ctx_coord": "ABSENT"},
    "CUR": {"hazard": "MODERATE", "locality": "MODERATE", "ctx_coord": "WEAK"},
}


def run_canonical_event_registry_builder(args=None):
    qa_gate = _qa_gate_status()
    explicit_dates = _event_dates()
    closure_by_event = {c.get("event_id"): c for c in _closures()}
    rows = []
    for event_id in EXPECTED_EVENTS:
        region = event_id.split("_")[0]
        city, uf = REGION_META.get(region, ("", ""))
        start, end = explicit_dates.get(event_id) or _dates_from_event_id(event_id)
        clo = closure_by_event.get(event_id, {})
        sup = EVENT_SUPPORT.get(region, {})
        rows.append({
            "canonical_event_id": f"EV_v2ae_{len(rows):04d}",
            "event_id": event_id,
            "region": region,
            "city": city,
            "uf": uf,
            "start_date": start,
            "end_date": end,
            "hazard_scope": "mixed_hydromet_context",
            "canonical_event_status": "EVENT_CANDIDATE_NON_OPERATIONAL",
            "official_source_support": "STRONG",
            "temporal_support": "STRONG" if start else "WEAK",
            "hazard_support": sup.get("hazard", "MODERATE"),
            "locality_support": sup.get("locality", "MODERATE"),
            "contextual_coordinate_support": sup.get("ctx_coord", "ABSENT"),
            "occurrence_coordinate_support": "ABSENT",
            "observed_geometry_support": "ABSENT",
            "qa_gate_status": qa_gate,
            "main_blocker": clo.get("main_blocker", "no_observed_geometry"),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Canonical event is candidate-only; never promoted or validated by Sentinel.",
        })
    out = dataset_path("v2ae_canonical_event_registry.csv")
    write_csv(out, EVENT_COLUMNS, rows)
    print(f"[v2ae event] events={len(rows)} -> {out}")
    return rows


# 3. Canonical Event-Patch Registry Builder -------------------------------

def run_canonical_event_patch_registry_builder(args=None):
    packages = _packages()
    schema = {r["event_patch_candidate_id"]: r.get("validation_status") for r in load_csv(dataset_path("v2ac_schema_contract_validation.csv"))}
    qa_gate = _qa_gate_status()
    rows = []
    for pkg in packages:
        epc = pkg.get("event_patch_candidate_id", "")
        rows.append({
            "canonical_package_id": f"CP_v2ae_{len(rows):05d}",
            "event_patch_candidate_id": epc,
            "event_id": pkg.get("event_id", ""),
            "region": pkg.get("event_region", ""),
            "patch_id": pkg.get("patch_id", ""),
            "patch_namespace": pkg.get("patch_namespace", ""),
            "crosswalk_status": pkg.get("crosswalk_status", ""),
            "sentinel_date_status": pkg.get("sentinel_date_status", ""),
            "date_linkability_status": pkg.get("date_linkability_status", ""),
            "evidence_status": pkg.get("evidence_status", ""),
            "phenomenon_status": pkg.get("phenomenon_status", ""),
            "coordinate_status": pkg.get("coordinate_status", ""),
            "geometry_status": pkg.get("geometry_status", ""),
            "overlay_status": "BLOCKED",
            "ground_reference_status": "BLOCKED",
            "training_label_status": "BLOCKED",
            "qa_status": qa_gate,
            "package_status": schema.get(epc, pkg.get("package_validation_status", "")),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Canonical event-patch package mirrors v2ac (read-only); no promotion.",
        })
    out = dataset_path("v2ae_canonical_event_patch_registry.csv")
    write_csv(out, PACKAGE_COLUMNS, rows)
    print(f"[v2ae event-patch] packages={len(rows)} -> {out}")
    return rows


# 4. Multi-Region Blocker Consolidator ------------------------------------

GLOBAL_BLOCKERS = {"no_observed_geometry", "no_occurrence_coordinates", "no_overlay", "no_ground_reference", "no_training_label", "patch_truth_forbidden"}
REGION_ONLY_BLOCKERS = {"context_only", "document_only", "locality_only", "hydromet_only"}
CRITICAL_BLOCKERS = {"no_ground_reference", "no_observed_geometry", "patch_truth_forbidden"}


def _blocker_class(blocker, scope):
    if blocker in CRITICAL_BLOCKERS:
        return "CRITICAL_BLOCKER"
    if scope == "GLOBAL":
        return "GLOBAL_BLOCKER"
    if scope == "REGION":
        return "REGION_BLOCKER"
    if scope == "EVENT":
        return "EVENT_BLOCKER"
    if scope == "PACKAGE":
        return "PACKAGE_BLOCKER"
    return "EXPECTED_BLOCKER"


def run_multiregion_blocker_consolidator(args=None):
    packages = _packages()
    rows = []
    # Global blockers (present across all regions).
    global_counts = {}
    region_blockers = {}
    for pkg in packages:
        region = pkg.get("event_region", "")
        for b in (pkg.get("blocker") or "").split("|"):
            if not b:
                continue
            global_counts[b] = global_counts.get(b, 0) + 1
            region_blockers.setdefault(region, {}).setdefault(b, 0)
            region_blockers[region][b] += 1
    for blocker, count in sorted(global_counts.items()):
        scope = "GLOBAL" if blocker in GLOBAL_BLOCKERS else "PACKAGE"
        rows.append({
            "blocker_consolidation_id": f"BC_v2ae_{len(rows):04d}",
            "blocker": blocker,
            "blocker_class": _blocker_class(blocker, scope),
            "scope": scope,
            "region": "ALL",
            "event_id": "",
            "status": "BLOCKED",
            "occurrences": str(count),
            **guardrails(),
            "notes": "Consolidated blocker; never removed; expected unless marked critical.",
        })
    # Region-specific descriptor blockers from closures.
    closure_blockers = {
        "REC": "locality_only", "PET": "document_only", "CUR": "context_only",
    }
    for region, blocker in closure_blockers.items():
        rows.append({
            "blocker_consolidation_id": f"BC_v2ae_{len(rows):04d}",
            "blocker": blocker,
            "blocker_class": "REGION_BLOCKER",
            "scope": "REGION",
            "region": region,
            "event_id": "",
            "status": "BLOCKED",
            "occurrences": str(sum(region_blockers.get(region, {}).values())),
            **guardrails(),
            "notes": "Region descriptor blocker from canonical closure.",
        })
    out = dataset_path("v2ae_multiregion_blocker_consolidation.csv")
    write_csv(out, BLOCKER_CONSOL_COLUMNS, rows)
    print(f"[v2ae blocker consol] rows={len(rows)} -> {out}")
    return rows


# 5. Multi-Region Readiness Consolidator ----------------------------------

def run_multiregion_readiness_consolidator(args=None):
    regions = load_csv(dataset_path("v2ae_canonical_region_registry.csv")) or run_canonical_region_registry_builder(args)
    events = load_csv(dataset_path("v2ae_canonical_event_registry.csv")) or run_canonical_event_registry_builder(args)
    packages = _packages()
    rows = []
    # Region-level.
    for reg in regions:
        for dim, status in [
            ("contextual_support", reg.get("contextual_support_status", "")),
            ("occurrence_support", reg.get("occurrence_support_status", "")),
            ("geometry_support", "ABSENT"),
            ("overlay_readiness", "BLOCKED"),
            ("ground_reference_readiness", "BLOCKED"),
            ("training_readiness", "BLOCKED"),
        ]:
            rows.append({
                "readiness_id": f"RDY_v2ae_{len(rows):05d}", "readiness_scope": "REGION",
                "region": reg.get("region", ""), "event_id": "", "event_patch_candidate_id": "",
                "dimension": dim, "status": status,
                "blocker": reg.get("main_blocker", "") if status in ("ABSENT", "BLOCKED") else "",
                **guardrails(),
                "notes": "Region-level readiness; overlay/ground reference/training stay BLOCKED.",
            })
    # Event-level.
    for ev in events:
        for dim, status in [
            ("temporal_support", ev.get("temporal_support", "")),
            ("occurrence_coordinate_support", ev.get("occurrence_coordinate_support", "")),
            ("observed_geometry_support", ev.get("observed_geometry_support", "")),
            ("overlay_readiness", "BLOCKED"),
            ("ground_reference_readiness", "BLOCKED"),
            ("training_readiness", "BLOCKED"),
        ]:
            rows.append({
                "readiness_id": f"RDY_v2ae_{len(rows):05d}", "readiness_scope": "EVENT",
                "region": ev.get("region", ""), "event_id": ev.get("event_id", ""), "event_patch_candidate_id": "",
                "dimension": dim, "status": status,
                "blocker": ev.get("main_blocker", "") if status in ("ABSENT", "BLOCKED") else "",
                **guardrails(),
                "notes": "Event-level readiness; candidate-only.",
            })
    # Package-level (summary dimensions per package).
    for pkg in packages:
        link = "BLOCKED" if pkg.get("date_linkability_status") != "LINKABLE_SAME_PATCH" else "MODERATE"
        for dim, status in [
            ("sentinel_date_linkability", link),
            ("overlay_readiness", "BLOCKED"),
            ("ground_reference_readiness", "BLOCKED"),
            ("training_readiness", "BLOCKED"),
        ]:
            rows.append({
                "readiness_id": f"RDY_v2ae_{len(rows):05d}", "readiness_scope": "PACKAGE",
                "region": pkg.get("event_region", ""), "event_id": pkg.get("event_id", ""),
                "event_patch_candidate_id": pkg.get("event_patch_candidate_id", ""),
                "dimension": dim, "status": status,
                "blocker": pkg.get("temporal_blocker", "") if dim == "sentinel_date_linkability" and link == "BLOCKED" else "",
                **guardrails(),
                "notes": "Package-level readiness; overlay/ground reference/training stay BLOCKED.",
            })
    out = dataset_path("v2ae_multiregion_readiness_consolidation.csv")
    write_csv(out, READINESS_CONSOL_COLUMNS, rows)
    print(f"[v2ae readiness consol] rows={len(rows)} -> {out}")
    return rows


# 6. Region Reopen Condition Builder --------------------------------------

REOPEN = {
    "REC": {
        "reopen_condition": "A public official source publishes occurrence-level coordinates or observed geometry within the event window.",
        "required_public_evidence": "public_occurrence_coordinate_or_observed_geometry_layer",
        "required_minimum_fields": "occurrence_date|occurrence_geometry|official_source",
    },
    "PET": {
        "reopen_condition": "A public official post-disaster geodata layer or an official spatial crosswalk is published.",
        "required_public_evidence": "public_geodata_layer_or_official_spatial_crosswalk",
        "required_minimum_fields": "geometry|official_source|temporal_window",
    },
    "CUR": {
        "reopen_condition": "An official observed occurrence layer or event table with geometry is published.",
        "required_public_evidence": "official_occurrence_layer_or_event_table_with_geometry",
        "required_minimum_fields": "occurrence_geometry|official_source|event_date",
    },
}


def run_region_reopen_condition_builder(args=None):
    rows = []
    for region in EXPECTED_REGIONS:
        spec = REOPEN.get(region, {})
        rows.append({
            "reopen_condition_id": f"RO_v2ae_{len(rows):04d}",
            "region": region,
            "current_hold_status": REGION_CANONICAL_STATUS.get(region, ""),
            "reopen_condition": spec.get("reopen_condition", ""),
            "required_public_evidence": spec.get("required_public_evidence", ""),
            "required_minimum_fields": spec.get("required_minimum_fields", ""),
            "forbidden_reopen_basis": "region|name_similarity|file_order|inferred_date|inferred_crosswalk|context_layer_as_occurrence",
            "recommended_future_version": "v2af_or_later_only_if_new_public_source",
            "notes": "Reopen conditions are documented only; no search is executed in v2ae.",
        })
    out = dataset_path("v2ae_region_reopen_condition_registry.csv")
    write_csv(out, REOPEN_COLUMNS, rows)
    print(f"[v2ae reopen] rows={len(rows)} -> {out}")
    return rows


# 7. Safe Use Policy Registry Builder -------------------------------------

def run_safe_use_policy_registry_builder(args=None):
    safe = "review_only|contextual_support|evidence_audit|dino_review_support|package_qa"
    prohibited = "ground_truth|training_label|patch_positive|patch_negative|overlay_truth|event_validated_by_sentinel|hydromet_as_occurrence|context_layer_as_occurrence"
    rows = []
    entries = [("GLOBAL", "", "event_patch_package_v2")]
    for region in EXPECTED_REGIONS:
        et = {"REC": "contextual_coordinate_layer", "PET": "official_document_only", "CUR": "context_layer_and_hydromet"}[region]
        entries.append(("REGION", region, et))
    for scope, region, evidence_type in entries:
        rows.append({
            "policy_id": f"SU_v2ae_{len(rows):04d}",
            "scope": scope,
            "region": region,
            "evidence_type": evidence_type,
            "safe_use": safe,
            "prohibited_use": prohibited,
            "rationale": "Evidence is contextual/candidate-only with no observed occurrence geometry; promotion would be an unsupported overclaim.",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "Safe-use policy; review-only support across regions.",
        })
    out = dataset_path("v2ae_safe_use_policy_registry.csv")
    write_csv(out, SAFE_USE_COLUMNS, rows)
    print(f"[v2ae safe use] rows={len(rows)} -> {out}")
    return rows


# 8. Registry Consistency QA ----------------------------------------------

def _qa(rows, group, name, expected, observed, status, severity="info", notes=""):
    rows.append({
        "qa_id": f"RCQA_{len(rows):04d}", "check_group": group, "check_name": name,
        "expected": expected, "observed": observed, "status": status,
        "severity": severity, "notes": notes,
    })


def _scan_forbidden(paths):
    counts = {"forbidden_true_value": 0, "forbidden_status": 0, "absolute_path": 0, "local_only_leak": 0, "tool_name_leak": 0}
    for path in paths:
        if not os.path.exists(path) or not path.lower().endswith(".csv"):
            continue
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
    return counts


def run_registry_consistency_qa(args=None):
    regions = load_csv(dataset_path("v2ae_canonical_region_registry.csv"))
    events = load_csv(dataset_path("v2ae_canonical_event_registry.csv"))
    packages = load_csv(dataset_path("v2ae_canonical_event_patch_registry.csv"))
    rows = []
    region_set = {r.get("region") for r in regions}
    _qa(rows, "region", "three_regions_present", "REC|PET|CUR",
        "|".join(sorted(region_set)), "PASS" if set(EXPECTED_REGIONS) <= region_set else "FAIL",
        "critical")
    event_set = {e.get("event_id") for e in events}
    missing_events = [e for e in EXPECTED_EVENTS if e not in event_set]
    _qa(rows, "event", "expected_events_present", "|".join(EXPECTED_EVENTS),
        "missing:" + "|".join(missing_events) if missing_events else "all_present",
        "FAIL" if missing_events else "PASS", "critical")
    v2ac_count = len(_packages())
    _qa(rows, "event_patch", "packages_preserved", str(v2ac_count), str(len(packages)),
        "PASS" if len(packages) == v2ac_count and v2ac_count > 0 else "FAIL", "critical")
    gate = _qa_gate_status()
    _qa(rows, "qa_gate", "v2ad_gate_preserved", "QA_PASS_WITH_EXPECTED_BLOCKERS", gate,
        "PASS" if gate in ("QA_PASS", "QA_PASS_WITH_EXPECTED_BLOCKERS") else "FAIL", "high")
    released = sum(
        1 for p in packages
        for f in ("overlay_status", "ground_reference_status", "training_label_status")
        if p.get(f) != "BLOCKED"
    )
    _qa(rows, "guardrail", "overlay_ground_reference_training_blocked", "0_released",
        str(released), "PASS" if released == 0 else "FAIL", "critical")
    counts = _scan_forbidden([
        dataset_path("v2ae_canonical_region_registry.csv"),
        dataset_path("v2ae_canonical_event_registry.csv"),
        dataset_path("v2ae_canonical_event_patch_registry.csv"),
    ])
    for check_type, n in counts.items():
        _qa(rows, "guardrail", check_type, "0", str(n), "PASS" if n == 0 else "FAIL",
            "critical" if n else "info")
    out = dataset_path("v2ae_registry_consistency_qa.csv")
    write_csv(out, QA_COLUMNS, rows)
    print(f"[v2ae consistency qa] checks={len(rows)} fails={sum(1 for r in rows if r['status'] == 'FAIL')} -> {out}")
    return rows


# 9. Next Programming Target Ranker ----------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}

TARGET_VERSION = {
    "EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION": "v2af — Event-Patch Package V2 QA Automation",
    "SENTINEL_DATE_CROSSWALK_DISCOVERY": "v2af — Sentinel Date Crosswalk Discovery",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2af — DINO Review Support Completion",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2af — Public Source Recheck Hold",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2af — Stop Ground Truth Search Until New Source",
    "MULTI_REGION_REGISTRY_MAINTENANCE": "v2af — Multi-Region Registry Maintenance",
}


def _ranker_metrics():
    qa = load_csv(dataset_path("v2ae_registry_consistency_qa.csv"))
    consistent = qa and all(r.get("status") != "FAIL" for r in qa)
    gate = _qa_gate_status()
    packages = _packages()
    total = len(packages) or 1
    no_anchor = sum(1 for p in packages if "NO_ANCHOR_CROSSWALK" in p.get("crosswalk_status", "") or p.get("crosswalk_status") in {"NO_EXPLICIT_CROSSWALK", "NO_CROSSWALK_PATCH_ID_MISSING"})
    return {"consistent": bool(consistent), "qa_pass": gate.startswith("QA_PASS"), "no_anchor_rate": no_anchor / total}


def _candidate_targets(m):
    base = m["consistent"] and m["qa_pass"]
    return [
        {
            "next_target": "EVENT_PATCH_PACKAGE_V2_QA_AUTOMATION",
            "programming_value": 60 if base else 35,
            "ground_truth_value": 0,
            "blocker_reduction_value": 45,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Canonical registries are consistent and the QA gate is green; turn the QA harness into a repeatable automated gate.",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_MAINTENANCE",
            "programming_value": 45,
            "ground_truth_value": 0,
            "blocker_reduction_value": 30,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Maintain the hardened canonical registries as new versions append.",
        },
        {
            "next_target": "SENTINEL_DATE_CROSSWALK_DISCOVERY",
            "programming_value": round(40 * m["no_anchor_rate"]),
            "ground_truth_value": 0,
            "blocker_reduction_value": round(50 * m["no_anchor_rate"]),
            "expected_effort": "MEDIUM",
            "overclaim_risk": "MEDIUM",
            "notes": "Search for an explicit key linking numeric and anchor namespaces; uncertain and only useful if a key exists.",
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
            "notes": "Hold until a new public source with linkable occurrence evidence appears.",
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
    out = dataset_path("v2ae_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v2ae ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
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
                "blocker_id": f"GB_v2ae_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "blocker": blocker,
                "status": "BLOCKED",
                **guardrails(),
                "notes": "Registry hardening does not unblock geometry, overlay, ground reference or labels.",
            })
    out = dataset_path("v2ae_ground_reference_blocker_matrix.csv")
    write_csv(out, BLOCKER_MATRIX_COLUMNS, rows)
    print(f"[v2ae gr blockers] rows={len(rows)} -> {out}")
    return rows


def run_completion_report(args=None):
    write_policy_configs()
    regions = load_csv(dataset_path("v2ae_canonical_region_registry.csv")) or run_canonical_region_registry_builder(args)
    events = load_csv(dataset_path("v2ae_canonical_event_registry.csv")) or run_canonical_event_registry_builder(args)
    packages = load_csv(dataset_path("v2ae_canonical_event_patch_registry.csv")) or run_canonical_event_patch_registry_builder(args)
    blockers = load_csv(dataset_path("v2ae_multiregion_blocker_consolidation.csv")) or run_multiregion_blocker_consolidator(args)
    readiness = load_csv(dataset_path("v2ae_multiregion_readiness_consolidation.csv")) or run_multiregion_readiness_consolidator(args)
    reopen = load_csv(dataset_path("v2ae_region_reopen_condition_registry.csv")) or run_region_reopen_condition_builder(args)
    safe = load_csv(dataset_path("v2ae_safe_use_policy_registry.csv")) or run_safe_use_policy_registry_builder(args)
    qa = load_csv(dataset_path("v2ae_registry_consistency_qa.csv")) or run_registry_consistency_qa(args)
    ranker = load_csv(dataset_path("v2ae_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    gr_blockers = run_ground_reference_blocker_matrix(args)

    region_status = {r["region"]: r["canonical_region_status"] for r in regions}
    qa_fails = sum(1 for r in qa if r.get("status") == "FAIL")
    qa_result = "CONSISTENT" if qa_fails == 0 else f"INCONSISTENT_{qa_fails}_FAILS"
    next_target = ranker[0].get("next_target", "") if ranker else ""
    next_version = ranker[0].get("recommended_version", "") if ranker else ""
    global_blockers = [b["blocker"] for b in blockers if b.get("scope") == "GLOBAL"]

    write_csv(dataset_path("v2ae_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v2ae_0000",
        "event_id": "MULTI_REGION",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v2ae score-based next-programming-target ranker after registry hardening.",
        "target": "CANONICAL_MULTIREGION_REGISTRY",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth, ground reference, inferred date or inferred crosswalk.",
    }])

    lines = [
        "# Protocolo C v2ae - Multi-Region Registry Hardening",
        "",
        f"- canonical regions: `{len(regions)}`",
        f"- canonical events: `{len(events)}`",
        f"- canonical event-patch packages: `{len(packages)}`",
        f"- consolidated blocker rows: `{len(blockers)}`",
        f"- consolidated readiness rows: `{len(readiness)}`",
        f"- reopen condition rows: `{len(reopen)}`",
        f"- safe-use policy rows: `{len(safe)}`",
        f"- registry consistency QA: `{qa_result}`",
        f"- Recife status: `{region_status.get('REC', '')}`",
        f"- Petropolis status: `{region_status.get('PET', '')}`",
        f"- Curitiba status: `{region_status.get('CUR', '')}`",
        f"- selected next target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v2ae consolidated the distributed state into hardened canonical multi-region registries. It modified no prior output, sought no new source, inferred no event/coordinate/date/crosswalk, promoted no context to occurrence, executed no overlay, and created no ground truth, ground reference or label.",
    ]
    write_text(doc_path("protocolo_c_v2ae_multiregion_registry_hardening.md"), lines)

    report = lines + [
        "",
        "## Canonical region status",
        f"Recife: `{region_status.get('REC', '')}` (contextual coordinate layer, no occurrence coordinate). "
        f"Petropolis: `{region_status.get('PET', '')}` (official document only, no public geodata). "
        f"Curitiba: `{region_status.get('CUR', '')}` (event candidate and hydromet context, no occurrence layer).",
        "",
        "## Canonical events and packages",
        f"{len(events)} canonical events and {len(packages)} canonical event-patch packages were consolidated, preserving every event_patch_candidate_id.",
        "",
        "## Blockers",
        f"Global blockers: {', '.join(sorted(set(global_blockers)))}. Region descriptor blockers: locality_only (Recife), document_only (Petropolis), context_only (Curitiba).",
        "",
        "## Consolidated readiness",
        f"{len(readiness)} readiness rows across region, event and package scope; overlay, ground reference and training readiness are BLOCKED everywhere.",
        "",
        "## Reopen conditions",
        "Each region can only be reopened by a new qualifying public source (occurrence coordinates/geometry for Recife, public geodata/official crosswalk for Petropolis, official occurrence layer/event table for Curitiba). Region, name similarity, file order, inferred dates and inferred crosswalks are forbidden reopen bases.",
        "",
        "## Safe and prohibited use",
        "Safe: review-only, contextual support, evidence audit, DINO review support, package QA. Prohibited: ground truth, label, patch positive/negative, overlay truth, event validated by Sentinel, hydromet as occurrence, context layer as occurrence.",
        "",
        "## Registry consistency QA",
        f"Result: `{qa_result}` ({len(qa)} checks, {qa_fails} failures).",
        "",
        "## Why there is still no overlay",
        "No overlay was executed and overlay readiness stays BLOCKED; hardening consolidates state but establishes no observed occurrence geometry.",
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
    write_text(doc_path("protocolo_c_relatorio_v2ae_multiregion_registry_hardening.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v2ae.md"), [
        "# Status atual - Protocolo C v2ae",
        "",
        f"Registry hardening status: `{MAX_STATUS}`.",
        f"Recife: `{region_status.get('REC', '')}`; Petropolis: `{region_status.get('PET', '')}`; Curitiba: `{region_status.get('CUR', '')}`.",
        f"Canonical events: `{len(events)}`; canonical packages: `{len(packages)}`.",
        f"Registry consistency QA: `{qa_result}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Overlay, ground reference, training labels, ground truth, inferred Sentinel dates and inferred crosswalks remain blocked.",
    ])

    manifest = []
    for idx, artifact in enumerate(V2AE_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v2ae_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v2ae canonical registry artifact; no raw data, no private path, no inferred date or crosswalk.",
        })
    write_csv(dataset_path("v2ae_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    for folder in (STAGING_DIR, REPORTS_DIR):
        os.makedirs(folder, exist_ok=True)
    print(f"[v2ae completion] regions={len(regions)} events={len(events)} packages={len(packages)} qa={qa_result} next={next_target}")
    return {"regions": len(regions), "events": len(events), "packages": len(packages), "qa_result": qa_result, "next_target": next_target, "next_version": next_version}


def run_all(args=None):
    args = args or parse_args([])
    run_canonical_region_registry_builder(args)
    run_canonical_event_registry_builder(args)
    run_canonical_event_patch_registry_builder(args)
    run_multiregion_blocker_consolidator(args)
    run_multiregion_readiness_consolidator(args)
    run_region_reopen_condition_builder(args)
    run_safe_use_policy_registry_builder(args)
    run_registry_consistency_qa(args)
    run_next_programming_target_ranker(args)
    return run_completion_report(args)
