#!/usr/bin/env python3
"""v1uz Curitiba context-only hold and multi-region priority re-ranking.

Pure registry / synthesis stage. It reads existing v1uy/v1ux/v1uw/v1us/v1uo/
v1uu/v1ur outputs without modifying them, consolidates Curitiba as a
context-only hold, hardens non-occurrence guards, refreshes the multi-region
closure, blocker and readiness matrices, and re-ranks the next real programming
target by score. No overlay, no ground reference, no label, no coordinate
inference, no raw data versioning, no web access.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v1uz"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"
REPORTS_DIR = "local_only/protocolo_c/curitiba_context_only_hold/reports/v1uz"

MAX_STATUS = "CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL"
RECIFE_STATUS = "RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL"
PETROPOLIS_STATUS = "PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA"

GUARDRAIL_COLUMNS = [
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "dino_usage",
    "no_overlay_executed", "no_coordinates_invented", "patch_bound_truth",
    "operational_validation", "event_candidate_only", "context_only_hold",
    "controlled_feature_download_executed", "geocoding_executed",
    "centroid_used", "raw_data_versioned",
]

# Guardrail columns whose only safe value is "false".
GUARDRAIL_MUST_BE_FALSE = {
    "ground_truth_operational", "can_create_ground_reference",
    "can_create_training_label", "can_reopen_protocol_b", "patch_bound_truth",
    "operational_validation", "controlled_feature_download_executed",
    "geocoding_executed", "centroid_used", "raw_data_versioned",
}

# Status tokens that must never appear as a final/affirmative status value.
FORBIDDEN_STATUS_TOKENS = [
    "GROUND_REFERENCE", "GROUND_TRUTH", "TRAINING_LABEL", "PATCH_POSITIVE",
    "PATCH_NEGATIVE", "OPERATIONAL_VALIDATED", "OBSERVED_FLOOD_LABEL",
    "FLOOD_DETECTED", "OCCURRENCE_GEOMETRY_VALIDATED",
]
FORBIDDEN_STATUS_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_STATUS_TOKENS) + r")\b")
TOOL_NAME_RE = re.compile(r"\b(claude|codex|llm|assistant|chatgpt|openai|anthropic|copilot|gemini)\b", re.I)
ABSOLUTE_PATH_RE = re.compile(r"(?:[A-Za-z]:\\|/Users/|/home/|/mnt/|\\\\)")

HOLD_COLUMNS = [
    "hold_id", "event_id", "candidate_event_id", "city", "uf",
    "event_candidate_status", "hydromet_support_status", "context_layer_status",
    "possible_occurrence_layer_status", "controlled_feature_download_status",
    "overlay_status", "ground_reference_status", "hold_status",
    "can_create_ground_reference", "can_create_training_label", "notes",
]
GUARD_COLUMNS = [
    "guard_id", "event_id", "evidence_type", "non_occurrence_reason",
    "prohibited_use", "safe_use", "can_create_ground_reference",
    "can_create_training_label", "patch_bound_truth", "notes",
]
EVENT_PATCH_HOLD_COLUMNS = [
    "hold_update_id", "event_patch_candidate_id", "event_id", "patch_id",
    "linkage_basis", "context_only_hold_status", "sentinel_date_status",
    "occurrence_geometry_status", "overlay_status", "ground_reference_status",
    "event_patch_candidate_only", "can_create_ground_reference",
    "can_create_training_label", "blocker", "notes",
]
CLOSURE_COLUMNS = [
    "closure_id", "region", "event_id", "closure_status", "best_evidence_type",
    "best_evidence_strength", "coordinate_status", "geometry_status",
    "overlay_status", "ground_reference_status", "main_blocker",
    "recommended_future_reopen_condition", "notes",
]
BLOCKER_MATRIX_COLUMNS = [
    "blocker_id", "region", "event_id", "blocker", "status", "applies",
    *GUARDRAIL_COLUMNS, "notes",
]
READINESS_SYNTH_COLUMNS = [
    "synthesis_id", "region", "event_id", "dimension", "classification",
    "basis", *GUARDRAIL_COLUMNS, "notes",
]
RANKER_COLUMNS = [
    "rank", "next_target", "programming_value", "ground_truth_value",
    "blocker_reduction_value", "expected_effort", "overclaim_risk",
    "recommended_version", "recommended_action", "notes",
]
TRANSITION_COLUMNS = [
    "transition_id", "selected_next_target", "selected_version", "reason",
    "required_inputs", "expected_outputs", "guardrails",
    "implementation_not_started", "notes",
]
AUDIT_COLUMNS = [
    "audit_id", "artifact", "check_type", "violation_count", "status", "notes",
]
NEXT_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]
MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UZ_ARTIFACTS = [
    "configs/protocolo_c/v1uz_curitiba_context_only_hold_policy.yaml",
    "configs/protocolo_c/v1uz_multiregion_closure_policy.yaml",
    "configs/protocolo_c/v1uz_multiregion_blocker_policy.yaml",
    "configs/protocolo_c/v1uz_next_programming_target_policy.yaml",
    "configs/protocolo_c/v1uz_version_transition_policy.yaml",
    "configs/protocolo_c/v1uz_guardrail_audit_policy.yaml",
    "datasets/protocolo_c/v1uz_curitiba_context_only_hold_registry.csv",
    "datasets/protocolo_c/v1uz_curitiba_non_occurrence_guard_registry.csv",
    "datasets/protocolo_c/v1uz_curitiba_event_patch_hold_update.csv",
    "datasets/protocolo_c/v1uz_multiregion_closure_status.csv",
    "datasets/protocolo_c/v1uz_multiregion_blocker_matrix.csv",
    "datasets/protocolo_c/v1uz_multiregion_readiness_synthesis.csv",
    "datasets/protocolo_c/v1uz_next_programming_target_ranker.csv",
    "datasets/protocolo_c/v1uz_version_transition_plan.csv",
    "datasets/protocolo_c/v1uz_guardrail_audit.csv",
    "datasets/protocolo_c/v1uz_next_actions_registry.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uz_curitiba_context_only_hold_multiregion_rerank.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uz_curitiba_context_only_hold_multiregion_rerank.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1uz.md",
]


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
        "event_candidate_only": "true",
        "context_only_hold": "true",
        "controlled_feature_download_executed": "false",
        "geocoding_executed": "false",
        "centroid_used": "false",
        "raw_data_versioned": "false",
    }


def write_policy_configs():
    policies = {
        "v1uz_curitiba_context_only_hold_policy.yaml": [
            "max_status: CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL",
            "context_is_not_occurrence: true",
            "hydromet_is_not_observed_occurrence: true",
            "controlled_feature_download_executed: false",
            "overlay_allowed: false",
            "ground_reference_allowed: false",
            "training_label_allowed: false",
        ],
        "v1uz_multiregion_closure_policy.yaml": [
            "recife_status: RECIFE_CONTEXTUAL_COORDINATE_LAYER_CONSOLIDATED_NON_OPERATIONAL",
            "petropolis_status: PETROPOLIS_DOCUMENT_ONLY_NO_GEODATA",
            "curitiba_status: CURITIBA_CONTEXT_ONLY_HOLD_NON_OPERATIONAL",
            "operational_validation: false",
        ],
        "v1uz_multiregion_blocker_policy.yaml": [
            "required_blockers:",
            "  - no_observed_geometry",
            "  - no_occurrence_coordinates",
            "  - no_sentinel_date",
            "  - no_overlay",
            "  - no_ground_reference",
            "  - no_training_label",
            "auto_unblock_allowed: false",
        ],
        "v1uz_next_programming_target_policy.yaml": [
            "ranking: score_based_not_hardcoded",
            "programming_weight: 0.5",
            "blocker_reduction_weight: 0.5",
            "effort_penalty: {LOW: 0, MEDIUM: 5, HIGH: 15}",
            "overclaim_penalty: {LOW: 0, MEDIUM: 10, HIGH: 25}",
        ],
        "v1uz_version_transition_policy.yaml": [
            "next_version_prefix: v2aa",
            "implementation_not_started: true",
            "selected_from: next_programming_target_ranker",
        ],
        "v1uz_guardrail_audit_policy.yaml": [
            "must_be_false:",
            "  - ground_truth_operational",
            "  - can_create_ground_reference",
            "  - can_create_training_label",
            "forbidden_status_tokens_blocked: true",
            "absolute_path_blocked: true",
            "local_only_leak_blocked: true",
            "tool_name_leak_blocked: true",
        ],
    }
    for name, lines in policies.items():
        write_text(config_path(name), lines)


# --------------------------------------------------------------------------
# Input accessors (read-only, never modify prior outputs)
# --------------------------------------------------------------------------

def curitiba_event_candidate():
    rows = load_csv(dataset_path("v1uw_curitiba_event_candidate_status.csv"))
    if rows:
        return rows[0]
    return {
        "candidate_event_id": "CE_v1uv_0000",
        "proposed_event_id": "CUR_2022_01_15",
        "status": "CURITIBA_EVENT_CANDIDATE_HYDROMET_SUPPORTED",
        "hydromet_support": "AVAILABLE",
    }


def curitiba_context_classes():
    return load_csv(dataset_path("v1uy_curitiba_context_layer_classification.csv"))


def curitiba_occurrence_audit():
    return load_csv(dataset_path("v1uy_curitiba_possible_occurrence_layer_audit.csv"))


def curitiba_download_plan():
    return load_csv(dataset_path("v1uy_curitiba_controlled_feature_download_plan.csv"))


def curitiba_prelinks():
    return load_csv(dataset_path("v1uw_curitiba_event_patch_prelink_update.csv"))


# --------------------------------------------------------------------------
# 1. Curitiba Context-Only Hold Builder
# --------------------------------------------------------------------------

def run_curitiba_context_only_hold_builder(args=None):
    write_policy_configs()
    cand = curitiba_event_candidate()
    classes = curitiba_context_classes()
    audit = curitiba_occurrence_audit()
    plans = curitiba_download_plan()
    context_layer_status = "CONTEXT_LAYERS_PRESENT" if classes else "NO_CONTEXT_LAYER"
    occ_status = (
        "POSSIBLE_OCCURRENCE_LAYER_ABSENT" if not audit
        else "POSSIBLE_OCCURRENCE_LAYER_CANDIDATE_UNCONFIRMED"
    )
    download_status = "NO_CONTROLLED_DOWNLOAD_RECOMMENDED"
    if any(p.get("plan_status") == "CONTROLLED_DOWNLOAD_CANDIDATE_FOR_V1UZ" for p in plans):
        download_status = "CONTROLLED_DOWNLOAD_CANDIDATE_NOT_EXECUTED"
    row = {
        "hold_id": "HOLD_v1uz_0000",
        "event_id": cand.get("proposed_event_id", "CUR_2022_01_15"),
        "candidate_event_id": cand.get("candidate_event_id", "CE_v1uv_0000"),
        "city": "Curitiba",
        "uf": "PR",
        "event_candidate_status": cand.get("status", "CURITIBA_EVENT_CANDIDATE_HYDROMET_SUPPORTED"),
        "hydromet_support_status": "HYDROMET_SUPPORT_PRESENT_NOT_OCCURRENCE",
        "context_layer_status": context_layer_status,
        "possible_occurrence_layer_status": occ_status,
        "controlled_feature_download_status": download_status,
        "overlay_status": "BLOCKED",
        "ground_reference_status": "BLOCKED",
        "hold_status": MAX_STATUS,
        "can_create_ground_reference": "false",
        "can_create_training_label": "false",
        "notes": "Curitiba consolidated as context-only hold; event candidate and hydromet support exist but no observed occurrence layer, no controlled download, no overlay, no ground reference.",
    }
    out = dataset_path("v1uz_curitiba_context_only_hold_registry.csv")
    write_csv(out, HOLD_COLUMNS, [row])
    print(f"[v1uz hold] status={row['hold_status']} -> {out}")
    return [row]


# --------------------------------------------------------------------------
# 2. Curitiba Non-Occurrence Guard Builder
# --------------------------------------------------------------------------

def run_curitiba_non_occurrence_guard_builder(args=None):
    cand = curitiba_event_candidate()
    event_id = cand.get("proposed_event_id", "CUR_2022_01_15")
    guards = [
        ("ADMINISTRATIVE_CONTEXT_LAYER", "administrative_boundary_is_not_an_observed_occurrence",
         "use_as_observed_occurrence_or_ground_reference", "context_reference_for_human_review_only"),
        ("DRAINAGE_CONTEXT_LAYER", "drainage_network_is_not_an_observed_occurrence",
         "use_as_observed_occurrence_or_ground_reference", "hydrographic_context_for_human_review_only"),
        ("OFFICIAL_ALERT_OR_NOTICE", "alert_or_notice_is_not_ground_truth",
         "use_as_ground_truth_or_training_label", "temporal_event_candidate_signal_only"),
        ("HYDROMET_SUPPORT", "hydromet_series_is_not_an_observed_occurrence_geometry",
         "use_as_observed_occurrence_or_overlay_input", "temporal_hazard_support_for_event_candidate_only"),
        ("CONTEXT_LAYER_METADATA", "context_layer_is_not_a_training_label",
         "use_as_training_label_or_patch_truth", "structural_context_for_review_only"),
        ("REGION_ONLY_EVENT_PATCH_LINKAGE", "region_only_linkage_is_not_patch_bound_truth",
         "use_as_patch_positive_or_patch_negative", "candidate_only_region_linkage_for_review"),
    ]
    rows = []
    for evidence_type, reason, prohibited, safe in guards:
        rows.append({
            "guard_id": f"GUARD_v1uz_{len(rows):04d}",
            "event_id": event_id,
            "evidence_type": evidence_type,
            "non_occurrence_reason": reason,
            "prohibited_use": prohibited,
            "safe_use": safe,
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "patch_bound_truth": "false",
            "notes": "Guard prevents promoting context/alert/hydromet/region-only evidence into occurrence, ground reference, label or patch truth.",
        })
    out = dataset_path("v1uz_curitiba_non_occurrence_guard_registry.csv")
    write_csv(out, GUARD_COLUMNS, rows)
    print(f"[v1uz guard] rows={len(rows)} -> {out}")
    return rows


# --------------------------------------------------------------------------
# 3. Curitiba Event-Patch Hold Updater
# --------------------------------------------------------------------------

def run_curitiba_event_patch_hold_updater(args=None):
    prelinks = curitiba_prelinks()
    us_candidates = {
        (r.get("region"), r.get("patch_id")): r.get("event_patch_candidate_id")
        for r in load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))
    }
    rows = []
    for pre in prelinks:
        patch_id = pre.get("patch_id", "")
        region = pre.get("region", "CUR")
        epc = us_candidates.get((region, patch_id)) or f"EPC_CUR_v1uz_{len(rows):05d}"
        rows.append({
            "hold_update_id": f"EPH_v1uz_{len(rows):05d}",
            "event_patch_candidate_id": epc,
            "event_id": pre.get("proposed_event_id", "CUR_2022_01_15"),
            "patch_id": patch_id,
            "linkage_basis": pre.get("linkage_basis", "REGION_ONLY_EVENT_CANDIDATE"),
            "context_only_hold_status": "CONTEXT_ONLY_HOLD",
            "sentinel_date_status": pre.get("sentinel_date_status", "SENTINEL_DATE_MISSING"),
            "occurrence_geometry_status": "NO_OCCURRENCE_GEOMETRY",
            "overlay_status": "BLOCKED",
            "ground_reference_status": "BLOCKED",
            "event_patch_candidate_only": "true",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "context_only_hold_sentinel_date_missing_no_occurrence_geometry",
            "notes": "Additive hold update; v1us/v1uw not modified; linkage stays region-only candidate.",
        })
    out = dataset_path("v1uz_curitiba_event_patch_hold_update.csv")
    write_csv(out, EVENT_PATCH_HOLD_COLUMNS, rows)
    print(f"[v1uz event-patch hold] rows={len(rows)} -> {out}")
    return rows


# --------------------------------------------------------------------------
# 4. Multi-Region Closure Status Builder
# --------------------------------------------------------------------------

REGION_CLOSURE = {
    "REC": {
        "closure_status": RECIFE_STATUS,
        "best_evidence_type": "CONTEXTUAL_COORDINATE_LAYER",
        "best_evidence_strength": "STRONG",
        "coordinate_status": "CONTEXTUAL_COORDINATE_ONLY_NO_OCCURRENCE",
        "main_blocker": "no_occurrence_coordinates",
        "recommended_future_reopen_condition": "public_occurrence_coordinate_or_observed_geometry_source_published",
    },
    "PET": {
        "closure_status": PETROPOLIS_STATUS,
        "best_evidence_type": "OFFICIAL_DOCUMENT_ONLY",
        "best_evidence_strength": "WEAK",
        "coordinate_status": "NO_COORDINATE_EVIDENCE",
        "main_blocker": "no_geodata",
        "recommended_future_reopen_condition": "public_geodata_or_occurrence_layer_published",
    },
    "CUR": {
        "closure_status": MAX_STATUS,
        "best_evidence_type": "PUBLIC_CONTEXT_LAYER_AND_HYDROMET",
        "best_evidence_strength": "MODERATE",
        "coordinate_status": "NO_OCCURRENCE_COORDINATE",
        "main_blocker": "no_occurrence_layer",
        "recommended_future_reopen_condition": "public_occurrence_layer_or_controlled_feature_source_published",
    },
}


def run_multiregion_closure_status_builder(args=None):
    registry = load_csv(dataset_path("v1uo_multiregion_event_registry.csv"))
    hold = load_csv(dataset_path("v1uz_curitiba_context_only_hold_registry.csv"))
    cur_event_id = hold[0].get("event_id", "CUR_2022_01_15") if hold else "CUR_2022_01_15"
    rows = []
    seen = set()
    for reg in registry:
        region = reg.get("region", "")
        spec = REGION_CLOSURE.get(region)
        if not spec:
            continue
        event_id = reg.get("event_id", "")
        if region == "CUR":
            event_id = cur_event_id
        key = (region, event_id)
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "closure_id": f"CLO_v1uz_{len(rows):04d}",
            "region": region,
            "event_id": event_id,
            "closure_status": spec["closure_status"],
            "best_evidence_type": spec["best_evidence_type"],
            "best_evidence_strength": spec["best_evidence_strength"],
            "coordinate_status": spec["coordinate_status"],
            "geometry_status": "NO_OBSERVED_GEOMETRY",
            "overlay_status": "BLOCKED",
            "ground_reference_status": "BLOCKED",
            "main_blocker": spec["main_blocker"],
            "recommended_future_reopen_condition": spec["recommended_future_reopen_condition"],
            "notes": "Non-operational regional closure; no overlay, no ground reference, no label, no coordinate inference.",
        })
    out = dataset_path("v1uz_multiregion_closure_status.csv")
    write_csv(out, CLOSURE_COLUMNS, rows)
    print(f"[v1uz closure] rows={len(rows)} -> {out}")
    return rows


# --------------------------------------------------------------------------
# 5. Multi-Region Blocker Matrix Builder
# --------------------------------------------------------------------------

ALL_BLOCKERS = [
    "no_observed_geometry", "no_occurrence_coordinates", "no_sentinel_date",
    "no_overlay", "no_ground_reference", "no_training_label", "locality_only",
    "context_only", "document_only", "hydromet_only", "patch_truth_forbidden",
]

REGION_BLOCKER_APPLIES = {
    "REC": {
        "no_observed_geometry", "no_occurrence_coordinates", "no_sentinel_date",
        "no_overlay", "no_ground_reference", "no_training_label", "locality_only",
        "patch_truth_forbidden",
    },
    "PET": {
        "no_observed_geometry", "no_occurrence_coordinates", "no_sentinel_date",
        "no_overlay", "no_ground_reference", "no_training_label", "document_only",
        "patch_truth_forbidden",
    },
    "CUR": {
        "no_observed_geometry", "no_occurrence_coordinates", "no_sentinel_date",
        "no_overlay", "no_ground_reference", "no_training_label", "context_only",
        "hydromet_only", "patch_truth_forbidden",
    },
}


def run_multiregion_blocker_matrix_builder(args=None):
    closures = load_csv(dataset_path("v1uz_multiregion_closure_status.csv"))
    rows = []
    for clo in closures:
        region = clo.get("region", "")
        event_id = clo.get("event_id", "")
        applies_set = REGION_BLOCKER_APPLIES.get(region, set(ALL_BLOCKERS))
        for blocker in ALL_BLOCKERS:
            applies = blocker in applies_set
            rows.append({
                "blocker_id": f"MB_v1uz_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "blocker": blocker,
                "status": "BLOCKED" if applies else "NOT_APPLICABLE",
                "applies": "true" if applies else "false",
                **guardrails(),
                "notes": "Multi-region blocker matrix; blockers are not auto-unblocked and never become occurrence/label/ground reference.",
            })
    out = dataset_path("v1uz_multiregion_blocker_matrix.csv")
    write_csv(out, BLOCKER_MATRIX_COLUMNS, rows)
    print(f"[v1uz blocker matrix] rows={len(rows)} -> {out}")
    return rows


# --------------------------------------------------------------------------
# 6. Multi-Region Readiness Synthesizer
# --------------------------------------------------------------------------

READINESS_DIMENSIONS = [
    "event_registry", "official_source", "temporal_support", "hazard_support",
    "locality_support", "contextual_coordinate_support",
    "occurrence_coordinate_support", "observed_geometry_support",
    "sentinel_date_support", "overlay_readiness", "ground_reference_readiness",
    "training_readiness",
]

REGION_READINESS = {
    "REC": {
        "event_registry": "STRONG", "official_source": "STRONG",
        "temporal_support": "STRONG", "hazard_support": "STRONG",
        "locality_support": "STRONG", "contextual_coordinate_support": "STRONG",
        "occurrence_coordinate_support": "ABSENT",
        "observed_geometry_support": "ABSENT", "sentinel_date_support": "WEAK",
        "overlay_readiness": "BLOCKED", "ground_reference_readiness": "BLOCKED",
        "training_readiness": "BLOCKED",
    },
    "PET": {
        "event_registry": "STRONG", "official_source": "STRONG",
        "temporal_support": "STRONG", "hazard_support": "MODERATE",
        "locality_support": "MODERATE", "contextual_coordinate_support": "ABSENT",
        "occurrence_coordinate_support": "ABSENT",
        "observed_geometry_support": "ABSENT", "sentinel_date_support": "WEAK",
        "overlay_readiness": "BLOCKED", "ground_reference_readiness": "BLOCKED",
        "training_readiness": "BLOCKED",
    },
    "CUR": {
        "event_registry": "MODERATE", "official_source": "STRONG",
        "temporal_support": "STRONG", "hazard_support": "MODERATE",
        "locality_support": "MODERATE", "contextual_coordinate_support": "WEAK",
        "occurrence_coordinate_support": "ABSENT",
        "observed_geometry_support": "ABSENT", "sentinel_date_support": "ABSENT",
        "overlay_readiness": "BLOCKED", "ground_reference_readiness": "BLOCKED",
        "training_readiness": "BLOCKED",
    },
}


def run_multiregion_readiness_synthesizer(args=None):
    closures = load_csv(dataset_path("v1uz_multiregion_closure_status.csv"))
    rows = []
    for clo in closures:
        region = clo.get("region", "")
        event_id = clo.get("event_id", "")
        dims = REGION_READINESS.get(region, {})
        for dim in READINESS_DIMENSIONS:
            classification = dims.get(dim, "UNKNOWN")
            rows.append({
                "synthesis_id": f"RS_v1uz_{len(rows):04d}",
                "region": region,
                "event_id": event_id,
                "dimension": dim,
                "classification": classification,
                "basis": "v1uz multi-region closure synthesis",
                **guardrails(),
                "notes": "Overlay, ground reference and training readiness remain BLOCKED for every region.",
            })
    out = dataset_path("v1uz_multiregion_readiness_synthesis.csv")
    write_csv(out, READINESS_SYNTH_COLUMNS, rows)
    print(f"[v1uz readiness] rows={len(rows)} -> {out}")
    return rows


# --------------------------------------------------------------------------
# 7. Next Programming Target Ranker
# --------------------------------------------------------------------------

EFFORT_PENALTY = {"LOW": 0, "MEDIUM": 5, "HIGH": 15}
OVERCLAIM_PENALTY = {"LOW": 0, "MEDIUM": 10, "HIGH": 25}


def _ranker_metrics():
    candidates = load_csv(dataset_path("v1us_event_patch_candidate_registry.csv"))
    total = len(candidates) or 1
    sentinel_missing = sum(1 for r in candidates if "SENTINEL_DATE" in (r.get("blocker") or ""))
    dino = load_csv(dataset_path("v1us_dino_review_support_attachment.csv"))
    dino_available = sum(
        1 for r in dino
        if (r.get("dino_review_support_status") or "") == "DINO_REVIEW_SUPPORT_AVAILABLE"
    )
    dino_total = len(dino) or 1
    return {
        "total": total,
        "sentinel_missing": sentinel_missing,
        "sentinel_ratio": sentinel_missing / total,
        "dino_remaining_ratio": max(0, (dino_total - dino_available)) / dino_total,
        "regions": 3,
    }


def _candidate_targets(m):
    sentinel_pct = round(m["sentinel_ratio"] * 100)
    dino_remaining_pct = round(m["dino_remaining_ratio"] * 100)
    return [
        {
            "next_target": "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES",
            "programming_value": sentinel_pct,
            "ground_truth_value": 0,
            "blocker_reduction_value": sentinel_pct,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": f"{m['sentinel_missing']} of {m['total']} event-patch candidates miss a Sentinel scene date; recovering it is metadata-only and unblocks temporal linkage.",
        },
        {
            "next_target": "MULTI_REGION_REGISTRY_HARDENING",
            "programming_value": 60,
            "ground_truth_value": 0,
            "blocker_reduction_value": 40,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Consolidates Recife/Petropolis/Curitiba registries; organizational value but does not remove occurrence blockers.",
        },
        {
            "next_target": "EVENT_PATCH_PACKAGE_SCHEMA_HARDENING",
            "programming_value": 55,
            "ground_truth_value": 0,
            "blocker_reduction_value": 35,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Hardens event-patch package schema; supports later Sentinel/temporal work without creating truth.",
        },
        {
            "next_target": "DINO_REVIEW_SUPPORT_COMPLETION",
            "programming_value": max(5, dino_remaining_pct),
            "ground_truth_value": 0,
            "blocker_reduction_value": 10,
            "expected_effort": "MEDIUM",
            "overclaim_risk": "LOW",
            "notes": "DINO review support is already attached for nearly all candidates; remaining completion is small and review-only.",
        },
        {
            "next_target": "PUBLIC_SOURCE_RECHECK_HOLD",
            "programming_value": 20,
            "ground_truth_value": 0,
            "blocker_reduction_value": 25,
            "expected_effort": "LOW",
            "overclaim_risk": "LOW",
            "notes": "Periodic recheck of public sources; held until a new public source appears.",
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


def _score(target):
    base = 0.5 * target["programming_value"] + 0.5 * target["blocker_reduction_value"]
    return base - EFFORT_PENALTY.get(target["expected_effort"], 5) - OVERCLAIM_PENALTY.get(target["overclaim_risk"], 10)


TARGET_VERSION = {
    "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES": "v2aa — Sentinel Date Recovery for Event-Patch Packages",
    "MULTI_REGION_REGISTRY_HARDENING": "v2aa — Multi-Region Registry Hardening",
    "EVENT_PATCH_PACKAGE_SCHEMA_HARDENING": "v2aa — Event-Patch Package Schema Hardening",
    "DINO_REVIEW_SUPPORT_COMPLETION": "v2aa — DINO Review Support Completion",
    "PUBLIC_SOURCE_RECHECK_HOLD": "v2aa — Public Source Recheck Hold",
    "STOP_GROUND_TRUTH_SEARCH_UNTIL_NEW_SOURCE": "v2aa — Ground Truth Search Hold",
}


def run_next_programming_target_ranker(args=None):
    metrics = _ranker_metrics()
    targets = _candidate_targets(metrics)
    targets.sort(key=_score, reverse=True)
    rows = []
    for idx, target in enumerate(targets, start=1):
        rows.append({
            "rank": str(idx),
            "next_target": target["next_target"],
            "programming_value": str(target["programming_value"]),
            "ground_truth_value": str(target["ground_truth_value"]),
            "blocker_reduction_value": str(target["blocker_reduction_value"]),
            "expected_effort": target["expected_effort"],
            "overclaim_risk": target["overclaim_risk"],
            "recommended_version": TARGET_VERSION.get(target["next_target"], ""),
            "recommended_action": "SELECTED_NEXT_TARGET" if idx == 1 else "RANKED_ALTERNATIVE",
            "notes": target["notes"],
        })
    out = dataset_path("v1uz_next_programming_target_ranker.csv")
    write_csv(out, RANKER_COLUMNS, rows)
    print(f"[v1uz ranker] selected={rows[0]['next_target'] if rows else 'none'} -> {out}")
    return rows


def selected_next_target():
    rows = load_csv(dataset_path("v1uz_next_programming_target_ranker.csv"))
    if not rows:
        rows = run_next_programming_target_ranker()
    return rows[0]


# --------------------------------------------------------------------------
# 8. Version Transition Planner
# --------------------------------------------------------------------------

def run_version_transition_planner(args=None):
    top = selected_next_target()
    target = top.get("next_target", "")
    version = TARGET_VERSION.get(target, "v2aa — Multi-Region Registry Hardening")
    inputs_map = {
        "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES":
            "v1us_event_patch_candidate_registry.csv|v1us_event_patch_readiness_matrix.csv|public Sentinel scene metadata catalog",
        "MULTI_REGION_REGISTRY_HARDENING":
            "v1uz_multiregion_closure_status.csv|v1uz_multiregion_readiness_synthesis.csv|v1uo_multiregion_event_registry.csv",
        "EVENT_PATCH_PACKAGE_SCHEMA_HARDENING":
            "v1us_event_patch_candidate_registry.csv|v1us_dino_review_support_attachment.csv",
        "DINO_REVIEW_SUPPORT_COMPLETION":
            "v1us_dino_review_support_attachment.csv",
    }
    outputs_map = {
        "SENTINEL_DATE_RECOVERY_FOR_EVENT_PATCH_PACKAGES":
            "sentinel_scene_date_resolution_registry|event_patch_temporal_linkage_update",
        "MULTI_REGION_REGISTRY_HARDENING":
            "hardened_multiregion_registry|consolidated_blocker_matrix",
        "EVENT_PATCH_PACKAGE_SCHEMA_HARDENING":
            "event_patch_package_schema|schema_validation_report",
        "DINO_REVIEW_SUPPORT_COMPLETION":
            "dino_review_support_completion_registry",
    }
    row = {
        "transition_id": "VT_v1uz_0000",
        "selected_next_target": target,
        "selected_version": version,
        "reason": f"Selected from v1uz score-based ranker (rank 1: {target}); {top.get('notes', '')}",
        "required_inputs": inputs_map.get(target, "v1uz registries"),
        "expected_outputs": outputs_map.get(target, "hardened registries"),
        "guardrails": "ground_truth_operational=false;can_create_ground_reference=false;can_create_training_label=false;no_overlay_executed=true;no_coordinates_invented=true",
        "implementation_not_started": "true",
        "notes": "Transition is planning-only; the next version is not implemented in v1uz.",
    }
    out = dataset_path("v1uz_version_transition_plan.csv")
    write_csv(out, TRANSITION_COLUMNS, [row])
    print(f"[v1uz transition] version={version} (not started) -> {out}")
    return [row]


# --------------------------------------------------------------------------
# 9. Guardrail Audit
# --------------------------------------------------------------------------

def _scan_file(path):
    """Return dict check_type -> violation_count for one file."""
    counts = {
        "forbidden_true_value": 0,
        "forbidden_status": 0,
        "absolute_path": 0,
        "local_only_leak": 0,
        "tool_name_leak": 0,
    }
    if not os.path.exists(path):
        return counts
    is_csv = path.lower().endswith(".csv")
    if is_csv:
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
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        counts["forbidden_status"] += len(FORBIDDEN_STATUS_RE.findall(text))
        counts["local_only_leak"] += text.count("local_only/") + text.count("local_only\\")
        counts["absolute_path"] += len(ABSOLUTE_PATH_RE.findall(text))
        counts["tool_name_leak"] += len(TOOL_NAME_RE.findall(text))
    return counts


def run_guardrail_audit(args=None, artifacts=None):
    if artifacts is None:
        artifacts = [artifact_path(a) for a in V1UZ_ARTIFACTS]
    rows = []
    for art in artifacts:
        rel = art.replace("\\", "/")
        counts = _scan_file(art)
        for check_type, violation_count in counts.items():
            status = "PASS" if violation_count == 0 else "FAIL"
            rows.append({
                "audit_id": f"AUD_v1uz_{len(rows):04d}",
                "artifact": os.path.basename(rel),
                "check_type": check_type,
                "violation_count": str(violation_count),
                "status": status,
                "notes": "Clean" if status == "PASS" else "Guardrail violation detected; artifact must not be versioned as-is.",
            })
    out = dataset_path("v1uz_guardrail_audit.csv")
    write_csv(out, AUDIT_COLUMNS, rows)
    fails = sum(1 for r in rows if r["status"] == "FAIL")
    print(f"[v1uz guardrail audit] rows={len(rows)} fails={fails} -> {out}")
    return rows


# --------------------------------------------------------------------------
# 10. Completion Report
# --------------------------------------------------------------------------

def run_completion_report(args=None):
    write_policy_configs()
    hold = load_csv(dataset_path("v1uz_curitiba_context_only_hold_registry.csv")) or run_curitiba_context_only_hold_builder(args)
    guards = load_csv(dataset_path("v1uz_curitiba_non_occurrence_guard_registry.csv")) or run_curitiba_non_occurrence_guard_builder(args)
    eph = load_csv(dataset_path("v1uz_curitiba_event_patch_hold_update.csv")) or run_curitiba_event_patch_hold_updater(args)
    closures = load_csv(dataset_path("v1uz_multiregion_closure_status.csv")) or run_multiregion_closure_status_builder(args)
    blockers = load_csv(dataset_path("v1uz_multiregion_blocker_matrix.csv")) or run_multiregion_blocker_matrix_builder(args)
    readiness = load_csv(dataset_path("v1uz_multiregion_readiness_synthesis.csv")) or run_multiregion_readiness_synthesizer(args)
    ranker = load_csv(dataset_path("v1uz_next_programming_target_ranker.csv")) or run_next_programming_target_ranker(args)
    transition = load_csv(dataset_path("v1uz_version_transition_plan.csv")) or run_version_transition_planner(args)

    top = ranker[0] if ranker else {}
    next_target = top.get("next_target", "")
    next_version = transition[0].get("selected_version", "") if transition else ""
    closure_summary = {r.get("region"): r.get("closure_status") for r in closures}

    write_csv(dataset_path("v1uz_next_actions_registry.csv"), NEXT_COLUMNS, [{
        "action_id": "NA_v1uz_0000",
        "event_id": hold[0].get("event_id", "CUR_2022_01_15") if hold else "CUR_2022_01_15",
        "action_type": next_target,
        "priority": "1",
        "description": "Selected from v1uz score-based next-programming-target ranker.",
        "target": "MULTI_REGION_EVENT_PATCH_PROGRAMMING",
        "status": "RECOMMENDED_NEXT_STEP",
        "notes": "No overlay, labels, ground truth or ground reference; next version not started.",
    }])

    lines = [
        "# Protocolo C v1uz - Curitiba Context-Only Hold and Multi-Region Priority Re-Ranking",
        "",
        f"- Curitiba hold status: `{hold[0].get('hold_status', MAX_STATUS) if hold else MAX_STATUS}`",
        f"- non-occurrence guards: `{len(guards)}`",
        f"- Curitiba event-patch hold updates: `{len(eph)}`",
        f"- multi-region closures: `{len(closures)}`",
        f"- multi-region blocker rows: `{len(blockers)}`",
        f"- multi-region readiness rows: `{len(readiness)}`",
        f"- Recife closure: `{closure_summary.get('REC', 'n/a')}`",
        f"- Petropolis closure: `{closure_summary.get('PET', 'n/a')}`",
        f"- Curitiba closure: `{closure_summary.get('CUR', 'n/a')}`",
        f"- selected next programming target: `{next_target}`",
        f"- suggested next version: `{next_version}`",
        "",
        "v1uz consolidated Curitiba as a context-only hold and re-ranked the next real programming target. It did not execute overlay, geocoding, centroid use, label creation, ground truth, ground reference, operational validation, DINO execution, model training, event inference, coordinate inference or raw data versioning.",
    ]
    write_text(doc_path("protocolo_c_v1uz_curitiba_context_only_hold_multiregion_rerank.md"), lines)

    report = lines + [
        "",
        "## Curitiba final status",
        f"Curitiba is `{MAX_STATUS}`. An official event candidate (`CUR_2022_01_15`) and hydromet support exist, and public/contextual layers (administrative, drainage, context) exist, but there is no observed occurrence layer, no possible-occurrence layer, no base for controlled feature download, no overlay preflight and no ground reference. Hydromet support is temporal hazard context only and is not an observed occurrence.",
        "",
        "## Why Curitiba entered context-only hold",
        "The v1uy deepening probed public geodata endpoints and classified layers as administrative, drainage or unknown context. No queryable occurrence table, endpoint or layer was found, so no controlled feature download was recommended. Context layers, alerts and hydromet series cannot be promoted to observed occurrence, ground reference or label.",
        "",
        "## Recife final status",
        f"Recife is `{RECIFE_STATUS}`. The strongest evidence is a contextual coordinate layer; there is no occurrence coordinate and no observed geometry, so overlay and ground reference remain blocked.",
        "",
        "## Petropolis final status",
        f"Petropolis is `{PETROPOLIS_STATUS}`. Only official documents are available with no public geodata and no observed geometry.",
        "",
        "## Multi-region blockers",
        "Every region shares `no_observed_geometry`, `no_occurrence_coordinates`, `no_sentinel_date`, `no_overlay`, `no_ground_reference` and `no_training_label`. Region-specific blockers are `locality_only` (Recife), `document_only` (Petropolis) and `context_only`/`hydromet_only` (Curitiba). `patch_truth_forbidden` applies to all.",
        "",
        "## Multi-region readiness",
        "Event registry, official source and temporal support are present across regions. Occurrence coordinate support, observed geometry support, overlay readiness, ground reference readiness and training readiness are absent or blocked for every region.",
        "",
        "## Next programming target",
        f"The score-based ranker selected `{next_target}`. Sentinel scene dates are missing for the large majority of event-patch candidates, and recovering them is metadata-only with low overclaim risk and high blocker reduction.",
        "",
        "## Suggested next version",
        f"`{next_version}` (planning only; implementation not started).",
        "",
        "## Why there is still no ground reference, overlay or label",
        "No region has an observed occurrence geometry tied to its event. Without observed occurrence geometry there is no basis for overlay, no basis for ground reference, and no basis for a training label. Creating any of them now would be an unsupported overclaim.",
    ]
    write_text(doc_path("protocolo_c_relatorio_v1uz_curitiba_context_only_hold_multiregion_rerank.md"), report)

    write_text(doc_path("protocolo_c_status_atual_v1uz.md"), [
        "# Status atual - Protocolo C v1uz",
        "",
        f"Curitiba status: `{MAX_STATUS}`.",
        f"Recife status: `{RECIFE_STATUS}`.",
        f"Petropolis status: `{PETROPOLIS_STATUS}`.",
        f"Selected next programming target: `{next_target}`.",
        f"Suggested next version: `{next_version}`.",
        "",
        "Ground truth, ground reference, labels, overlay, inferred coordinates, geocoding, centroid use and operational validation remain blocked for every region.",
    ])

    manifest = []
    for idx, artifact in enumerate(V1UZ_ARTIFACTS):
        real = artifact_path(artifact)
        if not os.path.exists(real):
            continue
        manifest.append({
            "artifact_id": f"MAN_v1uz_{idx:04d}",
            "artifact_path": artifact.replace("\\", "/"),
            "artifact_type": os.path.splitext(artifact)[1].lstrip(".") or "text",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(real)[:16],
            "file_size_bytes": str(os.path.getsize(real)),
            "is_versionable": "true",
            "reason": "v1uz registry/synthesis artifact; no raw data, no private path.",
        })
    write_csv(dataset_path("v1uz_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    print(f"[v1uz completion] next_target={next_target} version={next_version}")
    return {"next_target": next_target, "next_version": next_version, "closures": len(closures)}


def run_all(args=None):
    args = args or parse_args([])
    run_curitiba_context_only_hold_builder(args)
    run_curitiba_non_occurrence_guard_builder(args)
    run_curitiba_event_patch_hold_updater(args)
    run_multiregion_closure_status_builder(args)
    run_multiregion_blocker_matrix_builder(args)
    run_multiregion_readiness_synthesizer(args)
    run_next_programming_target_ranker(args)
    run_version_transition_planner(args)
    run_guardrail_audit(args)
    return run_completion_report(args)
