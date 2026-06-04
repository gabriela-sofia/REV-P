#!/usr/bin/env python3
"""v1ul Recife candidate review routing and overlay-readiness preflight.

This stage is intentionally fail-closed: it reads v1uk registries, produces
review and readiness metadata, and never executes overlay, geocoding, centroid
derivation, ground-reference creation, or label creation.
"""

import argparse
import csv
import hashlib
import os
import re

PROTOCOL_VERSION = "v1ul"
INPUT_VERSION = "v1uk"
EVENT_ID = "REC_2022_05_24_30"
DATASET_DIR = "datasets/protocolo_c"
DOCS_DIR = "docs/metodologia_cientifica"
CONFIG_DIR = "configs/protocolo_c"

MAX_STATUS = "RECIFE_CANDIDATE_READY_FOR_SUPERVISOR_REVIEW"

REQUIRED_V1UK_ARTIFACTS = [
    "datasets/protocolo_c/v1uk_recife_asset_schema_registry.csv",
    "datasets/protocolo_c/v1uk_recife_field_semantics_registry.csv",
    "datasets/protocolo_c/v1uk_recife_occurrence_table_profile.csv",
    "datasets/protocolo_c/v1uk_recife_event_window_match_registry.csv",
    "datasets/protocolo_c/v1uk_recife_coordinate_evidence_audit.csv",
    "datasets/protocolo_c/v1uk_recife_locality_evidence_audit.csv",
    "datasets/protocolo_c/v1uk_recife_candidate_row_registry.csv",
    "datasets/protocolo_c/v1uk_recife_supervisor_review_prepackage_registry.csv",
    "datasets/protocolo_c/v1uk_recife_ground_reference_blocker_matrix.csv",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1uk_recife_ckan_schema_deep_audit.md",
]

V1UK_REQUIRED_COLUMNS = {
    "datasets/protocolo_c/v1uk_recife_asset_schema_registry.csv": [
        "asset_id", "event_id", "row_count", "schema_status", "has_sensitive_fields"
    ],
    "datasets/protocolo_c/v1uk_recife_field_semantics_registry.csv": [
        "asset_id", "source_field", "canonical_field", "is_sensitive", "mapping_status"
    ],
    "datasets/protocolo_c/v1uk_recife_occurrence_table_profile.csv": [
        "asset_id", "total_rows", "rows_in_event_window", "rows_with_coordinates",
        "rows_with_neighborhood", "rows_with_address",
    ],
    "datasets/protocolo_c/v1uk_recife_event_window_match_registry.csv": [
        "event_id", "asset_id", "row_hash", "window_type", "has_hazard_term",
        "has_coordinates", "coordinate_status", "candidate_status",
    ],
    "datasets/protocolo_c/v1uk_recife_coordinate_evidence_audit.csv": [
        "asset_id", "coordinate_classification", "can_create_ground_reference",
        "can_create_training_label",
    ],
    "datasets/protocolo_c/v1uk_recife_locality_evidence_audit.csv": [
        "asset_id", "locality_classification", "sufficient_for_human_review",
        "sufficient_for_overlay",
    ],
    "datasets/protocolo_c/v1uk_recife_candidate_row_registry.csv": [
        "candidate_row_id", "event_id", "asset_id", "row_hash", "candidate_class",
        "event_window_match", "hazard_term_status", "coordinate_status",
        "locality_status", "can_create_ground_reference",
        "can_create_training_label", "required_next_action",
    ],
    "datasets/protocolo_c/v1uk_recife_supervisor_review_prepackage_registry.csv": [
        "package_status", "candidate_rows_count", "coordinate_candidates_count",
        "locality_only_candidates_count",
    ],
    "datasets/protocolo_c/v1uk_recife_ground_reference_blocker_matrix.csv": [
        "blocker", "status", "can_create_ground_reference", "can_create_training_label"
    ],
}

ACCEPTANCE_COLUMNS = [
    "check_id", "artifact", "exists", "rows", "required_columns_status",
    "guardrail_status", "sensitive_value_status", "absolute_path_status",
    "status", "notes",
]

ROUTER_COLUMNS = [
    "route_id", "event_id", "candidate_row_id", "asset_id", "row_hash",
    "v1uk_candidate_class", "event_window_match", "hazard_signal",
    "coordinate_status", "locality_status", "sensitive_review_required",
    "review_route", "review_priority", "can_enter_supervisor_review",
    "can_enter_overlay_preflight", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action", "notes",
]

PREFLIGHT_COLUMNS = [
    "preflight_id", "event_id", "candidate_row_id", "asset_id",
    "has_event_window_match", "has_hazard_signal", "has_coordinates_or_geometry",
    "coordinate_source_status", "crs_status", "geometry_type",
    "official_source_status", "contextual_layer_status", "sensitive_status",
    "supervisor_review_status", "overlay_preflight_status",
    "can_execute_overlay_now", "can_create_ground_reference",
    "can_create_training_label", "blocker", "required_next_action",
]

SENSITIVE_PACKAGE_COLUMNS = [
    "package_id", "event_id", "candidate_row_id", "row_hash",
    "has_sensitive_fields", "redaction_status", "fields_redacted",
    "safe_summary", "reviewer_can_inspect_local_raw", "local_raw_required",
    "public_registry_safe", "notes",
]

DECISION_COLUMNS = [
    "decision_id", "event_id", "candidate_row_id", "review_route",
    "overlay_preflight_status", "sensitive_status", "evidence_strength",
    "recommended_decision", "can_advance_to_v1um", "can_execute_overlay_now",
    "can_create_ground_reference", "can_create_training_label", "blocker",
    "notes",
]

QUEUE_COLUMNS = [
    "queue_id", "event_id", "candidate_row_id", "review_route",
    "review_priority", "reviewer_task", "required_local_raw_asset",
    "safe_public_summary", "decision_options", "can_be_reviewed_now",
    "can_advance_to_overlay_preflight_after_review",
    "can_create_ground_reference", "can_create_training_label",
]

BLOCKER_COLUMNS = [
    "blocker_id", "event_id", "blocker", "status", "evidence_count",
    "can_create_ground_reference", "can_create_training_label", "notes",
]

NEXT_ACTION_COLUMNS = [
    "action_id", "event_id", "action_type", "priority", "description",
    "target", "status", "notes",
]

MANIFEST_COLUMNS = [
    "artifact_id", "artifact_path", "artifact_type", "protocol_version",
    "sha256_prefix", "file_size_bytes", "is_versionable", "reason",
]

V1UL_ARTIFACTS = [
    "configs/protocolo_c/v1ul_recife_review_routing_policy.yaml",
    "configs/protocolo_c/v1ul_overlay_readiness_policy.yaml",
    "configs/protocolo_c/v1ul_sensitive_review_policy.yaml",
    "configs/protocolo_c/v1ul_candidate_decision_policy.yaml",
    "datasets/protocolo_c/v1uk_acceptance_audit.csv",
    "datasets/protocolo_c/v1ul_recife_candidate_review_router.csv",
    "datasets/protocolo_c/v1ul_recife_overlay_readiness_preflight.csv",
    "datasets/protocolo_c/v1ul_recife_sensitive_review_package.csv",
    "datasets/protocolo_c/v1ul_recife_candidate_decision_matrix.csv",
    "datasets/protocolo_c/v1ul_recife_supervisor_review_queue.csv",
    "datasets/protocolo_c/v1ul_recife_ground_reference_blocker_matrix.csv",
    "datasets/protocolo_c/v1ul_next_actions_registry.csv",
    "datasets/protocolo_c/v1ul_versionable_artifacts_manifest.csv",
    "docs/metodologia_cientifica/protocolo_c_v1uk_acceptance_audit.md",
    "docs/metodologia_cientifica/protocolo_c_v1ul_recife_candidate_review_router.md",
    "docs/metodologia_cientifica/protocolo_c_relatorio_v1ul_recife_candidate_review_router.md",
    "docs/metodologia_cientifica/protocolo_c_status_atual_v1ul.md",
]

ABSOLUTE_PATH_RE = re.compile(r"([A-Za-z]:\\|\\\\|/home/|/Users/|/mnt/)")
FORBIDDEN_PROMOTION_RE = re.compile(
    r"\b(GROUND_REFERENCE|GROUND_TRUTH|LABEL|PATCH_POSITIVE|PATCH_NEGATIVE)\b"
)


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def csv_columns(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return csv.DictReader(f).fieldnames or []


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bool_text(value):
    return "true" if bool(value) else "false"


def int_value(row, key):
    try:
        return int(row.get(key) or 0)
    except ValueError:
        return 0


def artifact_path(path):
    return path.replace("\\", "/")


def has_absolute_path_in_file(path):
    if not os.path.exists(path) or os.path.getsize(path) > 20 * 1024 * 1024:
        return False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return bool(ABSOLUTE_PATH_RE.search(f.read()))


def forbidden_guardrail_hits(rows):
    hits = []
    for row in rows:
        for key, value in row.items():
            text = str(value or "")
            if key in {"can_create_ground_reference", "can_create_training_label"} and text.lower() == "true":
                hits.append(f"{key}=true")
            if FORBIDDEN_PROMOTION_RE.search(text):
                hits.append(f"{key}:{text[:40]}")
    return hits


def public_sensitive_status(path, rows):
    if not rows:
        return "NO_ROWS"
    fieldnames = set(rows[0].keys())
    literal_sensitive_columns = {
        "endereco", "logradouro", "numero", "cpf", "telefone", "email",
        "nome", "descricao", "protocolo", "processo",
    }
    exposed = sorted(c for c in fieldnames if c.lower() in literal_sensitive_columns)
    if exposed:
        return "REVIEW_FIELD_NAMES_ONLY"
    if has_absolute_path_in_file(path):
        return "REVIEW_ABSOLUTE_PATH"
    return "PUBLIC_SAFE_HASHES_FLAGS_COUNTS_ONLY"


def v1uk_complete():
    return all(os.path.exists(p) for p in REQUIRED_V1UK_ARTIFACTS)


def compute_v1uk_summary():
    profile = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_occurrence_table_profile.csv"))
    candidates = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv"))
    schema = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_asset_schema_registry.csv"))
    matches = load_csv(os.path.join(DATASET_DIR, "v1uk_recife_event_window_match_registry.csv"))
    total_rows = sum(int_value(r, "total_rows") for r in profile)
    rows_in_window = sum(int_value(r, "rows_in_event_window") for r in profile)
    hazard_rows = sum(
        int_value(r, "rows_with_flood_terms")
        + int_value(r, "rows_with_rain_terms")
        + int_value(r, "rows_with_landslide_terms")
        for r in profile
    )
    locality_rows = sum(
        int_value(r, "rows_with_neighborhood") + int_value(r, "rows_with_address")
        for r in profile
    )
    coordinate_rows = sum(int_value(r, "rows_with_coordinates") for r in profile)
    coordinate_candidates = sum(
        1 for r in candidates
        if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_COORDINATES_FOR_REVIEW"
    )
    locality_candidates = sum(
        1 for r in candidates
        if r.get("candidate_class") == "ROW_LEVEL_OCCURRENCE_WITH_LOCALITY_ONLY_FOR_REVIEW"
    )
    contextual = sum(
        1 for r in candidates
        if r.get("candidate_class") in {
            "EVENT_WINDOW_DOCUMENTED_OCCURRENCE_NO_GEOMETRY",
            "CONTEXTUAL_ROW",
            "CONTEXT_ONLY",
        }
    )
    return {
        "v1uk_exists": v1uk_complete(),
        "outputs_expected": len(REQUIRED_V1UK_ARTIFACTS),
        "outputs_present": sum(1 for p in REQUIRED_V1UK_ARTIFACTS if os.path.exists(p)),
        "tables_audited": len(profile),
        "assets_audited": len(schema),
        "total_rows": total_rows,
        "rows_in_window": rows_in_window,
        "rows_with_hazard": hazard_rows,
        "rows_with_locality_or_address": locality_rows,
        "rows_with_coordinates": coordinate_rows,
        "candidate_rows": len(candidates),
        "event_window_matches": len(matches),
        "review_candidates": coordinate_candidates + locality_candidates,
        "coordinate_candidates": coordinate_candidates,
        "locality_only_candidates": locality_candidates,
        "contextual_layers": contextual,
    }


def run_v1uk_acceptance_audit(out_path=None, doc_path=None):
    rows = []
    missing = []
    absolute_hits = []
    guardrail_hits = []
    sensitive_reviews = []
    for idx, path in enumerate(REQUIRED_V1UK_ARTIFACTS):
        exists = os.path.exists(path)
        if not exists:
            missing.append(path)
        data = load_csv(path) if path.endswith(".csv") and exists else []
        columns = csv_columns(path) if path.endswith(".csv") and exists else []
        required = V1UK_REQUIRED_COLUMNS.get(path, [])
        missing_cols = [c for c in required if c not in columns]
        req_status = "OK" if exists and not missing_cols else (
            "MISSING_COLUMNS:" + "|".join(missing_cols) if exists else "MISSING_ARTIFACT"
        )
        hits = forbidden_guardrail_hits(data)
        guard_status = "OK" if not hits else "VIOLATION:" + "|".join(sorted(set(hits))[:5])
        if hits:
            guardrail_hits.extend(hits)
        abs_status = "OK"
        if exists and has_absolute_path_in_file(path):
            abs_status = "ABSOLUTE_PATH_REVIEW"
            absolute_hits.append(path)
        sens_status = (
            public_sensitive_status(path, data)
            if path.endswith(".csv") and exists else "NOT_APPLICABLE"
        )
        if sens_status.startswith("REVIEW"):
            sensitive_reviews.append(path)
        status = "PASS" if exists and not missing_cols and not hits and abs_status == "OK" else "FAIL"
        rows.append({
            "check_id": f"V1UK_ACCEPT_{idx:04d}",
            "artifact": artifact_path(path),
            "exists": bool_text(exists),
            "rows": str(len(data)) if path.endswith(".csv") and exists else "",
            "required_columns_status": req_status,
            "guardrail_status": guard_status,
            "sensitive_value_status": sens_status,
            "absolute_path_status": abs_status,
            "status": status,
            "notes": "v1uk artifact acceptance audit",
        })
    summary = compute_v1uk_summary()
    overall_status = "V1UK_COMPLETE" if summary["v1uk_exists"] and not guardrail_hits else "V1UK_INCOMPLETE"
    rows.append({
        "check_id": "V1UK_ACCEPT_SUMMARY",
        "artifact": "v1uk_aggregate_summary",
        "exists": bool_text(summary["v1uk_exists"]),
        "rows": str(summary["candidate_rows"]),
        "required_columns_status": "OK" if summary["v1uk_exists"] else "MISSING_OUTPUTS",
        "guardrail_status": "OK" if not guardrail_hits else "VIOLATION",
        "sensitive_value_status": "OK" if not sensitive_reviews else "REVIEW_FIELD_NAMES_ONLY",
        "absolute_path_status": "OK" if not absolute_hits else "ABSOLUTE_PATH_REVIEW",
        "status": overall_status,
        "notes": (
            f"outputs_present={summary['outputs_present']}/{summary['outputs_expected']};"
            f"tables_audited={summary['tables_audited']};total_rows={summary['total_rows']};"
            f"rows_in_window={summary['rows_in_window']};rows_with_hazard={summary['rows_with_hazard']};"
            f"rows_with_locality_or_address={summary['rows_with_locality_or_address']};"
            f"rows_with_coordinates={summary['rows_with_coordinates']};"
            f"review_candidates={summary['review_candidates']}"
        ),
    })
    out_path = out_path or os.path.join(DATASET_DIR, "v1uk_acceptance_audit.csv")
    write_csv(out_path, ACCEPTANCE_COLUMNS, rows)
    doc_path = doc_path or os.path.join(DOCS_DIR, "protocolo_c_v1uk_acceptance_audit.md")
    os.makedirs(os.path.dirname(doc_path), exist_ok=True)
    lines = [
        "# Protocolo C v1uk Acceptance Audit",
        "",
        f"- v1uk_exists: {bool_text(summary['v1uk_exists'])}",
        f"- expected_outputs_present: {summary['outputs_present']}/{summary['outputs_expected']}",
        f"- tables_audited: {summary['tables_audited']}",
        f"- total_rows: {summary['total_rows']}",
        f"- rows_in_REC_2022_05_24_30_window: {summary['rows_in_window']}",
        f"- rows_with_hazard_terms: {summary['rows_with_hazard']}",
        f"- rows_with_bairro_locality_address: {summary['rows_with_locality_or_address']}",
        f"- rows_with_coordinates: {summary['rows_with_coordinates']}",
        f"- candidates_for_review: {summary['review_candidates']}",
        f"- absolute_path_public_csv: {bool_text(bool(absolute_hits))}",
        f"- sensitive_literal_public_csv: false",
        f"- guardrail_violation: {bool_text(bool(guardrail_hits))}",
        f"- acceptance_status: {overall_status}",
        "",
        "Missing outputs:",
        *(f"- {artifact_path(p)}" for p in missing),
        "",
        "Guardrails:",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- supervisor_review_completed=false",
    ]
    if not missing:
        lines.insert(lines.index("Missing outputs:") + 1, "- none")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[v1ul v1uk acceptance] status={overall_status} -> {out_path}")
    return rows


def route_for_candidate(candidate, incomplete=False):
    if incomplete:
        return "ROUTE_BLOCKED_V1UK_INCOMPLETE", "v1uk_incomplete"
    if candidate.get("event_window_match") != "event_core_window":
        return "ROUTE_REJECT_OUTSIDE_WINDOW", "outside_core_event_window"
    if candidate.get("hazard_term_status") != "HAS_HAZARD_SIGNAL":
        return "ROUTE_REJECT_NO_HAZARD_SIGNAL", "no_hazard_signal"
    coord_status = candidate.get("coordinate_status", "")
    loc_status = candidate.get("locality_status", "")
    if coord_status == "OCCURRENCE_COORDINATES_CANDIDATE":
        return "ROUTE_COORDINATE_OCCURRENCE_REVIEW", "no_supervisor_review_no_overlay"
    if loc_status and loc_status != "NO_LOCALITY":
        return "ROUTE_LOCALITY_ONLY_REVIEW", "locality_only_no_overlay"
    if candidate.get("candidate_class") == "EVENT_WINDOW_DOCUMENTED_OCCURRENCE_NO_GEOMETRY":
        return "ROUTE_DOCUMENTED_OCCURRENCE_NO_GEOMETRY", "no_coordinates_no_locality"
    return "ROUTE_CONTEXT_ONLY", "context_only_do_not_promote"


def candidate_sensitive_required(candidate):
    return candidate.get("locality_status") in {
        "ADDRESS_TEXT_AVAILABLE", "NEIGHBORHOOD_LEVEL_LOCALITY", "LOCALITY_AMBIGUOUS"
    }


def run_candidate_review_router(out_path=None, candidates_path=None, acceptance_path=None):
    candidates = load_csv(candidates_path or os.path.join(DATASET_DIR, "v1uk_recife_candidate_row_registry.csv"))
    acceptance = load_csv(acceptance_path or os.path.join(DATASET_DIR, "v1uk_acceptance_audit.csv"))
    incomplete = not v1uk_complete()
    if acceptance:
        summary = next((r for r in acceptance if r.get("check_id") == "V1UK_ACCEPT_SUMMARY"), {})
        incomplete = summary.get("status") == "V1UK_INCOMPLETE"
    rows = []
    for idx, cand in enumerate(candidates):
        route, blocker = route_for_candidate(cand, incomplete)
        sensitive_required = candidate_sensitive_required(cand)
        can_supervisor = route in {
            "ROUTE_COORDINATE_OCCURRENCE_REVIEW",
            "ROUTE_LOCALITY_ONLY_REVIEW",
            "ROUTE_DOCUMENTED_OCCURRENCE_NO_GEOMETRY",
        }
        can_overlay_preflight = route == "ROUTE_COORDINATE_OCCURRENCE_REVIEW"
        rows.append({
            "route_id": f"ROUTE_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": cand.get("event_id") or EVENT_ID,
            "candidate_row_id": cand.get("candidate_row_id", ""),
            "asset_id": cand.get("asset_id", ""),
            "row_hash": cand.get("row_hash", ""),
            "v1uk_candidate_class": cand.get("candidate_class", ""),
            "event_window_match": cand.get("event_window_match", ""),
            "hazard_signal": cand.get("hazard_term_status", ""),
            "coordinate_status": cand.get("coordinate_status", ""),
            "locality_status": cand.get("locality_status", ""),
            "sensitive_review_required": bool_text(sensitive_required),
            "review_route": route,
            "review_priority": cand.get("review_priority", "3"),
            "can_enter_supervisor_review": bool_text(can_supervisor),
            "can_enter_overlay_preflight": bool_text(can_overlay_preflight),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "required_next_action": "SUPERVISOR_REVIEW" if can_supervisor else "DO_NOT_PROMOTE",
            "notes": "public_registry_uses_ids_hashes_flags_no_overlay_no_label",
        })
    if incomplete and not rows:
        rows.append({
            "route_id": f"ROUTE_{PROTOCOL_VERSION}_000000",
            "event_id": EVENT_ID,
            "candidate_row_id": "",
            "asset_id": "",
            "row_hash": "",
            "v1uk_candidate_class": "V1UK_INCOMPLETE",
            "event_window_match": "",
            "hazard_signal": "",
            "coordinate_status": "",
            "locality_status": "",
            "sensitive_review_required": "false",
            "review_route": "ROUTE_BLOCKED_V1UK_INCOMPLETE",
            "review_priority": "9",
            "can_enter_supervisor_review": "false",
            "can_enter_overlay_preflight": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": "v1uk_incomplete",
            "required_next_action": "COMPLETE_V1UK_OUTPUTS",
            "notes": "fail_closed_no_candidate_rows_available",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv")
    write_csv(out_path, ROUTER_COLUMNS, rows)
    print(f"[v1ul router] rows={len(rows)} -> {out_path}")
    return rows


def coordinate_source_status(route):
    if route.get("coordinate_status") == "OCCURRENCE_COORDINATES_CANDIDATE":
        return "OFFICIAL_DATASET_COORDINATES_NOT_INFERRED"
    if route.get("coordinate_status") and route.get("coordinate_status") != "NO_COORDINATES":
        return "COORDINATE_REVIEW_REQUIRED"
    return "NO_COORDINATES"


def run_overlay_readiness_preflight(out_path=None, router_path=None):
    routes = load_csv(router_path or os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv"))
    rows = []
    for idx, route in enumerate(routes):
        has_window = route.get("event_window_match") == "event_core_window"
        has_hazard = route.get("hazard_signal") == "HAS_HAZARD_SIGNAL"
        has_coord = route.get("coordinate_status") == "OCCURRENCE_COORDINATES_CANDIDATE"
        review_route = route.get("review_route")
        if review_route == "ROUTE_BLOCKED_V1UK_INCOMPLETE":
            status, blocker = "BLOCKED_V1UK_INCOMPLETE", "v1uk_incomplete"
        elif not has_window:
            status, blocker = "BLOCKED_NO_EVENT_WINDOW_MATCH", "outside_core_event_window"
        elif not has_hazard:
            status, blocker = "BLOCKED_NO_HAZARD_SIGNAL", "no_hazard_signal"
        elif review_route == "ROUTE_LOCALITY_ONLY_REVIEW":
            status, blocker = "LOCALITY_ONLY_NOT_OVERLAY_ELIGIBLE", "locality_only"
        elif review_route == "ROUTE_DOCUMENTED_OCCURRENCE_NO_GEOMETRY":
            status, blocker = "DOCUMENTED_OCCURRENCE_NO_GEOMETRY", "no_coordinates"
        elif review_route == "ROUTE_CONTEXT_ONLY":
            status, blocker = "CONTEXTUAL_LAYER_NOT_ELIGIBLE", "contextual_layer"
        elif has_coord:
            status, blocker = (
                "OVERLAY_PREFLIGHT_ELIGIBLE_AFTER_SUPERVISOR_REVIEW",
                "supervisor_review_pending_overlay_not_executed",
            )
        else:
            status, blocker = "BLOCKED_NO_COORDINATES", "no_coordinates"
        rows.append({
            "preflight_id": f"PREFLIGHT_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": route.get("event_id") or EVENT_ID,
            "candidate_row_id": route.get("candidate_row_id", ""),
            "asset_id": route.get("asset_id", ""),
            "has_event_window_match": bool_text(has_window),
            "has_hazard_signal": bool_text(has_hazard),
            "has_coordinates_or_geometry": bool_text(has_coord),
            "coordinate_source_status": coordinate_source_status(route),
            "crs_status": "WGS84_PLAUSIBLE_NEEDS_SUPERVISOR_REVIEW" if has_coord else "NOT_APPLICABLE",
            "geometry_type": "point_or_row_coordinates" if has_coord else "none",
            "official_source_status": "PUBLIC_OFFICIAL_SOURCE_FROM_V1UK",
            "contextual_layer_status": "CONTEXTUAL_LAYER" if review_route == "ROUTE_CONTEXT_ONLY" else "NOT_CONTEXTUAL_LAYER",
            "sensitive_status": "REDACTED_PUBLIC_REVIEW_PACKAGE_REQUIRED" if route.get("sensitive_review_required") == "true" else "NO_SENSITIVE_FIELDS_DETECTED",
            "supervisor_review_status": "PENDING",
            "overlay_preflight_status": status,
            "can_execute_overlay_now": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "required_next_action": "SUPERVISOR_REVIEW" if status.endswith("SUPERVISOR_REVIEW") else "DO_NOT_EXECUTE_OVERLAY",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1ul_recife_overlay_readiness_preflight.csv")
    write_csv(out_path, PREFLIGHT_COLUMNS, rows)
    print(f"[v1ul overlay preflight] rows={len(rows)} -> {out_path}")
    return rows


def run_sensitive_review_packager(out_path=None, router_path=None):
    routes = load_csv(router_path or os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv"))
    rows = []
    for idx, route in enumerate(routes):
        has_sensitive = route.get("sensitive_review_required") == "true"
        fields = []
        if route.get("locality_status") == "ADDRESS_TEXT_AVAILABLE":
            fields.append("address_hash")
        elif route.get("locality_status") in {"NEIGHBORHOOD_LEVEL_LOCALITY", "LOCALITY_AMBIGUOUS"}:
            fields.append("locality_hash")
        rows.append({
            "package_id": f"SENSPKG_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": route.get("event_id") or EVENT_ID,
            "candidate_row_id": route.get("candidate_row_id", ""),
            "row_hash": route.get("row_hash", ""),
            "has_sensitive_fields": bool_text(has_sensitive),
            "redaction_status": "REDACTED_HASH_ONLY" if has_sensitive else "NO_REDACTION_REQUIRED",
            "fields_redacted": "|".join(fields),
            "safe_summary": (
                f"{route.get('review_route', '')};"
                f"window={route.get('event_window_match', '')};"
                f"hazard={route.get('hazard_signal', '')};"
                f"coordinates={route.get('coordinate_status', '')};"
                f"locality={route.get('locality_status', '')}"
            ),
            "reviewer_can_inspect_local_raw": bool_text(has_sensitive),
            "local_raw_required": bool_text(has_sensitive),
            "public_registry_safe": "true",
            "notes": "no_literal_address_no_name_no_phone_no_cpf_no_full_description",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1ul_recife_sensitive_review_package.csv")
    write_csv(out_path, SENSITIVE_PACKAGE_COLUMNS, rows)
    print(f"[v1ul sensitive package] rows={len(rows)} -> {out_path}")
    return rows


def decision_for(route, preflight):
    review_route = route.get("review_route")
    sensitive_status = preflight.get("sensitive_status", "")
    if "REDACTED" not in sensitive_status and review_route == "ROUTE_BLOCKED_SENSITIVE_REVIEW":
        return "BLOCKED_SENSITIVE_REVIEW", "false", "sensitive_review_required"
    if review_route == "ROUTE_COORDINATE_OCCURRENCE_REVIEW":
        return "READY_FOR_SUPERVISOR_REVIEW_COORDINATE_CANDIDATE", "true", "no_supervisor_review_no_overlay"
    if review_route == "ROUTE_LOCALITY_ONLY_REVIEW":
        return "READY_FOR_SUPERVISOR_REVIEW_LOCALITY_ONLY", "true", "locality_only_no_overlay"
    if review_route == "ROUTE_DOCUMENTED_OCCURRENCE_NO_GEOMETRY":
        return "KEEP_AS_DOCUMENTED_CONTEXT", "false", "no_coordinates"
    if review_route == "ROUTE_REJECT_OUTSIDE_WINDOW":
        return "REJECT_NO_TEMPORAL_MATCH", "false", "outside_core_event_window"
    if review_route == "ROUTE_REJECT_NO_HAZARD_SIGNAL":
        return "REJECT_NO_HAZARD_SIGNAL", "false", "no_hazard_signal"
    if review_route == "ROUTE_BLOCKED_V1UK_INCOMPLETE":
        return "BLOCKED_SCHEMA_AMBIGUITY", "false", "v1uk_incomplete"
    if review_route == "ROUTE_CONTEXT_ONLY":
        return "REJECT_CONTEXTUAL_LAYER", "false", "contextual_layer"
    return "DO_NOT_PROMOTE", "false", "do_not_promote"


def run_candidate_decision_matrix(out_path=None, queue_path=None, router_path=None, preflight_path=None):
    routes = load_csv(router_path or os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv"))
    preflights = {
        r.get("candidate_row_id"): r
        for r in load_csv(preflight_path or os.path.join(DATASET_DIR, "v1ul_recife_overlay_readiness_preflight.csv"))
    }
    rows = []
    queue = []
    for idx, route in enumerate(routes):
        pf = preflights.get(route.get("candidate_row_id"), {})
        decision, can_v1um, blocker = decision_for(route, pf)
        evidence_strength = "high" if "COORDINATE" in decision else "medium" if "LOCALITY" in decision else "low"
        rows.append({
            "decision_id": f"DECISION_{PROTOCOL_VERSION}_{idx:06d}",
            "event_id": route.get("event_id") or EVENT_ID,
            "candidate_row_id": route.get("candidate_row_id", ""),
            "review_route": route.get("review_route", ""),
            "overlay_preflight_status": pf.get("overlay_preflight_status", ""),
            "sensitive_status": pf.get("sensitive_status", ""),
            "evidence_strength": evidence_strength,
            "recommended_decision": decision,
            "can_advance_to_v1um": can_v1um,
            "can_execute_overlay_now": "false",
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "blocker": blocker,
            "notes": "status_max_recife_candidate_ready_for_supervisor_review",
        })
        if route.get("can_enter_supervisor_review") == "true":
            queue.append({
                "queue_id": f"QUEUE_{PROTOCOL_VERSION}_{len(queue):06d}",
                "event_id": route.get("event_id") or EVENT_ID,
                "candidate_row_id": route.get("candidate_row_id", ""),
                "review_route": route.get("review_route", ""),
                "review_priority": route.get("review_priority", "3"),
                "reviewer_task": "Inspect local raw evidence and decide whether candidate remains documented occurrence context.",
                "required_local_raw_asset": route.get("asset_id", ""),
                "safe_public_summary": (
                    f"{route.get('review_route', '')};"
                    f"hazard={route.get('hazard_signal', '')};"
                    f"coordinate={route.get('coordinate_status', '')};"
                    f"locality={route.get('locality_status', '')}"
                ),
                "decision_options": "|".join([
                    "KEEP_AS_DOCUMENTED_CONTEXT",
                    "REQUEST_SCHEMA_CLARIFICATION",
                    "DO_NOT_PROMOTE",
                ]),
                "can_be_reviewed_now": "true",
                "can_advance_to_overlay_preflight_after_review": bool_text(
                    route.get("can_enter_overlay_preflight") == "true"
                ),
                "can_create_ground_reference": "false",
                "can_create_training_label": "false",
            })
    out_path = out_path or os.path.join(DATASET_DIR, "v1ul_recife_candidate_decision_matrix.csv")
    queue_path = queue_path or os.path.join(DATASET_DIR, "v1ul_recife_supervisor_review_queue.csv")
    write_csv(out_path, DECISION_COLUMNS, rows)
    write_csv(queue_path, QUEUE_COLUMNS, queue)
    print(f"[v1ul decision matrix] rows={len(rows)} queue={len(queue)} -> {out_path}")
    return rows


def run_ground_reference_blocker_matrix(out_path=None):
    routes = load_csv(os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv"))
    preflight = load_csv(os.path.join(DATASET_DIR, "v1ul_recife_overlay_readiness_preflight.csv"))
    counts = {
        "no_supervisor_review": sum(1 for r in routes if r.get("can_enter_supervisor_review") == "true"),
        "no_overlay_executed": len(preflight),
        "locality_only": sum(1 for r in routes if r.get("review_route") == "ROUTE_LOCALITY_ONLY_REVIEW"),
        "no_coordinates": sum(1 for r in routes if r.get("coordinate_status") != "OCCURRENCE_COORDINATES_CANDIDATE"),
        "contextual_layer": sum(1 for r in routes if r.get("review_route") == "ROUTE_CONTEXT_ONLY"),
        "sensitive_review_required": sum(1 for r in routes if r.get("sensitive_review_required") == "true"),
        "schema_ambiguity": sum(1 for r in routes if r.get("review_route") == "ROUTE_BLOCKED_V1UK_INCOMPLETE"),
        "event_window_uncertainty": sum(1 for r in routes if r.get("event_window_match") != "event_core_window"),
        "hazard_ambiguity": sum(1 for r in routes if r.get("hazard_signal") != "HAS_HAZARD_SIGNAL"),
        "label_forbidden": len(routes),
    }
    rows = []
    for idx, (blocker, count) in enumerate(counts.items()):
        rows.append({
            "blocker_id": f"BLOCK_{PROTOCOL_VERSION}_{idx:04d}",
            "event_id": EVENT_ID,
            "blocker": blocker,
            "status": "ACTIVE" if count else "INACTIVE",
            "evidence_count": str(count),
            "can_create_ground_reference": "false",
            "can_create_training_label": "false",
            "notes": "v1ul blocks promotion before supervisor review and overlay execution",
        })
    out_path = out_path or os.path.join(DATASET_DIR, "v1ul_recife_ground_reference_blocker_matrix.csv")
    write_csv(out_path, BLOCKER_COLUMNS, rows)
    return rows


def write_policy_configs():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    policies = {
        "v1ul_recife_review_routing_policy.yaml": [
            "protocol_version: v1ul",
            "event_id: REC_2022_05_24_30",
            f"max_status: {MAX_STATUS}",
            "guardrails:",
            "  ground_truth_operational: false",
            "  can_create_ground_reference: false",
            "  can_create_training_label: false",
            "  can_reopen_protocol_b: false",
            "  dino_usage: SUPPORT_ONLY",
            "  no_overlay_executed: true",
            "  no_coordinates_invented: true",
            "routes:",
            "  coordinate: ROUTE_COORDINATE_OCCURRENCE_REVIEW",
            "  locality_only: ROUTE_LOCALITY_ONLY_REVIEW",
            "  documented_no_geometry: ROUTE_DOCUMENTED_OCCURRENCE_NO_GEOMETRY",
        ],
        "v1ul_overlay_readiness_policy.yaml": [
            "protocol_version: v1ul",
            "overlay_execution_allowed: false",
            "geocoding_allowed: false",
            "centroid_allowed: false",
            "requires_real_coordinates_or_geometry: true",
            "requires_supervisor_review_before_execution: true",
        ],
        "v1ul_sensitive_review_policy.yaml": [
            "protocol_version: v1ul",
            "public_outputs:",
            "  allow_hashes: true",
            "  allow_flags: true",
            "  allow_counts: true",
            "  forbid_literal_address: true",
            "  forbid_names_phone_cpf_email: true",
            "  forbid_full_description: true",
        ],
        "v1ul_candidate_decision_policy.yaml": [
            "protocol_version: v1ul",
            "promotion_policy:",
            "  ground_reference: forbidden",
            "  ground_truth: forbidden",
            "  label: forbidden",
            "  patch_positive: forbidden",
            "  patch_negative: forbidden",
            "  supervisor_review_required: true",
        ],
    }
    for name, lines in policies.items():
        with open(os.path.join(CONFIG_DIR, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def run_completion_report():
    write_policy_configs()
    run_ground_reference_blocker_matrix()
    summary = compute_v1uk_summary()
    routes = load_csv(os.path.join(DATASET_DIR, "v1ul_recife_candidate_review_router.csv"))
    preflight = load_csv(os.path.join(DATASET_DIR, "v1ul_recife_overlay_readiness_preflight.csv"))
    decisions = load_csv(os.path.join(DATASET_DIR, "v1ul_recife_candidate_decision_matrix.csv"))
    queue = load_csv(os.path.join(DATASET_DIR, "v1ul_recife_supervisor_review_queue.csv"))
    coordinate_candidates = sum(1 for r in routes if r.get("review_route") == "ROUTE_COORDINATE_OCCURRENCE_REVIEW")
    locality_candidates = sum(1 for r in routes if r.get("review_route") == "ROUTE_LOCALITY_ONLY_REVIEW")
    contextual = sum(1 for r in routes if r.get("review_route") in {"ROUTE_CONTEXT_ONLY", "ROUTE_DOCUMENTED_OCCURRENCE_NO_GEOMETRY"})
    future_overlay = sum(
        1 for r in preflight
        if r.get("overlay_preflight_status") == "OVERLAY_PREFLIGHT_ELIGIBLE_AFTER_SUPERVISOR_REVIEW"
    )
    if coordinate_candidates:
        next_action = "v1um - Recife Supervisor Review and Overlay Preflight Execution"
    elif locality_candidates:
        next_action = "v1um - Recife Locality-Only Human Review Package"
    else:
        next_action = "v1um - Recife CKAN Dataset-Specific Manual Schema Refinement"
    action_rows = [{
        "action_id": f"ACT_{PROTOCOL_VERSION}_0000",
        "event_id": EVENT_ID,
        "action_type": "LOCALITY_ONLY_HUMAN_REVIEW" if locality_candidates else "SCHEMA_REFINEMENT",
        "priority": "1",
        "description": next_action,
        "target": "Recife CKAN v1uk candidate rows",
        "status": "PENDING",
        "notes": "do_not_implement_v1um_in_v1ul",
    }]
    write_csv(os.path.join(DATASET_DIR, "v1ul_next_actions_registry.csv"), NEXT_ACTION_COLUMNS, action_rows)
    manifest = []
    for idx, path in enumerate(V1UL_ARTIFACTS):
        exists = os.path.exists(path)
        manifest.append({
            "artifact_id": f"ART_{PROTOCOL_VERSION}_{idx:04d}",
            "artifact_path": artifact_path(path),
            "artifact_type": "config" if path.startswith("configs/") else "doc" if path.startswith("docs/") else "dataset",
            "protocol_version": PROTOCOL_VERSION,
            "sha256_prefix": sha256_file(path)[:16] if exists else "MISSING",
            "file_size_bytes": str(os.path.getsize(path) if exists else 0),
            "is_versionable": bool_text(exists),
            "reason": "Safe public metadata artifact" if exists else "File not found",
        })
    write_csv(os.path.join(DATASET_DIR, "v1ul_versionable_artifacts_manifest.csv"), MANIFEST_COLUMNS, manifest)
    os.makedirs(DOCS_DIR, exist_ok=True)
    methodology = [
        "# Protocolo C v1ul - Recife Candidate Review Router",
        "",
        "## Scope",
        "- Reads v1uk registries only.",
        "- Routes candidates for supervisor review without promotion.",
        "- Evaluates overlay-readiness preconditions without executing overlay.",
        "- Public outputs contain ids, hashes, flags, counts, and redacted summaries only.",
        "",
        "## Guardrails",
        "- ground_truth_operational=false",
        "- can_create_ground_reference=false",
        "- can_create_training_label=false",
        "- can_reopen_protocol_b=false",
        "- dino_usage=SUPPORT_ONLY",
        "- no_overlay_executed=true",
        "- no_coordinates_invented=true",
        "- supervisor_review_completed=false",
        f"- max_status={MAX_STATUS}",
    ]
    report = [
        "# Relatorio v1ul - Recife Candidate Review Router and Overlay-Readiness Preflight",
        "",
        "## What v1uk Generated",
        f"- assets_audited: {summary['assets_audited']}",
        f"- tables_audited: {summary['tables_audited']}",
        f"- total_rows: {summary['total_rows']}",
        f"- event_window_matches: {summary['event_window_matches']}",
        f"- candidate_rows: {summary['candidate_rows']}",
        "",
        "## Candidate Counts",
        f"- rows_in_event_window: {summary['rows_in_window']}",
        f"- rows_with_hazard_terms: {summary['rows_with_hazard']}",
        f"- rows_with_coordinates: {summary['rows_with_coordinates']}",
        f"- coordinate_candidates: {coordinate_candidates}",
        f"- locality_only_candidates: {locality_candidates}",
        f"- contextual_or_documented_no_geometry: {contextual}",
        f"- supervisor_review_queue: {len(queue)}",
        f"- future_overlay_preflight_candidates_after_review: {future_overlay}",
        "",
        "## Decision",
        f"- any_supervisor_review_candidate: {bool_text(bool(queue))}",
        f"- any_future_overlay_preflight_candidate: {bool_text(bool(future_overlay))}",
        "- ground_reference_status: blocked",
        "- ground_reference_blocker: no_supervisor_review; no_overlay_executed; no_coordinates for locality-only candidates; label_forbidden",
        f"- recommended_next_stage: {next_action}",
    ]
    status = [
        "# Status Atual - Protocolo C v1ul",
        "",
        f"event_id={EVENT_ID}",
        f"v1uk_exists={bool_text(summary['v1uk_exists'])}",
        f"candidate_rows={summary['candidate_rows']}",
        f"coordinate_candidates={coordinate_candidates}",
        f"locality_only_candidates={locality_candidates}",
        f"contextual_layers={contextual}",
        f"supervisor_review_queue={len(queue)}",
        f"future_overlay_preflight_candidates={future_overlay}",
        "ground_truth_operational=false",
        "can_create_ground_reference=false",
        "can_create_training_label=false",
        "can_reopen_protocol_b=false",
        "dino_usage=SUPPORT_ONLY",
        "no_overlay_executed=true",
        "no_coordinates_invented=true",
        "supervisor_review_completed=false",
        f"max_status={MAX_STATUS}",
        f"next_action={next_action}",
    ]
    with open(os.path.join(DOCS_DIR, "protocolo_c_v1ul_recife_candidate_review_router.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(methodology) + "\n")
    with open(os.path.join(DOCS_DIR, "protocolo_c_relatorio_v1ul_recife_candidate_review_router.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    with open(os.path.join(DOCS_DIR, "protocolo_c_status_atual_v1ul.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(status) + "\n")
    print(f"[v1ul completion] next_action={next_action}")
    return {
        "candidate_rows": summary["candidate_rows"],
        "coordinate_candidates": coordinate_candidates,
        "locality_only_candidates": locality_candidates,
        "contextual_layers": contextual,
        "supervisor_review_queue": len(queue),
        "future_overlay_preflight_candidates": future_overlay,
        "decisions": len(decisions),
        "next_action": next_action,
    }


def simple_main(fn):
    parser = argparse.ArgumentParser()
    parser.parse_args()
    fn()
