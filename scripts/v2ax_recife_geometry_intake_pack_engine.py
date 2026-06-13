#!/usr/bin/env python3
"""v2ax Recife Geometry Intake, Documentation, Validation and Replay Pack.

Builds a fail-closed operational manual-intake pack from v2aw/v2av/v2au outputs.
It preserves filled manual rows across replays, validates only supplied geometry,
and never generates geometry, labels, final ground truth, training, or automatic C4.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict


STAGE = "v2ax"
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_FILE))
CONFIG_NAME = "v2ax_recife_geometry_intake_pack_config.json"
UNKNOWN = "UNKNOWN"


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


def resolve_dirs():
    return (
        os.environ.get("DATASET_DIR") or project_path("datasets"),
        os.environ.get("OUTPUT_DIR") or project_path("outputs_public"),
        os.environ.get("CONFIG_DIR") or project_path("configs"),
        os.environ.get("DOCS_DIR") or project_path("docs"),
    )


DEFAULT_CONFIG = {
    "offline_mode": True, "strict_mode": True, "priority_region": "Recife",
    "expected_priority_patches": 55, "expected_recife_events": 3,
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "preferred_crs": "EPSG:4326", "target_crs_for_area": "EPSG:3857",
    "allowed_patch_geometry_formats": ["bbox", "wkt", "geojson_inline", "geojson_file"],
    "allowed_event_geometry_formats": ["wkt", "geojson_inline", "geojson_file"],
    "allow_auto_geometry_generation": False, "allow_point_as_patch_boundary": False,
    "allow_point_as_event_polygon": False, "allow_default_patch_size": False,
    "manual_pack_dir": "datasets/manual_intake/recife_p1",
    "examples_dir": "datasets/examples/v2ax_recife_geometry_pack",
}

IN_PATCH_TEMPLATE = "v2aw_patch_geometry_sources_template.csv"
IN_EVENT_TEMPLATE = "v2aw_event_geometry_sources_template.csv"
IN_READINESS = "v2aw_recife_p1_geometry_readiness.csv"
IN_VALIDATION = "v2aw_geometry_source_validation_registry.csv"
IN_RECOVERY = "v2av_patch_boundary_recovery_queue.csv"
IN_PACKAGES = "v2at_event_patch_package_registry.csv"
IN_OVERLAY_QUEUE = "v2au_overlay_review_queue.csv"

MANUAL_SUBDIR = os.path.join("manual_intake", "recife_p1")
EXAMPLE_SUBDIR = os.path.join("examples", "v2ax_recife_geometry_pack")
PATCH_INTAKE = "recife_p1_patch_geometry_intake.csv"
EVENT_INTAKE = "recife_p1_event_geometry_intake.csv"
PATCH_CHECKLIST = "recife_p1_patch_checklist.csv"
EVENT_CHECKLIST = "recife_p1_event_checklist.csv"
PACKAGE_MATRIX = "recife_p1_package_review_matrix.csv"
COLLECTION_PLAN = "recife_p1_geometry_collection_plan.csv"
MANUAL_VALIDATION = "recife_p1_manual_validation_results.csv"
MANUAL_README = "recife_p1_readme.md"
OUT_MANIFEST = "v2ax_recife_manual_intake_manifest.csv"
OUT_VALIDATION = "v2ax_recife_manual_intake_validation.csv"
OUT_V2AW_PATCH = "v2ax_ready_to_feed_v2aw_patch_sources.csv"
OUT_V2AW_EVENT = "v2ax_ready_to_feed_v2aw_event_sources.csv"
OUT_V2AV_PATCH = "v2ax_ready_to_feed_v2av_patch_sources.csv"
OUT_V2AU_GEOM = "v2ax_ready_to_feed_v2au_geometry_sources.csv"

REPORT_REL = os.path.join("execution_reports", "v2ax_recife_geometry_intake_pack_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2ax_recife_geometry_intake_pack_summary.json")
LOG_REL = os.path.join("logs_summary", "v2ax_recife_geometry_intake_pack.txt")

PATCH_COLUMNS = [
    "intake_id", "patch_id", "region", "city", "priority_rank", "package_count",
    "required_geometry_kind", "source_type", "geometry_value", "geometry_path", "crs",
    "provenance_type", "provenance_note", "source_document", "source_document_page",
    "source_document_url_or_path", "digitized_by", "digitized_at", "source_confidence",
    "license_status", "review_status", "validation_status", "blocking_reason", "notes",
]
EVENT_COLUMNS = [
    "event_intake_id", "event_id", "region", "city", "hazard_type", "linked_packages_count",
    "required_geometry_kind", "source_type", "geometry_value", "geometry_path", "crs",
    "event_geometry_role", "source_id", "source_name", "provenance_type", "provenance_note",
    "source_document", "source_document_page", "source_document_url_or_path", "digitized_by",
    "digitized_at", "source_confidence", "license_status", "review_status",
    "validation_status", "blocking_reason", "notes",
]
PATCH_CHECK_COLUMNS = [
    "check_id", "patch_id", "priority_rank", "check_item", "is_required", "current_status",
    "blocking_reason", "how_to_complete", "notes",
]
EVENT_CHECK_COLUMNS = [
    "check_id", "event_id", "check_item", "is_required", "current_status",
    "blocking_reason", "how_to_complete", "notes",
]
MATRIX_COLUMNS = [
    "matrix_id", "package_id", "patch_id", "event_id", "region", "city", "hazard_type",
    "priority_rank", "current_allowed_use", "current_promotion_decision", "needs_patch_boundary",
    "needs_event_polygon", "has_patch_boundary_source", "has_event_polygon_source",
    "ready_for_v2aw", "ready_for_v2av", "ready_for_v2au", "remaining_blocker", "next_action", "notes",
]
PLAN_COLUMNS = [
    "collection_task_id", "target_type", "target_id", "region", "city", "priority_rank",
    "recommended_source_class", "recommended_source_name", "what_to_collect", "accepted_formats",
    "required_crs", "why_needed", "current_blocker", "collection_status", "notes",
]
MANIFEST_COLUMNS = [
    "intake_manifest_id", "package_id", "patch_id", "event_id", "region", "city",
    "hazard_type", "priority_rank", "manual_patch_file", "manual_event_file",
    "needs_patch_boundary", "needs_event_geometry", "patch_boundary_status",
    "event_geometry_status", "ready_for_v2aw", "ready_for_v2av", "ready_for_v2au",
    "blocking_reason", "next_action", "notes",
]
VALIDATION_COLUMNS = [
    "validation_id", "target_type", "target_id", "package_id", "patch_id", "event_id",
    "source_file", "geometry_present", "geometry_format", "geometry_format_valid",
    "crs_present", "crs_accepted", "provenance_present", "license_present",
    "review_status_valid", "is_point", "is_polygon_or_bbox", "can_feed_v2aw",
    "can_feed_v2av", "can_feed_v2au", "blocking_reason", "recommended_fix", "notes",
]
V2AW_PATCH_COLUMNS = [
    "geometry_source_id", "linked_patch_id", "region", "city", "priority_rank", "source_type",
    "geometry_value", "geometry_path", "crs", "provenance_type", "provenance_note", "digitized_by",
    "digitized_at", "source_document", "source_document_page", "source_confidence",
    "license_status", "review_status", "notes",
]
V2AW_EVENT_COLUMNS = [
    "event_geometry_source_id", "linked_event_id", "region", "city", "hazard_type", "source_type",
    "geometry_value", "geometry_path", "crs", "event_geometry_role", "source_id", "source_name",
    "provenance_type", "provenance_note", "digitized_by", "digitized_at", "source_document",
    "source_document_page", "source_confidence", "license_status", "review_status", "notes",
]
V2AV_PATCH_COLUMNS = [
    "patch_id", "region", "city", "source_file", "source_field", "source_type",
    "geometry_value", "geometry_path", "crs", "center_lat", "center_lon", "size_meters",
    "source_confidence",
]
V2AU_GEOM_COLUMNS = [
    "geometry_role", "linked_event_id", "linked_patch_id", "source_id", "source_name",
    "geometry_type", "geometry_format", "geometry_value", "geometry_path", "crs",
    "latitude", "longitude",
]


def clean(value):
    return "" if value is None else str(value).strip()


def b(value):
    return "true" if bool(value) else "false"


def stable_id(prefix, *parts, length=12):
    raw = "|".join(clean(part) for part in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:length]


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def normalise_crs(value):
    value = clean(value).upper().replace(" ", "")
    if value.isdigit():
        return "EPSG:" + value
    return value


def geometry_kind(source_type, value, geometry_path, dataset_dir):
    source_type, value = clean(source_type).lower(), clean(value)
    try:
        if source_type == "bbox":
            nums = [float(x) for x in re.split(r"[\s,]+", value) if x]
            return ("polygon", len(nums) == 4 and nums[0] < nums[2] and nums[1] < nums[3])
        if source_type == "wkt":
            upper = value.upper()
            if upper.startswith("POINT"):
                return "point", bool(re.search(r"POINT\s*\([^)]+\)", upper))
            if upper.startswith("POLYGON"):
                return "polygon", value.count(",") >= 3 and "((" in value and "))" in value
            return None, False
        if source_type in ("geojson_inline", "geojson_file"):
            if source_type == "geojson_file":
                path = clean(geometry_path)
                path = path if os.path.isabs(path) else os.path.join(dataset_dir, path)
                with open(path, encoding="utf-8") as handle:
                    obj = json.load(handle)
            else:
                obj = json.loads(value)
            geom = obj.get("geometry", obj)
            kind = clean(geom.get("type")).lower()
            if kind == "point":
                return "point", bool(geom.get("coordinates"))
            if kind in ("polygon", "multipolygon"):
                return "polygon", bool(geom.get("coordinates"))
    except (ValueError, OSError, json.JSONDecodeError, AttributeError, TypeError):
        return None, False
    return None, False


def merge_preserved(defaults, existing, id_field, columns):
    previous = {clean(row.get(id_field)): row for row in existing if clean(row.get(id_field))}
    merged = []
    for default in defaults:
        row = dict(default)
        old = previous.get(clean(default.get(id_field)))
        if old:
            for column in columns:
                if column not in ("validation_status", "blocking_reason", "notes") and clean(old.get(column)):
                    row[column] = old[column]
        merged.append(row)
    return merged


def load_config(config_dir):
    config = dict(DEFAULT_CONFIG)
    path = os.path.join(config_dir, CONFIG_NAME)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def load_inputs(dataset_dir):
    mapping = {
        "patch_template": IN_PATCH_TEMPLATE, "event_template": IN_EVENT_TEMPLATE,
        "readiness": IN_READINESS, "validation": IN_VALIDATION, "recovery": IN_RECOVERY,
        "packages": IN_PACKAGES, "overlay_queue": IN_OVERLAY_QUEUE,
    }
    inputs, found = {}, []
    for key, name in mapping.items():
        inputs[key] = load_csv(os.path.join(dataset_dir, name))
        if inputs[key]:
            found.append(name)
    return inputs, found


def recife_packages(inputs, config):
    region = config["priority_region"]
    priority = {clean(row.get("linked_patch_id")) for row in inputs["patch_template"]
                if clean(row.get("region")) == region}
    rows = [row for row in inputs["packages"]
            if clean(row.get("region")) == region and clean(row.get("patch_id")) in priority]
    return sorted(rows, key=lambda row: (clean(row.get("patch_id")), clean(row.get("event_id")),
                                         clean(row.get("package_id"))))


def build_patch_defaults(inputs, packages):
    counts = Counter(clean(row.get("patch_id")) for row in packages)
    rows = []
    for source in inputs["patch_template"]:
        if clean(source.get("region")) != "Recife":
            continue
        patch_id = clean(source.get("linked_patch_id"))
        rows.append({
            "intake_id": stable_id("V2AX_PATCH_", patch_id), "patch_id": patch_id,
            "region": "Recife", "city": clean(source.get("city")) or "Recife",
            "priority_rank": clean(source.get("priority_rank")) or "1",
            "package_count": str(counts[patch_id]), "required_geometry_kind": "patch_boundary_polygon",
            "source_type": "missing", "geometry_value": "", "geometry_path": "", "crs": "",
            "provenance_type": "unknown", "provenance_note": "", "source_document": "",
            "source_document_page": "", "source_document_url_or_path": "", "digitized_by": "",
            "digitized_at": "", "source_confidence": "", "license_status": "",
            "review_status": "not_started", "validation_status": "BLOCKED_PENDING_MANUAL_GEOMETRY",
            "blocking_reason": "NO_PATCH_BOUNDARY_SOURCE_PROVIDED",
            "notes": "Manual intake row; geometry is never auto-generated.",
        })
    return sorted(rows, key=lambda row: (int(row["priority_rank"]), row["patch_id"]))


def build_event_defaults(packages):
    grouped = {}
    for package in packages:
        event_id = clean(package.get("event_id"))
        grouped.setdefault(event_id, package)
    counts = Counter(clean(row.get("event_id")) for row in packages)
    rows = []
    for event_id, package in grouped.items():
        rows.append({
            "event_intake_id": stable_id("V2AX_EVENT_", event_id), "event_id": event_id,
            "region": "Recife", "city": clean(package.get("city")) or "Recife",
            "hazard_type": clean(package.get("hazard_type")), "linked_packages_count": str(counts[event_id]),
            "required_geometry_kind": "observed_event_polygon", "source_type": "missing",
            "geometry_value": "", "geometry_path": "", "crs": "",
            "event_geometry_role": "observed_event_polygon", "source_id": "", "source_name": "",
            "provenance_type": "unknown", "provenance_note": "", "source_document": "",
            "source_document_page": "", "source_document_url_or_path": "", "digitized_by": "",
            "digitized_at": "", "source_confidence": "", "license_status": "",
            "review_status": "not_started", "validation_status": "BLOCKED_PENDING_MANUAL_GEOMETRY",
            "blocking_reason": "NO_VALID_OBSERVED_EVENT_POLYGON",
            "notes": "Requires a real observed-event polygon; a CPRM point is anchor-only.",
        })
    return sorted(rows, key=lambda row: row["event_id"])


def validate_intake(target_type, row, package_id, config, dataset_dir):
    source_type = clean(row.get("source_type")).lower() or "missing"
    value, path = clean(row.get("geometry_value")), clean(row.get("geometry_path"))
    present = source_type != "missing" and bool(value or path)
    allowed = config["allowed_patch_geometry_formats"] if target_type == "patch" \
        else config["allowed_event_geometry_formats"]
    format_valid = source_type in allowed
    kind, geometry_valid = geometry_kind(source_type, value, path, dataset_dir) \
        if present and format_valid else (None, False)
    crs = normalise_crs(row.get("crs"))
    crs_present = bool(crs and crs != UNKNOWN)
    crs_accepted = crs in config["accepted_crs"]
    provenance = bool(clean(row.get("provenance_note")) and clean(row.get("provenance_type"))
                      not in ("", "unknown"))
    license_present = bool(clean(row.get("license_status")))
    review_valid = clean(row.get("review_status")) in (
        "provided_unreviewed", "format_validated", "approved_for_v2av", "approved_for_v2au")
    is_point, is_polygon = kind == "point", kind == "polygon"
    if not present:
        block, fix = "BLOCKED_PENDING_MANUAL_GEOMETRY", "Provide real geometry; do not infer it."
    elif not format_valid or not geometry_valid:
        block, fix = "BLOCKED_INVALID_GEOMETRY", "Fix geometry format or payload."
    elif not crs_present or not crs_accepted:
        block, fix = "BLOCKED_UNKNOWN_CRS", "Provide an accepted verified CRS."
    elif is_point and target_type == "patch":
        block, fix = "BLOCKED_POINT_NOT_PATCH_BOUNDARY", "Provide a polygon/bbox, not a point."
    elif is_point:
        block, fix = "POINT_ANCHOR_NOT_OVERLAY", "Provide an observed-event polygon."
    elif not is_polygon:
        block, fix = "BLOCKED_INVALID_GEOMETRY", "Provide polygon geometry."
    elif not provenance:
        block, fix = "BLOCKED_MISSING_PROVENANCE", "Document source type and provenance note."
    elif not license_present:
        block, fix = "BLOCKED_MISSING_LICENSE", "Document license status."
    elif not review_valid:
        block, fix = "BLOCKED_PENDING_REVIEW", "Set a valid reviewed intake status."
    else:
        block, fix = "", ""
    usable = not block
    target_id = clean(row.get("patch_id")) if target_type == "patch" else clean(row.get("event_id"))
    return {
        "validation_id": stable_id("V2AX_VAL_", target_type, target_id),
        "target_type": target_type, "target_id": target_id, "package_id": package_id,
        "patch_id": target_id if target_type == "patch" else "",
        "event_id": target_id if target_type == "event" else "",
        "source_file": PATCH_INTAKE if target_type == "patch" else EVENT_INTAKE,
        "geometry_present": b(present), "geometry_format": source_type,
        "geometry_format_valid": b(format_valid and geometry_valid), "crs_present": b(crs_present),
        "crs_accepted": b(crs_accepted), "provenance_present": b(provenance),
        "license_present": b(license_present), "review_status_valid": b(review_valid),
        "is_point": b(is_point), "is_polygon_or_bbox": b(is_polygon),
        "can_feed_v2aw": b(usable), "can_feed_v2av": b(usable and target_type == "patch"),
        "can_feed_v2au": b(usable), "blocking_reason": block, "recommended_fix": fix,
        "notes": "Validation is fail-closed; usable geometry is not a label or final ground truth.",
    }


PATCH_CHECKS = [
    ("confirm_patch_id", "Confirmar patch_id", "Confirm patch identity against v2aw/v2av."),
    ("obtain_real_boundary", "Obter boundary vetorial real", "Collect real patch boundary."),
    ("record_format", "Registrar formato bbox/WKT/GeoJSON", "Fill source_type."),
    ("record_crs", "Registrar CRS", "Fill a verified accepted CRS."),
    ("record_provenance", "Registrar fonte/proveniencia", "Fill provenance fields."),
    ("record_license", "Registrar licenca", "Fill license_status."),
    ("record_operator", "Registrar responsavel", "Fill digitized_by."),
    ("record_date", "Registrar data de digitalizacao", "Fill digitized_at."),
    ("validate_geometry", "Validar geometria", "Run v2ax validation."),
    ("reject_point", "Validar que nao e ponto/centroide", "Provide area polygon, never centroid."),
    ("approve_replay", "Aprovar para v2aw/v2av", "Human review then approved status."),
]
EVENT_CHECKS = [
    ("confirm_event", "Confirmar evento", "Confirm event identity."),
    ("confirm_hazard", "Confirmar fenomeno", "Confirm hazard semantics."),
    ("obtain_observed_polygon", "Obter poligono observado real", "Collect observed event polygon."),
    ("separate_context", "Distinguir observado de risco/contexto", "Do not promote context geometry."),
    ("record_crs", "Registrar CRS", "Fill a verified accepted CRS."),
    ("record_source", "Registrar fonte", "Fill provenance fields."),
    ("record_license", "Registrar licenca", "Fill license_status."),
    ("record_operator", "Registrar responsavel", "Fill digitized_by."),
    ("reject_cprm_overlay", "Validar que ponto CPRM nao e overlay", "Keep CPRM point anchor-only."),
    ("approve_v2au", "Aprovar para v2au", "Human review then approved status."),
]


def build_checklists(patches, events, validation):
    by_target = {(row["target_type"], row["target_id"]): row for row in validation}
    patch_rows, event_rows = [], []
    for patch in patches:
        val = by_target[("patch", patch["patch_id"])]
        for code, item, how in PATCH_CHECKS:
            patch_rows.append({
                "check_id": stable_id("V2AX_PCHK_", patch["patch_id"], code),
                "patch_id": patch["patch_id"], "priority_rank": patch["priority_rank"],
                "check_item": item, "is_required": "true",
                "current_status": "completed" if val["can_feed_v2av"] == "true" else "pending",
                "blocking_reason": val["blocking_reason"], "how_to_complete": how,
                "notes": "Checklist does not authorize labels or automatic C4.",
            })
    for event in events:
        val = by_target[("event", event["event_id"])]
        for code, item, how in EVENT_CHECKS:
            event_rows.append({
                "check_id": stable_id("V2AX_ECHK_", event["event_id"], code),
                "event_id": event["event_id"], "check_item": item, "is_required": "true",
                "current_status": "completed" if val["can_feed_v2au"] == "true" else "pending",
                "blocking_reason": val["blocking_reason"], "how_to_complete": how,
                "notes": "Observed polygon remains review-only evidence.",
            })
    return patch_rows, event_rows


def build_matrix(packages, patch_validation, event_validation):
    rows = []
    for package in packages:
        patch_id, event_id = clean(package.get("patch_id")), clean(package.get("event_id"))
        pv, ev = patch_validation[patch_id], event_validation[event_id]
        patch_ok, event_ok = pv["can_feed_v2av"] == "true", ev["can_feed_v2au"] == "true"
        ready_v2au = patch_ok and event_ok
        blocker = pv["blocking_reason"] if not patch_ok else ev["blocking_reason"] if not event_ok else ""
        rows.append({
            "matrix_id": stable_id("V2AX_MATRIX_", package.get("package_id")),
            "package_id": clean(package.get("package_id")), "patch_id": patch_id, "event_id": event_id,
            "region": "Recife", "city": clean(package.get("city")) or "Recife",
            "hazard_type": clean(package.get("hazard_type")), "priority_rank": "1",
            "current_allowed_use": clean(package.get("allowed_use")),
            "current_promotion_decision": clean(package.get("promotion_decision")),
            "needs_patch_boundary": b(not patch_ok), "needs_event_polygon": b(not event_ok),
            "has_patch_boundary_source": pv["geometry_present"],
            "has_event_polygon_source": ev["geometry_present"], "ready_for_v2aw": b(patch_ok or event_ok),
            "ready_for_v2av": b(patch_ok), "ready_for_v2au": b(ready_v2au),
            "remaining_blocker": blocker,
            "next_action": "Replay v2aw then v2av then v2au under human review" if ready_v2au
            else "Complete and validate the missing real geometry fields",
            "notes": "Maximum downstream decision is C4_CANDIDATE_REQUIRES_HUMAN_REVIEW.",
        })
    return rows


def build_plan(patches, events, validation):
    by_target = {(row["target_type"], row["target_id"]): row for row in validation}
    rows = []
    for patch in patches:
        val = by_target[("patch", patch["patch_id"])]
        rows.append({
            "collection_task_id": stable_id("V2AX_TASK_", "patch", patch["patch_id"]),
            "target_type": "patch", "target_id": patch["patch_id"], "region": "Recife",
            "city": patch["city"], "priority_rank": patch["priority_rank"],
            "recommended_source_class": "patch_generation_metadata|gis_export|manual_digitization",
            "recommended_source_name": "Sentinel patch generation metadata or verified GIS export",
            "what_to_collect": "Real patch boundary polygon", "accepted_formats": "bbox|wkt|geojson",
            "required_crs": "EPSG:4326|EPSG:3857|EPSG:31982|EPSG:31983",
            "why_needed": "Required for v2av boundary validation and v2au overlay.",
            "current_blocker": val["blocking_reason"],
            "collection_status": "ready_for_replay" if val["can_feed_v2av"] == "true" else "pending_manual_collection",
            "notes": "Do not use centroid/default size; recommended sources are not assumed to exist.",
        })
    for event in events:
        val = by_target[("event", event["event_id"])]
        rows.append({
            "collection_task_id": stable_id("V2AX_TASK_", "event", event["event_id"]),
            "target_type": "event", "target_id": event["event_id"], "region": "Recife",
            "city": event["city"], "priority_rank": "1",
            "recommended_source_class": "official_vector|manual_digitization",
            "recommended_source_name": "Charter/EMS/VHR verified product",
            "what_to_collect": "Real observed-event polygon", "accepted_formats": "wkt|geojson",
            "required_crs": "EPSG:4326|EPSG:3857|EPSG:31982|EPSG:31983",
            "why_needed": "Required for v2au patch-event overlay.",
            "current_blocker": val["blocking_reason"],
            "collection_status": "ready_for_replay" if val["can_feed_v2au"] == "true" else "pending_manual_collection",
            "notes": "Media/social/EM-DAT cannot close geometry alone; quickview is not verified product.",
        })
    return rows


def build_manifest(matrix):
    rows = []
    for row in matrix:
        rows.append({
            "intake_manifest_id": stable_id("V2AX_MAN_", row["package_id"]),
            "package_id": row["package_id"], "patch_id": row["patch_id"], "event_id": row["event_id"],
            "region": row["region"], "city": row["city"], "hazard_type": row["hazard_type"],
            "priority_rank": row["priority_rank"],
            "manual_patch_file": f"datasets/manual_intake/recife_p1/{PATCH_INTAKE}",
            "manual_event_file": f"datasets/manual_intake/recife_p1/{EVENT_INTAKE}",
            "needs_patch_boundary": row["needs_patch_boundary"],
            "needs_event_geometry": row["needs_event_polygon"],
            "patch_boundary_status": "READY" if row["ready_for_v2av"] == "true" else "BLOCKED",
            "event_geometry_status": "READY" if row["has_event_polygon_source"] == "true" else "BLOCKED",
            "ready_for_v2aw": row["ready_for_v2aw"], "ready_for_v2av": row["ready_for_v2av"],
            "ready_for_v2au": row["ready_for_v2au"], "blocking_reason": row["remaining_blocker"],
            "next_action": row["next_action"], "notes": row["notes"],
        })
    return rows


def build_exports(patches, events, validation):
    valid = {(row["target_type"], row["target_id"]): row for row in validation}
    aw_patch, aw_event, av_patch, au_geom = [], [], [], []
    for row in patches:
        if valid[("patch", row["patch_id"])]["can_feed_v2av"] != "true":
            continue
        aw_patch.append({
            "geometry_source_id": row["intake_id"], "linked_patch_id": row["patch_id"],
            "region": row["region"], "city": row["city"], "priority_rank": row["priority_rank"],
            "source_type": row["source_type"], "geometry_value": row["geometry_value"],
            "geometry_path": row["geometry_path"], "crs": row["crs"],
            "provenance_type": row["provenance_type"], "provenance_note": row["provenance_note"],
            "digitized_by": row["digitized_by"], "digitized_at": row["digitized_at"],
            "source_document": row["source_document"], "source_document_page": row["source_document_page"],
            "source_confidence": row["source_confidence"], "license_status": row["license_status"],
            "review_status": row["review_status"], "notes": "Validated v2ax export; not a label.",
        })
        av_patch.append({
            "patch_id": row["patch_id"], "region": row["region"], "city": row["city"],
            "source_file": f"datasets/manual_intake/recife_p1/{PATCH_INTAKE}",
            "source_field": "geometry_value|geometry_path", "source_type": row["source_type"],
            "geometry_value": row["geometry_value"], "geometry_path": row["geometry_path"],
            "crs": row["crs"], "center_lat": "", "center_lon": "", "size_meters": "",
            "source_confidence": row["source_confidence"],
        })
        au_geom.append({
            "geometry_role": "patch_boundary", "linked_event_id": "", "linked_patch_id": row["patch_id"],
            "source_id": row["intake_id"], "source_name": row["source_document"],
            "geometry_type": "polygon", "geometry_format": row["source_type"],
            "geometry_value": row["geometry_value"], "geometry_path": row["geometry_path"],
            "crs": row["crs"], "latitude": "", "longitude": "",
        })
    for row in events:
        if valid[("event", row["event_id"])]["can_feed_v2au"] != "true":
            continue
        aw_event.append({
            "event_geometry_source_id": row["event_intake_id"], "linked_event_id": row["event_id"],
            "region": row["region"], "city": row["city"], "hazard_type": row["hazard_type"],
            "source_type": row["source_type"], "geometry_value": row["geometry_value"],
            "geometry_path": row["geometry_path"], "crs": row["crs"],
            "event_geometry_role": row["event_geometry_role"], "source_id": row["source_id"],
            "source_name": row["source_name"], "provenance_type": row["provenance_type"],
            "provenance_note": row["provenance_note"], "digitized_by": row["digitized_by"],
            "digitized_at": row["digitized_at"], "source_document": row["source_document"],
            "source_document_page": row["source_document_page"], "source_confidence": row["source_confidence"],
            "license_status": row["license_status"], "review_status": row["review_status"],
            "notes": "Validated v2ax event export; not ground truth.",
        })
        au_geom.append({
            "geometry_role": "event_observed_geometry", "linked_event_id": row["event_id"],
            "linked_patch_id": "", "source_id": row["source_id"] or row["event_intake_id"],
            "source_name": row["source_name"] or row["source_document"], "geometry_type": "polygon",
            "geometry_format": row["source_type"], "geometry_value": row["geometry_value"],
            "geometry_path": row["geometry_path"], "crs": row["crs"], "latitude": "", "longitude": "",
        })
    return aw_patch, aw_event, av_patch, au_geom


def schema(title, columns, enums=None, booleans=None):
    enums, booleans = enums or {}, booleans or set()
    properties = {}
    for column in columns:
        prop = {"type": "string"}
        if column in enums:
            prop["enum"] = enums[column]
        if column in booleans:
            prop["enum"] = ["true", "false"]
        properties[column] = prop
    return {
        "$schema": "http://json-schema.org/draft-07/schema#", "title": title,
        "description": "v2ax fail-closed row contract. No operational label, training, final ground truth, or automatic C4.",
        "type": "object", "required": columns, "additionalProperties": False,
        "properties": properties,
    }


def write_schemas(dataset_dir):
    schema_dir = os.path.join(dataset_dir, "schemas")
    specs = {
        "v2ax_recife_manual_intake_manifest.schema.json": (MANIFEST_COLUMNS, {
            "ready_for_v2aw", "ready_for_v2av", "ready_for_v2au", "needs_patch_boundary",
            "needs_event_geometry"}),
        "v2ax_recife_manual_intake_validation.schema.json": (VALIDATION_COLUMNS, {
            "geometry_present", "geometry_format_valid", "crs_present", "crs_accepted",
            "provenance_present", "license_present", "review_status_valid", "is_point",
            "is_polygon_or_bbox", "can_feed_v2aw", "can_feed_v2av", "can_feed_v2au"}),
        "v2ax_recife_p1_patch_geometry_intake.schema.json": (PATCH_COLUMNS, set()),
        "v2ax_recife_p1_event_geometry_intake.schema.json": (EVENT_COLUMNS, set()),
        "v2ax_recife_p1_package_review_matrix.schema.json": (MATRIX_COLUMNS, {
            "needs_patch_boundary", "needs_event_polygon", "has_patch_boundary_source",
            "has_event_polygon_source", "ready_for_v2aw", "ready_for_v2av", "ready_for_v2au"}),
        "v2ax_recife_p1_geometry_collection_plan.schema.json": (PLAN_COLUMNS, set()),
    }
    for name, (columns, bools) in specs.items():
        enums = {}
        if "source_type" in columns:
            enums["source_type"] = ["missing", "bbox", "wkt", "geojson_inline", "geojson_file"]
        write_text(os.path.join(schema_dir, name),
                   json.dumps(schema(name[:-12], columns, enums, bools), indent=2) + "\n")


def write_examples(dataset_dir):
    directory = os.path.join(dataset_dir, EXAMPLE_SUBDIR)
    write_text(os.path.join(directory, "README.md"), """# v2ax synthetic examples

All files are synthetic and use only `SYNTHETIC_*` IDs. Never mix them with real intake.
Valid examples demonstrate accepted formats. Invalid examples demonstrate fail-closed blockers.
Nothing here is a label, final ground truth, or training target.
""")
    patch_header = ",".join(PATCH_COLUMNS) + "\n"
    common = "SYNTHETIC_INTAKE,SYNTHETIC_PATCH_001,Example,Example,1,1,patch_boundary_polygon"
    tail = ",manual_digitization,SYNTHETIC EXAMPLE,example_doc,,,example,2026-01-01,EXAMPLE,example_license,provided_unreviewed,,,SYNTHETIC EXAMPLE\n"
    write_text(os.path.join(directory, "synthetic_patch_boundary_bbox_example.csv"),
               patch_header + common + ',bbox,"0,0,10,10",,EPSG:3857' + tail)
    write_text(os.path.join(directory, "synthetic_patch_boundary_wkt_example.csv"),
               patch_header + common + ',wkt,"POLYGON((0 0,10 0,10 10,0 10,0 0))",,EPSG:3857' + tail)
    write_text(os.path.join(directory, "synthetic_invalid_point_as_patch_boundary.csv"),
               patch_header + common + ',wkt,"POINT(1 2)",,EPSG:4326' + tail)
    write_text(os.path.join(directory, "synthetic_unknown_crs_blocked_example.csv"),
               patch_header + common + ',bbox,"0,0,10,10",,UNKNOWN' + tail)
    geojson = {"type": "Feature", "properties": {"id": "SYNTHETIC_PATCH_001", "synthetic": True},
               "geometry": {"type": "Polygon",
                            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]]}}
    write_text(os.path.join(directory, "synthetic_patch_boundary_geojson_example.geojson"),
               json.dumps(geojson, indent=2) + "\n")
    event_header = ",".join(EVENT_COLUMNS) + "\n"
    event = ('SYNTHETIC_EVENT_INTAKE,SYNTHETIC_EVENT_001,Example,Example,urban_flood,1,'
             'observed_event_polygon,wkt,"POLYGON((0 0,20 0,20 20,0 20,0 0))",,EPSG:3857,'
             'observed_event_polygon,SYNTHETIC_SOURCE,Synthetic source,manual_digitization,'
             'SYNTHETIC EXAMPLE,example_doc,,,example,2026-01-01,EXAMPLE,example_license,'
             'provided_unreviewed,,,SYNTHETIC EXAMPLE\n')
    write_text(os.path.join(directory, "synthetic_event_polygon_wkt_example.csv"), event_header + event)


def write_docs(docs_dir, observed_events):
    mismatch = ("O checkout atual comprova apenas 1 evento Recife nos pacotes, embora a configuracao espere 3. "
                "A v2ax registra a divergencia e nao inventa eventos." if observed_events != 3 else "")
    docs = {
        "v2ax_recife_geometry_intake_workflow.md": f"""# v2ax - Fluxo de intake geometrico Recife

A v2ax prepara os 55 patches Recife P1 e os eventos Recife comprovados para preenchimento manual.
Recife P1 e prioridade porque os pacotes `candidate_reference` seguem bloqueados sem boundary.
{mismatch}

Preencha os CSVs em `datasets/manual_intake/recife_p1/`, rode
`python scripts/run_v2ax_recife_geometry_intake_pack.py`, revise os blockers e use somente os
exports validados. Depois execute v2aw, v2av e v2au, nessa ordem. O fluxo nao cria label,
ground truth final, treino ou C4 automatico.
""",
        "v2ax_manual_digitization_protocol.md": """# v2ax - Protocolo de digitalizacao manual

Digitalize apenas boundary real de patch ou poligono observado real de evento, com fonte auditavel.
Patch boundary, evento observado, area de risco e contexto sao papeis distintos. Ponto/centroide nao
serve como poligono; ponto CPRM permanece anchor. CRS e obrigatorio. Registre proveniencia, documento,
operador, data, confianca e licenca. Quickview nao e produto verificado. Ausencia nunca vira negativo.
""",
        "v2ax_geometry_quality_checklist.md": """# v2ax - Checklist de qualidade geometrica

Verifique CRS aceito, validade do poligono, area computavel, fonte, proveniencia, licenca e revisao
humana. Blockers comuns: geometria ausente/invalida, CRS UNKNOWN, ponto como boundary, ponto de evento
como overlay, proveniencia/licenca ausente. Nenhuma aprovacao cria label ou ground truth final.
""",
        "v2ax_operator_handoff.md": """# v2ax - Handoff para operador

Abra o CSV de intake, preencha `source_type`, `geometry_value` ou `geometry_path`, `crs`,
proveniencia, documento, licenca, operador e status de revisao. Salve e rode
`python scripts/run_v2ax_recife_geometry_intake_pack.py`. Leia
`datasets/v2ax_recife_manual_intake_validation.csv`; corrija o blocker indicado. Somente exports
com linhas validadas podem alimentar v2aw/v2av/v2au.
""",
    }
    for name, text in docs.items():
        write_text(os.path.join(docs_dir, name), text)


def build_summary(inputs_found, patches, events, packages, validation, checklist_count, config):
    blocks = Counter(row["blocking_reason"] for row in validation if row["blocking_reason"])
    patch_sources = sum(row["target_type"] == "patch" and row["geometry_present"] == "true"
                        for row in validation)
    event_sources = sum(row["target_type"] == "event" and row["geometry_present"] == "true"
                        for row in validation)
    ready_aw = sum(row["can_feed_v2aw"] == "true" for row in validation)
    ready_av = sum(row["can_feed_v2av"] == "true" for row in validation)
    ready_au = sum(row["can_feed_v2au"] == "true" and row["target_type"] == "event"
                   for row in validation)
    mismatch = len(events) != int(config["expected_recife_events"])
    return {
        "stage": STAGE, "stage_scope": "v2ax_recife_geometry_intake_pack",
        "namespace_collision_note": "The repository also contains the pre-existing Protocolo C "
        "hydrometeorological v2ax track; full artifact names keep both scopes separate.",
        "status": "OK_WITH_EXPECTED_BLOCKERS_AND_EVENT_COUNT_MISMATCH"
        if mismatch else "OK_WITH_EXPECTED_BLOCKERS",
        "total_recife_p1_patches": len(patches), "total_recife_events": len(events),
        "expected_recife_events": int(config["expected_recife_events"]),
        "recife_event_count_mismatch": mismatch, "total_packages_covered": len(packages),
        "manual_patch_rows_created": len(patches), "manual_event_rows_created": len(events),
        "manual_checklist_rows_created": checklist_count, "patch_sources_provided": patch_sources,
        "event_sources_provided": event_sources, "ready_for_v2aw_count": ready_aw,
        "ready_for_v2av_count": ready_av, "ready_for_v2au_count": ready_au,
        "blocked_pending_manual_geometry_count": blocks["BLOCKED_PENDING_MANUAL_GEOMETRY"],
        "blocked_unknown_crs_count": blocks["BLOCKED_UNKNOWN_CRS"],
        "blocked_invalid_geometry_count": blocks["BLOCKED_INVALID_GEOMETRY"],
        "event_count_mismatch_blocker": "EXPECTED_RECIFE_EVENTS_NOT_FOUND" if mismatch else "",
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "RECIFE_GEOMETRY_INTAKE_PACK_READY_NOT_FOR_TRAINING",
        "inputs_found": inputs_found,
    }


def build_report(summary):
    return f"""# v2ax - Recife Geometry Intake Pack

## Objetivo
Transformar a ausencia de geometria real em fluxo manual auditavel e fail-closed.

Escopo completo: `{summary['stage_scope']}`. O repositorio tambem possui uma trilha v2ax
hidrometeorologica preexistente em Protocolo C; os nomes completos mantem os escopos separados.

## Entradas
{os.linesep.join('- `' + item + '`' for item in summary['inputs_found'])}

## Contagens
- Patches Recife P1: **{summary['total_recife_p1_patches']}**
- Eventos Recife comprovados: **{summary['total_recife_events']}** (esperados no config: {summary['expected_recife_events']})
- Pacotes cobertos: **{summary['total_packages_covered']}**
- Geometrias de patch fornecidas: **{summary['patch_sources_provided']}**
- Geometrias de evento fornecidas: **{summary['event_sources_provided']}**
- Prontos para v2aw/v2av/v2au: **{summary['ready_for_v2aw_count']} / {summary['ready_for_v2av_count']} / {summary['ready_for_v2au_count']}**

## Blockers
- Pendentes de geometria manual: **{summary['blocked_pending_manual_geometry_count']}**
- CRS desconhecido: **{summary['blocked_unknown_crs_count']}**
- Geometria invalida: **{summary['blocked_invalid_geometry_count']}**
- Divergencia de eventos Recife: `{summary['event_count_mismatch_blocker'] or 'NONE'}`

Preencha `datasets/manual_intake/recife_p1/`, rode novamente a v2ax e use apenas exports validados.
Depois execute v2aw, v2av e v2au sob revisao humana.

## Guardrails
Nenhum label, modelo, treino supervisionado, ground truth final ou promocao C4 automatica foi criado.
`can_train_model=false`; `can_create_operational_labels=false`.
"""


def log_lines(summary):
    lines = [
        f"[v2ax] patches={summary['total_recife_p1_patches']} events={summary['total_recife_events']} "
        f"expected_events={summary['expected_recife_events']} packages={summary['total_packages_covered']}",
        f"[v2ax] patch_sources={summary['patch_sources_provided']} event_sources={summary['event_sources_provided']}",
        f"[v2ax] ready_v2aw={summary['ready_for_v2aw_count']} ready_v2av={summary['ready_for_v2av_count']} "
        f"ready_v2au={summary['ready_for_v2au_count']}",
        f"[v2ax] pending_manual={summary['blocked_pending_manual_geometry_count']} "
        f"event_mismatch={summary['recife_event_count_mismatch']}",
        "[v2ax] can_train_model=False can_create_operational_labels=False",
        f"[v2ax] status={summary['status']}",
    ]
    return "\n".join(lines) + "\n"


def run(dataset_dir=None, output_dir=None, config_dir=None, docs_dir=None):
    env_dataset, env_output, env_config, env_docs = resolve_dirs()
    dataset_dir, output_dir = dataset_dir or env_dataset, output_dir or env_output
    config_dir, docs_dir = config_dir or env_config, docs_dir or env_docs
    config = load_config(config_dir)
    inputs, found = load_inputs(dataset_dir)
    packages = recife_packages(inputs, config)

    manual_dir = os.path.join(dataset_dir, MANUAL_SUBDIR)
    patch_path, event_path = os.path.join(manual_dir, PATCH_INTAKE), os.path.join(manual_dir, EVENT_INTAKE)
    patches = merge_preserved(build_patch_defaults(inputs, packages), load_csv(patch_path),
                              "intake_id", PATCH_COLUMNS)
    events = merge_preserved(build_event_defaults(packages), load_csv(event_path),
                             "event_intake_id", EVENT_COLUMNS)
    package_by_patch = {clean(row.get("patch_id")): clean(row.get("package_id")) for row in packages}
    package_by_event = {clean(row.get("event_id")): clean(row.get("package_id")) for row in packages}
    validation = [
        validate_intake("patch", row, package_by_patch.get(row["patch_id"], ""), config, dataset_dir)
        for row in patches
    ] + [
        validate_intake("event", row, package_by_event.get(row["event_id"], ""), config, dataset_dir)
        for row in events
    ]
    validation.sort(key=lambda row: (row["target_type"], row["target_id"]))
    by_target = {(row["target_type"], row["target_id"]): row for row in validation}
    for row in patches:
        val = by_target[("patch", row["patch_id"])]
        row["validation_status"] = "READY_FOR_REPLAY" if val["can_feed_v2av"] == "true" \
            else "BLOCKED_PENDING_MANUAL_GEOMETRY"
        row["blocking_reason"] = val["blocking_reason"]
    for row in events:
        val = by_target[("event", row["event_id"])]
        row["validation_status"] = "READY_FOR_REPLAY" if val["can_feed_v2au"] == "true" \
            else "BLOCKED_PENDING_MANUAL_GEOMETRY"
        row["blocking_reason"] = val["blocking_reason"]

    patch_validation = {row["target_id"]: row for row in validation if row["target_type"] == "patch"}
    event_validation = {row["target_id"]: row for row in validation if row["target_type"] == "event"}
    patch_checks, event_checks = build_checklists(patches, events, validation)
    matrix = build_matrix(packages, patch_validation, event_validation)
    plan = build_plan(patches, events, validation)
    manifest = build_manifest(matrix)
    aw_patch, aw_event, av_patch, au_geom = build_exports(patches, events, validation)

    write_csv(patch_path, PATCH_COLUMNS, patches)
    write_csv(event_path, EVENT_COLUMNS, events)
    write_csv(os.path.join(manual_dir, PATCH_CHECKLIST), PATCH_CHECK_COLUMNS, patch_checks)
    write_csv(os.path.join(manual_dir, EVENT_CHECKLIST), EVENT_CHECK_COLUMNS, event_checks)
    write_csv(os.path.join(manual_dir, PACKAGE_MATRIX), MATRIX_COLUMNS, matrix)
    write_csv(os.path.join(manual_dir, COLLECTION_PLAN), PLAN_COLUMNS, plan)
    write_csv(os.path.join(manual_dir, MANUAL_VALIDATION), VALIDATION_COLUMNS, validation)
    write_text(os.path.join(manual_dir, MANUAL_README), """# Recife P1 manual geometry intake

Fill only real geometry with verified CRS, provenance, license and human review.
Run `python scripts/run_v2ax_recife_geometry_intake_pack.py` after editing.
Use only rows exported by v2ax. Missing data remains blocked and is never inferred.
""")
    write_csv(os.path.join(dataset_dir, OUT_MANIFEST), MANIFEST_COLUMNS, manifest)
    write_csv(os.path.join(dataset_dir, OUT_VALIDATION), VALIDATION_COLUMNS, validation)
    write_csv(os.path.join(dataset_dir, OUT_V2AW_PATCH), V2AW_PATCH_COLUMNS, aw_patch)
    write_csv(os.path.join(dataset_dir, OUT_V2AW_EVENT), V2AW_EVENT_COLUMNS, aw_event)
    write_csv(os.path.join(dataset_dir, OUT_V2AV_PATCH), V2AV_PATCH_COLUMNS, av_patch)
    write_csv(os.path.join(dataset_dir, OUT_V2AU_GEOM), V2AU_GEOM_COLUMNS, au_geom)
    write_schemas(dataset_dir)
    write_examples(dataset_dir)
    write_docs(docs_dir, len(events))

    summary = build_summary(found, patches, events, packages, validation,
                            len(patch_checks) + len(event_checks), config)
    write_text(os.path.join(output_dir, SUMMARY_REL), json.dumps(summary, indent=2) + "\n")
    write_text(os.path.join(output_dir, REPORT_REL), build_report(summary))
    write_text(os.path.join(output_dir, LOG_REL), log_lines(summary))
    sys.stdout.write(log_lines(summary))
    return 0, summary


def main(_argv=None):
    code, _ = run()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
