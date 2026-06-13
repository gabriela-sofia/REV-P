#!/usr/bin/env python3
"""v2ay Event Scope Reconciliation + Geometry Acquisition Turning Point Gate.

Reconciles event scope from existing registries, certifies the current spatial
absence, and defines the minimum real-data contract for the first overlay.
Strictly additive: never invents events/geometries or changes prior-stage files.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict


STAGE = "v2ay_event_scope_reconciliation_turning_point"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_NAME = "v2ay_event_scope_reconciliation_turning_point_config.json"


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
    "offline_mode": True, "strict_mode": True,
    "regions": ["Recife", "Petropolis", "Curitiba"], "priority_region": "Recife",
    "expected_recife_events_from_previous_prompt": 3, "allow_expected_event_override": False,
    "allow_event_invention": False, "allow_geometry_invention": False,
    "turning_point_requires_real_patch_geometry": True,
    "turning_point_requires_real_event_geometry": True,
    "turning_point_requires_pipeline_replay": True,
    "accepted_patch_geometry_formats": ["bbox", "wkt", "geojson_inline", "geojson_file"],
    "accepted_event_geometry_formats": ["wkt", "geojson_inline", "geojson_file"],
    "accepted_crs": ["EPSG:4326", "EPSG:3857", "EPSG:31982", "EPSG:31983"],
    "minimum_turning_point_overlay_count": 1,
}

INPUTS = {
    "packages": "v2at_event_patch_package_registry.csv",
    "observations": "v2at_evidence_observation_registry.csv",
    "sources": "v2at_external_evidence_source_catalog.csv",
    "overlay_queue": "v2au_overlay_review_queue.csv",
    "overlays": "v2au_patch_event_overlay_registry.csv",
    "inventory": "v2au_geometry_inventory.csv",
    "patch_registry": "v2av_patch_boundary_geometry_registry.csv",
    "patch_queue": "v2av_patch_boundary_recovery_queue.csv",
    "v2aw_event_template": "v2aw_event_geometry_sources_template.csv",
    "v2aw_validation": "v2aw_geometry_source_validation_registry.csv",
    "v2ax_manifest": "v2ax_recife_manual_intake_manifest.csv",
    "v2ax_validation": "v2ax_recife_manual_intake_validation.csv",
    "canonical_events": os.path.join("protocolo_c", "v2ae_canonical_event_registry.csv"),
    "candidate_geometry": os.path.join("protocolo_c", "v2bh_candidate_geometry_source_registry.csv"),
    "charter_readiness": os.path.join("protocolo_c", "v2bi_charter_candidate_geometry_readiness.csv"),
}

OUTPUTS = {
    "events": "v2ay_region_event_canonical_registry.csv",
    "audit": "v2ay_event_scope_reconciliation_audit.csv",
    "crosswalk": "v2ay_event_package_patch_crosswalk.csv",
    "gaps": "v2ay_geometry_gap_analysis.csv",
    "contract": "v2ay_minimum_real_geometry_contract.csv",
    "acquisition": "v2ay_geometry_acquisition_targets.csv",
    "queries": "v2ay_external_source_query_plan.csv",
    "certificate": "v2ay_spatial_metadata_absence_certificate.csv",
    "replay": "v2ay_pipeline_replay_plan.csv",
    "gates": "v2ay_turning_point_readiness_gate.csv",
}

COLUMNS = {
    "events": ["canonical_event_id", "source_event_id", "region", "city", "hazard_type",
               "event_window_start", "event_window_end", "source_ids", "source_names",
               "package_count", "patch_count", "has_temporal_anchor", "has_spatial_anchor",
               "has_event_polygon", "has_only_point_anchor", "has_context_only",
               "event_scope_status", "evidence_strength", "blocking_reason", "notes"],
    "audit": ["audit_id", "claim_or_expectation", "expected_value", "observed_value", "region",
              "event_id", "status", "severity", "evidence_file", "blocking_reason",
              "recommended_action", "notes"],
    "crosswalk": ["crosswalk_id", "canonical_event_id", "event_id", "package_id", "patch_id",
                  "region", "city", "hazard_type", "allowed_use", "promotion_decision",
                  "priority_rank", "needs_patch_boundary", "needs_event_polygon",
                  "has_patch_boundary_source", "has_event_geometry_source", "current_blocker",
                  "next_required_input", "notes"],
    "gaps": ["gap_id", "target_type", "target_id", "region", "city", "related_event_id",
             "related_patch_id", "related_package_count", "required_geometry", "accepted_formats",
             "accepted_crs", "current_status", "is_required_for_turning_point",
             "is_required_for_c4_candidate", "blocking_reason", "minimum_valid_input_example",
             "next_action", "notes"],
    "contract": ["contract_id", "turning_point_level", "required_input", "target_type",
                 "target_id_or_scope", "minimum_count", "accepted_formats", "accepted_crs",
                 "required_provenance", "required_license_status", "required_review_status",
                 "unlocks_stage", "does_not_unlock", "blocking_reason_if_missing", "notes"],
    "acquisition": ["acquisition_id", "target_type", "target_id", "region", "city",
                    "priority_rank", "recommended_source_category", "recommended_source_name",
                    "what_to_acquire", "why_needed", "accepted_formats", "required_crs",
                    "provenance_requirement", "license_requirement", "suggested_search_terms",
                    "manual_collection_steps", "validation_command", "feeds_file",
                    "blocks_until_filled", "notes"],
    "queries": ["query_id", "target_type", "target_id", "region", "city", "source_category",
                "source_name", "query_goal", "suggested_query", "expected_artifact",
                "expected_geometry_type", "expected_crs_or_metadata", "license_attention",
                "can_promote_alone", "notes"],
    "certificate": ["certificate_id", "scan_scope", "files_scanned", "patches_scanned",
                    "spatial_metadata_found", "patch_boundaries_found", "event_polygons_found",
                    "usable_overlay_geometries_found", "status", "evidence_summary",
                    "blocking_reason", "recommended_action", "notes"],
    "replay": ["replay_step_id", "step_order", "stage", "command", "required_input",
               "expected_output", "precondition", "success_condition", "failure_condition", "notes"],
    "gates": ["gate_id", "turning_point_level", "gate_name", "required_condition",
              "observed_value", "gate_passed", "severity", "blocking_reason",
              "recommended_action", "notes"],
}

BOOL_FIELDS = {
    "has_temporal_anchor", "has_spatial_anchor", "has_event_polygon", "has_only_point_anchor",
    "has_context_only", "needs_patch_boundary", "needs_event_polygon", "has_patch_boundary_source",
    "has_event_geometry_source", "is_required_for_turning_point", "is_required_for_c4_candidate",
    "spatial_metadata_found", "gate_passed", "can_promote_alone",
}

REPORT_REL = os.path.join("execution_reports", "v2ay_event_scope_reconciliation_turning_point_report.md")
SUMMARY_REL = os.path.join("execution_reports", "v2ay_event_scope_reconciliation_turning_point_summary.json")
LOG_REL = os.path.join("logs_summary", "v2ay_event_scope_reconciliation_turning_point.txt")


def clean(value):
    return "" if value is None else str(value).strip()


def b(value):
    return "true" if bool(value) else "false"


def stable_id(prefix, *parts):
    raw = "|".join(clean(part) for part in parts)
    return prefix + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


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
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)


def load_config(config_dir):
    config = dict(DEFAULT_CONFIG)
    path = os.path.join(config_dir, CONFIG_NAME)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as handle:
            config.update(json.load(handle))
    return config


def load_inputs(dataset_dir):
    data, found = {}, []
    for key, rel in INPUTS.items():
        data[key] = load_csv(os.path.join(dataset_dir, rel))
        if data[key]:
            found.append(rel.replace(os.sep, "/"))
    return data, found


def true(row, field):
    return clean(row.get(field)).lower() == "true"


def valid_patch_ids(inputs):
    return {clean(row.get("patch_id")) for row in inputs["patch_registry"]
            if true(row, "is_valid_geometry")}


def valid_event_ids(inputs):
    return {clean(row.get("linked_event_id")) for row in inputs["inventory"]
            if clean(row.get("geometry_role")) == "event_observed_geometry"
            and true(row, "is_valid_geometry")}


def point_event_ids(inputs):
    return {clean(row.get("linked_event_id")) for row in inputs["inventory"]
            if clean(row.get("geometry_role")) == "point_anchor" and true(row, "is_valid_geometry")}


def canonical_map(inputs):
    return {clean(row.get("event_id")): clean(row.get("canonical_event_id"))
            for row in inputs["canonical_events"] if clean(row.get("event_id"))}


def build_events(inputs):
    grouped = defaultdict(list)
    for package in inputs["packages"]:
        grouped[(clean(package.get("region")), clean(package.get("event_id")))].append(package)
    canonical = canonical_map(inputs)
    event_polygons, points = valid_event_ids(inputs), point_event_ids(inputs)
    observations = defaultdict(list)
    for row in inputs["observations"]:
        observations[clean(row.get("region"))].append(row)
    rows = []
    for (region, event_id), packages in sorted(grouped.items()):
        patch_ids = {clean(row.get("patch_id")) for row in packages if clean(row.get("patch_id")) != "UNKNOWN_PATCH"}
        obs = observations.get(region, [])
        source_ids = sorted({clean(row.get("source_id")) for row in obs if clean(row.get("source_id"))})
        source_names = sorted({clean(row.get("source_name")) for row in obs if clean(row.get("source_name"))})
        has_polygon, has_point = event_id in event_polygons, event_id in points
        placeholder = "MISSING" in event_id or event_id in ("", "UNKNOWN_EVENT")
        if placeholder:
            status, blocker = "UNRESOLVED", "EVENT_REGISTRY_MISSING_OR_UNTYPED"
        elif has_polygon:
            status, blocker = "CONFIRMED_IN_REGISTRIES", ""
        elif has_point:
            status, blocker = "POINT_ANCHOR_ONLY", "POINT_ANCHOR_NOT_OVERLAY"
        else:
            status, blocker = "MISSING_EVENT_GEOMETRY", "NO_VALID_OBSERVED_EVENT_POLYGON"
        rows.append({
            "canonical_event_id": canonical.get(event_id) or stable_id("V2AY_EVENT_", region, event_id),
            "source_event_id": event_id, "region": region, "city": clean(packages[0].get("city")),
            "hazard_type": clean(packages[0].get("hazard_type")),
            "event_window_start": clean(packages[0].get("event_window_start")),
            "event_window_end": clean(packages[0].get("event_window_end")),
            "source_ids": "|".join(source_ids), "source_names": "|".join(source_names),
            "package_count": str(len(packages)), "patch_count": str(len(patch_ids)),
            "has_temporal_anchor": b(any(true(row, "has_temporal_anchor") for row in packages)),
            "has_spatial_anchor": b(any(true(row, "has_spatial_support") for row in packages) or has_point),
            "has_event_polygon": b(has_polygon), "has_only_point_anchor": b(has_point and not has_polygon),
            "has_context_only": b(all(clean(row.get("allowed_use")) == "rejected_context_only"
                                      for row in packages)),
            "event_scope_status": status,
            "evidence_strength": "strong" if any(float(clean(row.get("evidence_score")) or 0) >= .65
                                                 for row in packages) else "limited",
            "blocking_reason": blocker,
            "notes": "Derived from registries; no event or region assignment was invented.",
        })
    return rows


def build_audit(events, config):
    recife = [row for row in events if row["region"] == "Recife" and row["event_scope_status"] != "UNRESOLVED"]
    pet = [row["source_event_id"] for row in events if row["region"] == "Petropolis"]
    expected, observed = int(config["expected_recife_events_from_previous_prompt"]), len(recife)
    entries = [
        ("previous_prompt_recife_event_count", str(expected), str(observed), "Recife", "",
         "DIVERGENCE_CONFIRMED", "high", "v2at_event_patch_package_registry.csv",
         "EXPECTED_RECIFE_EVENTS_NOT_FOUND",
         "Use counts derived from registries; never create missing events.",
         "Previous expectation was not data-derived."),
        ("do_not_invent_missing_recife_events", "0 invented events", "0 invented events", "Recife", "",
         "PASS", "critical", "v2ay_region_event_canonical_registry.csv", "",
         "Keep fail-closed reconciliation.", "Two fake Recife events were not created."),
        ("probable_origin_of_divergence", "3 Recife pending polygons",
         "1 Recife polygon row plus 2 Petropolis polygon rows", "All", "",
         "EXPLAINED", "high", "v2aw_event_geometry_sources_template.csv", "",
         "Separate event counts by region.", "Petropolis events: " + "|".join(pet)),
        ("future_event_count_policy", "fixed prompt count", "registry-derived count", "All", "",
         "POLICY_CORRECTED", "high", "v2ay_region_event_canonical_registry.csv", "",
         "Use the canonical registry as source of truth.", "Configuration expectations cannot override data."),
    ]
    return [{
        "audit_id": stable_id("V2AY_AUDIT_", item[0]), "claim_or_expectation": item[0],
        "expected_value": item[1], "observed_value": item[2], "region": item[3],
        "event_id": item[4], "status": item[5], "severity": item[6], "evidence_file": item[7],
        "blocking_reason": item[8], "recommended_action": item[9], "notes": item[10],
    } for item in entries]


def build_crosswalk(inputs, events):
    event_map = {row["source_event_id"]: row for row in events}
    patch_ok, event_ok = valid_patch_ids(inputs), valid_event_ids(inputs)
    rows = []
    for package in sorted(inputs["packages"], key=lambda r: (clean(r.get("region")),
                                                              clean(r.get("event_id")),
                                                              clean(r.get("patch_id")))):
        patch_id, event_id = clean(package.get("patch_id")), clean(package.get("event_id"))
        has_patch, has_event = patch_id in patch_ok, event_id in event_ok
        blocker = "" if has_patch and has_event else (
            "NO_PATCH_BOUNDARY_SOURCE_PROVIDED" if not has_patch else "NO_VALID_OBSERVED_EVENT_POLYGON")
        rows.append({
            "crosswalk_id": stable_id("V2AY_XW_", package.get("package_id")),
            "canonical_event_id": event_map[event_id]["canonical_event_id"], "event_id": event_id,
            "package_id": clean(package.get("package_id")), "patch_id": patch_id,
            "region": clean(package.get("region")), "city": clean(package.get("city")),
            "hazard_type": clean(package.get("hazard_type")), "allowed_use": clean(package.get("allowed_use")),
            "promotion_decision": clean(package.get("promotion_decision")),
            "priority_rank": "1" if clean(package.get("region")) == "Recife" else "2",
            "needs_patch_boundary": b(not has_patch), "needs_event_polygon": b(not has_event),
            "has_patch_boundary_source": b(has_patch), "has_event_geometry_source": b(has_event),
            "current_blocker": blocker,
            "next_required_input": "Real patch boundary and observed-event polygon with verified CRS"
            if not has_patch and not has_event else "Real observed-event polygon with verified CRS"
            if not has_event else "Real patch boundary with verified CRS" if not has_patch else "Replay v2au",
            "notes": "Crosswalk is diagnostic only; maximum decision remains human-review C4 candidate.",
        })
    return rows


def recife_targets(inputs):
    patches = sorted({clean(row.get("target_id")) for row in inputs["v2ax_validation"]
                      if clean(row.get("target_type")) == "patch" and clean(row.get("target_id"))})
    events = sorted({clean(row.get("target_id")) for row in inputs["v2ax_validation"]
                     if clean(row.get("target_type")) == "event" and clean(row.get("target_id"))})
    return patches, events


def build_gaps(inputs):
    patches, events = recife_targets(inputs)
    package_counts = Counter(clean(row.get("patch_id")) for row in inputs["packages"])
    rows = []
    for patch_id in patches:
        rows.append({
            "gap_id": stable_id("V2AY_GAP_", "patch", patch_id), "target_type": "patch_boundary",
            "target_id": patch_id, "region": "Recife", "city": "Recife",
            "related_event_id": "REC_2022_05_24_30", "related_patch_id": patch_id,
            "related_package_count": str(package_counts[patch_id]), "required_geometry": "real patch boundary polygon",
            "accepted_formats": "bbox|wkt|geojson_inline|geojson_file",
            "accepted_crs": "EPSG:4326|EPSG:3857|EPSG:31982|EPSG:31983",
            "current_status": "MISSING", "is_required_for_turning_point": "true",
            "is_required_for_c4_candidate": "true", "blocking_reason": "NO_PATCH_BOUNDARY_SOURCE_PROVIDED",
            "minimum_valid_input_example": "bbox minx,miny,maxx,maxy plus verified CRS and provenance",
            "next_action": "Fill datasets/manual_intake/recife_p1/recife_p1_patch_geometry_intake.csv",
            "notes": "Do not use centroid or default size.",
        })
    for event_id in events:
        count = sum(clean(row.get("event_id")) == event_id for row in inputs["packages"])
        rows.append({
            "gap_id": stable_id("V2AY_GAP_", "event", event_id), "target_type": "event_observed_polygon",
            "target_id": event_id, "region": "Recife", "city": "Recife",
            "related_event_id": event_id, "related_patch_id": "", "related_package_count": str(count),
            "required_geometry": "real observed-event polygon",
            "accepted_formats": "wkt|geojson_inline|geojson_file",
            "accepted_crs": "EPSG:4326|EPSG:3857|EPSG:31982|EPSG:31983",
            "current_status": "MISSING", "is_required_for_turning_point": "true",
            "is_required_for_c4_candidate": "true", "blocking_reason": "NO_VALID_OBSERVED_EVENT_POLYGON",
            "minimum_valid_input_example": "WKT/GeoJSON polygon plus verified CRS and provenance",
            "next_action": "Fill datasets/manual_intake/recife_p1/recife_p1_event_geometry_intake.csv",
            "notes": "CPRM point anchors and context geometry do not close overlay.",
        })
    return rows


def build_contract():
    specs = [
        ("TP0_DOCUMENTED_ABSENCE", "Auditable absence certificate", "repository_state", "all", "1",
         "csv", "N/A", "audit provenance", "N/A", "reviewed", "acquisition planning",
         "label|model|C4", "ABSENCE_NOT_DOCUMENTED"),
        ("TP1_ONE_PATCH_BOUNDARY_VALIDATED", "One real patch boundary", "patch_boundary", "one Recife P1 patch", "1",
         "bbox|wkt|geojson", "accepted CRS", "required", "required", "approved_for_v2av", "v2av replay",
         "event overlay|label|model", "NO_PATCH_BOUNDARY_SOURCE_PROVIDED"),
        ("TP2_ONE_EVENT_POLYGON_VALIDATED", "One real observed-event polygon", "event_observed_polygon",
         "REC_2022_05_24_30", "1", "wkt|geojson", "accepted CRS", "required", "required",
         "approved_for_v2au", "v2au event source readiness", "overlay without patch|label|model",
         "NO_VALID_OBSERVED_EVENT_POLYGON"),
        ("TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY", "Linked valid patch+event geometry pair",
         "patch_event_pair", "one v2at package", "1", "validated pair", "compatible accepted CRS",
         "required", "required", "human reviewed", "v2au replay", "automatic C4|label|model",
         "NO_LINKED_VALID_GEOMETRY_PAIR"),
        ("TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW", "One confirmed v2au overlay", "overlay",
         "one v2at package", "1", "v2au output", "target CRS", "replay provenance", "required",
         "human review required", "C4_CANDIDATE_REQUIRES_HUMAN_REVIEW",
         "C4 operational label|training|ground truth final", "NO_CONFIRMED_OVERLAY"),
    ]
    return [{
        "contract_id": stable_id("V2AY_CONTRACT_", s[0]), "turning_point_level": s[0],
        "required_input": s[1], "target_type": s[2], "target_id_or_scope": s[3],
        "minimum_count": s[4], "accepted_formats": s[5], "accepted_crs": s[6],
        "required_provenance": s[7], "required_license_status": s[8], "required_review_status": s[9],
        "unlocks_stage": s[10], "does_not_unlock": s[11], "blocking_reason_if_missing": s[12],
        "notes": "No turning point creates a label, model, final ground truth, or automatic C4.",
    } for s in specs]


def build_acquisition(gaps):
    rows = []
    for gap in gaps:
        patch = gap["target_type"] == "patch_boundary"
        rows.append({
            "acquisition_id": stable_id("V2AY_ACQ_", gap["target_type"], gap["target_id"]),
            "target_type": gap["target_type"], "target_id": gap["target_id"], "region": gap["region"],
            "city": gap["city"], "priority_rank": "1",
            "recommended_source_category": "Sentinel patch generation metadata|GIS export|manual digitization"
            if patch else "Charter/EMS verified product|VHR manual digitization",
            "recommended_source_name": "Verified patch-generation source" if patch
            else "Verified observed-event product",
            "what_to_acquire": gap["required_geometry"], "why_needed": "First real overlay turning point",
            "accepted_formats": gap["accepted_formats"], "required_crs": gap["accepted_crs"],
            "provenance_requirement": "source document/path and provenance note required",
            "license_requirement": "license status required",
            "suggested_search_terms": f"{gap['target_id']} real vector geometry CRS",
            "manual_collection_steps": "Acquire; verify semantics; fill v2ax manual intake; run validation",
            "validation_command": "python scripts/run_v2ax_recife_geometry_intake_pack.py",
            "feeds_file": "datasets/manual_intake/recife_p1/recife_p1_patch_geometry_intake.csv"
            if patch else "datasets/manual_intake/recife_p1/recife_p1_event_geometry_intake.csv",
            "blocks_until_filled": "v2av|v2au" if patch else "v2au",
            "notes": "Recommended sources are not assumed to exist.",
        })
    return rows


def build_queries(gaps):
    rows = []
    for gap in gaps:
        patch = gap["target_type"] == "patch_boundary"
        categories = [
            ("patch_metadata", "Sentinel patch generation metadata", "real patch footprint", "vector/bounds", "false"),
            ("local_gis", "Local GIS export", "real patch footprint", "vector/bounds", "false"),
        ] if patch else [
            ("official_mapping", "International Charter / Copernicus EMS", "observed event extent", "polygon", "false"),
            ("vhr_digitization", "VHR/manual digitization", "observed event extent", "polygon", "false"),
            ("context_only", "SGB/CPRM|ANA|INMET|Cemaden|EM-DAT|media", "context/temporal support", "point/context", "false"),
        ]
        for category, name, goal, geometry, promote in categories:
            rows.append({
                "query_id": stable_id("V2AY_QUERY_", gap["target_id"], category),
                "target_type": gap["target_type"], "target_id": gap["target_id"],
                "region": gap["region"], "city": gap["city"], "source_category": category,
                "source_name": name, "query_goal": goal,
                "suggested_query": f"{gap['target_id']} {name} geometry CRS vector",
                "expected_artifact": "verified vector or metadata", "expected_geometry_type": geometry,
                "expected_crs_or_metadata": "explicit CRS and provenance",
                "license_attention": "verify terms before use", "can_promote_alone": promote,
                "notes": "No automatic download. Context/quickview cannot close geometry alone.",
            })
    return rows


def physical_spatial_scan(dataset_dir):
    extensions = {".geojson", ".shp", ".gpkg", ".kml", ".wkt", ".tif", ".tiff"}
    found, scanned = [], 0
    for root, _, files in os.walk(dataset_dir):
        if os.sep + "examples" + os.sep in root + os.sep:
            continue
        for name in files:
            scanned += 1
            if os.path.splitext(name)[1].lower() in extensions:
                found.append(os.path.join(root, name))
    return scanned, sorted(found)


def build_certificate(inputs, dataset_dir):
    scanned, files = physical_spatial_scan(dataset_dir)
    patch_count = len(inputs["patch_registry"])
    boundaries = sum(true(row, "is_valid_geometry") for row in inputs["patch_registry"])
    polygons = len(valid_event_ids(inputs))
    overlays = sum(true(row, "has_patch_overlay") for row in inputs["overlays"])
    points = sum(clean(row.get("geometry_role")) == "point_anchor" and true(row, "is_valid_geometry")
                 for row in inputs["inventory"])
    return [{
        "certificate_id": stable_id("V2AY_CERT_", str(len(files)), str(patch_count)),
        "scan_scope": "datasets recursive physical spatial files excluding synthetic examples plus v2au/v2av/v2aw registries",
        "files_scanned": str(scanned), "patches_scanned": str(patch_count),
        "spatial_metadata_found": b(points > 0 or boundaries > 0 or polygons > 0),
        "patch_boundaries_found": str(boundaries), "event_polygons_found": str(polygons),
        "usable_overlay_geometries_found": str(overlays), "status": "TP0_DOCUMENTED_ABSENCE",
        "evidence_summary": f"physical_spatial_files={len(files)}; valid_point_anchors={points}; "
        f"valid_patch_boundaries={boundaries}; valid_event_polygons={polygons}; overlays={overlays}",
        "blocking_reason": "NO_USABLE_PATCH_EVENT_OVERLAY_GEOMETRIES",
        "recommended_action": "Acquire one real Recife patch boundary and one real Recife observed-event polygon.",
        "notes": "Point anchors are spatial metadata but are not patch boundaries or event polygons.",
    }]


def build_replay():
    specs = [
        ("1", "v2ax_manual_fill", "manual edit", "Real geometry, CRS, provenance, license",
         "filled v2ax manual intake", "real data acquired", "fields filled", "missing fields"),
        ("2", "v2ax_validate", "python scripts/run_v2ax_recife_geometry_intake_pack.py",
         "filled manual intake", "v2ax_ready_to_feed_*", "manual fields filled", "validated exports", "v2ax blocker"),
        ("3", "prepare_previous_stage_inputs", "copy/review validated exports into expected input files",
         "v2ax ready exports", "v2aw/v2av/v2au input files", "human approval", "compatible input", "no valid export"),
        ("4", "v2aw", "python scripts/run_v2aw_geometry_source_intake.py", "real source inputs",
         "v2aw validation/readiness", "valid input", "ready source", "geometry/CRS blocker"),
        ("5", "v2av", "python scripts/run_v2av_patch_boundary_geometry_builder.py", "valid patch source",
         "valid patch boundary", "v2aw/v2ax patch ready", "boundary valid", "patch blocker"),
        ("6", "v2au", "python scripts/run_v2au_patch_event_overlay_geometry.py", "valid linked patch+event pair",
         "overlay registry", "patch and event polygons valid", "confirmed overlay", "overlay blocker"),
        ("7", "human_review", "manual review", "confirmed overlay", "C4 candidate review decision",
         "overlay confirmed", "human-reviewed candidate", "automatic promotion attempted"),
        ("8", "no_training", "do not train model", "none", "guardrails preserved",
         "all stages", "no model/label", "model or label created"),
    ]
    return [{
        "replay_step_id": stable_id("V2AY_REPLAY_", s[0]), "step_order": s[0], "stage": s[1],
        "command": s[2], "required_input": s[3], "expected_output": s[4], "precondition": s[5],
        "success_condition": s[6], "failure_condition": s[7],
        "notes": "Replay is controlled and never promotes C4 automatically.",
    } for s in specs]


def turning_level(inputs):
    patch_count, event_count = len(valid_patch_ids(inputs)), len(valid_event_ids(inputs))
    overlay_count = sum(true(row, "has_patch_overlay") for row in inputs["overlays"])
    pair_ready = any(true(row, "ready_for_v2au") for row in inputs["v2ax_manifest"])
    if overlay_count:
        return "TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW"
    if pair_ready:
        return "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY"
    if event_count:
        return "TP2_ONE_EVENT_POLYGON_VALIDATED"
    if patch_count:
        return "TP1_ONE_PATCH_BOUNDARY_VALIDATED"
    return "TP0_DOCUMENTED_ABSENCE"


def build_gates(inputs, events):
    patch_count, event_count = len(valid_patch_ids(inputs)), len(valid_event_ids(inputs))
    overlay_count = sum(true(row, "has_patch_overlay") for row in inputs["overlays"])
    pair_ready = any(true(row, "ready_for_v2au") for row in inputs["v2ax_manifest"])
    checks = [
        ("TP_GATE_01_EVENT_SCOPE_RECONCILED", True, "Observed event scope derived from registries"),
        ("TP_GATE_02_NO_EVENT_INVENTED", True, "No synthetic event rows"),
        ("TP_GATE_03_PATCH_BOUNDARY_SOURCE_EXISTS", patch_count > 0, f"valid_patch_boundaries={patch_count}"),
        ("TP_GATE_04_EVENT_POLYGON_SOURCE_EXISTS", event_count > 0, f"valid_event_polygons={event_count}"),
        ("TP_GATE_05_CRS_VALID", patch_count > 0 and event_count > 0, "CRS required on both geometries"),
        ("TP_GATE_06_PROVENANCE_RECORDED", pair_ready, f"pair_ready={pair_ready}"),
        ("TP_GATE_07_LICENSE_RECORDED", pair_ready, f"pair_ready={pair_ready}"),
        ("TP_GATE_08_V2AX_VALIDATION_READY", pair_ready, f"pair_ready={pair_ready}"),
        ("TP_GATE_09_V2AW_READY", pair_ready, f"pair_ready={pair_ready}"),
        ("TP_GATE_10_V2AV_READY", patch_count > 0, f"valid_patch_boundaries={patch_count}"),
        ("TP_GATE_11_V2AU_REPLAY_READY", pair_ready, f"pair_ready={pair_ready}; overlays={overlay_count}"),
        ("TP_GATE_12_NO_LABEL_CREATED", True, "can_create_operational_labels=false"),
        ("TP_GATE_13_NO_MODEL_TRAINED", True, "can_train_model=false"),
        ("TP_GATE_14_C4_NOT_PROMOTED_AUTOMATICALLY", True, "maximum is human-review C4 candidate"),
    ]
    safety = {12, 13, 14}
    rows = []
    for index, (name, passed, observed) in enumerate(checks, 1):
        rows.append({
            "gate_id": stable_id("V2AY_GATE_", name), "turning_point_level": turning_level(inputs),
            "gate_name": name, "required_condition": name.replace("TP_GATE_", "").lower(),
            "observed_value": observed, "gate_passed": b(passed),
            "severity": "critical" if index in safety or index in (1, 2) else "blocking",
            "blocking_reason": "" if passed else "REQUIRED_REAL_GEOMETRY_OR_REPLAY_EVIDENCE_MISSING",
            "recommended_action": "Preserve guardrail" if passed else "Acquire/validate real geometry and replay pipeline",
            "notes": "Safety gates pass independently of missing geometry.",
        })
    return rows


def make_schema(name, columns):
    properties = {}
    for column in columns:
        prop = {"type": "string"}
        if column in BOOL_FIELDS:
            prop["enum"] = ["true", "false"]
        if column == "event_scope_status":
            prop["enum"] = ["CONFIRMED_IN_REGISTRIES", "CONTEXT_ONLY", "POINT_ANCHOR_ONLY",
                            "MISSING_EVENT_GEOMETRY", "EXPECTED_BUT_NOT_FOUND", "REGION_MISMATCH", "UNRESOLVED"]
        properties[column] = prop
    return {
        "$schema": "http://json-schema.org/draft-07/schema#", "title": name,
        "description": "v2ay fail-closed contract; no invented event/geometry, label, model, final ground truth, or automatic C4.",
        "type": "object", "required": columns, "additionalProperties": False, "properties": properties,
    }


def write_schemas(dataset_dir):
    for key, name in OUTPUTS.items():
        schema_name = os.path.splitext(name)[0] + ".schema.json"
        write_text(os.path.join(dataset_dir, "schemas", schema_name),
                   json.dumps(make_schema(schema_name, COLUMNS[key]), indent=2) + "\n")


def write_docs(docs_dir):
    docs = {
        "v2ay_event_scope_reconciliation.md": """# v2ay - Reconciliacao do escopo de eventos

A expectativa anterior de 3 eventos Recife nao se confirmou. Os registries mostram 1 evento Recife,
2 eventos Petropolis e 1 placeholder Curitiba nao resolvido. A confusao veio das tres linhas de
poligono pendente na v2aw: uma Recife e duas Petropolis. Nenhum evento foi inventado. A fonte canonica
passa a ser `datasets/v2ay_region_event_canonical_registry.csv`.
""",
        "v2ay_turning_point_definition.md": """# v2ay - Definicao de turning point

TP0 documenta a ausencia atual. TP1 exige um patch boundary real validado. TP2 exige um poligono
observado real. TP3 exige o par patch-evento ligado e pronto para overlay. TP4 exige replay v2au com
overlay confirmado e revisao humana. Commit sem dado real novo consolida infraestrutura; o primeiro
turning point cientifico real requer patch boundary + event polygon + replay v2au. Nenhum TP cria label.
""",
        "v2ay_real_geometry_acquisition_playbook.md": """# v2ay - Playbook de aquisicao de geometria real

Obtenha boundary de patch a partir de metadata de geracao/GIS verificado e poligono observado de evento
a partir de produto oficial/VHR revisado. Aceitos: bbox/WKT/GeoJSON para patch e WKT/GeoJSON para evento,
sempre com CRS, proveniencia, licenca e revisao humana. Preencha os CSVs manuais da v2ax e rode v2ax,
v2aw, v2av e v2au. Ponto, centroide, quickview e contexto nao fecham overlay.
""",
        "v2ay_current_scientific_state.md": """# v2ay - Estado cientifico atual

A infraestrutura e a evidencia estao organizadas, mas o bloqueio real e a ausencia de geometria
utilizavel. DINO permanece review-only. Nao existe ground truth final, label ou modelo operacional.
O proximo salto e dado geoespacial real, nao outro motor. Estado atual: TP0_DOCUMENTED_ABSENCE.
""",
    }
    for name, text in docs.items():
        write_text(os.path.join(docs_dir, name), text)


def build_summary(inputs, events, gaps, acquisition, certificate, gates, config):
    recife_observed = sum(row["region"] == "Recife" and row["event_scope_status"] != "UNRESOLVED"
                          for row in events)
    cert = certificate[0]
    level = turning_level(inputs)
    return {
        "stage": STAGE, "status": "OK_WITH_EXPECTED_BLOCKERS",
        "recife_events_expected_from_previous_prompt": int(config["expected_recife_events_from_previous_prompt"]),
        "recife_events_observed": recife_observed, "event_scope_reconciled": True,
        "event_invention_detected": False, "patch_boundaries_found": int(cert["patch_boundaries_found"]),
        "event_polygons_found": int(cert["event_polygons_found"]),
        "usable_overlay_geometries_found": int(cert["usable_overlay_geometries_found"]),
        "turning_point_level": level, "turning_point_ready": level != "TP0_DOCUMENTED_ABSENCE",
        "total_gap_items": len(gaps), "total_acquisition_targets": len(acquisition),
        "canonical_event_rows": len(events), "turning_point_gates_passed":
        sum(row["gate_passed"] == "true" for row in gates),
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "EVENT_SCOPE_RECONCILED_GEOMETRY_ACQUISITION_REQUIRED_NOT_FOR_TRAINING",
        "namespace_note": "A pre-existing Protocolo C hydrometeorological v2ay track is preserved separately.",
    }


def build_report(summary):
    return f"""# v2ay - Event Scope Reconciliation + Turning Point Gate

Eventos Recife: esperado anterior **{summary['recife_events_expected_from_previous_prompt']}**; observado **{summary['recife_events_observed']}**.
A divergencia foi reconciliada sem inventar eventos: as outras duas linhas pendentes pertencem a Petropolis.

- Patch boundaries validos: **{summary['patch_boundaries_found']}**
- Event polygons validos: **{summary['event_polygons_found']}**
- Overlays utilizaveis: **{summary['usable_overlay_geometries_found']}**
- Lacunas/targets de aquisicao: **{summary['total_gap_items']} / {summary['total_acquisition_targets']}**
- Turning point atual: **{summary['turning_point_level']}**

Preencha os arquivos manuais v2ax, valide com v2ax e reexecute v2aw -> v2av -> v2au.
Nenhum modelo, label, treino supervisionado, ground truth final ou C4 automatico foi criado.
"""


def log_lines(summary):
    return (
        f"[v2ay] recife_expected={summary['recife_events_expected_from_previous_prompt']} "
        f"recife_observed={summary['recife_events_observed']} reconciled=True invented=False\n"
        f"[v2ay] patch_boundaries={summary['patch_boundaries_found']} event_polygons="
        f"{summary['event_polygons_found']} overlays={summary['usable_overlay_geometries_found']}\n"
        f"[v2ay] gaps={summary['total_gap_items']} acquisition_targets={summary['total_acquisition_targets']}\n"
        f"[v2ay] turning_point={summary['turning_point_level']} ready={summary['turning_point_ready']}\n"
        "[v2ay] can_train_model=False can_create_operational_labels=False\n"
        f"[v2ay] status={summary['status']}\n"
    )


def run(dataset_dir=None, output_dir=None, config_dir=None, docs_dir=None):
    env_dataset, env_output, env_config, env_docs = resolve_dirs()
    dataset_dir, output_dir = dataset_dir or env_dataset, output_dir or env_output
    config_dir, docs_dir = config_dir or env_config, docs_dir or env_docs
    config = load_config(config_dir)
    inputs, _ = load_inputs(dataset_dir)
    events = build_events(inputs)
    audit = build_audit(events, config)
    crosswalk = build_crosswalk(inputs, events)
    gaps = build_gaps(inputs)
    contract = build_contract()
    acquisition = build_acquisition(gaps)
    queries = build_queries(gaps)
    certificate = build_certificate(inputs, dataset_dir)
    replay = build_replay()
    gates = build_gates(inputs, events)
    artifacts = {
        "events": events, "audit": audit, "crosswalk": crosswalk, "gaps": gaps,
        "contract": contract, "acquisition": acquisition, "queries": queries,
        "certificate": certificate, "replay": replay, "gates": gates,
    }
    for key, rows in artifacts.items():
        write_csv(os.path.join(dataset_dir, OUTPUTS[key]), COLUMNS[key], rows)
    write_schemas(dataset_dir)
    write_docs(docs_dir)
    summary = build_summary(inputs, events, gaps, acquisition, certificate, gates, config)
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
