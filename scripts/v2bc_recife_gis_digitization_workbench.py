#!/usr/bin/env python3
"""Build a context-only GIS package for manual Recife digitization."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "v2bc_recife_gis_digitization_workbench_config.json"
POLICY_COLUMNS = "policy_id geometry_source geometry_role allowed_use not_allowed_use can_feed_v2aw can_feed_v2av can_feed_v2au can_support_digitization can_be_ground_truth blocking_reason notes".split()
INVENTORY_COLUMNS = "risk_area_id source_file feature_index geometry_type crs area_m2 bbox_minx bbox_miny bbox_maxx bbox_maxy attributes_summary geometry_hash allowed_use blocking_reason notes".split()
AOI_COLUMNS = "aoi_id source_basis target_patch_id target_event_id target_package_id geometry_type crs bbox_minx bbox_miny bbox_maxx bbox_maxy area_m2 aoi_role can_feed_overlay blocking_reason notes".split()
TASK_COLUMNS = "task_id task_type target_id priority_package_id priority_patch_id priority_event_id input_layers output_expected output_file_to_fill required_format required_crs step_by_step validation_command blocking_reason notes".split()
UPDATE_COLUMNS = "update_id target_type target_id source_file_to_edit field_name current_value expected_value_or_instruction is_required blocking_reason notes".split()
MANIFEST_COLUMNS = "manifest_id artifact_type artifact_path artifact_role source_file can_be_used_for_digitization can_feed_pipeline blocking_reason notes".split()
QUICKVIEW_COLUMNS = "quickview_id source_file source_name target_event_id visual_support_role is_georeferenced can_extract_geometry can_feed_event_polygon blocking_reason recommended_manual_use notes".split()
TABLES = {
    "v2bc_contextual_geometry_use_policy.csv": POLICY_COLUMNS,
    "v2bc_recife_risk_area_context_inventory.csv": INVENTORY_COLUMNS,
    "v2bc_digitization_support_aoi_registry.csv": AOI_COLUMNS,
    "v2bc_manual_digitization_task_queue.csv": TASK_COLUMNS,
    "v2bc_fill_this_update_plan.csv": UPDATE_COLUMNS,
    "v2bc_gis_workbench_manifest.csv": MANIFEST_COLUMNS,
    "v2bc_quickview_visual_support_registry.csv": QUICKVIEW_COLUMNS,
}


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def stable_id(prefix, *parts):
    return f"{prefix}_{hashlib.sha256('|'.join(clean(x) for x in parts).encode()).hexdigest()[:12]}"


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def resolve_dirs(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    external = Path(external_dir or os.getenv("EXTERNAL_DIR") or dataset / "external_sources" / "recife_minimal_tp").resolve()
    return {"dataset": dataset, "output": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve(),
            "config": Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve(),
            "external": external, "raw": external / "raw", "workbench": dataset / "gis_workbench" / "recife_minimal_tp",
            "docs": Path(docs_dir or ROOT / "docs").resolve()}


def load_config(path):
    return json.loads((path / CONFIG_NAME).read_text(encoding="utf-8"))


def load_risk_features(raw):
    path = raw / "recife_defesa_civil_risk_areas_geojson.geojson"
    obj = json.loads(path.read_text(encoding="utf-8"))
    return path, obj.get("features", [])


def point(feature):
    coords = feature.get("geometry", {}).get("coordinates", [])
    return coords[:2] if len(coords) >= 2 else [None, None]


def normalize_context(features):
    normalized, rows, coords = [], [], []
    for index, feature in enumerate(features):
        x, y = point(feature)
        if x is not None:
            coords.append((x, y))
        props = dict(feature.get("properties", {}))
        props.update({"geometry_role": "context_risk_location_point", "allowed_use": "context_only",
                      "can_feed_pipeline": False, "can_be_ground_truth": False,
                      "source_note": "Municipal risk-location point; not an observed-event polygon."})
        geometry = feature.get("geometry")
        normalized.append({"type": "Feature", "id": stable_id("V2BC_RISK", index, x, y),
                           "properties": props, "geometry": geometry})
        summary = "; ".join(f"{k}={clean(v)}" for k, v in sorted(feature.get("properties", {}).items()))
        ghash = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
        rows.append({"risk_area_id": stable_id("V2BC_RISK", index, x, y),
            "source_file": "raw/recife_defesa_civil_risk_areas_geojson.geojson", "feature_index": str(index),
            "geometry_type": clean(geometry.get("type")) if geometry else "null", "crs": "EPSG:4326_INFERRED_FROM_LON_LAT_FIELDS",
            "area_m2": "0", "bbox_minx": x, "bbox_miny": y, "bbox_maxx": x, "bbox_maxy": y,
            "attributes_summary": summary, "geometry_hash": ghash, "allowed_use": "context_only",
            "blocking_reason": "POINT_CONTEXT_NOT_OBSERVED_EVENT_POLYGON",
            "notes": "Source calls these risk areas, but supplied geometry is a point location."})
    bbox = [min(x for x, _ in coords), min(y for _, y in coords), max(x for x, _ in coords), max(y for _, y in coords)]
    return normalized, rows, bbox


def feature_collection(features, name, role):
    return {"type": "FeatureCollection", "name": name,
            "properties": {"geometry_role": role, "can_feed_pipeline": False}, "features": features}


def polygon_from_bbox(bbox, properties):
    minx, miny, maxx, maxy = bbox
    return {"type": "Feature", "properties": properties,
            "geometry": {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy],
                                                               [minx, maxy], [minx, miny]]]}}


def policies():
    specs = [
        ("risk_points", "context_risk_location_point", "context_only", "event_polygon|patch_boundary|ground_truth", True, "POINT_CONTEXT_NOT_OBSERVED_EVENT_POLYGON"),
        ("Charter quickview", "visual_support_only", "manual visual reference", "verified_product|automatic_geometry", True, "QUICKVIEW_NOT_GEOREFERENCED"),
        ("SGB study", "temporal_contextual_support", "documentary support", "event_polygon|patch_boundary", True, "DOCUMENT_NOT_GEOMETRY"),
        ("CPRM points", "point_anchor_only", "spatial anchor/context", "event_polygon|overlay", True, "POINT_ANCHOR_NOT_OVERLAY"),
    ]
    return [{"policy_id": stable_id("V2BC_POLICY", source), "geometry_source": source, "geometry_role": role,
             "allowed_use": allowed, "not_allowed_use": denied, "can_feed_v2aw": "false", "can_feed_v2av": "false",
             "can_feed_v2au": "false", "can_support_digitization": b(support), "can_be_ground_truth": "false",
             "blocking_reason": blocker, "notes": "No automatic promotion."}
            for source, role, allowed, denied, support, blocker in specs]


def tasks(config):
    common = {"priority_package_id": config["priority_package_id"], "priority_patch_id": config["priority_patch_id"],
              "priority_event_id": config["priority_event_id"], "required_crs": "|".join(config["accepted_crs"]),
              "validation_command": "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate",
              "blocking_reason": "REAL_HUMAN_REVIEWED_GEOMETRY_REQUIRED", "notes": "Context cannot be promoted automatically."}
    specs = [
        ("digitize_patch_boundary", "REC_00019", "layers/recife_risk_areas_context.geojson|external Sentinel/GIS source", "real patch boundary", "datasets/external_sources/recife_minimal_tp/patch_boundary_REC_00019/FILL_THIS_PATCH_BOUNDARY.csv", "wkt|geojson_file|geojson_inline|bbox", "Verify patch identity; digitize real footprint; save; record CRS/provenance."),
        ("digitize_observed_event_polygon", "REC_2022_05_24_30", "layers/recife_risk_areas_context.geojson|maps/charter_758_recife_quickview.png|verified event product", "real observed-event polygon", "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/FILL_THIS_EVENT_POLYGON.csv", "wkt|geojson_file|geojson_inline", "Use verified observed evidence; digitize polygon; save; record CRS/provenance."),
        ("review_context_intersection", "PKG_34713b8aab96", "layers/recife_risk_areas_context.geojson", "context review note", "", "review_note", "Review intersections as context only; do not promote."),
        ("update_fill_this", "PKG_34713b8aab96", "manual digitized files", "completed FILL_THIS files", "FILL_THIS_PATCH_BOUNDARY.csv|FILL_THIS_EVENT_POLYGON.csv", "csv", "Fill geometry path, CRS, provenance and provided_unreviewed."),
        ("run_v2ba_v2az", "PKG_34713b8aab96", "completed FILL_THIS files", "validated readiness/replay audit", "", "command", "Run v2ba validate; v2az dry_run; only then controlled replay."),
    ]
    return [dict(common, task_id=stable_id("V2BC_TASK", kind), task_type=kind, target_id=target,
                 input_layers=inputs, output_expected=expected, output_file_to_fill=fill, required_format=fmt,
                 step_by_step=steps) for kind, target, inputs, expected, fill, fmt, steps in specs]


def update_plan(config):
    rows = []
    specs = [("patch_boundary", config["priority_patch_id"], "datasets/external_sources/recife_minimal_tp/patch_boundary_REC_00019/FILL_THIS_PATCH_BOUNDARY.csv"),
             ("event_polygon", config["priority_event_id"], "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/FILL_THIS_EVENT_POLYGON.csv")]
    fields = [("source_type", "wkt, geojson_file, geojson_inline or bbox"), ("geometry_path", "Path to real saved geometry"),
              ("crs", "Verified accepted CRS"), ("provenance_note", "Source and digitization provenance"),
              ("source_document", "Verified source document/reference"), ("review_status", "provided_unreviewed")]
    for target_type, target_id, source in specs:
        for field, instruction in fields:
            rows.append({"update_id": stable_id("V2BC_UPDATE", target_id, field), "target_type": target_type,
                "target_id": target_id, "source_file_to_edit": source, "field_name": field,
                "current_value": "missing" if field == "source_type" else "", "expected_value_or_instruction": instruction,
                "is_required": "true", "blocking_reason": "REAL_GEOMETRY_AND_PROVENANCE_REQUIRED",
                "notes": "Do not fill from context-only layers."})
    return rows


def draft_rows(config, target_type, target_id):
    return [{"target_type": target_type, "target_id": target_id, "package_id": config["priority_package_id"],
             "source_type": "missing", "geometry_value": "", "geometry_path": "", "crs": "UNKNOWN",
             "provenance_note": "", "source_document": "", "review_status": "not_started",
             "accepted_source_types": "wkt|geojson_file|geojson_inline|bbox" if target_type == "patch_boundary" else "wkt|geojson_file|geojson_inline",
             "accepted_crs": "|".join(config["accepted_crs"]), "blocking_reason": "REAL_GEOMETRY_REQUIRED",
             "instructions": "Digitize from verified source, save geometry, record CRS/provenance, then set provided_unreviewed."}]


def schema(columns):
    bools = {x for x in columns if x.startswith("can_") or x == "is_required"}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": list(columns),
            "additionalProperties": False, "properties": {x: {"type": "boolean" if x in bools else "string"} for x in columns}}


def run(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dirs = resolve_dirs(dataset_dir, output_dir, config_dir, external_dir, docs_dir)
    config = load_config(dirs["config"])
    wb, layers, qgis, maps, manual = dirs["workbench"], dirs["workbench"]/"layers", dirs["workbench"]/"qgis", dirs["workbench"]/"maps", dirs["workbench"]/"manual_digitization"
    for path in (layers, qgis, maps, manual):
        path.mkdir(parents=True, exist_ok=True)
    source_path, features = load_risk_features(dirs["raw"])
    normalized, inventory, bbox = normalize_context(features)
    write_text(layers/"recife_risk_areas_context.geojson", json.dumps(feature_collection(normalized, "recife_risk_areas_context", "context_only"), ensure_ascii=False, indent=2))
    write_csv(layers/"recife_risk_areas_context_index.csv", INVENTORY_COLUMNS, inventory)
    placeholders = [
        {"type": "Feature", "properties": {"target_type": "patch_boundary", "target_id": config["priority_patch_id"], "package_id": config["priority_package_id"], "geometry_role": "missing_patch_boundary", "can_feed_pipeline": False}, "geometry": None},
        {"type": "Feature", "properties": {"target_type": "event_polygon", "target_id": config["priority_event_id"], "package_id": config["priority_package_id"], "geometry_role": "missing_event_polygon", "can_feed_pipeline": False}, "geometry": None},
    ]
    write_text(layers/"recife_minimal_tp_targets.geojson", json.dumps(feature_collection(placeholders, "recife_minimal_tp_targets", "missing_targets"), indent=2))
    write_text(layers/"recife_missing_geometries_placeholders.geojson", json.dumps(feature_collection(placeholders, "recife_missing_geometries_placeholders", "missing_geometry"), indent=2))
    aoi_feature = polygon_from_bbox(bbox, {"geometry_role": "digitization_support_only", "can_feed_overlay": False, "source_basis": "envelope_of_400_context_points"})
    write_text(layers/"recife_digitization_aoi_context.geojson", json.dumps(feature_collection([aoi_feature], "recife_digitization_aoi_context", "digitization_support_only"), indent=2))
    aoi = [{"aoi_id": stable_id("V2BC_AOI", *bbox), "source_basis": "envelope_of_400_context_points",
            "target_patch_id": config["priority_patch_id"], "target_event_id": config["priority_event_id"],
            "target_package_id": config["priority_package_id"], "geometry_type": "Polygon", "crs": "EPSG:4326",
            "bbox_minx": bbox[0], "bbox_miny": bbox[1], "bbox_maxx": bbox[2], "bbox_maxy": bbox[3], "area_m2": "",
            "aoi_role": "digitization_support_only", "can_feed_overlay": "false",
            "blocking_reason": "CONTEXT_ENVELOPE_NOT_OPERATIONAL_GEOMETRY", "notes": "AOI is not event extent or patch boundary."}]
    quickview_src = dirs["raw"]/"charter_758_recife_quickview.png"
    shutil.copy2(quickview_src, maps/"charter_758_recife_quickview.png")
    quickview = [{"quickview_id": stable_id("V2BC_QV", quickview_src.name), "source_file": "raw/charter_758_recife_quickview.png",
        "source_name": "Charter 758 Recife quickview", "target_event_id": config["priority_event_id"],
        "visual_support_role": "visual_support_only", "is_georeferenced": "false", "can_extract_geometry": "false",
        "can_feed_event_polygon": "false", "blocking_reason": "QUICKVIEW_NOT_GEOREFERENCED_OR_VERIFIED_VECTOR",
        "recommended_manual_use": "Visual orientation only alongside verified/georeferenced source.", "notes": "Never trace as final geometry without verified georeferencing."}]
    draft_cols = "target_type target_id package_id source_type geometry_value geometry_path crs provenance_note source_document review_status accepted_source_types accepted_crs blocking_reason instructions".split()
    write_csv(manual/"FILL_THIS_PATCH_BOUNDARY_REC_00019.draft.csv", draft_cols, draft_rows(config, "patch_boundary", config["priority_patch_id"]))
    write_csv(manual/"FILL_THIS_EVENT_POLYGON_REC_2022_05_24_30.draft.csv", draft_cols, draft_rows(config, "event_polygon", config["priority_event_id"]))
    qgs_xml = """<?xml version="1.0" encoding="UTF-8"?><qgis projectname="recife_minimal_tp_digitization" version="3.34"><title>Context only - manual digitization required</title><projectlayers/></qgis>"""
    write_text(qgis/"recife_minimal_tp_digitization.qgs", qgs_xml)
    write_text(qgis/"README_QGIS.md", "# QGIS\n\nOpen the `.qgs`, then add the four GeoJSON layers from `../layers/` and the quickview from `../maps/`. Create new polygon layers for `REC_00019` and `REC_2022_05_24_30`; do not trace context as ground truth.\n")
    write_text(wb/"README.md", "# Recife minimal TP GIS workbench\n\nThis package contains 400 municipal risk-location points, a context AOI, null-geometry target placeholders, a non-georeferenced Charter quickview, QGIS instructions and manual digitization drafts. All current layers are context/digitization support only. Real patch and observed-event polygons remain missing.\n")
    policies_rows, task_rows, update_rows = policies(), tasks(config), update_plan(config)
    artifacts = [
        ("layer", "layers/recife_risk_areas_context.geojson", "context_only", "raw/recife_defesa_civil_risk_areas_geojson.geojson", True, False, "RISK_POINTS_NOT_EVENT_POLYGON"),
        ("layer", "layers/recife_minimal_tp_targets.geojson", "missing_targets", "", True, False, "NULL_GEOMETRY_PLACEHOLDERS"),
        ("layer", "layers/recife_missing_geometries_placeholders.geojson", "missing_geometry", "", True, False, "NULL_GEOMETRY_PLACEHOLDERS"),
        ("layer", "layers/recife_digitization_aoi_context.geojson", "digitization_support_only", "risk point envelope", True, False, "AOI_NOT_OPERATIONAL_GEOMETRY"),
        ("map", "maps/charter_758_recife_quickview.png", "visual_support_only", "raw/charter_758_recife_quickview.png", True, False, "QUICKVIEW_NOT_GEOREFERENCED"),
        ("qgis_project", "qgis/recife_minimal_tp_digitization.qgs", "digitization_workbench", "", True, False, "MANUAL_DIGITIZATION_REQUIRED"),
        ("draft", "manual_digitization/FILL_THIS_PATCH_BOUNDARY_REC_00019.draft.csv", "manual_fill_draft", "", True, False, "REAL_GEOMETRY_REQUIRED"),
        ("draft", "manual_digitization/FILL_THIS_EVENT_POLYGON_REC_2022_05_24_30.draft.csv", "manual_fill_draft", "", True, False, "REAL_GEOMETRY_REQUIRED"),
    ]
    manifest = [{"manifest_id": stable_id("V2BC_MANIFEST", path), "artifact_type": kind, "artifact_path": path,
        "artifact_role": role, "source_file": source, "can_be_used_for_digitization": b(digitize), "can_feed_pipeline": b(feed),
        "blocking_reason": blocker, "notes": "Workbench artifact; no automatic promotion."}
        for kind, path, role, source, digitize, feed, blocker in artifacts]
    rows_by_name = {"v2bc_contextual_geometry_use_policy.csv": policies_rows,
        "v2bc_recife_risk_area_context_inventory.csv": inventory, "v2bc_digitization_support_aoi_registry.csv": aoi,
        "v2bc_manual_digitization_task_queue.csv": task_rows, "v2bc_fill_this_update_plan.csv": update_rows,
        "v2bc_gis_workbench_manifest.csv": manifest, "v2bc_quickview_visual_support_registry.csv": quickview}
    for name, columns in TABLES.items():
        write_csv(dirs["dataset"]/name, columns, rows_by_name[name])
        write_text(dirs["dataset"]/"schemas"/f"{Path(name).stem}.schema.json", json.dumps(schema(columns), indent=2))
    counts = Counter(clean(f.get("properties", {}).get("Descrição")) for f in features)
    docs = {
        "v2bc_recife_gis_digitization_workbench.md": f"# v2bc Recife GIS digitization workbench\n\nv2bc packages the four v2bb downloads for manual GIS work. The municipal source contains {len(features)} risk-location points, not area polygons and not the observed event. The package includes context layers, null placeholders, AOI support, quickview and drafts. Real patch/event geometry remains missing.\n",
        "v2bc_contextual_geometry_policy.md": "# Contextual geometry policy\n\nRisk-location point is not event polygon. A non-georeferenced quickview is not a verified vector product. A point anchor is not a polygon. Temporal/documentary support is not ground truth. No context layer is automatically promoted.\n",
        "v2bc_qgis_digitization_steps.md": "# QGIS digitization steps\n\n1. Open the QGIS project.\n2. Load `recife_risk_areas_context.geojson`.\n3. Load the Charter PNG for visual reference only.\n4. Create polygon layer `patch_boundary_REC_00019` from a verified patch source.\n5. Create polygon layer `event_polygon_REC_2022_05_24_30` from verified observed evidence.\n6. Save as GeoJSON with explicit CRS.\n7. Fill the original FILL_THIS files.\n8. Run v2ba validation.\n",
        "v2bc_after_manual_digitization_replay.md": "# After manual digitization\n\nRun v2ba validate, then v2az dry_run and controlled replay. Verify TP1/TP2/TP3/TP4 and stop at `C4_CANDIDATE_REQUIRES_HUMAN_REVIEW` with human review. No label or training is authorized.\n",
    }
    for name, text in docs.items():
        write_text(dirs["docs"]/name, text)
    summary = {"stage": "v2bc_recife_gis_digitization_workbench", "status": "OK_WITH_EXPECTED_BLOCKERS",
        "priority_package_id": config["priority_package_id"], "priority_patch_id": config["priority_patch_id"],
        "priority_event_id": config["priority_event_id"], "risk_context_features": len(features), "quickview_files": 1,
        "context_layers_written": 5, "gis_workbench_artifacts": len(manifest), "manual_digitization_tasks": len(task_rows),
        "operational_patch_boundaries": 0, "operational_event_polygons": 0, "ready_feeds_created": 0,
        "turning_point_level": "TP0_CONTEXTUAL_GIS_WORKBENCH_READY", "turning_point_ready": False,
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "CONTEXTUAL_GIS_WORKBENCH_READY_FOR_MANUAL_DIGITIZATION_NOT_FOR_TRAINING"}
    write_text(dirs["output"]/"execution_reports"/"v2bc_recife_gis_digitization_workbench_summary.json", json.dumps(summary, indent=2))
    write_text(dirs["output"]/"execution_reports"/"v2bc_recife_gis_digitization_workbench_report.md",
        f"# v2bc report\n\nProcessed {len(features)} municipal risk-location points as context only. Attribute values for Descrição: {dict(counts)}. Created {len(manifest)} GIS artifacts and {len(task_rows)} manual tasks. No operational geometry or feed was created.\n")
    write_text(dirs["output"]/"logs_summary"/"v2bc_recife_gis_digitization_workbench.txt",
        "\n".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in summary.items()))
    print(f"[v2bc] risk_context_features={len(features)} context_layers=5 artifacts={len(manifest)} tasks={len(task_rows)}")
    print("[v2bc] operational_patch=0 operational_event=0 feeds=0")
    print("[v2bc] turning_point=TP0_CONTEXTUAL_GIS_WORKBENCH_READY")
    return 0, summary


if __name__ == "__main__":
    raise SystemExit(run()[0])
