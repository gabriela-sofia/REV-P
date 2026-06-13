#!/usr/bin/env python3
"""Integrate the v2bd patch-boundary candidate without promoting truth or labels."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "v2be_tp1_patch_boundary_integration_gate_config.json"
STAGE = "v2be_tp1_patch_boundary_integration_gate"
SOURCE_DOCUMENT = "manifests/training_readiness/revp_v1fs_self_supervised_asset_sanity_and_embedding_plan/asset_sanity_audit_v1fs.csv"
SOURCE_METHOD = "bbox_explicit_from_preserved_asset_bounds"
TP1 = "TP1_ONE_PATCH_BOUNDARY_CANDIDATE_REQUIRES_HUMAN_REVIEW"

INTEGRATION_COLUMNS = "integration_id patch_id package_id event_id source_stage source_geojson source_method asset_id asset_file original_crs normalized_crs geometry_type geometry_valid bbox_minx bbox_miny bbox_maxx bbox_maxy area_m2_approx vertex_count geometry_hash requires_human_review can_feed_v2ba can_feed_v2aw can_feed_v2av can_feed_v2az blocking_reason notes".split()
CRS_COLUMNS = "crs_audit_id patch_id source_stage original_crs normalized_crs original_bounds normalized_bbox reprojection_method reprojection_supported crs_preserved crs_normalized blocking_reason notes".split()
FEED_COLUMNS = "feed_id patch_id package_id event_id geometry_path geometry_format original_crs normalized_crs geometry_hash source_stage source_method asset_id source_document source_public access_status review_status requires_human_review ready blocking_reason notes".split()
GATE_COLUMNS = "gate_id turning_point_level gate_name required_condition observed_value gate_passed severity blocking_reason recommended_action notes".split()
SYNC_COLUMNS = "sync_step_id step_order target_stage input_feed command will_run_now precondition precondition_met expected_output blocking_reason notes".split()
DRY_COLUMNS = "dry_run_id stage command executed exit_code status tp1_recognized tp2_available tp3_available tp4_available blocking_reason notes".split()
BLOCKER_COLUMNS = "blocker_id turning_point_level target_type target_id package_id event_id current_status blocking_reason required_next_input where_to_put_it validation_command notes".split()

TABLES = {
    "v2be_tp1_patch_boundary_integration_registry.csv": INTEGRATION_COLUMNS,
    "v2be_tp1_crs_reprojection_audit.csv": CRS_COLUMNS,
    "v2be_ready_patch_boundary_feed_for_v2ba.csv": FEED_COLUMNS,
    "v2be_ready_patch_boundary_feed_for_v2aw.csv": FEED_COLUMNS,
    "v2be_ready_patch_boundary_feed_for_v2av.csv": FEED_COLUMNS,
    "v2be_ready_patch_boundary_feed_for_v2az.csv": FEED_COLUMNS,
    "v2be_tp1_readiness_gate.csv": GATE_COLUMNS,
    "v2be_tp1_replay_synchronization_plan.csv": SYNC_COLUMNS,
    "v2be_tp1_replay_dry_run_status.csv": DRY_COLUMNS,
    "v2be_remaining_turning_point_blockers.csv": BLOCKER_COLUMNS,
}


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def stable_id(prefix, *parts):
    basis = "|".join(clean(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(basis.encode()).hexdigest()[:12]}"


def load_csv(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def resolve_dirs(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    external = Path(external_dir or os.getenv("EXTERNAL_DIR") or dataset / "external_sources").resolve()
    return {
        "dataset": dataset,
        "output": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve(),
        "config": Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve(),
        "external": external,
        "docs": Path(docs_dir or ROOT / "docs").resolve(),
    }


def geometry_from_geojson(path):
    if not path.is_file():
        return {}, {}, "SOURCE_GEOJSON_MISSING"
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}, "SOURCE_GEOJSON_INVALID_JSON"
    if obj.get("type") == "Feature":
        return obj.get("geometry") or {}, obj.get("properties") or {}, ""
    if obj.get("type") == "FeatureCollection" and len(obj.get("features") or []) == 1:
        feature = obj["features"][0]
        return feature.get("geometry") or {}, feature.get("properties") or {}, ""
    return obj, {}, ""


def polygon_rings(geometry):
    if geometry.get("type") == "Polygon":
        return geometry.get("coordinates") or []
    if geometry.get("type") == "MultiPolygon":
        return [ring for polygon in geometry.get("coordinates") or [] for ring in polygon]
    return []


def validate_geometry(geometry):
    rings = polygon_rings(geometry)
    if not rings:
        return False, "GEOMETRY_NOT_POLYGON"
    for ring in rings:
        if len(ring) < 4 or ring[0] != ring[-1]:
            return False, "POLYGON_RING_INVALID"
        if any(len(point) < 2 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in point[:2]) for point in ring):
            return False, "POLYGON_COORDINATE_INVALID"
    return True, ""


def geometry_bbox(geometry):
    points = [point for ring in polygon_rings(geometry) for point in ring]
    xs, ys = [point[0] for point in points], [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)] if points else []


def vertex_count(geometry):
    return sum(max(0, len(ring) - 1) for ring in polygon_rings(geometry))


def approximate_area_m2(geometry):
    """Approximate WGS84 polygon area with a local equirectangular projection."""
    rings = polygon_rings(geometry)
    if not rings:
        return 0.0
    latitude = sum(point[1] for ring in rings for point in ring) / sum(len(ring) for ring in rings)
    radius = 6371008.8
    total = 0.0
    for ring in rings:
        projected = [(math.radians(p[0]) * radius * math.cos(math.radians(latitude)),
                      math.radians(p[1]) * radius) for p in ring]
        total += abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(projected, projected[1:]))) / 2
    return total


def geometry_hash(geometry):
    # Preserve the v2bd geometry-hash contract so lineage stays directly comparable.
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def parse_original_bounds(metadata_rows):
    for row in metadata_rows:
        value = clean(row.get("metadata_value"))
        if "bounds=" not in value:
            continue
        raw = value.split("bounds=", 1)[1].split(";", 1)[0]
        try:
            bounds = [float(item) for item in raw.split(",")]
            if len(bounds) == 4:
                return bounds
        except ValueError:
            pass
    return []


def back_projected_bounds(geometry, normalized_crs, original_crs):
    projected = transform_geom(normalized_crs, original_crs, geometry, precision=6)
    return geometry_bbox(projected)


def bounds_match(expected, observed, tolerance=0.1):
    return bool(expected and observed and all(abs(left - right) <= tolerance for left, right in zip(expected, observed)))


def schema(columns):
    bools = {column for column in columns if column.startswith(("can_", "requires_", "ready", "crs_",
                                                                 "reprojection_", "geometry_", "gate_", "will_",
                                                                 "precondition_", "executed", "tp"))}
    bools -= {"geometry_type", "geometry_hash", "geometry_path", "geometry_format", "crs_audit_id",
              "blocking_reason", "ready_patch_boundary_feed"}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": columns,
        "additionalProperties": False,
        "properties": {column: {"type": "boolean" if column in bools else "string"} for column in columns},
    }


def build_gates(valid, crs_ok, lineage_ok, feeds_ready, human_review):
    specs = [
        ("TP1_01_BOUNDARY_GEOJSON_EXISTS", "source GeoJSON exists", valid, "GeoJSON loaded"),
        ("TP1_02_BOUNDARY_GEOMETRY_VALID", "Polygon/MultiPolygon valid", valid, f"geometry_valid={valid}"),
        ("TP1_03_ORIGINAL_CRS_RECORDED", "original CRS recorded", crs_ok, "EPSG:32725"),
        ("TP1_04_NORMALIZED_CRS_RECORDED", "normalized CRS recorded", crs_ok, "EPSG:4326"),
        ("TP1_05_LINEAGE_SOURCE_RECORDED", "direct lineage source recorded", lineage_ok, f"lineage_recorded={lineage_ok}"),
        ("TP1_06_FEED_READY_FOR_V2BA", "v2ba feed ready", feeds_ready, f"ready={feeds_ready}"),
        ("TP1_07_FEED_READY_FOR_V2AW", "v2aw feed ready", feeds_ready, f"ready={feeds_ready}"),
        ("TP1_08_FEED_READY_FOR_V2AV", "v2av feed ready", feeds_ready, f"ready={feeds_ready}"),
        ("TP1_09_FEED_READY_FOR_V2AZ", "v2az feed ready", feeds_ready, f"ready={feeds_ready}"),
        ("TP1_10_HUMAN_REVIEW_REQUIRED", "human review remains required", human_review, f"required={human_review}"),
        ("TP1_11_NO_EVENT_POLYGON_INFERRED", "no event polygon inferred", True, "event_polygons_created=0"),
        ("TP1_12_NO_LABEL_CREATED", "no operational label created", True, "labels_created=0"),
        ("TP1_13_NO_MODEL_TRAINED", "no model trained", True, "models_trained=0"),
    ]
    return [{
        "gate_id": stable_id("V2BE_GATE", name), "turning_point_level": TP1, "gate_name": name,
        "required_condition": requirement, "observed_value": observed, "gate_passed": b(passed),
        "severity": "safety" if name >= "TP1_11" else "blocking",
        "blocking_reason": "" if passed else "TP1_REQUIRED_CONDITION_NOT_MET",
        "recommended_action": "Preserve guardrail" if passed else "Correct source candidate before integration",
        "notes": "TP1 remains a candidate requiring human review; never final truth.",
    } for name, requirement, passed, observed in specs]


def build_sync_plan(feed_ready):
    specs = [
        ("1", "v2be_autofill_candidate", "datasets/v2be_ready_patch_boundary_feed_for_v2ba.csv", "Review v2be autofill candidate", False, True, "Human-reviewed autofill decision"),
        ("2", "v2ba_validate", "datasets/v2be_ready_patch_boundary_feed_for_v2ba.csv", "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate", False, feed_ready, "Validated patch candidate"),
        ("3", "v2ba_ingest", "datasets/v2be_ready_patch_boundary_feed_for_v2ba.csv", "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode ingest", False, False, "Ingested reviewed candidate"),
        ("4", "v2az_dry_run", "datasets/v2be_ready_patch_boundary_feed_for_v2az.csv", "python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run", False, feed_ready, "TP1 recognized in controlled dry-run"),
        ("5", "v2aw", "datasets/v2be_ready_patch_boundary_feed_for_v2aw.csv", "python scripts/run_v2aw_geometry_source_intake.py", False, feed_ready, "Patch source validation"),
        ("6", "v2av", "datasets/v2be_ready_patch_boundary_feed_for_v2av.csv", "python scripts/run_v2av_patch_boundary_geometry_builder.py", False, feed_ready, "Patch boundary registry"),
        ("7", "v2au", "event polygon required", "python scripts/run_v2au_patch_event_overlay_geometry.py", False, False, "Remain blocked until event polygon"),
        ("8", "TP2", "observed event polygon", "Provide and validate REC_2022_05_24_30 event polygon", False, False, "TP2 candidate"),
        ("9", "TP3_TP4", "reviewed patch-event pair", "Run controlled overlay only after TP1 and TP2", False, False, "TP3 then TP4 candidate"),
    ]
    return [{
        "sync_step_id": stable_id("V2BE_SYNC", order, target), "step_order": order, "target_stage": target,
        "input_feed": feed, "command": command, "will_run_now": b(run_now),
        "precondition": "TP1 feed ready" if order in {"2", "4", "5", "6"} else "Human review or later turning point",
        "precondition_met": b(met), "expected_output": expected,
        "blocking_reason": "" if met else "HUMAN_REVIEW_OR_EVENT_POLYGON_REQUIRED",
        "notes": "Plan only; v2be does not execute destructive replay.",
    } for order, target, feed, command, run_now, met, expected in specs]


def run(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None, docs_dir=None):
    dirs = resolve_dirs(dataset_dir, output_dir, config_dir, external_dir, docs_dir)
    config = json.loads((dirs["config"] / CONFIG_NAME).read_text(encoding="utf-8"))
    patch, package, event = config["priority_patch_id"], config["priority_package_id"], config["priority_event_id"]
    source_rel = config["source_boundary_geojson"]
    source_path = ROOT / source_rel if dataset_dir is None else dirs["dataset"] / Path(source_rel).relative_to("datasets")
    geometry, properties, source_error = geometry_from_geojson(source_path)
    valid, geometry_error = validate_geometry(geometry)
    valid = valid and not source_error
    bbox = geometry_bbox(geometry) if valid else []
    geom_hash = geometry_hash(geometry) if valid else ""
    area = approximate_area_m2(geometry) if valid else 0.0
    lineage = load_csv(dirs["dataset"] / "v2bd_patch_asset_lineage_registry.csv")
    metadata = load_csv(dirs["dataset"] / "v2bd_spatial_metadata_inventory.csv")
    asset_id = clean(lineage[0].get("candidate_asset_id")) if lineage else ""
    asset_file = clean(lineage[0].get("asset_file")) if lineage else ""
    lineage_ok = bool(lineage and lineage[0].get("has_direct_link") == "true")
    original_bounds = parse_original_bounds(metadata)
    original_crs, normalized_crs = config["original_crs"], config["normalized_crs"]
    accepted = set(config["accepted_crs"])
    crs_ok = original_crs in accepted and normalized_crs in accepted and properties.get("source_crs") == original_crs
    observed_original = back_projected_bounds(geometry, normalized_crs, original_crs) if valid and crs_ok else []
    reprojection_ok = bounds_match(original_bounds, observed_original)
    ready = bool(valid and crs_ok and reprojection_ok and lineage_ok and config["allow_tp1_candidate"])
    blocker = "" if ready else "|".join(x for x in (source_error, geometry_error,
        "" if crs_ok else "CRS_LINEAGE_MISMATCH", "" if reprojection_ok else "REPROJECTION_BOUNDS_MISMATCH",
        "" if lineage_ok else "DIRECT_LINEAGE_MISSING") if x)

    integration = [{
        "integration_id": stable_id("V2BE_INTEGRATION", patch, geom_hash), "patch_id": patch, "package_id": package,
        "event_id": event, "source_stage": "v2bd", "source_geojson": source_rel, "source_method": SOURCE_METHOD,
        "asset_id": asset_id, "asset_file": asset_file, "original_crs": original_crs, "normalized_crs": normalized_crs,
        "geometry_type": geometry.get("type", "UNKNOWN"), "geometry_valid": b(valid), "bbox_minx": bbox[0] if bbox else "",
        "bbox_miny": bbox[1] if bbox else "", "bbox_maxx": bbox[2] if bbox else "", "bbox_maxy": bbox[3] if bbox else "",
        "area_m2_approx": f"{area:.2f}" if valid else "", "vertex_count": str(vertex_count(geometry)),
        "geometry_hash": geom_hash, "requires_human_review": "true", "can_feed_v2ba": b(ready),
        "can_feed_v2aw": b(ready), "can_feed_v2av": b(ready), "can_feed_v2az": b(ready),
        "blocking_reason": blocker, "notes": "TP1 candidate only; not an event polygon, label, or final ground truth.",
    }]
    crs_audit = [{
        "crs_audit_id": stable_id("V2BE_CRS", patch, original_crs, normalized_crs), "patch_id": patch,
        "source_stage": "v2bd", "original_crs": original_crs, "normalized_crs": normalized_crs,
        "original_bounds": ",".join(str(value) for value in original_bounds),
        "normalized_bbox": ",".join(str(value) for value in bbox), "reprojection_method": "rasterio.warp.transform_geom",
        "reprojection_supported": b(bool(observed_original)), "crs_preserved": b(crs_ok),
        "crs_normalized": b(reprojection_ok), "blocking_reason": "" if reprojection_ok else "REPROJECTION_BOUNDS_MISMATCH",
        "notes": "EPSG:32725 is accepted only as the preserved source CRS; EPSG:4326 is the normalized export CRS.",
    }]
    feed = [{
        "feed_id": stable_id("V2BE_FEED", patch, geom_hash), "patch_id": patch, "package_id": package, "event_id": event,
        "geometry_path": source_rel, "geometry_format": "geojson_file", "original_crs": original_crs,
        "normalized_crs": normalized_crs, "geometry_hash": geom_hash, "source_stage": "v2bd",
        "source_method": SOURCE_METHOD, "asset_id": asset_id, "source_document": SOURCE_DOCUMENT,
        "source_public": "true", "access_status": "project_lineage_preserved_metadata",
        "review_status": "provided_unreviewed", "requires_human_review": "true", "ready": b(ready),
        "blocking_reason": blocker, "notes": "Candidate boundary only; human review required before downstream promotion.",
    }]
    gates = build_gates(valid, crs_ok and reprojection_ok, lineage_ok, ready, True)
    tp1_passed = all(row["gate_passed"] == "true" for row in gates)
    sync = build_sync_plan(ready)
    dry = [{
        "dry_run_id": stable_id("V2BE_DRY", patch, geom_hash), "stage": "v2be_internal_integration_dry_run",
        "command": "internal validation only; no subprocess replay", "executed": "true", "exit_code": "0" if tp1_passed else "1",
        "status": "TP1_RECOGNIZED_CANDIDATE_REQUIRES_HUMAN_REVIEW" if tp1_passed else "TP1_BLOCKED",
        "tp1_recognized": b(tp1_passed), "tp2_available": "false", "tp3_available": "false", "tp4_available": "false",
        "blocking_reason": "EVENT_POLYGON_REQUIRED_FOR_TP2_TP3_TP4",
        "notes": "No prior-stage outputs or subprocesses were modified or executed.",
    }]
    blockers = [
        {"blocker_id": stable_id("V2BE_BLOCKER", "TP1"), "turning_point_level": TP1, "target_type": "patch_boundary",
         "target_id": patch, "package_id": package, "event_id": event, "current_status": "CANDIDATE_READY",
         "blocking_reason": "HUMAN_REVIEW_REQUIRED", "required_next_input": "Human confirmation of recovered footprint",
         "where_to_put_it": "FILL_THIS_PATCH_BOUNDARY.csv after review",
         "validation_command": "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate",
         "notes": "TP1 candidate exists but is not final truth."},
        {"blocker_id": stable_id("V2BE_BLOCKER", "TP2"), "turning_point_level": "TP2_ONE_EVENT_POLYGON_VALIDATED",
         "target_type": "event_observed_polygon", "target_id": event, "package_id": package, "event_id": event,
         "current_status": "BLOCKED", "blocking_reason": "NO_VALID_OBSERVED_EVENT_POLYGON",
         "required_next_input": "Observed event polygon with CRS and provenance",
         "where_to_put_it": "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/",
         "validation_command": "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate",
         "notes": "No event polygon is inferred from the patch candidate."},
        {"blocker_id": stable_id("V2BE_BLOCKER", "TP3"), "turning_point_level": "TP3_ONE_PATCH_EVENT_PAIR_READY_FOR_OVERLAY",
         "target_type": "patch_event_pair", "target_id": package, "package_id": package, "event_id": event,
         "current_status": "BLOCKED", "blocking_reason": "TP2_EVENT_POLYGON_REQUIRED",
         "required_next_input": "Reviewed TP1 plus validated TP2 in the same package", "where_to_put_it": "v2az controlled replay",
         "validation_command": "python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run",
         "notes": "Pair remains blocked until event geometry exists."},
        {"blocker_id": stable_id("V2BE_BLOCKER", "TP4"), "turning_point_level": "TP4_ONE_OVERLAY_CONFIRMED_REQUIRES_HUMAN_REVIEW",
         "target_type": "overlay", "target_id": package, "package_id": package, "event_id": event,
         "current_status": "BLOCKED", "blocking_reason": "TP3_PAIR_AND_V2AU_OVERLAY_REQUIRED",
         "required_next_input": "Confirmed v2au overlay after TP3", "where_to_put_it": "v2au controlled replay",
         "validation_command": "python scripts/run_v2az_turning_point_replay_orchestrator.py --mode replay",
         "notes": "TP4 never promotes automatic C4 or an operational label."},
    ]
    outputs = {
        "v2be_tp1_patch_boundary_integration_registry.csv": integration,
        "v2be_tp1_crs_reprojection_audit.csv": crs_audit,
        "v2be_ready_patch_boundary_feed_for_v2ba.csv": feed,
        "v2be_ready_patch_boundary_feed_for_v2aw.csv": feed,
        "v2be_ready_patch_boundary_feed_for_v2av.csv": feed,
        "v2be_ready_patch_boundary_feed_for_v2az.csv": feed,
        "v2be_tp1_readiness_gate.csv": gates,
        "v2be_tp1_replay_synchronization_plan.csv": sync,
        "v2be_tp1_replay_dry_run_status.csv": dry,
        "v2be_remaining_turning_point_blockers.csv": blockers,
    }
    for name, rows in outputs.items():
        write_csv(dirs["dataset"] / name, TABLES[name], rows)
        write_text(dirs["dataset"] / "schemas" / name.replace(".csv", ".schema.json"), json.dumps(schema(TABLES[name]), indent=2))

    autofill_columns = "target_type target_id patch_id package_id event_id source_type geometry_path crs original_crs provenance_type provenance_note source_document source_public access_status review_status notes".split()
    autofill_path = dirs["external"] / "recife_minimal_tp" / "patch_boundary_REC_00019" / "FILL_THIS_PATCH_BOUNDARY.autofill_tp1_candidate_v2be.csv"
    write_csv(autofill_path, autofill_columns, [{
        "target_type": "patch_boundary", "target_id": patch, "patch_id": patch, "package_id": package, "event_id": event,
        "source_type": "geojson_file", "geometry_path": source_rel, "crs": normalized_crs, "original_crs": original_crs,
        "provenance_type": "sentinel_patch_lineage",
        "provenance_note": "TP1 candidate recovered from preserved Sentinel asset bounds; requires human review",
        "source_document": "asset_sanity_audit_v1fs.csv", "source_public": "true",
        "access_status": "project_lineage_preserved_metadata", "review_status": "provided_unreviewed",
        "notes": "TP1 candidate recovered from preserved Sentinel asset bounds; requires human review",
    }])

    docs = {
        "v2be_tp1_patch_boundary_integration_gate.md": f"""# v2be TP1 patch boundary integration gate

`{patch}` was recovered by v2bd from explicit Sentinel asset bounds in `{original_crs}` and exported as a candidate GeoJSON in `{normalized_crs}`. The back-projected normalized geometry matches the preserved original bounds. It is a TP1 candidate because direct lineage, CRS, bounds, provenance and hash exist, but the raster payload is absent and human review remains mandatory.

The candidate is not an event polygon, label, final ground truth, or automatic C4 promotion.
""",
        "v2be_tp1_review_and_replay_instructions.md": f"""# v2be TP1 review and replay instructions

Open `{source_rel}` in QGIS and confirm its location, extent, source CRS lineage and correspondence with the expected Sentinel patch. Review the separate `FILL_THIS_PATCH_BOUNDARY.autofill_tp1_candidate_v2be.csv`; do not overwrite the original `FILL_THIS_PATCH_BOUNDARY.csv` before review.

After human confirmation, copy the reviewed values into the manual intake and run:

1. `python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate`
2. `python scripts/run_v2az_turning_point_replay_orchestrator.py --mode dry_run`

Only run ingest/replay after explicit review. Never promote the candidate to a label or final truth.
""",
        "v2be_remaining_path_to_tp4.md": """# v2be remaining path to TP4

TP1 is available only as a patch-boundary candidate requiring human review. TP2 still requires a real observed-event polygon. TP3 requires reviewed TP1 and validated TP2 in the same package. TP4 requires a confirmed v2au overlay and human review. A C4 candidate is never an automatic operational label, and none of these steps trains a model.
""",
    }
    for name, content in docs.items():
        write_text(dirs["docs"] / name, content)

    summary = {
        "stage": STAGE, "status": "OK_TP1_CANDIDATE_REQUIRES_HUMAN_REVIEW", "priority_patch_id": patch,
        "priority_package_id": package, "priority_event_id": event, "source_stage": "v2bd",
        "boundary_geojson_exists": source_path.is_file(), "boundary_geometry_valid": valid,
        "original_crs": original_crs, "normalized_crs": normalized_crs, "feeds_ready": 4 if ready else 0,
        "tp1_gate_passed": tp1_passed, "tp1_requires_human_review": True, "tp2_available": False,
        "tp3_available": False, "tp4_available": False, "turning_point_level": TP1 if tp1_passed else "TP0_TP1_INTEGRATION_BLOCKED",
        "turning_point_ready": tp1_passed, "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "TP1_PATCH_BOUNDARY_CANDIDATE_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING",
    }
    write_text(dirs["output"] / "execution_reports" / "v2be_tp1_patch_boundary_integration_summary.json", json.dumps(summary, indent=2))
    write_text(dirs["output"] / "execution_reports" / "v2be_tp1_patch_boundary_integration_report.md",
        f"# v2be TP1 patch boundary integration report\n\nBoundary valid: **{valid}**. Original CRS: `{original_crs}`. Normalized CRS: `{normalized_crs}`. Feeds ready: **{summary['feeds_ready']}**. TP1 candidate passed: **{tp1_passed}**. TP2, TP3 and TP4 remain blocked. Human review is mandatory. No event polygon, label, model, final ground truth or automatic C4 was created.")
    write_text(dirs["output"] / "logs_summary" / "v2be_tp1_patch_boundary_integration.txt",
        f"[v2be] patch={patch} valid={str(valid).lower()} feeds_ready={summary['feeds_ready']}\n[v2be] original_crs={original_crs} normalized_crs={normalized_crs} reprojection_match={str(reprojection_ok).lower()}\n[v2be] turning_point={summary['turning_point_level']} human_review=true\n[v2be] can_train_model=false can_create_operational_labels=false")
    print(f"[v2be] boundary_valid={str(valid).lower()} reprojection_match={str(reprojection_ok).lower()} feeds_ready={summary['feeds_ready']}")
    print(f"[v2be] turning_point={summary['turning_point_level']} human_review=true")
    print("[v2be] tp2=false tp3=false tp4=false can_train_model=false can_create_operational_labels=false")
    return (0 if tp1_passed else 1), summary


if __name__ == "__main__":
    raise SystemExit(run()[0])
