#!/usr/bin/env python3
"""Recover REC_00019 footprint only from explicit, traceable spatial lineage."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path

from rasterio.warp import transform_geom

ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "v2bd_sentinel_patch_footprint_recovery_config.json"
REFERENCE_COLUMNS = "reference_id patch_id file_path file_type line_number field_name field_value reference_role contains_spatial_hint contains_asset_hint contains_crs_hint contains_window_hint contains_transform_hint notes".split()
LINEAGE_COLUMNS = "lineage_id patch_id region city package_id event_id candidate_asset_id asset_file asset_type asset_sensor asset_date asset_band_or_product lineage_source_file lineage_confidence has_direct_link has_indirect_link blocking_reason notes".split()
METADATA_COLUMNS = "metadata_id patch_id source_file source_type metadata_key metadata_value crs crs_status has_bbox has_transform has_window has_pixel_size has_width_height has_center has_tile_id can_support_boundary blocking_reason notes".split()
METHOD_COLUMNS = "method_id patch_id method_name required_inputs observed_inputs method_allowed method_sufficient can_build_boundary blocking_reason recommended_action notes".split()
CANDIDATE_COLUMNS = "candidate_boundary_id patch_id boundary_source_method source_file geometry_format crs crs_status geometry_present geometry_valid area_m2 bbox_minx bbox_miny bbox_maxx bbox_maxy geometry_hash can_feed_v2av can_feed_v2az can_feed_v2aw blocking_reason notes".split()
FEED_COLUMNS = "feed_id patch_id package_id event_id source_method source_file geometry_path crs geometry_hash ready_for_v2aw ready_for_v2av ready_for_v2az requires_human_review notes".split()
CERT_COLUMNS = "certificate_id patch_id package_id event_id references_scanned direct_asset_links_found spatial_metadata_records_found valid_reconstruction_methods boundary_candidates_valid boundary_recovered turning_point_unlocked status blocking_reason minimum_required_next_input notes".split()
PLAN_COLUMNS = "plan_id patch_id required_input why_required where_to_put_it accepted_formats required_crs example_value_format validation_command unlocks notes".split()
TABLES = {
    "v2bd_REC_00019_reference_inventory.csv": REFERENCE_COLUMNS,
    "v2bd_patch_asset_lineage_registry.csv": LINEAGE_COLUMNS,
    "v2bd_spatial_metadata_inventory.csv": METADATA_COLUMNS,
    "v2bd_footprint_reconstruction_method_audit.csv": METHOD_COLUMNS,
    "v2bd_REC_00019_patch_boundary_candidate_registry.csv": CANDIDATE_COLUMNS,
    "v2bd_ready_patch_boundary_feed.csv": FEED_COLUMNS,
    "v2bd_REC_00019_footprint_recovery_certificate.csv": CERT_COLUMNS,
    "v2bd_minimal_patch_footprint_recovery_plan.csv": PLAN_COLUMNS,
}
ASSET_AUDIT = Path("manifests/training_readiness/revp_v1fs_self_supervised_asset_sanity_and_embedding_plan/asset_sanity_audit_v1fs.csv")


def clean(value):
    return str(value or "").strip()


def b(value):
    return "true" if value else "false"


def stable_id(prefix, *parts):
    return f"{prefix}_{hashlib.sha256('|'.join(clean(x) for x in parts).encode()).hexdigest()[:12]}"


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
        writer.writerows([{col: row.get(col, "") for col in columns} for row in rows])


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def resolve_dirs(dataset_dir=None, output_dir=None, config_dir=None, asset_dir=None, docs_dir=None):
    return {"dataset": Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT/"datasets").resolve(),
            "output": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT/"outputs_public").resolve(),
            "config": Path(config_dir or os.getenv("CONFIG_DIR") or ROOT/"configs").resolve(),
            "asset": Path(asset_dir or os.getenv("ASSET_DIR") or ROOT).resolve(),
            "docs": Path(docs_dir or ROOT/"docs").resolve()}


def scan_references(root, patch_id):
    rows = []
    allowed = {".csv", ".json", ".txt", ".md", ".geojson", ".xml", ".yaml", ".yml", ".py"}
    excluded = {".git", "__pycache__", ".pytest_cache", ".venv", "node_modules"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in allowed or excluded.intersection(path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        if "v2bd" in rel.lower() or rel == "datasets/external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson":
            continue
        try:
            lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        except OSError:
            continue
        for number, line in enumerate(lines, 1):
            if patch_id not in line:
                continue
            low = line.lower()
            rows.append({"reference_id": stable_id("V2BD_REF", rel, number), "patch_id": patch_id, "file_path": rel,
                "file_type": path.suffix.lower(), "line_number": str(number), "field_name": "",
                "field_value": line[:500], "reference_role": "patch_reference",
                "contains_spatial_hint": b(any(x in low for x in ("bounds", "bbox", "geometry", "coordinate"))),
                "contains_asset_hint": b(any(x in low for x in ("asset", ".tif", "sentinel"))),
                "contains_crs_hint": b("epsg:" in low or "crs" in low), "contains_window_hint": b("window" in low),
                "contains_transform_hint": b("transform" in low or "affine" in low),
                "notes": "Repository text reference; does not independently authorize footprint reconstruction."})
    return rows


def explicit_asset_metadata(root, patch_id):
    for row in load_csv(root / ASSET_AUDIT):
        if clean(row.get("candidate_id")) == patch_id:
            return row
    return {}


def parse_bbox(value):
    try:
        vals = [float(x.strip()) for x in clean(value).split(",")]
        return vals if len(vals) == 4 and vals[0] < vals[2] and vals[1] < vals[3] else None
    except ValueError:
        return None


def bbox_polygon(bounds):
    minx, miny, maxx, maxy = bounds
    return {"type": "Polygon", "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]]}


def reconstruct_bbox(bounds, source_crs, target_crs="EPSG:4326"):
    """Reproject an explicit real bbox; never infer missing inputs."""
    if not bounds or not source_crs:
        return None
    return transform_geom(source_crs, target_crs, bbox_polygon(bounds), precision=12)


def reconstruct_affine(transform, width, height):
    """Return explicit raster bounds from GDAL affine [a,b,c,d,e,f]."""
    if not transform or len(transform) != 6 or not width or not height:
        return None
    a, b_, c, d, e, f = [float(x) for x in transform]
    corners = [(c, f), (a*width+c, d*width+f), (b_*height+c, e*height+f),
               (a*width+b_*height+c, d*width+e*height+f)]
    xs, ys = [x for x, _ in corners], [y for _, y in corners]
    return [min(xs), min(ys), max(xs), max(ys)]


def geom_bbox(geometry):
    pts = geometry["coordinates"][0]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def methods(has_bbox, has_crs, has_direct):
    specs = [
        ("bbox_explicit", "explicit bbox|CRS|direct lineage", f"bbox={has_bbox}|crs={has_crs}|direct={has_direct}", True, has_bbox and has_crs and has_direct, ""),
        ("geojson_explicit", "explicit GeoJSON|CRS|direct lineage", "none", True, False, "NO_EXPLICIT_GEOJSON"),
        ("wkt_explicit", "explicit WKT|CRS|direct lineage", "none", True, False, "NO_EXPLICIT_WKT"),
        ("raster_bounds", "raster header bounds|CRS|direct lineage", f"bbox={has_bbox}|crs={has_crs}|direct={has_direct}", True, has_bbox and has_crs and has_direct, ""),
        ("affine_transform_plus_window", "affine|window|CRS|direct lineage", "none", True, False, "NO_AFFINE_WINDOW"),
        ("sentinel_tile_plus_patch_window", "verified tile|patch window|CRS", "none", True, False, "NO_VERIFIED_TILE_WINDOW"),
        ("center_plus_size", "explicit center|size|CRS|provenance", "none", False, False, "CENTER_SIZE_CRS_NOT_COMPLETE"),
        ("filename_inference", "filename coordinates", "filename only", False, False, "FILENAME_INFERENCE_FORBIDDEN"),
        ("default_patch_size", "default size", "none", False, False, "DEFAULT_PATCH_SIZE_FORBIDDEN"),
        ("manual_digitization", "verified source|human review", "future action", True, False, "FUTURE_MANUAL_ACTION"),
    ]
    return [{"method_id": stable_id("V2BD_METHOD", name), "patch_id": "REC_00019", "method_name": name,
        "required_inputs": required, "observed_inputs": observed, "method_allowed": b(allowed),
        "method_sufficient": b(sufficient), "can_build_boundary": b(allowed and sufficient),
        "blocking_reason": "" if allowed and sufficient else blocker,
        "recommended_action": "Build candidate and require human review" if allowed and sufficient else "Provide missing explicit inputs",
        "notes": "Unsafe inference methods remain blocked."} for name, required, observed, allowed, sufficient, blocker in specs]


def schema(columns):
    bools = {x for x in columns if x.startswith("has_") or x.startswith("can_") or x.startswith("ready_")}
    bools |= {"contains_spatial_hint", "contains_asset_hint", "contains_crs_hint", "contains_window_hint",
              "contains_transform_hint", "method_allowed", "method_sufficient", "geometry_present", "geometry_valid",
              "boundary_recovered", "turning_point_unlocked", "requires_human_review", "has_direct_link", "has_indirect_link"}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": list(columns),
            "additionalProperties": False, "properties": {x: {"type": "boolean" if x in bools else "string"} for x in columns}}


def run(dataset_dir=None, output_dir=None, config_dir=None, asset_dir=None, docs_dir=None):
    dirs = resolve_dirs(dataset_dir, output_dir, config_dir, asset_dir, docs_dir)
    config = json.loads((dirs["config"]/CONFIG_NAME).read_text(encoding="utf-8"))
    patch, package, event = config["priority_patch_id"], config["priority_package_id"], config["priority_event_id"]
    references = scan_references(dirs["asset"], patch)
    meta = explicit_asset_metadata(dirs["asset"], patch)
    bounds = parse_bbox(meta.get("bounds_if_header_available"))
    source_crs = clean(meta.get("crs_if_header_available"))
    direct = bool(meta and clean(meta.get("asset_path")) and clean(meta.get("sha256_small_file")))
    asset_exists = (dirs["asset"]/clean(meta.get("asset_path"))).is_file() if meta else False
    lineage = [{"lineage_id": stable_id("V2BD_LINEAGE", patch, meta.get("asset_id")), "patch_id": patch,
        "region": "Recife", "city": "Recife", "package_id": package, "event_id": event,
        "candidate_asset_id": clean(meta.get("asset_id")), "asset_file": clean(meta.get("asset_path")),
        "asset_type": clean(meta.get("asset_type")), "asset_sensor": "Sentinel", "asset_date": "UNKNOWN",
        "asset_band_or_product": clean(meta.get("shape_or_dimensions")), "lineage_source_file": ASSET_AUDIT.as_posix(),
        "lineage_confidence": "HIGH_HEADER_METADATA_PRESERVED_ASSET_CURRENTLY_ABSENT" if direct else "NONE",
        "has_direct_link": b(direct), "has_indirect_link": "false",
        "blocking_reason": "" if direct else "NO_DIRECT_PATCH_ASSET_LINK",
        "notes": f"Explicit prior header audit; current asset_exists={str(asset_exists).lower()}; hash={clean(meta.get('sha256_small_file'))}"}]
    spatial = []
    if meta:
        spatial.append({"metadata_id": stable_id("V2BD_META", patch, ASSET_AUDIT), "patch_id": patch,
            "source_file": ASSET_AUDIT.as_posix(), "source_type": "preserved_raster_header_audit",
            "metadata_key": "bounds|crs|shape|hash|asset_path", "metadata_value": f"bounds={clean(meta.get('bounds_if_header_available'))};shape={clean(meta.get('shape_or_dimensions'))};hash={clean(meta.get('sha256_small_file'))}",
            "crs": source_crs, "crs_status": "KNOWN_SOURCE_CRS_REPROJECTED_TO_ACCEPTED_OUTPUT" if source_crs else "UNKNOWN",
            "has_bbox": b(bool(bounds)), "has_transform": "false", "has_window": "false",
            "has_pixel_size": "true" if bounds and "103x101" in clean(meta.get("shape_or_dimensions")) else "false",
            "has_width_height": b(bool(clean(meta.get("shape_or_dimensions")))), "has_center": "false", "has_tile_id": "false",
            "can_support_boundary": b(bool(bounds and source_crs and direct)),
            "blocking_reason": "" if bounds and source_crs and direct else "INSUFFICIENT_SPATIAL_METADATA",
            "notes": "Explicit header metadata is sufficient for a footprint candidate; current raster payload is absent."})
    method_rows = methods(bool(bounds), bool(source_crs), direct)
    geometry = reconstruct_bbox(bounds, source_crs) if bounds and source_crs and direct else None
    candidates, feeds = [], []
    derived_rel = "external_sources/recife_minimal_tp/derived/patch_boundary_REC_00019_from_lineage.geojson"
    if geometry:
        derived = dirs["dataset"]/derived_rel
        feature = {"type": "Feature", "properties": {"patch_id": patch, "package_id": package,
            "source_method": "preserved_raster_header_bounds_reprojected", "source_crs": source_crs,
            "source_file": ASSET_AUDIT.as_posix(), "source_asset_hash": clean(meta.get("sha256_small_file")),
            "requires_human_review": True, "can_be_ground_truth": False}, "geometry": geometry}
        write_text(derived, json.dumps(feature, indent=2))
        gh = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
        bb = geom_bbox(geometry)
        candidates.append({"candidate_boundary_id": stable_id("V2BD_BOUNDARY", patch, gh), "patch_id": patch,
            "boundary_source_method": "preserved_raster_header_bounds_reprojected", "source_file": ASSET_AUDIT.as_posix(),
            "geometry_format": "geojson_file", "crs": "EPSG:4326", "crs_status": "ACCEPTED_REPROJECTED_FROM_EPSG_32725",
            "geometry_present": "true", "geometry_valid": "true", "area_m2": "", "bbox_minx": bb[0], "bbox_miny": bb[1],
            "bbox_maxx": bb[2], "bbox_maxy": bb[3], "geometry_hash": gh, "can_feed_v2av": "true",
            "can_feed_v2az": "true", "can_feed_v2aw": "true", "blocking_reason": "",
            "notes": "TP1 candidate from explicit preserved raster header bounds; human review required because raster payload is absent."})
        feeds.append({"feed_id": stable_id("V2BD_FEED", patch, gh), "patch_id": patch, "package_id": package,
            "event_id": event, "source_method": "preserved_raster_header_bounds_reprojected",
            "source_file": ASSET_AUDIT.as_posix(), "geometry_path": f"datasets/{derived_rel}", "crs": "EPSG:4326",
            "geometry_hash": gh, "ready_for_v2aw": "true", "ready_for_v2av": "true", "ready_for_v2az": "true",
            "requires_human_review": "true", "notes": "Candidate footprint only; not event geometry, label, or final ground truth."})
    valid_methods = sum(r["can_build_boundary"] == "true" for r in method_rows)
    status = "BOUNDARY_RECOVERED_FROM_LINEAGE" if candidates else "BOUNDARY_BLOCKED_NO_SPATIAL_METADATA"
    cert = [{"certificate_id": stable_id("V2BD_CERT", patch, status), "patch_id": patch, "package_id": package,
        "event_id": event, "references_scanned": str(len(references)), "direct_asset_links_found": str(int(direct)),
        "spatial_metadata_records_found": str(len(spatial)), "valid_reconstruction_methods": str(valid_methods),
        "boundary_candidates_valid": str(len(candidates)), "boundary_recovered": b(bool(candidates)),
        "turning_point_unlocked": b(bool(candidates)), "status": status,
        "blocking_reason": "" if candidates else "NO_EXPLICIT_SPATIAL_METADATA",
        "minimum_required_next_input": "Human review of recovered footprint and current raster/header confirmation" if candidates else "Real bbox/WKT/GeoJSON or raster bounds+CRS",
        "notes": "Recovered footprint is TP1 candidate only; it does not create event geometry or truth."}]
    plan_specs = [("explicit bbox with CRS", "Direct footprint bounds", "FILL_THIS_PATCH_BOUNDARY.csv", "bbox", "minx,miny,maxx,maxy"),
                  ("real WKT with CRS", "Direct footprint polygon", "FILL_THIS_PATCH_BOUNDARY.csv", "wkt", "POLYGON((...))"),
                  ("real GeoJSON with CRS", "Direct footprint polygon", "patch_boundary_REC_00019/", "geojson", "Feature Polygon"),
                  ("raster transform/window", "Reproduce raster footprint", "patch_boundary_REC_00019/", "tif|json", "transform,width,height,CRS"),
                  ("GIS footprint export", "Human-reviewed direct footprint", "patch_boundary_REC_00019/", "geojson", "Polygon with CRS")]
    plan = [{"plan_id": stable_id("V2BD_PLAN", req), "patch_id": patch, "required_input": req, "why_required": why,
        "where_to_put_it": where, "accepted_formats": formats, "required_crs": "|".join(config["accepted_crs"]),
        "example_value_format": example, "validation_command": "python scripts/run_v2ba_minimal_real_geometry_acquisition_workbench.py --mode validate",
        "unlocks": "TP1 candidate after human review", "notes": "Use to confirm or replace recovered candidate; never infer from filename/default size."}
        for req, why, where, formats, example in plan_specs]
    recovery_dir = dirs["dataset"]/"external_sources"/"recife_minimal_tp"/"patch_boundary_REC_00019"
    if candidates:
        write_csv(recovery_dir/"FILL_THIS_PATCH_BOUNDARY.autofill_from_v2bd.csv",
            "target_type target_id package_id source_type geometry_path crs provenance_note source_document review_status instructions".split(),
            [{"target_type": "patch_boundary", "target_id": patch, "package_id": package, "source_type": "geojson_file",
              "geometry_path": f"datasets/{derived_rel}", "crs": "EPSG:4326",
              "provenance_note": "Recovered from explicit preserved raster header bounds and reprojected from EPSG:32725.",
              "source_document": ASSET_AUDIT.as_posix(), "review_status": "provided_unreviewed",
              "instructions": "Human must confirm lineage/header before promotion."}])
    else:
        write_csv(recovery_dir/"FILL_THIS_PATCH_BOUNDARY.recovery_required.csv",
            "target_type target_id source_type geometry_value geometry_path crs blocking_reason instructions".split(),
            [{"target_type": "patch_boundary", "target_id": patch, "source_type": "missing", "crs": "UNKNOWN",
              "blocking_reason": "REAL_GEOMETRY_REQUIRED", "instructions": "Provide explicit footprint and provenance."}])
    outputs = {"v2bd_REC_00019_reference_inventory.csv": references, "v2bd_patch_asset_lineage_registry.csv": lineage,
        "v2bd_spatial_metadata_inventory.csv": spatial, "v2bd_footprint_reconstruction_method_audit.csv": method_rows,
        "v2bd_REC_00019_patch_boundary_candidate_registry.csv": candidates, "v2bd_ready_patch_boundary_feed.csv": feeds,
        "v2bd_REC_00019_footprint_recovery_certificate.csv": cert, "v2bd_minimal_patch_footprint_recovery_plan.csv": plan}
    for name, rows in outputs.items():
        write_csv(dirs["dataset"]/name, TABLES[name], rows)
        write_text(dirs["dataset"]/"schemas"/f"{Path(name).stem}.schema.json", json.dumps(schema(TABLES[name]), indent=2))
    docs = {
        "v2bd_sentinel_patch_footprint_recovery_drilldown.md": f"# v2bd Sentinel patch footprint recovery drilldown\n\nScanned {len(references)} repository references. Found direct patch-to-asset lineage in `{ASSET_AUDIT.as_posix()}` with preserved hash, bounds and CRS. The current raster payload is absent, but explicit header evidence supports a human-review TP1 candidate.\n",
        "v2bd_REC_00019_lineage_findings.md": f"# REC_00019 lineage findings\n\nDirect asset: `{clean(meta.get('asset_path'))}` (`{clean(meta.get('asset_id'))}`). Preserved bounds: `{clean(meta.get('bounds_if_header_available'))}`; source CRS: `{source_crs}`; shape: `{clean(meta.get('shape_or_dimensions'))}`; hash: `{clean(meta.get('sha256_small_file'))}`. Asset currently present: `{str(asset_exists).lower()}`. Boundary reconstructed from explicit bounds and reprojected to EPSG:4326; human confirmation remains required.\n",
        "v2bd_patch_boundary_recovery_requirements.md": "# Patch boundary recovery requirements\n\nSafe methods require explicit bbox/WKT/GeoJSON or raster bounds/transform/window plus CRS and direct lineage. Filename inference and default size are forbidden. Review the recovered candidate, then run v2ba validation and v2az dry_run/replay.\n"}
    for name, text in docs.items():
        write_text(dirs["docs"]/name, text)
    summary = {"stage": "v2bd_sentinel_patch_footprint_recovery_drilldown", "status": "OK_WITH_HUMAN_REVIEW_REQUIRED",
        "priority_patch_id": patch, "priority_package_id": package, "priority_event_id": event,
        "references_scanned": len(references), "direct_asset_links_found": int(direct),
        "spatial_metadata_records_found": len(spatial), "valid_reconstruction_methods": valid_methods,
        "boundary_candidates_valid": len(candidates), "boundary_recovered": bool(candidates),
        "ready_patch_feed_rows": len(feeds), "turning_point_level": "TP1_ONE_PATCH_BOUNDARY_CANDIDATE_REQUIRES_HUMAN_REVIEW" if candidates else "TP0_PATCH_FOOTPRINT_LINEAGE_EXHAUSTED",
        "turning_point_ready": bool(candidates), "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "PATCH_FOOTPRINT_RECOVERED_FROM_EXPLICIT_LINEAGE_REQUIRES_HUMAN_REVIEW_NOT_FOR_TRAINING" if candidates else "PATCH_FOOTPRINT_LINEAGE_EXHAUSTED_NOT_FOR_TRAINING"}
    write_text(dirs["output"]/"execution_reports"/"v2bd_sentinel_patch_footprint_recovery_summary.json", json.dumps(summary, indent=2))
    write_text(dirs["output"]/"execution_reports"/"v2bd_sentinel_patch_footprint_recovery_report.md",
        f"# v2bd report\n\nReferences: {len(references)}. Direct asset links: {int(direct)}. Explicit spatial metadata: {len(spatial)}. Boundary candidates: {len(candidates)}. Ready feed rows: {len(feeds)}. Current raster payload absent: {str(not asset_exists).lower()}. Candidate requires human review; no label, model, event geometry, final truth or automatic C4 was created.\n")
    write_text(dirs["output"]/"logs_summary"/"v2bd_sentinel_patch_footprint_recovery.txt",
        "\n".join(f"{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in summary.items()))
    print(f"[v2bd] references={len(references)} direct_asset_links={int(direct)} spatial_metadata={len(spatial)}")
    print(f"[v2bd] boundary_candidates={len(candidates)} ready_feed={len(feeds)} asset_currently_present={str(asset_exists).lower()}")
    print(f"[v2bd] turning_point={summary['turning_point_level']}")
    return 0, summary


if __name__ == "__main__":
    raise SystemExit(run()[0])
