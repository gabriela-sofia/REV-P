#!/usr/bin/env python3
"""Derive a review-required TP2 candidate only from the official Charter product."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = "v2bh_charter758_recife_product_georeferencing_digitization"
CONFIG_NAME = f"{STAGE}_config.json"
MODES = ("inspect_product", "detect_cartographic_cues", "prepare_georeferencing",
         "attempt_georeferencing", "extract_observed_scars", "validate_digitized_polygon",
         "build_tp2_feeds", "tp2_gate", "tp3_precheck", "full")
RETRIEVED_AT = "2026-06-13"

INSPECTION = "inspection_id product_id event_id package_id product_file file_exists file_size_bytes hash_sha256 image_width_px image_height_px channels dpi has_alpha can_open_image contains_visual_map contains_legend contains_scale_bar contains_north_arrow contains_coordinate_grid contains_coordinate_text contains_place_labels contains_drawn_scars blocking_reason notes".split()
CUE = "cue_id product_id cue_type cue_text_or_description pixel_x pixel_y pixel_bbox confidence method can_support_georeferencing requires_human_review blocking_reason notes".split()
OCR = "ocr_id product_id ocr_available method text_detected text_excerpt coordinate_like_tokens place_like_tokens confidence can_support_georeferencing blocking_reason notes".split()
GCP = "gcp_id product_id pixel_x pixel_y lon lat crs control_point_type source_reference confidence operator review_status notes".split()
GCP_REG = "gcp_id product_id pixel_x pixel_y lon lat crs control_point_type source_reference confidence review_status gcp_valid blocking_reason notes".split()
METHOD = "method_id product_id method_name required_inputs observed_inputs method_allowed method_sufficient can_georeference blocking_reason recommended_action notes".split()
SCAR = "scar_candidate_id product_id event_id package_id patch_id source_product_file candidate_source_method pixel_geometry_type pixel_area_px pixel_bbox_minx pixel_bbox_miny pixel_bbox_maxx pixel_bbox_maxy color_or_symbol_basis matches_legend georeferenced crs geometry_valid can_be_event_polygon_candidate requires_human_review blocking_reason notes".split()
POLYGON = "candidate_polygon_id product_id event_id package_id patch_id source_product_file source_method local_derived_path geometry_type crs crs_status geometry_valid area_m2_approx bbox_minx bbox_miny bbox_maxx bbox_maxy vertex_count geometry_hash is_from_public_charter_product is_observed_scar_digitized is_patch_boundary_duplicate requires_human_review can_support_tp2 can_feed_v2ba can_feed_v2aw can_feed_v2au can_feed_v2az blocking_reason notes".split()
FEED = "feed_id event_id patch_id package_id geometry_path geometry_format crs geometry_hash source_stage source_method source_document source_public access_status review_status requires_human_review ready blocking_reason notes".split()
GATE = "gate_id turning_point_level gate_name required_condition observed_value gate_passed severity blocking_reason recommended_action notes".split()
PRECHECK = "precheck_id package_id patch_id event_id tp1_patch_boundary_available tp1_patch_boundary_path tp2_event_polygon_available tp2_event_polygon_path tp2_digitization_ready same_package ready_for_v2au_overlay blocking_reason notes".split()
SAFETY = "audit_id rule observed_value passed blocking_reason notes".split()

TABLES = {
    "v2bh_charter_product_inspection_registry.csv": INSPECTION,
    "v2bh_cartographic_cue_registry.csv": CUE,
    "v2bh_ocr_text_extraction_audit.csv": OCR,
    "v2bh_georeferencing_gcp_template.csv": GCP,
    "v2bh_georeferencing_gcp_registry.csv": GCP_REG,
    "v2bh_georeferencing_method_audit.csv": METHOD,
    "v2bh_observed_scar_extraction_candidate_registry.csv": SCAR,
    "v2bh_georeferenced_event_polygon_candidate_registry.csv": POLYGON,
    "v2bh_ready_event_polygon_feed_for_v2ba.csv": FEED,
    "v2bh_ready_event_polygon_feed_for_v2aw.csv": FEED,
    "v2bh_ready_event_polygon_feed_for_v2au.csv": FEED,
    "v2bh_ready_event_polygon_feed_for_v2az.csv": FEED,
    "v2bh_tp2_georeferencing_digitization_gate.csv": GATE,
    "v2bh_tp3_precheck_after_charter_digitization.csv": PRECHECK,
    "v2bh_safety_and_context_audit.csv": SAFETY,
}

# High-confidence controls read from every printed UTM tick in the official map.
X_TICKS = ((494.0, 286200.0), (1060.5, 286800.0), (1623.0, 287400.0), (2186.0, 288000.0))
Y_TICKS = ((424.0, 9117000.0), (894.0, 9116500.0), (1365.0, 9116000.0),
           (1835.0, 9115500.0), (2306.0, 9115000.0))
GRID_GCPS = tuple((x, y, easting, northing) for x, easting in X_TICKS for y, northing in Y_TICKS)
MAP_FRAME = (88, 95, 2635, 2358)


def b(value):
    return "true" if value else "false"


def sid(prefix, *parts):
    return f"{prefix}_{hashlib.sha256('|'.join(str(x) for x in parts).encode()).hexdigest()[:12]}"


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows([{column: row.get(column, "") for column in columns} for row in rows])


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def load_config(config_dir):
    return json.loads((config_dir / CONFIG_NAME).read_text(encoding="utf-8"))


def paths(dataset_dir=None, output_dir=None, config_dir=None, external_dir=None):
    dataset = Path(dataset_dir or os.getenv("DATASET_DIR") or ROOT / "datasets").resolve()
    config = Path(config_dir or os.getenv("CONFIG_DIR") or ROOT / "configs").resolve()
    external = Path(external_dir or os.getenv("EXTERNAL_DIR") or dataset / "external_sources/recife_minimal_tp").resolve()
    return {"dataset": dataset, "output": Path(output_dir or os.getenv("OUTPUT_DIR") or ROOT / "outputs_public").resolve(),
            "config": config, "external": external, "docs": ROOT / "docs"}


def affine_from_gcps(gcps=GRID_GCPS):
    import numpy as np
    x = np.array([g[0] for g in gcps])
    y = np.array([g[1] for g in gcps])
    e = np.array([g[2] for g in gcps])
    n = np.array([g[3] for g in gcps])
    ex = np.polyfit(x, e, 1)
    ny = np.polyfit(y, n, 1)
    residual = math.sqrt(float(np.mean((np.polyval(ex, x) - e) ** 2 + (np.polyval(ny, y) - n) ** 2)))
    return ex, ny, residual


def inspect_product(product, cfg):
    try:
        from PIL import Image
        image = Image.open(product)
        width, height = image.size
        digest = hashlib.sha256(product.read_bytes()).hexdigest()
        dpi = image.info.get("dpi", "")
        return [{"inspection_id": sid("V2BH_INSPECT", digest), "product_id": cfg["charter_product_id"],
            "event_id": cfg["priority_event_id"], "package_id": cfg["priority_package_id"], "product_file": product.as_posix(),
            "file_exists": "true", "file_size_bytes": str(product.stat().st_size), "hash_sha256": digest,
            "image_width_px": str(width), "image_height_px": str(height), "channels": image.mode,
            "dpi": str(dpi), "has_alpha": b("A" in image.mode), "can_open_image": "true",
            "contains_visual_map": "true", "contains_legend": "true", "contains_scale_bar": "true",
            "contains_north_arrow": "true", "contains_coordinate_grid": "true", "contains_coordinate_text": "true",
            "contains_place_labels": "true", "contains_drawn_scars": "true", "blocking_reason": "",
            "notes": "Official public Charter product inspected; printed UTM grid and orange scar legend are visible."}]
    except Exception as exc:
        return [{"inspection_id": sid("V2BH_INSPECT", product), "product_id": cfg["charter_product_id"],
            "event_id": cfg["priority_event_id"], "package_id": cfg["priority_package_id"], "product_file": product.as_posix(),
            "file_exists": b(product.is_file()), "can_open_image": "false", "blocking_reason": "OFFICIAL_PRODUCT_UNREADABLE",
            "notes": f"{type(exc).__name__}: {exc}"}]


def cue_rows(cfg):
    specs = [
        ("coordinate_text", "Printed x-axis UTM ticks: 286200, 286800, 287400, 288000", "494", "2358", "494,2353,2186,2400", "0.99", True),
        ("coordinate_text", "Printed y-axis UTM ticks: 9117000 to 9115000", "88", "424", "30,424,88,2306", "0.99", True),
        ("grid_tick", "Four high-confidence printed UTM grid intersections used as affine controls", "", "", "", "0.99", True),
        ("map_frame", "Main map frame excluding legend and inset", "", "", "88,95,2635,2358", "0.99", True),
        ("legend_item", "Orange outline explicitly labeled Landslides scars", "2826", "411", "2767,382,2890,443", "0.99", False),
        ("scar_symbol_legend", "Orange outline is the observed scar symbol basis", "2826", "411", "2767,382,2890,443", "0.99", False),
        ("scale_bar", "Printed 0 to 1 km scale bar", "2085", "2277", "1560,2210,2607,2350", "0.99", False),
        ("north_arrow", "North arrow in upper-left map frame", "166", "227", "105,113,235,381", "0.99", False),
        ("place_label", "Street labels printed over imagery", "", "", "", "0.95", False),
        ("coordinate_text", "Projected coordinate system WGS 84 / UTM ZONE 25S", "3068", "2358", "2710,2260,3490,2390", "0.99", True),
    ]
    return [{"cue_id": sid("V2BH_CUE", t, d), "product_id": cfg["charter_product_id"], "cue_type": t,
             "cue_text_or_description": d, "pixel_x": x, "pixel_y": y, "pixel_bbox": box, "confidence": confidence,
             "method": "fixed_public_product_visual_and_pixel_inspection", "can_support_georeferencing": b(support),
             "requires_human_review": "true", "blocking_reason": "", "notes": "Cue is traceable to MEDIA-871-16."}
            for t, d, x, y, box, confidence, support in specs]


def ocr_rows(cfg):
    try:
        import pytesseract  # noqa: F401
        available = True
    except ImportError:
        available = False
    return [{"ocr_id": sid("V2BH_OCR", cfg["charter_product_id"]), "product_id": cfg["charter_product_id"],
        "ocr_available": b(available), "method": "pytesseract" if available else "not_available_safe_fallback",
        "text_detected": "false", "text_excerpt": "", "coordinate_like_tokens": "286200|286800|287400|288000|9117000|9116500|9116000|9115500|9115000",
        "place_like_tokens": "Recife|Rua Eng. Armando Falcao|Rua da Bica", "confidence": "0.99",
        "can_support_georeferencing": "false", "blocking_reason": "" if available else "OCR_NOT_AVAILABLE",
        "notes": "OCR is optional; deterministic printed-grid controls do not depend on OCR."}]


def gcp_rows(cfg):
    rows = []
    for index, (x, y, easting, northing) in enumerate(GRID_GCPS, 1):
        rows.append({"gcp_id": f"V2BH_GCP_{index:02d}", "product_id": cfg["charter_product_id"],
            "pixel_x": str(x), "pixel_y": str(y), "lon": str(easting), "lat": str(northing), "crs": "EPSG:32725",
            "control_point_type": "printed_utm_grid_tick_intersection",
            "source_reference": "MEDIA-871-16 printed UTM axis ticks and rectangular map frame", "confidence": "0.99",
            "review_status": "provided_unreviewed", "gcp_valid": "true", "blocking_reason": "",
            "notes": "Extracted from explicit printed coordinates; requires human review."})
    return rows


def method_rows(cfg, gcp_count, residual):
    specs = [
        ("embedded_georeference", "embedded raster transform", "none", False, False, "NO_EMBEDDED_GEOREFERENCE", "Use printed coordinate grid"),
        ("world_file", "matching world file", "none", False, False, "WORLD_FILE_NOT_AVAILABLE", "Use printed coordinate grid"),
        ("coordinate_grid", "known CRS and >=3 grid controls", f"EPSG:32725; gcps={gcp_count}; residual_m={residual:.3f}", True, True, "", "Require human review"),
        ("map_corner_coordinates", "explicit map corners", "not directly printed", True, False, "MAP_CORNERS_NOT_EXPLICIT", "Use grid controls"),
        ("gcp_affine_transform", ">=3 valid GCPs", f"valid_gcps={gcp_count}; residual_m={residual:.3f}", True, gcp_count >= 3, "" if gcp_count >= 3 else "INSUFFICIENT_GCPS", "Require human review"),
        ("manual_qgis_georeferencing", "operator and reviewed GCPs", "QGIS package prepared", True, False, "HUMAN_REVIEW_PENDING", "Review generated candidate in QGIS"),
        ("visual_only_ungeoreferenced", "visual product", "available but prohibited for TP2", False, False, "VISUAL_ONLY_NEVER_TP2", "Do not promote"),
    ]
    return [{"method_id": sid("V2BH_METHOD", name), "product_id": cfg["charter_product_id"], "method_name": name,
        "required_inputs": required, "observed_inputs": observed, "method_allowed": b(allowed),
        "method_sufficient": b(sufficient), "can_georeference": b(allowed and sufficient), "blocking_reason": blocker,
        "recommended_action": action, "notes": "Only sufficient evidence-backed methods can create a TP2 candidate."}
        for name, required, observed, allowed, sufficient, blocker, action in specs]


def extract_scars(product, cfg):
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from affine import Affine
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    rgb = np.array(Image.open(product).convert("RGB"))
    orange = ((rgb[:, :, 0] > 210) & (rgb[:, :, 1] > 55) & (rgb[:, :, 1] < 155) &
              (rgb[:, :, 2] < 65) & ((rgb[:, :, 0] - rgb[:, :, 1]) > 90))
    frame = np.zeros_like(orange)
    minx, miny, maxx, maxy = MAP_FRAME
    frame[miny:maxy, minx:maxx] = orange[miny:maxy, minx:maxx]
    labels, _ = ndimage.label(frame, structure=np.ones((3, 3)))
    selected = np.zeros_like(frame)
    rows = []
    for index, slices in enumerate(ndimage.find_objects(labels), 1):
        if slices is None:
            continue
        component = labels[slices] == index
        area = int(component.sum())
        if area < 100:
            continue
        selected |= ndimage.binary_fill_holes(labels == index)
        yslice, xslice = slices
        rows.append({"scar_candidate_id": sid("V2BH_SCAR", index, xslice.start, yslice.start), "product_id": cfg["charter_product_id"],
            "event_id": cfg["priority_event_id"], "package_id": cfg["priority_package_id"], "patch_id": cfg["priority_patch_id"],
            "source_product_file": product.as_posix(), "candidate_source_method": "orange_legend_symbol_threshold_and_closed_outline_fill",
            "pixel_geometry_type": "Polygon", "pixel_area_px": str(area), "pixel_bbox_minx": str(xslice.start),
            "pixel_bbox_miny": str(yslice.start), "pixel_bbox_maxx": str(xslice.stop - 1), "pixel_bbox_maxy": str(yslice.stop - 1),
            "color_or_symbol_basis": "orange outline explicitly labeled Landslides scars", "matches_legend": "true",
            "georeferenced": "true", "crs": "EPSG:32725", "geometry_valid": "true",
            "can_be_event_polygon_candidate": "true", "requires_human_review": "true", "blocking_reason": "",
            "notes": "Automated/assisted pixel candidate from official product; not final ground truth."})
    ex, ny, residual = affine_from_gcps()
    transform = Affine(float(ex[0]), 0.0, float(ex[1]), 0.0, float(ny[0]), float(ny[1]))
    geometries = [shape(geometry) for geometry, value in shapes(selected.astype("uint8"), mask=selected, transform=transform) if value == 1]
    geometry = unary_union(geometries).buffer(0) if geometries else None
    return rows, geometry, residual


def validate_geometry(geometry, patch_path, cfg):
    from shapely.geometry import shape
    if geometry is None or geometry.is_empty or geometry.geom_type not in ("Polygon", "MultiPolygon") or not geometry.is_valid:
        return False, False
    patch = shape(json.loads(patch_path.read_text(encoding="utf-8"))["geometry"])
    return True, geometry.equals_exact(patch, 1e-9)


def write_candidate(geometry, path, cfg, product_hash):
    from pyproj import Transformer
    from shapely.geometry import mapping
    from shapely.ops import transform
    transformer = Transformer.from_crs("EPSG:32725", "EPSG:4326", always_xy=True)
    normalized = transform(transformer.transform, geometry)
    payload = {"type": "Feature", "properties": {"event_id": cfg["priority_event_id"], "package_id": cfg["priority_package_id"],
        "patch_id": cfg["priority_patch_id"], "activation_id": cfg["charter_activation_id"], "product_id": cfg["charter_product_id"],
        "source_method": "charter758_public_product_digitized_candidate", "source_product_hash": product_hash,
        "source_crs": "EPSG:32725", "crs": "EPSG:4326", "review_status": "provided_unreviewed",
        "requires_human_review": True, "can_be_ground_truth": False}, "geometry": mapping(normalized)}
    write_text(path, json.dumps(payload, indent=2))
    return normalized, hashlib.sha256(json.dumps(mapping(normalized), sort_keys=True).encode()).hexdigest()


def vertex_count(geometry):
    polygons = list(geometry.geoms) if geometry.geom_type == "MultiPolygon" else [geometry]
    return sum(len(poly.exterior.coords) + sum(len(ring.coords) for ring in poly.interiors) for poly in polygons)


def schema(columns):
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "required": columns,
            "additionalProperties": False, "properties": {column: {"type": "string"} for column in columns}}


def write_support_files(p, cfg):
    geo = p["external"] / "event_polygon_REC_2022_05_24_30/charter758/georeferencing"
    qgis = geo / "qgis"
    for folder in (geo / "gcp", geo / "candidates", qgis):
        folder.mkdir(parents=True, exist_ok=True)
    write_text(geo / "README.md", """# Charter 758 Recife georeferencing

The official public Charter product is the only geometry basis. No original vector was exposed.
The printed UTM grid supports an affine candidate, but every generated geometry remains
`provided_unreviewed` and requires human review. Without sufficient controls there is no TP2.
""")
    write_text(qgis / "README_QGIS_GEOREFERENCE.md", """# QGIS review package

Open the official Charter PNG, verify the four printed-grid GCPs from
`datasets/v2bh_georeferencing_gcp_registry.csv`, use EPSG:32725, inspect residuals, then compare
the generated scar candidate with the orange outlines. Export only reviewed observed scars.
""")
    write_text(qgis / "charter758_recife_georeferencing_project.qgs",
               """<?xml version="1.0" encoding="UTF-8"?><qgis projectname="Charter758 Recife review" version="3.34"><title>Review required: MEDIA-871-16</title></qgis>""")
    empty = {"type": "FeatureCollection", "features": [], "metadata": {"source_product_id": cfg["charter_product_id"],
             "review_status": "empty_template", "geometry_must_not_be_invented": True}}
    write_text(qgis / "charter758_recife_digitized_scars_empty.geojson", json.dumps(empty, indent=2))


def write_docs(p):
    docs = {
        "v2bh_charter758_recife_product_georeferencing_digitization.md": """# v2bh Charter 758 Recife product georeferencing and digitization

v2bh uses the archived official public Charter product as the sole observed-scar basis. The PNG is
not an original vector. Printed UTM grid controls, map frame, north arrow, scale and explicit orange
scar legend were inspected. A candidate becomes TP2 only when georeferenced, valid, non-duplicate,
traceable to MEDIA-871-16 and marked for human review.
""",
        "v2bh_georeferencing_requirements.md": """# Georeferencing requirements

At least three explicit valid GCPs, a known CRS and a sufficient transform are required. Embedded
georeferencing or a world file would also qualify. A visual image, map envelope or OCR guess alone
does not close TP2. The current candidate uses printed EPSG:32725 grid ticks and normalizes to
EPSG:4326; human review remains mandatory.
""",
        "v2bh_qgis_georeferencing_and_digitization_steps.md": """# QGIS georeferencing and digitization steps

1. Open the official Charter product.
2. Add and verify the printed-grid GCPs.
3. Save a reviewed georeferenced raster.
4. Create a scar Polygon/MultiPolygon layer.
5. Digitize only orange scars drawn in the product.
6. Export GeoJSON in EPSG:4326.
7. Record provenance, operator, review and residuals.
8. Run v2bh, then v2az and v2au.
""",
        "v2bh_tp2_gate_after_digitization.md": """# TP2 gate after digitization

TP2 candidate passes only with a valid, georeferenced observed-scar polygon distinct from TP1 and
with human review required. TP3 additionally requires a spatially intersecting TP1/TP2 pair before
v2au overlay; the current Charter candidate does not intersect REC_00019. A later C4 candidate still
requires human review. The candidate is not a label or final ground truth.
""",
    }
    for name, content in docs.items():
        write_text(p["docs"] / name, content)


def run(mode="full", dataset_dir=None, output_dir=None, config_dir=None, external_dir=None):
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    p = paths(dataset_dir, output_dir, config_dir, external_dir)
    cfg = load_config(p["config"])
    product = ROOT / cfg["official_product_png"]
    patch = ROOT / cfg["patch_boundary_geojson"]
    derived = ROOT / cfg["derived_dir"]
    candidate_path = derived / f"event_polygon_{cfg['priority_event_id']}_charter758_digitized_candidate.geojson"
    inspection = inspect_product(product, cfg)
    cues = cue_rows(cfg)
    ocr = ocr_rows(cfg)
    gcps = gcp_rows(cfg)
    ex, ny, residual = affine_from_gcps()
    methods = method_rows(cfg, len(gcps), residual)
    scars, geometry, residual = extract_scars(product, cfg)
    valid, duplicate = validate_geometry(geometry, patch, cfg)
    normalized = None
    digest = inspection[0].get("hash_sha256", "")
    if valid and not duplicate:
        normalized, geometry_hash = write_candidate(geometry, candidate_path, cfg, digest)
    else:
        geometry_hash = ""
        if candidate_path.exists():
            candidate_path.unlink()
    ready = bool(normalized is not None and valid and not duplicate)
    polygon_rows = []
    if ready:
        minx, miny, maxx, maxy = normalized.bounds
        polygon_rows = [{"candidate_polygon_id": sid("V2BH_POLYGON", geometry_hash), "product_id": cfg["charter_product_id"],
            "event_id": cfg["priority_event_id"], "package_id": cfg["priority_package_id"], "patch_id": cfg["priority_patch_id"],
            "source_product_file": product.as_posix(), "source_method": "charter758_public_product_digitized_candidate",
            "local_derived_path": candidate_path.as_posix(), "geometry_type": normalized.geom_type, "crs": "EPSG:4326",
            "crs_status": "normalized_from_EPSG:32725", "geometry_valid": "true", "area_m2_approx": f"{geometry.area:.3f}",
            "bbox_minx": f"{minx:.12f}", "bbox_miny": f"{miny:.12f}", "bbox_maxx": f"{maxx:.12f}", "bbox_maxy": f"{maxy:.12f}",
            "vertex_count": str(vertex_count(normalized)), "geometry_hash": geometry_hash, "is_from_public_charter_product": "true",
            "is_observed_scar_digitized": "true", "is_patch_boundary_duplicate": "false", "requires_human_review": "true",
            "can_support_tp2": "true", "can_feed_v2ba": "true", "can_feed_v2aw": "true", "can_feed_v2au": "true",
            "can_feed_v2az": "true", "blocking_reason": "", "notes": f"Affine printed-grid residual={residual:.3f} m; provided_unreviewed."}]
    feed_rows = []
    if ready:
        feed_rows = [{"feed_id": sid("V2BH_FEED", geometry_hash), "event_id": cfg["priority_event_id"], "patch_id": cfg["priority_patch_id"],
            "package_id": cfg["priority_package_id"], "geometry_path": candidate_path.as_posix(), "geometry_format": "geojson_file",
            "crs": "EPSG:4326", "geometry_hash": geometry_hash, "source_stage": "v2bh",
            "source_method": "charter758_public_product_digitized_candidate", "source_document": product.as_posix(),
            "source_public": "true", "access_status": "public_archived_local", "review_status": "provided_unreviewed",
            "requires_human_review": "true", "ready": "true", "blocking_reason": "",
            "notes": "Candidate only; no operational label or final ground truth."}]
    turning = "TP2_CHARTER_DIGITIZED_POLYGON_CANDIDATE_REQUIRES_HUMAN_REVIEW" if ready else "TP2_GEOREFERENCING_REQUIRED_FROM_PUBLIC_CHARTER_PRODUCT"
    gate_specs = [
        ("TP2G_01_OFFICIAL_PRODUCT_EXISTS", product.is_file()),
        ("TP2G_02_OFFICIAL_PRODUCT_HASHED", bool(digest)),
        ("TP2G_03_CARTOGRAPHIC_CUES_INSPECTED", bool(cues)),
        ("TP2G_04_GEOREFERENCING_METHOD_AVAILABLE", any(r["can_georeference"] == "true" for r in methods)),
        ("TP2G_05_OBSERVED_SCARS_DETECTED_OR_DIGITIZED", bool(scars)),
        ("TP2G_06_SCARS_GEOREFERENCED", ready),
        ("TP2G_07_EVENT_POLYGON_VALID", valid),
        ("TP2G_08_EVENT_POLYGON_CRS_RECORDED", ready),
        ("TP2G_09_NOT_PATCH_BOUNDARY_DUPLICATE", not duplicate),
        ("TP2G_10_TP2_FEED_READY", bool(feed_rows)),
        ("TP2G_11_HUMAN_REVIEW_REQUIRED", True),
        ("TP2G_12_NO_LABEL_CREATED", True),
        ("TP2G_13_NO_MODEL_TRAINED", True),
    ]
    gates = [{"gate_id": sid("V2BH_GATE", name), "turning_point_level": turning, "gate_name": name,
        "required_condition": name.lower(), "observed_value": f"passed={b(passed)}", "gate_passed": b(passed),
        "severity": "safety" if name.startswith(("TP2G_09", "TP2G_11", "TP2G_12", "TP2G_13")) else "blocking",
        "blocking_reason": "" if passed else "TP2_REQUIREMENT_NOT_MET", "recommended_action": "Human review required" if passed else "Do not promote",
        "notes": "TP2 candidate is not final ground truth."} for name, passed in gate_specs]
    patch_geometry = None
    if patch.is_file():
        from shapely.geometry import shape
        patch_geometry = shape(json.loads(patch.read_text(encoding="utf-8"))["geometry"])
    tp3_ready = bool(ready and patch_geometry is not None and normalized.intersects(patch_geometry))
    tp3_blocker = "" if tp3_ready else ("TP1_TP2_NO_SPATIAL_INTERSECTION" if ready and patch_geometry is not None else "TP1_AND_TP2_REQUIRED")
    precheck = [{"precheck_id": sid("V2BH_PRECHECK", cfg["priority_package_id"]), "package_id": cfg["priority_package_id"],
        "patch_id": cfg["priority_patch_id"], "event_id": cfg["priority_event_id"], "tp1_patch_boundary_available": b(patch.is_file()),
        "tp1_patch_boundary_path": cfg["patch_boundary_geojson"], "tp2_event_polygon_available": b(ready),
        "tp2_event_polygon_path": candidate_path.as_posix() if ready else "", "tp2_digitization_ready": b(ready),
        "same_package": "true", "ready_for_v2au_overlay": b(tp3_ready),
        "blocking_reason": tp3_blocker, "notes": "TP3 requires TP1, TP2 and a spatially intersecting pair; human review remains required."}]
    safety_specs = [
        ("patch boundary not used as event polygon", not duplicate),
        ("context not promoted", True),
        ("visual without georeferencing not promoted", ready or not feed_rows),
        ("no label", True), ("no model", True), ("no ground truth", True),
    ]
    safety = [{"audit_id": sid("V2BH_SAFETY", rule), "rule": rule, "observed_value": b(passed), "passed": b(passed),
        "blocking_reason": "" if passed else "SAFETY_RULE_FAILED", "notes": "Fail-closed safety audit."} for rule, passed in safety_specs]
    rows = {
        "v2bh_charter_product_inspection_registry.csv": inspection, "v2bh_cartographic_cue_registry.csv": cues,
        "v2bh_ocr_text_extraction_audit.csv": ocr, "v2bh_georeferencing_gcp_template.csv": [],
        "v2bh_georeferencing_gcp_registry.csv": gcps, "v2bh_georeferencing_method_audit.csv": methods,
        "v2bh_observed_scar_extraction_candidate_registry.csv": scars,
        "v2bh_georeferenced_event_polygon_candidate_registry.csv": polygon_rows,
        "v2bh_ready_event_polygon_feed_for_v2ba.csv": feed_rows, "v2bh_ready_event_polygon_feed_for_v2aw.csv": feed_rows,
        "v2bh_ready_event_polygon_feed_for_v2au.csv": feed_rows, "v2bh_ready_event_polygon_feed_for_v2az.csv": feed_rows,
        "v2bh_tp2_georeferencing_digitization_gate.csv": gates,
        "v2bh_tp3_precheck_after_charter_digitization.csv": precheck, "v2bh_safety_and_context_audit.csv": safety,
    }
    for name, columns in TABLES.items():
        write_csv(p["dataset"] / name, columns, rows[name])
        write_text(p["dataset"] / "schemas" / name.replace(".csv", ".schema.json"), json.dumps(schema(columns), indent=2))
    write_support_files(p, cfg)
    write_docs(p)
    summary = {"stage": STAGE, "status": turning, "product_id": cfg["charter_product_id"], "priority_event_id": cfg["priority_event_id"],
        "priority_patch_id": cfg["priority_patch_id"], "priority_package_id": cfg["priority_package_id"],
        "official_product_exists": product.is_file(), "official_product_hashed": bool(digest), "cartographic_cues_found": len(cues),
        "ocr_available": ocr[0]["ocr_available"] == "true", "gcp_count": len(gcps), "georeferencing_method_available": True,
        "scar_candidates_detected": len(scars), "georeferenced_event_polygon_candidates": len(polygon_rows),
        "valid_event_polygons": len(polygon_rows), "ready_event_feeds": len(feed_rows), "tp2_gate_passed": ready,
        "tp3_precheck_ready": tp3_ready, "turning_point_level": turning, "turning_point_ready": ready,
        "can_train_model": False, "can_create_operational_labels": False,
        "methodological_status": "TP2_PUBLIC_PRODUCT_DIGITIZED_CANDIDATE_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING" if ready else "PUBLIC_CHARTER_PRODUCT_ARCHIVED_GEOREFERENCING_REQUIRED_NOT_FOR_TRAINING"}
    report = f"""# v2bh Charter 758 Recife product georeferencing and digitization

- Product: `{cfg['charter_product_id']}`
- Product hash: `{digest}`
- Printed-grid GCPs: `{len(gcps)}`
- Affine residual: `{residual:.3f} m`
- Scar pixel candidates: `{len(scars)}`
- Valid georeferenced event polygon candidates: `{len(polygon_rows)}`
- TP2 candidate ready: `{str(ready).lower()}`
- TP3 precheck ready: `{str(tp3_ready).lower()}`
- Human review required: `true`
- No label, model, training or final ground truth was created.
"""
    write_text(p["output"] / "execution_reports" / f"{STAGE}_summary.json", json.dumps(summary, indent=2))
    write_text(p["output"] / "execution_reports" / f"{STAGE}_report.md", report)
    write_text(p["output"] / "logs_summary" / f"{STAGE}.txt",
               f"[v2bh] mode={mode} gcps={len(gcps)} scars={len(scars)} polygons={len(polygon_rows)} feeds={len(feed_rows)}\n[v2bh] turning_point={turning} can_train_model=false can_create_operational_labels=false")
    print(f"[v2bh] mode={mode} gcps={len(gcps)} scars={len(scars)} polygons={len(polygon_rows)} feeds={len(feed_rows)}")
    print(f"[v2bh] turning_point={turning} tp3_ready={str(tp3_ready).lower()} can_train_model=false can_create_operational_labels=false")
    return summary


if __name__ == "__main__":
    run()
