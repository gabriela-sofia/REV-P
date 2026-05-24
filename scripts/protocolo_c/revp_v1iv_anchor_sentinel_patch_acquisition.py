"""
REV-P v1iv - OFFICIAL_ANCHOR_SENTINEL_PATCH_ACQUISITION_AND_QA

Creates a reproducible Google Earth Engine acquisition package for the official
CPRM anchor at Moinho Preto, Petropolis. If Earth Engine is installed and
authenticated, the script searches Sentinel-2 SR Harmonized scenes, downloads a
centered 96 x 96 patch, and emits local QA. If Earth Engine is not ready, it
fails closed with GEE_AUTH_REQUIRED and writes reproducible export plans.

Fixed invariants:
  - no label is created
  - no target is created
  - no model is trained
  - Protocol B remains closed
  - raw raster artifacts stay under local_runs/
  - public metadata is written only when a real patch has QA PASS
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iv"
PATCH_DIR = LOCAL_RUN_DIR / "anchor_centered_patch"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

STAGE = "v1iv"
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
ANCHOR = {
    "anchor_id": "ANCHOR_PET2022_CPRM_ANEXOII_19022022",
    "source_documented_event_unit_id": "PET2022_CPRM_ANEXOII_19022022",
    "region": "PET",
    "municipality": "Petropolis",
    "locality_text_sanitized": "Moinho Preto",
    "anchor_date": "2022-02-19",
    "anchor_latitude": -22.484251,
    "anchor_longitude": -43.211257,
    "phenomenon_group": "MOVEMENT_OF_MASS",
}
WINDOWS = [
    {
        "window_label": "PRE_EVENT",
        "temporal_relation_to_event": "PRE_EVENT",
        "start": "2022-02-01",
        "end": "2022-02-15",
    },
    {
        "window_label": "POST_EVENT_SURVEY",
        "temporal_relation_to_event": "POST_EVENT_OR_SURVEY_WINDOW",
        "start": "2022-02-15",
        "end": "2022-03-06",
    },
]
PATCH_SIZE_PX = 96
PATCH_SIZE_M = 960
PATCH_RADIUS_M = PATCH_SIZE_M / 2
PATCH_CRS = "EPSG:32723"

REGISTRY_FIELDS = [
    "reference_patch_id",
    "anchor_id",
    "source_documented_event_unit_id",
    "region",
    "municipality",
    "locality_text_sanitized",
    "anchor_date",
    "anchor_latitude",
    "anchor_longitude",
    "phenomenon_group",
    "sensor",
    "gee_collection",
    "scene_id_sanitized",
    "scene_date",
    "temporal_relation_to_event",
    "bands_available",
    "patch_generated_locally",
    "patch_centered_on_anchor",
    "center_error_m",
    "patch_size_px",
    "patch_size_m",
    "crs",
    "resolution_m",
    "valid_pixel_fraction",
    "cloud_cover_metadata",
    "public_versioning_status",
    "can_be_reference_patch_candidate",
    "can_be_ground_reference_candidate",
    "can_be_operational_ground_truth",
    "can_create_training_label",
    "can_train_model",
    "can_reopen_protocol_b",
    "primary_blocker",
    "minimum_evidence_needed",
    "notes",
]

SCENE_FIELDS = [
    "search_status",
    "window_label",
    "temporal_relation_to_event",
    "gee_collection",
    "scene_id_sanitized",
    "scene_date",
    "cloud_cover_metadata",
    "mgrs_tile",
    "processing_baseline",
    "product_id_sanitized",
    "bands_available",
    "date_delta_days_from_anchor",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def scene_date_from_millis(ms: Any) -> str:
    if ms in ("", None):
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def delta_days(scene_date: str) -> str:
    if not scene_date:
        return ""
    try:
        event_dt = datetime.fromisoformat(ANCHOR["anchor_date"]).date()
        scene_dt = datetime.fromisoformat(scene_date).date()
        return str((scene_dt - event_dt).days)
    except ValueError:
        return ""


def clean_scene_id(scene_id: str) -> str:
    return scene_id.replace("COPERNICUS/S2_SR_HARMONIZED/", "")


def prepare_output_dir(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1iv").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    PATCH_DIR.mkdir(parents=True, exist_ok=True)


def check_gee() -> tuple[dict[str, Any], Any | None]:
    status: dict[str, Any] = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "earthengine_api_installed": False,
        "earthengine_api_version": "",
        "gee_authenticated": False,
        "gee_available": False,
        "status": "GEE_AUTH_REQUIRED",
        "error_type": "",
        "error_message": "",
        "notes": "No external data are assumed unless Earth Engine initialization succeeds.",
    }

    if importlib.util.find_spec("ee") is None:
        status["error_type"] = "EARTHENGINE_API_NOT_INSTALLED"
        status["error_message"] = "Python package earthengine-api is not installed."
        return status, None

    status["earthengine_api_installed"] = True
    try:
        status["earthengine_api_version"] = importlib.metadata.version("earthengine-api")
    except importlib.metadata.PackageNotFoundError:
        status["earthengine_api_version"] = "UNKNOWN"

    try:
        ee = importlib.import_module("ee")
        ee.Initialize()
        status["gee_authenticated"] = True
        status["gee_available"] = True
        status["status"] = "GEE_AVAILABLE"
        return status, ee
    except Exception as exc:  # Earth Engine uses several exception classes across versions.
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        return status, None


def build_plan_js() -> str:
    bands_js = json.dumps(BANDS)
    windows_js = json.dumps(WINDOWS, indent=2)
    return f"""// REV-P {STAGE} reproducible Earth Engine export plan.
// Run in the Earth Engine Code Editor only after authenticating GEE.
// Raw exports must remain local-only and must not be committed.

var anchor = ee.Geometry.Point([{ANCHOR['anchor_longitude']}, {ANCHOR['anchor_latitude']}]);
var collectionId = '{COLLECTION}';
var bands = {bands_js};
var windows = {windows_js};
var patchCrs = '{PATCH_CRS}';
var patchSizePx = {PATCH_SIZE_PX};
var patchRadiusM = {PATCH_RADIUS_M};

var region = anchor.transform(patchCrs, 1).buffer(patchRadiusM).bounds(1)
  .transform('EPSG:4326', 1);

function candidates(windowSpec) {{
  return ee.ImageCollection(collectionId)
    .filterBounds(anchor)
    .filterDate(windowSpec.start, windowSpec.end)
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .map(function(image) {{
      return image.set('revp_window_label', windowSpec.window_label)
        .set('revp_temporal_relation_to_event', windowSpec.temporal_relation_to_event);
    }});
}}

var merged = ee.ImageCollection([]);
windows.forEach(function(windowSpec) {{
  merged = merged.merge(candidates(windowSpec));
}});

var selected = ee.Image(merged.sort('CLOUDY_PIXEL_PERCENTAGE').first()).select(bands);
print('REV-P {STAGE} selected scene', selected);

Export.image.toDrive({{
  image: selected,
  description: 'revp_{STAGE}_anchor_s2_patch',
  folder: 'revp_{STAGE}_manual_download',
  fileNamePrefix: 'revp_{STAGE}_anchor_centered_s2_patch',
  region: region,
  crs: patchCrs,
  dimensions: '{PATCH_SIZE_PX}x{PATCH_SIZE_PX}',
  maxPixels: 1000000
}});
"""


def build_plan_py() -> str:
    return f'''"""
REV-P {STAGE} reproducible Earth Engine Python export plan.
Authenticate with: earthengine authenticate
Then run from the REV-P repository root.
"""

import pathlib
import urllib.request
import zipfile

import ee

ANCHOR_LON = {ANCHOR["anchor_longitude"]}
ANCHOR_LAT = {ANCHOR["anchor_latitude"]}
COLLECTION = "{COLLECTION}"
BANDS = {BANDS!r}
PATCH_CRS = "{PATCH_CRS}"
PATCH_SIZE_PX = {PATCH_SIZE_PX}
PATCH_RADIUS_M = {PATCH_RADIUS_M}
WINDOWS = {WINDOWS!r}
OUT_DIR = pathlib.Path("local_runs/protocolo_c/{STAGE}/manual_export_download")

ee.Initialize()
anchor = ee.Geometry.Point([ANCHOR_LON, ANCHOR_LAT])
region = anchor.transform(PATCH_CRS, 1).buffer(PATCH_RADIUS_M).bounds(1).transform("EPSG:4326", 1)

merged = ee.ImageCollection([])
for window in WINDOWS:
    candidates = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(anchor)
        .filterDate(window["start"], window["end"])
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .map(lambda image: image.set("revp_window_label", window["window_label"])
             .set("revp_temporal_relation_to_event", window["temporal_relation_to_event"]))
    )
    merged = merged.merge(candidates)

selected = ee.Image(merged.sort("CLOUDY_PIXEL_PERCENTAGE").first()).select(BANDS)
url = selected.getDownloadURL({{
    "name": "revp_{STAGE}_anchor_centered_s2_patch",
    "bands": BANDS,
    "region": region,
    "crs": PATCH_CRS,
    "dimensions": f"{{PATCH_SIZE_PX}}x{{PATCH_SIZE_PX}}",
    "format": "GEO_TIFF",
}})

OUT_DIR.mkdir(parents=True, exist_ok=True)
zip_path = OUT_DIR / "revp_{STAGE}_anchor_centered_s2_patch.zip"
urllib.request.urlretrieve(url, zip_path)
with zipfile.ZipFile(zip_path) as zf:
    zf.extractall(OUT_DIR)
print(f"Downloaded to {{OUT_DIR}}")
'''


def write_export_plans() -> None:
    (LOCAL_RUN_DIR / "v1iv_gee_export_plan.js").write_text(build_plan_js(), encoding="utf-8")
    (LOCAL_RUN_DIR / "v1iv_gee_export_plan.py").write_text(build_plan_py(), encoding="utf-8")


def blocked_registry_row(
    blocker: str,
    minimum_evidence_needed: str,
    notes: str,
    selected: dict[str, Any] | None = None,
    qa: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = selected or {}
    qa = qa or {}
    return {
        "reference_patch_id": "",
        "anchor_id": ANCHOR["anchor_id"],
        "source_documented_event_unit_id": ANCHOR["source_documented_event_unit_id"],
        "region": ANCHOR["region"],
        "municipality": ANCHOR["municipality"],
        "locality_text_sanitized": ANCHOR["locality_text_sanitized"],
        "anchor_date": ANCHOR["anchor_date"],
        "anchor_latitude": ANCHOR["anchor_latitude"],
        "anchor_longitude": ANCHOR["anchor_longitude"],
        "phenomenon_group": ANCHOR["phenomenon_group"],
        "sensor": "Sentinel-2 MSI",
        "gee_collection": COLLECTION,
        "scene_id_sanitized": selected.get("scene_id_sanitized", ""),
        "scene_date": selected.get("scene_date", ""),
        "temporal_relation_to_event": selected.get("temporal_relation_to_event", ""),
        "bands_available": selected.get("bands_available", ""),
        "patch_generated_locally": bool_text(False),
        "patch_centered_on_anchor": bool_text(False),
        "center_error_m": qa.get("center_error_m", ""),
        "patch_size_px": PATCH_SIZE_PX,
        "patch_size_m": PATCH_SIZE_M,
        "crs": qa.get("crs", ""),
        "resolution_m": qa.get("resolution_m", ""),
        "valid_pixel_fraction": qa.get("valid_pixel_fraction", ""),
        "cloud_cover_metadata": selected.get("cloud_cover_metadata", ""),
        "public_versioning_status": "METADATA_ONLY",
        "can_be_reference_patch_candidate": bool_text(False),
        "can_be_ground_reference_candidate": bool_text(False),
        "can_be_operational_ground_truth": bool_text(False),
        "can_create_training_label": bool_text(False),
        "can_train_model": bool_text(False),
        "can_reopen_protocol_b": bool_text(False),
        "primary_blocker": blocker,
        "minimum_evidence_needed": minimum_evidence_needed,
        "notes": notes,
    }


def pass_registry_row(selected: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    row = blocked_registry_row(
        blocker="NONE",
        minimum_evidence_needed="Reference candidate only; independent interpretation remains required before any later claim.",
        notes="Patch exists locally with QA PASS; no label or target is created.",
        selected=selected,
        qa=qa,
    )
    row.update(
        {
            "reference_patch_id": "REFPATCH_PET2022_CPRM_MOINHO_PRETO_S2_V1IV",
            "patch_generated_locally": bool_text(True),
            "patch_centered_on_anchor": bool_text(True),
            "can_be_reference_patch_candidate": bool_text(True),
            "can_be_ground_reference_candidate": bool_text(True),
        }
    )
    return row


def write_blocked_outputs(blocker: str, availability: dict[str, Any]) -> None:
    search_row = {
        "search_status": "SKIPPED_GEE_AUTH_REQUIRED",
        "window_label": "",
        "temporal_relation_to_event": "",
        "gee_collection": COLLECTION,
        "scene_id_sanitized": "",
        "scene_date": "",
        "cloud_cover_metadata": "",
        "mgrs_tile": "",
        "processing_baseline": "",
        "product_id_sanitized": "",
        "bands_available": "",
        "date_delta_days_from_anchor": "",
        "notes": availability.get("error_type", "GEE unavailable"),
    }
    write_csv(LOCAL_RUN_DIR / "v1iv_sentinel_scene_search.csv", [search_row], SCENE_FIELDS)

    decision = {
        "decision_status": "NO_SCENE_SELECTED",
        "selection_reason": blocker,
        "scene_id_sanitized": "",
        "scene_date": "",
        "temporal_relation_to_event": "",
        "cloud_cover_metadata": "",
        "bands_available": "",
        "notes": "Earth Engine authentication is required before scene search and local export.",
    }
    write_csv(
        LOCAL_RUN_DIR / "v1iv_selected_scene_decision.csv",
        [decision],
        list(decision.keys()),
    )

    row = blocked_registry_row(
        blocker=blocker,
        minimum_evidence_needed="Authenticate Google Earth Engine and rerun v1iv to search and export the Sentinel-2 patch.",
        notes="No Sentinel scene or raster patch was generated in this run.",
    )
    write_csv(LOCAL_RUN_DIR / "v1iv_anchor_patch_manifest_local.csv", [row], REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1iv_reference_patch_readiness_decision.csv", [row], REGISTRY_FIELDS)

    qa_rows = [
        {"check": "gee_available", "status": "FAIL", "detail": blocker},
        {"check": "patch_generated", "status": "FAIL", "detail": "No patch generated."},
        {"check": "center_correct", "status": "NOT_RUN", "detail": "No patch generated."},
        {"check": "bands_present", "status": "NOT_RUN", "detail": ",".join(BANDS)},
        {"check": "crs_resolution", "status": "NOT_RUN", "detail": "No patch generated."},
        {"check": "valid_pixel_fraction", "status": "NOT_RUN", "detail": "No patch generated."},
        {"check": "cloud_metadata", "status": "NOT_RUN", "detail": "No scene selected."},
        {"check": "temporal_relation_to_event", "status": "NOT_RUN", "detail": "No scene selected."},
        {"check": "no_nan_inf", "status": "NOT_RUN", "detail": "No patch generated."},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "No public v1iv output written."},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
    ]
    write_csv(LOCAL_RUN_DIR / "v1iv_qa.csv", qa_rows, ["check", "status", "detail"])
    write_csv(LOCAL_RUN_DIR / "v1iv_patch_quality_audit.csv", qa_rows, ["check", "status", "detail"])

    summary = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "status": blocker,
        "gee_available": False,
        "earthengine_api_installed": availability.get("earthengine_api_installed", False),
        "gee_authenticated": availability.get("gee_authenticated", False),
        "sentinel_scenes_found": 0,
        "selected_scene_id_sanitized": "",
        "patch_generated": False,
        "bands_present": [],
        "patch_qa_status": "NOT_RUN",
        "can_be_reference_patch_candidate": False,
        "can_be_ground_reference_candidate": False,
        "can_be_operational_ground_truth": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_reopen_protocol_b": False,
        "public_versioning_status": "METADATA_ONLY",
        "primary_blocker": blocker,
        "commit_warranted": False,
    }
    write_json(LOCAL_RUN_DIR / "v1iv_summary.json", summary)


def feature_to_scene_row(feature: dict[str, Any], window: dict[str, str]) -> dict[str, Any]:
    props = feature.get("properties", {})
    scene_id = feature.get("id", props.get("system:index", ""))
    scene_date = scene_date_from_millis(props.get("system:time_start"))
    cloud = props.get("CLOUDY_PIXEL_PERCENTAGE", "")
    return {
        "search_status": "FOUND",
        "window_label": window["window_label"],
        "temporal_relation_to_event": window["temporal_relation_to_event"],
        "gee_collection": COLLECTION,
        "scene_id_sanitized": clean_scene_id(str(scene_id)),
        "scene_date": scene_date,
        "cloud_cover_metadata": cloud,
        "mgrs_tile": props.get("MGRS_TILE", ""),
        "processing_baseline": props.get("PROCESSING_BASELINE", ""),
        "product_id_sanitized": props.get("PRODUCT_ID", ""),
        "bands_available": ",".join(BANDS),
        "date_delta_days_from_anchor": delta_days(scene_date),
        "notes": "Sorted by CLOUDY_PIXEL_PERCENTAGE within window.",
    }


def search_scenes(ee: Any) -> list[dict[str, Any]]:
    point = ee.Geometry.Point([ANCHOR["anchor_longitude"], ANCHOR["anchor_latitude"]])
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        collection = (
            ee.ImageCollection(COLLECTION)
            .filterBounds(point)
            .filterDate(window["start"], window["end"])
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        info = collection.limit(20).getInfo()
        for feature in info.get("features", []):
            rows.append(feature_to_scene_row(feature, window))
    rows.sort(key=lambda row: (float(row["cloud_cover_metadata"]), row["scene_date"]))
    return rows


def select_scene(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda row: (
            float(row["cloud_cover_metadata"]) if row["cloud_cover_metadata"] != "" else 9999.0,
            0 if row["window_label"] == "POST_EVENT_SURVEY" else 1,
            row["scene_date"],
        ),
    )[0]


def write_selected_decision(selected: dict[str, Any] | None, scene_count: int) -> None:
    if selected is None:
        row = {
            "decision_status": "NO_SCENE_SELECTED",
            "selection_reason": "NO_SENTINEL2_SR_HARMONIZED_SCENE_FOUND_FOR_ANCHOR_WINDOWS",
            "scene_id_sanitized": "",
            "scene_date": "",
            "temporal_relation_to_event": "",
            "cloud_cover_metadata": "",
            "bands_available": "",
            "notes": f"scenes_found={scene_count}",
        }
    else:
        row = {
            "decision_status": "SCENE_SELECTED",
            "selection_reason": "Lowest CLOUDY_PIXEL_PERCENTAGE across requested windows; tie favors post-event/survey window.",
            "scene_id_sanitized": selected["scene_id_sanitized"],
            "scene_date": selected["scene_date"],
            "temporal_relation_to_event": selected["temporal_relation_to_event"],
            "cloud_cover_metadata": selected["cloud_cover_metadata"],
            "bands_available": selected["bands_available"],
            "notes": f"scenes_found={scene_count}",
        }
    write_csv(LOCAL_RUN_DIR / "v1iv_selected_scene_decision.csv", [row], list(row.keys()))


def download_patch(ee: Any, selected: dict[str, Any]) -> Path:
    point = ee.Geometry.Point([ANCHOR["anchor_longitude"], ANCHOR["anchor_latitude"]])
    region = (
        point.transform(PATCH_CRS, 1)
        .buffer(PATCH_RADIUS_M)
        .bounds(1)
        .transform("EPSG:4326", 1)
    )
    scene_asset = f"{COLLECTION}/{selected['scene_id_sanitized']}"
    image = ee.Image(scene_asset).select(BANDS)
    url = image.getDownloadURL(
        {
            "name": "revp_v1iv_anchor_centered_s2_patch",
            "bands": BANDS,
            "region": region,
            "crs": PATCH_CRS,
            "dimensions": f"{PATCH_SIZE_PX}x{PATCH_SIZE_PX}",
            "format": "GEO_TIFF",
        }
    )
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = PATCH_DIR / "revp_v1iv_anchor_centered_s2_patch.zip"
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(PATCH_DIR)
    tifs = sorted(PATCH_DIR.glob("*.tif")) + sorted(PATCH_DIR.glob("*.tiff"))
    if not tifs:
        raise RuntimeError("GEE download completed but no GeoTIFF was found in the archive.")
    return tifs[0]


def qa_patch(tif_path: Path, selected: dict[str, Any]) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    qa: dict[str, Any] = {
        "patch_local_private_path": str(tif_path),
        "center_error_m": "",
        "crs": "",
        "resolution_m": "",
        "valid_pixel_fraction": "",
        "bands_present": "",
        "width_px": "",
        "height_px": "",
        "has_nan_inf": "",
    }
    rows: list[dict[str, str]] = []

    try:
        import numpy as np
        import rasterio
        from rasterio.warp import transform
    except ImportError as exc:
        rows.append({"check": "rasterio_numpy_available", "status": "FAIL", "detail": str(exc)})
        return "FAIL", qa, rows

    with rasterio.open(tif_path) as src:
        data = src.read(masked=True)
        qa["crs"] = str(src.crs) if src.crs else ""
        qa["resolution_m"] = f"{abs(src.res[0]):.4f}" if src.res else ""
        qa["width_px"] = src.width
        qa["height_px"] = src.height
        qa["bands_present"] = ",".join(BANDS[: src.count])
        center_x = (src.bounds.left + src.bounds.right) / 2
        center_y = (src.bounds.bottom + src.bounds.top) / 2
        if src.crs:
            anchor_x, anchor_y = transform(
                "EPSG:4326",
                src.crs,
                [ANCHOR["anchor_longitude"]],
                [ANCHOR["anchor_latitude"]],
            )
            qa["center_error_m"] = f"{math.hypot(center_x - anchor_x[0], center_y - anchor_y[0]):.4f}"
        valid_mask = ~np.ma.getmaskarray(data)
        finite_mask = np.isfinite(np.ma.filled(data, np.nan))
        valid_fraction = float(np.count_nonzero(valid_mask & finite_mask) / data.size)
        qa["valid_pixel_fraction"] = f"{valid_fraction:.6f}"
        qa["has_nan_inf"] = bool_text(not bool(np.all(finite_mask[valid_mask])))

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": check, "status": "PASS" if ok else "FAIL", "detail": detail})

    add("patch_generated", tif_path.exists(), tif_path.name)
    add("center_correct", qa["center_error_m"] != "" and float(qa["center_error_m"]) <= 15.0, str(qa["center_error_m"]))
    add("bands_present", qa["bands_present"] == ",".join(BANDS), qa["bands_present"])
    add("patch_size_px", qa["width_px"] == PATCH_SIZE_PX and qa["height_px"] == PATCH_SIZE_PX, f"{qa['width_px']}x{qa['height_px']}")
    add("crs_present", bool(qa["crs"]), qa["crs"])
    add("resolution_m", qa["resolution_m"] != "" and 8.0 <= float(qa["resolution_m"]) <= 12.5, str(qa["resolution_m"]))
    add("valid_pixel_fraction", qa["valid_pixel_fraction"] != "" and float(qa["valid_pixel_fraction"]) >= 0.80, str(qa["valid_pixel_fraction"]))
    add("cloud_metadata", selected.get("cloud_cover_metadata", "") != "", str(selected.get("cloud_cover_metadata", "")))
    add("temporal_relation_to_event", selected.get("temporal_relation_to_event", "") != "", selected.get("temporal_relation_to_event", ""))
    add("no_nan_inf", qa["has_nan_inf"] == "false", qa["has_nan_inf"])
    add("no_private_path_in_public_outputs", True, "Public outputs use sanitized metadata only.")
    add("can_create_training_label_false", True, "false")
    add("can_train_model_false", True, "false")
    add("can_reopen_protocol_b_false", True, "false")

    status = "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"
    return status, qa, rows


def write_schema() -> None:
    schema_rows = [
        {"field": field, "description": f"REV-P {STAGE} official anchor Sentinel patch registry field."}
        for field in REGISTRY_FIELDS
    ]
    write_csv(
        SCHEMAS_DIR / "official_anchor_sentinel_patch_schema.csv",
        schema_rows,
        ["field", "description"],
    )


def write_public_registry(row: dict[str, Any]) -> None:
    write_csv(DATASETS_DIR / "official_anchor_sentinel_patch_registry.csv", [row], REGISTRY_FIELDS)
    write_schema()


def run_available_flow(args: argparse.Namespace, availability: dict[str, Any], ee: Any) -> None:
    scenes = search_scenes(ee)
    if not scenes:
        no_scene = blocked_registry_row(
            "NO_SENTINEL2_SR_HARMONIZED_SCENE_FOUND_FOR_ANCHOR_WINDOWS",
            "A Sentinel-2 SR Harmonized scene intersecting the anchor in the requested windows.",
            "Earth Engine was available, but no scene was returned by the requested filters.",
        )
        write_csv(LOCAL_RUN_DIR / "v1iv_sentinel_scene_search.csv", [], SCENE_FIELDS)
        write_selected_decision(None, 0)
        write_csv(LOCAL_RUN_DIR / "v1iv_anchor_patch_manifest_local.csv", [no_scene], REGISTRY_FIELDS)
        write_csv(LOCAL_RUN_DIR / "v1iv_reference_patch_readiness_decision.csv", [no_scene], REGISTRY_FIELDS)
        qa_rows = [{"check": "scene_found", "status": "FAIL", "detail": no_scene["primary_blocker"]}]
        write_csv(LOCAL_RUN_DIR / "v1iv_qa.csv", qa_rows, ["check", "status", "detail"])
        write_csv(LOCAL_RUN_DIR / "v1iv_patch_quality_audit.csv", qa_rows, ["check", "status", "detail"])
        write_json(
            LOCAL_RUN_DIR / "v1iv_summary.json",
            {
                "stage": STAGE,
                "timestamp": utc_now(),
                "status": no_scene["primary_blocker"],
                "gee_available": True,
                "sentinel_scenes_found": 0,
                "selected_scene_id_sanitized": "",
                "patch_generated": False,
                "patch_qa_status": "NOT_RUN",
                "can_be_reference_patch_candidate": False,
                "can_be_ground_reference_candidate": False,
                "can_be_operational_ground_truth": False,
                "can_create_training_label": False,
                "can_train_model": False,
                "can_reopen_protocol_b": False,
                "public_versioning_status": "METADATA_ONLY",
                "primary_blocker": no_scene["primary_blocker"],
                "commit_warranted": False,
            },
        )
        return

    write_csv(LOCAL_RUN_DIR / "v1iv_sentinel_scene_search.csv", scenes, SCENE_FIELDS)
    selected = select_scene(scenes)
    write_selected_decision(selected, len(scenes))
    assert selected is not None

    patch_path: Path | None = None
    qa_status = "NOT_RUN"
    qa: dict[str, Any] = {}
    qa_rows: list[dict[str, str]] = []
    blocker = ""
    if args.try_local_export:
        try:
            patch_path = download_patch(ee, selected)
            qa_status, qa, qa_rows = qa_patch(patch_path, selected)
            blocker = "NONE" if qa_status == "PASS" else "PATCH_QA_FAILED"
        except Exception as exc:
            blocker = "GEE_LOCAL_EXPORT_FAILED"
            qa_rows = [
                {"check": "local_export", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"},
                {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
                {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
                {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
            ]
    else:
        blocker = "LOCAL_EXPORT_NOT_REQUESTED"
        qa_rows = [{"check": "local_export", "status": "NOT_RUN", "detail": "Pass --try-local-export to download patch."}]

    write_csv(LOCAL_RUN_DIR / "v1iv_qa.csv", qa_rows, ["check", "status", "detail"])
    write_csv(LOCAL_RUN_DIR / "v1iv_patch_quality_audit.csv", qa_rows, ["check", "status", "detail"])

    if patch_path and qa_status == "PASS":
        registry_row = pass_registry_row(selected, qa)
    else:
        registry_row = blocked_registry_row(
            blocker=blocker,
            minimum_evidence_needed="Successful local Sentinel-2 patch export with QA PASS.",
            notes="Scene metadata may exist, but reference patch candidacy remains blocked until QA passes.",
            selected=selected,
            qa=qa,
        )

    write_csv(LOCAL_RUN_DIR / "v1iv_anchor_patch_manifest_local.csv", [registry_row], REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1iv_reference_patch_readiness_decision.csv", [registry_row], REGISTRY_FIELDS)

    if patch_path and qa_status == "PASS":
        write_public_registry(registry_row)

    summary = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "status": "PATCH_QA_PASS" if qa_status == "PASS" else blocker,
        "gee_available": availability.get("gee_available", False),
        "earthengine_api_installed": availability.get("earthengine_api_installed", False),
        "gee_authenticated": availability.get("gee_authenticated", False),
        "sentinel_scenes_found": len(scenes),
        "selected_scene_id_sanitized": selected.get("scene_id_sanitized", ""),
        "selected_scene_date": selected.get("scene_date", ""),
        "selected_scene_cloud_cover_metadata": selected.get("cloud_cover_metadata", ""),
        "patch_generated": bool(patch_path),
        "bands_present": qa.get("bands_present", "").split(",") if qa.get("bands_present") else [],
        "patch_qa_status": qa_status,
        "can_be_reference_patch_candidate": qa_status == "PASS",
        "can_be_ground_reference_candidate": qa_status == "PASS",
        "can_be_operational_ground_truth": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_reopen_protocol_b": False,
        "public_versioning_status": "METADATA_ONLY",
        "primary_blocker": "NONE" if qa_status == "PASS" else blocker,
        "commit_warranted": qa_status == "PASS",
    }
    write_json(LOCAL_RUN_DIR / "v1iv_summary.json", summary)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Clear local v1iv outputs before running.")
    parser.add_argument("--check-gee", action="store_true", help="Check earthengine-api and authentication.")
    parser.add_argument("--build-gee-export", action="store_true", help="Write reproducible GEE export plans.")
    parser.add_argument("--try-local-export", action="store_true", help="Try to download the selected GEE patch locally.")
    parser.add_argument("--emit-qa", action="store_true", help="Emit QA CSV outputs.")
    parser.add_argument("--emit-manifest", action="store_true", help="Emit manifest/readiness outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    prepare_output_dir(args.force)
    if args.build_gee_export:
        write_export_plans()

    availability, ee = check_gee()
    write_json(LOCAL_RUN_DIR / "v1iv_gee_availability_check.json", availability)

    if not availability.get("gee_available") or ee is None:
        if args.build_gee_export:
            write_export_plans()
        write_blocked_outputs("GEE_AUTH_REQUIRED", availability)
    else:
        if args.build_gee_export:
            write_export_plans()
        run_available_flow(args, availability, ee)

    summary = json.loads((LOCAL_RUN_DIR / "v1iv_summary.json").read_text(encoding="utf-8"))
    print("=" * 72)
    print("REV-P v1iv OFFICIAL_ANCHOR_SENTINEL_PATCH_ACQUISITION_AND_QA")
    print("=" * 72)
    print(f"GEE available: {summary.get('gee_available')}")
    print(f"Sentinel scenes found: {summary.get('sentinel_scenes_found')}")
    print(f"Selected scene: {summary.get('selected_scene_id_sanitized', '')}")
    print(f"Patch generated: {summary.get('patch_generated')}")
    print(f"Bands present: {summary.get('bands_present')}")
    print(f"Patch QA: {summary.get('patch_qa_status')}")
    print(f"Reference patch candidate: {summary.get('can_be_reference_patch_candidate')}")
    print(f"Primary blocker: {summary.get('primary_blocker')}")
    print(f"Commit warranted: {summary.get('commit_warranted')}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
