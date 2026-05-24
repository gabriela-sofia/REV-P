"""
REV-P v1ix - GEE_SENTINEL_PATCH_EXPORT_FOR_OFFICIAL_ANCHOR.

Searches Google Earth Engine for Sentinel-2 SR Harmonized pre-event and
post-event scenes at the official CPRM anchor, then tries to download small
anchor-centered patches. If authentication or direct download is unavailable,
the script fails closed with a controlled status and reproducible export plans.

Invariants:
  - no label, target, training, or Protocol B reopening is created
  - rasters stay under local_runs/
  - public registry metadata is written only after real local patch QA PASS
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.metadata
import json
import math
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ix"
PATCH_DIR = LOCAL_RUN_DIR / "anchor_centered_patch"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

STAGE = "v1ix"
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
GEE_SOURCE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
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
EVENT_DATE = "2022-02-15"
SURVEY_DATE = "2022-02-19"
WINDOWS = [
    {
        "window_label": "PRE_EVENT",
        "temporal_relation_to_event": "PRE_EVENT",
        "start_inclusive": "2022-02-01",
        "end_inclusive": "2022-02-14",
        "gee_end_exclusive": "2022-02-15",
    },
    {
        "window_label": "POST_EVENT_SURVEY",
        "temporal_relation_to_event": "POST_EVENT_OR_SURVEY_WINDOW",
        "start_inclusive": "2022-02-15",
        "end_inclusive": "2022-03-05",
        "gee_end_exclusive": "2022-03-06",
    },
]
PATCH_SIZE_PX = 96
PATCH_SIZE_M = 960
PATCH_RADIUS_M = PATCH_SIZE_M / 2
PATCH_CRS = "EPSG:32723"
LOCAL_RASTER_SUFFIX = ".local_geotiff"

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
    "date_delta_days_from_event",
    "date_delta_days_from_survey",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


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


def clean_scene_id(scene_id: str) -> str:
    return scene_id.replace(f"{COLLECTION}/", "")


def date_from_millis(ms: Any) -> str:
    if ms in ("", None):
        return ""
    try:
        return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).date().isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def delta_days(scene_date: str, reference_date: str) -> str:
    if not scene_date:
        return ""
    try:
        return str((datetime.fromisoformat(scene_date).date() - datetime.fromisoformat(reference_date).date()).days)
    except ValueError:
        return ""


def prepare_output_dir(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / STAGE).resolve()
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
        "authentication_instruction": "earthengine authenticate",
        "error_type": "",
        "error_message": "",
        "notes": "Earth Engine must initialize before any Sentinel-2 search or download.",
    }

    if importlib.util.find_spec("ee") is None:
        status["error_type"] = "EARTHENGINE_API_NOT_INSTALLED"
        status["error_message"] = "Install with: python -m pip install earthengine-api"
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
        status["authentication_instruction"] = ""
        return status, ee
    except Exception as exc:
        status["error_type"] = type(exc).__name__
        status["error_message"] = str(exc)
        return status, None


def build_plan_js() -> str:
    bands_js = json.dumps(BANDS)
    source_bands_js = json.dumps(GEE_SOURCE_BANDS)
    windows_js = json.dumps(WINDOWS, indent=2)
    return f"""// REV-P {STAGE} reproducible Earth Engine export plan.
// Run in the Earth Engine Code Editor after authenticating GEE.
// Raw exports must remain local-only and must not be committed.

var anchor = ee.Geometry.Point([{ANCHOR['anchor_longitude']}, {ANCHOR['anchor_latitude']}]);
var collectionId = '{COLLECTION}';
var bands = {bands_js};
var sourceBands = {source_bands_js};
var windows = {windows_js};
var patchCrs = '{PATCH_CRS}';
var patchSizePx = {PATCH_SIZE_PX};
var patchRadiusM = {PATCH_RADIUS_M};
var region = anchor.transform(patchCrs, 1).buffer(patchRadiusM).bounds(1)
  .transform('EPSG:4326', 1);

function candidates(windowSpec) {{
  return ee.ImageCollection(collectionId)
    .filterBounds(anchor)
    .filterDate(windowSpec.start_inclusive, windowSpec.gee_end_exclusive)
    .sort('CLOUDY_PIXEL_PERCENTAGE')
    .map(function(image) {{
      return image.set('revp_window_label', windowSpec.window_label)
        .set('revp_temporal_relation_to_event', windowSpec.temporal_relation_to_event);
    }});
}}

windows.forEach(function(windowSpec) {{
  var selected = ee.Image(candidates(windowSpec).first()).select(sourceBands, bands);
  print('REV-P {STAGE} selected ' + windowSpec.window_label, selected);
  Export.image.toDrive({{
    image: selected,
    description: 'revp_{STAGE}_' + windowSpec.window_label.toLowerCase() + '_anchor_s2_patch',
    folder: 'revp_{STAGE}_manual_download',
    fileNamePrefix: 'revp_{STAGE}_' + windowSpec.window_label.toLowerCase() + '_anchor_centered_s2_patch',
    region: region,
    crs: patchCrs,
    dimensions: patchSizePx + 'x' + patchSizePx,
    maxPixels: 1000000
  }});
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
GEE_SOURCE_BANDS = {GEE_SOURCE_BANDS!r}
PATCH_CRS = "{PATCH_CRS}"
PATCH_SIZE_PX = {PATCH_SIZE_PX}
PATCH_RADIUS_M = {PATCH_RADIUS_M}
WINDOWS = {WINDOWS!r}
OUT_DIR = pathlib.Path("local_runs/protocolo_c/{STAGE}/manual_export_download")

ee.Initialize()
anchor = ee.Geometry.Point([ANCHOR_LON, ANCHOR_LAT])
region = anchor.transform(PATCH_CRS, 1).buffer(PATCH_RADIUS_M).bounds(1).transform("EPSG:4326", 1)

for window in WINDOWS:
    collection = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(anchor)
        .filterDate(window["start_inclusive"], window["gee_end_exclusive"])
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    selected = ee.Image(collection.first()).select(GEE_SOURCE_BANDS, BANDS)
    name = f"revp_{STAGE}_{{window['window_label'].lower()}}_anchor_centered_s2_patch"
    url = selected.getDownloadURL({{
        "name": name,
        "bands": BANDS,
        "region": region,
        "crs": PATCH_CRS,
        "dimensions": f"{{PATCH_SIZE_PX}}x{{PATCH_SIZE_PX}}",
        "format": "GEO_TIFF",
    }})
    target_dir = OUT_DIR / window["window_label"].lower()
    target_dir.mkdir(parents=True, exist_ok=True)
    download_path = target_dir / f"{{name}}.download"
    urllib.request.urlretrieve(url, download_path)
    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as zf:
            zf.extractall(target_dir)
    else:
        download_path.rename(target_dir / f"{{name}}.tif")
    print(f"Downloaded {{window['window_label']}} to {{target_dir}}")
'''


def write_export_plans() -> None:
    (LOCAL_RUN_DIR / "v1ix_gee_export_plan.js").write_text(build_plan_js(), encoding="utf-8")
    (LOCAL_RUN_DIR / "v1ix_gee_export_plan.py").write_text(build_plan_py(), encoding="utf-8")


def registry_row(
    temporal_relation: str,
    blocker: str,
    minimum_evidence_needed: str,
    notes: str,
    selected: dict[str, Any] | None = None,
    qa: dict[str, Any] | None = None,
    reference_patch_id: str = "",
) -> dict[str, Any]:
    selected = selected or {}
    qa = qa or {}
    qa_pass = blocker == "NONE"
    return {
        "reference_patch_id": reference_patch_id if qa_pass else "",
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
        "temporal_relation_to_event": selected.get("temporal_relation_to_event", temporal_relation),
        "bands_available": qa.get("bands_present", selected.get("bands_available", "")),
        "patch_generated_locally": bool_text(qa_pass),
        "patch_centered_on_anchor": bool_text(qa_pass),
        "center_error_m": qa.get("center_error_m", ""),
        "patch_size_px": PATCH_SIZE_PX,
        "patch_size_m": PATCH_SIZE_M,
        "crs": qa.get("crs", ""),
        "resolution_m": qa.get("resolution_m", ""),
        "valid_pixel_fraction": qa.get("valid_pixel_fraction", ""),
        "cloud_cover_metadata": selected.get("cloud_cover_metadata", ""),
        "public_versioning_status": "METADATA_ONLY",
        "can_be_reference_patch_candidate": bool_text(qa_pass),
        "can_be_ground_reference_candidate": bool_text(qa_pass),
        "can_be_operational_ground_truth": bool_text(False),
        "can_create_training_label": bool_text(False),
        "can_train_model": bool_text(False),
        "can_reopen_protocol_b": bool_text(False),
        "primary_blocker": blocker,
        "minimum_evidence_needed": minimum_evidence_needed,
        "notes": notes,
    }


def write_blocked_outputs(blocker: str, availability: dict[str, Any]) -> None:
    search_rows = [
        {
            "search_status": "SKIPPED_GEE_AUTH_REQUIRED",
            "window_label": window["window_label"],
            "temporal_relation_to_event": window["temporal_relation_to_event"],
            "gee_collection": COLLECTION,
            "bands_available": ",".join(BANDS),
            "notes": availability.get("error_type", blocker),
        }
        for window in WINDOWS
    ]
    write_csv(LOCAL_RUN_DIR / "v1ix_sentinel2_scene_search.csv", search_rows, SCENE_FIELDS)

    decision_rows = [
        {
            "decision_status": "NO_SCENE_SELECTED",
            "window_label": window["window_label"],
            "temporal_relation_to_event": window["temporal_relation_to_event"],
            "selection_reason": blocker,
            "scene_id_sanitized": "",
            "scene_date": "",
            "cloud_cover_metadata": "",
            "bands_available": "",
            "notes": "Run earthengine authenticate before Sentinel-2 search and local export.",
        }
        for window in WINDOWS
    ]
    write_csv(LOCAL_RUN_DIR / "v1ix_selected_scene_decision.csv", decision_rows, list(decision_rows[0].keys()))

    manifest_rows = [
        registry_row(
            window["temporal_relation_to_event"],
            blocker,
            "Authenticate Google Earth Engine and rerun v1ix.",
            "No Sentinel-2 scene or raster patch was generated in this run.",
        )
        for window in WINDOWS
    ]
    write_csv(LOCAL_RUN_DIR / "v1ix_anchor_patch_manifest_local.csv", manifest_rows, REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1ix_reference_patch_pair_readiness.csv", manifest_rows, REGISTRY_FIELDS)

    qa_rows = [
        {"check": "gee_available", "status": "FAIL", "detail": blocker},
        {"check": "patch_pre_generated", "status": "FAIL", "detail": "No patch generated."},
        {"check": "patch_post_generated", "status": "FAIL", "detail": "No patch generated."},
        {"check": "bands_present", "status": "NOT_RUN", "detail": ",".join(BANDS)},
        {"check": "center_correct", "status": "NOT_RUN", "detail": "No patch generated."},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "No public v1ix output written."},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
    ]
    write_csv(LOCAL_RUN_DIR / "v1ix_qa.csv", qa_rows, ["check", "status", "detail"])
    write_csv(LOCAL_RUN_DIR / "v1ix_patch_quality_audit.csv", qa_rows, ["check", "status", "detail"])
    write_csv(
        LOCAL_RUN_DIR / "v1ix_patch_download_log.csv",
        [{"window_label": window["window_label"], "status": "SKIPPED", "detail": blocker} for window in WINDOWS],
        ["window_label", "status", "detail"],
    )
    write_json(
        LOCAL_RUN_DIR / "v1ix_summary.json",
        {
            "stage": STAGE,
            "timestamp": utc_now(),
            "status": blocker,
            "earthengine_api_installed": availability.get("earthengine_api_installed", False),
            "earthengine_api_version": availability.get("earthengine_api_version", ""),
            "gee_authenticated": availability.get("gee_authenticated", False),
            "gee_available": availability.get("gee_available", False),
            "authentication_instruction": "earthengine authenticate",
            "pre_sentinel_scenes_found": 0,
            "post_sentinel_scenes_found": 0,
            "selected_pre_scene_id_sanitized": "",
            "selected_post_scene_id_sanitized": "",
            "pre_patch_generated": False,
            "post_patch_generated": False,
            "bands_present": [],
            "patch_qa_status": "NOT_RUN",
            "can_be_reference_patch_candidate": False,
            "can_be_operational_ground_truth": False,
            "can_create_training_label": False,
            "can_train_model": False,
            "can_reopen_protocol_b": False,
            "primary_blocker": blocker,
            "commit_warranted": False,
        },
    )


def feature_to_scene_row(feature: dict[str, Any], window: dict[str, str]) -> dict[str, Any]:
    props = feature.get("properties", {})
    scene_id = feature.get("id", props.get("system:index", ""))
    scene_date = date_from_millis(props.get("system:time_start"))
    return {
        "search_status": "FOUND",
        "window_label": window["window_label"],
        "temporal_relation_to_event": window["temporal_relation_to_event"],
        "gee_collection": COLLECTION,
        "scene_id_sanitized": clean_scene_id(str(scene_id)),
        "scene_date": scene_date,
        "cloud_cover_metadata": props.get("CLOUDY_PIXEL_PERCENTAGE", ""),
        "mgrs_tile": props.get("MGRS_TILE", ""),
        "processing_baseline": props.get("PROCESSING_BASELINE", ""),
        "product_id_sanitized": props.get("PRODUCT_ID", ""),
        "bands_available": ",".join(BANDS),
        "date_delta_days_from_event": delta_days(scene_date, EVENT_DATE),
        "date_delta_days_from_survey": delta_days(scene_date, SURVEY_DATE),
        "notes": "Candidate returned by GEE and sorted within its window.",
    }


def search_scenes(ee: Any) -> list[dict[str, Any]]:
    point = ee.Geometry.Point([ANCHOR["anchor_longitude"], ANCHOR["anchor_latitude"]])
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        collection = (
            ee.ImageCollection(COLLECTION)
            .filterBounds(point)
            .filterDate(window["start_inclusive"], window["gee_end_exclusive"])
            .sort("CLOUDY_PIXEL_PERCENTAGE")
        )
        info = collection.limit(30).getInfo()
        features = info.get("features", [])
        rows.extend(feature_to_scene_row(feature, window) for feature in features)
    rows.sort(
        key=lambda row: (
            row["window_label"],
            float(row["cloud_cover_metadata"]) if row["cloud_cover_metadata"] != "" else 9999.0,
            row["scene_date"],
        )
    )
    return rows


def select_best_by_window(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for window in WINDOWS:
        candidates = [row for row in rows if row["window_label"] == window["window_label"]]
        if not candidates:
            continue
        selected[window["window_label"]] = sorted(
            candidates,
            key=lambda row: (
                float(row["cloud_cover_metadata"]) if row["cloud_cover_metadata"] != "" else 9999.0,
                abs(int(row["date_delta_days_from_event"])) if row["date_delta_days_from_event"] else 9999,
                row["scene_date"],
            ),
        )[0]
    return selected


def write_selected_decision(selected_by_window: dict[str, dict[str, Any]], scene_rows: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        selected = selected_by_window.get(window["window_label"])
        count = len([row for row in scene_rows if row["window_label"] == window["window_label"]])
        if selected is None:
            rows.append(
                {
                    "decision_status": "NO_SCENE_SELECTED",
                    "window_label": window["window_label"],
                    "temporal_relation_to_event": window["temporal_relation_to_event"],
                    "selection_reason": "NO_SENTINEL2_SR_HARMONIZED_SCENE_FOUND_FOR_WINDOW",
                    "scene_id_sanitized": "",
                    "scene_date": "",
                    "cloud_cover_metadata": "",
                    "bands_available": "",
                    "notes": f"scenes_found={count}",
                }
            )
        else:
            rows.append(
                {
                    "decision_status": "SCENE_SELECTED",
                    "window_label": window["window_label"],
                    "temporal_relation_to_event": selected["temporal_relation_to_event"],
                    "selection_reason": "Lowest CLOUDY_PIXEL_PERCENTAGE in requested temporal window.",
                    "scene_id_sanitized": selected["scene_id_sanitized"],
                    "scene_date": selected["scene_date"],
                    "cloud_cover_metadata": selected["cloud_cover_metadata"],
                    "bands_available": selected["bands_available"],
                    "notes": f"scenes_found={count}",
                }
            )
    write_csv(LOCAL_RUN_DIR / "v1ix_selected_scene_decision.csv", rows, list(rows[0].keys()))


def _collect_tif_extensions(target_dir: Path) -> list[Path]:
    return sorted(target_dir.glob("*.tif")) + sorted(target_dir.glob("*.tiff"))


def _collect_local_rasters(target_dir: Path) -> list[Path]:
    return sorted(target_dir.glob(f"*{LOCAL_RASTER_SUFFIX}"))


def download_patch(ee: Any, selected: dict[str, Any]) -> Path:
    point = ee.Geometry.Point([ANCHOR["anchor_longitude"], ANCHOR["anchor_latitude"]])
    region = point.transform(PATCH_CRS, 1).buffer(PATCH_RADIUS_M).bounds(1).transform("EPSG:4326", 1)
    image = ee.Image(f"{COLLECTION}/{selected['scene_id_sanitized']}").select(GEE_SOURCE_BANDS, BANDS)
    safe_relation = selected["temporal_relation_to_event"].lower()
    name = f"revp_{STAGE}_{safe_relation}_anchor_centered_s2_patch"
    target_dir = PATCH_DIR / safe_relation
    target_dir.mkdir(parents=True, exist_ok=True)
    url = image.getDownloadURL(
        {
            "name": name,
            "bands": BANDS,
            "region": region,
            "crs": PATCH_CRS,
            "dimensions": f"{PATCH_SIZE_PX}x{PATCH_SIZE_PX}",
            "format": "GEO_TIFF",
        }
    )
    download_path = target_dir / f"{name}.download"
    urllib.request.urlretrieve(url, download_path)
    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as archive:
            archive.extractall(target_dir)
        for tif in _collect_tif_extensions(target_dir):
            tif.replace(tif.with_suffix(LOCAL_RASTER_SUFFIX))
    else:
        download_path.replace(target_dir / f"{name}{LOCAL_RASTER_SUFFIX}")
    rasters = _collect_local_rasters(target_dir)
    if not rasters:
        raise RuntimeError("GEE download completed but no local raster was found.")
    return rasters[0]


def qa_patch(tif_path: Path, selected: dict[str, Any]) -> tuple[str, dict[str, Any], list[dict[str, str]]]:
    qa: dict[str, Any] = {
        "local_private_path": str(tif_path),
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
        rows.append({"check": f"{selected['window_label']}_rasterio_numpy_available", "status": "FAIL", "detail": str(exc)})
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
        float_data = data.astype("float64")
        finite_mask = np.isfinite(np.ma.filled(float_data, np.nan))
        valid_denominator = data.size if data.size else 1
        qa["valid_pixel_fraction"] = f"{float(np.count_nonzero(valid_mask & finite_mask) / valid_denominator):.6f}"
        qa["has_nan_inf"] = bool_text(not bool(np.all(finite_mask[valid_mask])))

    prefix = selected["window_label"]

    def add(check: str, ok: bool, detail: str) -> None:
        rows.append({"check": f"{prefix}_{check}", "status": "PASS" if ok else "FAIL", "detail": detail})

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
    return ("PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL"), qa, rows


def write_schema() -> None:
    write_csv(
        SCHEMAS_DIR / "official_anchor_sentinel_patch_schema.csv",
        [{"field": field, "description": f"REV-P {STAGE} official anchor Sentinel patch metadata field."} for field in REGISTRY_FIELDS],
        ["field", "description"],
    )


def write_public_registry(rows: list[dict[str, Any]]) -> None:
    write_csv(DATASETS_DIR / "official_anchor_sentinel_patch_registry.csv", rows, REGISTRY_FIELDS)
    write_schema()


def run_available_flow(args: argparse.Namespace, availability: dict[str, Any]) -> None:
    ee = importlib.import_module("ee")
    scene_rows = search_scenes(ee) if args.search_sentinel2 else []
    write_csv(LOCAL_RUN_DIR / "v1ix_sentinel2_scene_search.csv", scene_rows, SCENE_FIELDS)
    selected_by_window = select_best_by_window(scene_rows)
    write_selected_decision(selected_by_window, scene_rows)

    download_logs: list[dict[str, Any]] = []
    qa_rows: list[dict[str, str]] = []
    manifest_rows: list[dict[str, Any]] = []
    patch_paths: dict[str, Path] = {}
    qa_by_window: dict[str, dict[str, Any]] = {}
    qa_status_by_window: dict[str, str] = {}

    for window in WINDOWS:
        selected = selected_by_window.get(window["window_label"])
        relation = window["temporal_relation_to_event"]
        if selected is None:
            row = registry_row(
                relation,
                "NO_SENTINEL2_SR_HARMONIZED_SCENE_FOUND_FOR_WINDOW",
                "At least one Sentinel-2 SR Harmonized scene in this temporal window.",
                "No local patch was generated for this temporal window.",
            )
            manifest_rows.append(row)
            continue
        if not args.download_small_patches:
            row = registry_row(
                relation,
                "LOCAL_DOWNLOAD_NOT_REQUESTED",
                "Rerun with --download-small-patches.",
                "Scene metadata exists, but local patch export was not requested.",
                selected=selected,
            )
            manifest_rows.append(row)
            download_logs.append({"window_label": window["window_label"], "status": "NOT_RUN", "detail": row["primary_blocker"]})
            continue
        try:
            tif_path = download_patch(ee, selected)
            status, qa, rows = qa_patch(tif_path, selected)
            patch_paths[window["window_label"]] = tif_path
            qa_by_window[window["window_label"]] = qa
            qa_status_by_window[window["window_label"]] = status
            qa_rows.extend(rows)
            blocker = "NONE" if status == "PASS" else "PATCH_QA_FAILED"
            reference_patch_id = f"REFPATCH_PET2022_CPRM_MOINHO_PRETO_S2_{STAGE.upper()}_{window['window_label']}"
            manifest_rows.append(
                registry_row(
                    relation,
                    blocker,
                    "Independent interpretation remains required before any later claim." if blocker == "NONE" else "Successful local Sentinel-2 patch export with QA PASS.",
                    "Patch exists locally with QA PASS; no label or target is created." if blocker == "NONE" else "Patch was downloaded but did not pass QA.",
                    selected=selected,
                    qa=qa,
                    reference_patch_id=reference_patch_id,
                )
            )
            download_logs.append({"window_label": window["window_label"], "status": "DOWNLOADED", "detail": tif_path.name})
        except Exception as exc:
            qa_status_by_window[window["window_label"]] = "NOT_RUN"
            qa_rows.append({"check": f"{window['window_label']}_local_download", "status": "FAIL", "detail": f"{type(exc).__name__}: {exc}"})
            manifest_rows.append(
                registry_row(
                    relation,
                    "EXPORT_TASK_REQUIRED",
                    "Use the generated GEE JS/PY export plan or rerun after resolving direct download failure.",
                    "Scene was selected, but direct small-patch download did not complete.",
                    selected=selected,
                )
            )
            download_logs.append({"window_label": window["window_label"], "status": "FAILED", "detail": f"{type(exc).__name__}: {exc}"})

    qa_rows.extend(
        [
            {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "Public outputs use sanitized metadata only."},
            {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
            {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
            {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
        ]
    )

    write_csv(LOCAL_RUN_DIR / "v1ix_patch_download_log.csv", download_logs, ["window_label", "status", "detail"])
    write_csv(LOCAL_RUN_DIR / "v1ix_qa.csv", qa_rows, ["check", "status", "detail"])
    write_csv(LOCAL_RUN_DIR / "v1ix_patch_quality_audit.csv", qa_rows, ["check", "status", "detail"])
    write_csv(LOCAL_RUN_DIR / "v1ix_anchor_patch_manifest_local.csv", manifest_rows, REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1ix_reference_patch_pair_readiness.csv", manifest_rows, REGISTRY_FIELDS)

    pre_generated = "PRE_EVENT" in patch_paths
    post_generated = "POST_EVENT_SURVEY" in patch_paths
    all_downloaded = pre_generated and post_generated
    all_qa_pass = all(qa_status_by_window.get(window["window_label"]) == "PASS" for window in WINDOWS)
    if all_downloaded and all_qa_pass:
        write_public_registry(manifest_rows)

    pre_count = len([row for row in scene_rows if row["window_label"] == "PRE_EVENT"])
    post_count = len([row for row in scene_rows if row["window_label"] == "POST_EVENT_SURVEY"])
    primary_blockers = sorted({row["primary_blocker"] for row in manifest_rows if row["primary_blocker"] != "NONE"})
    status = "PATCH_PAIR_QA_PASS" if all_downloaded and all_qa_pass else (primary_blockers[0] if primary_blockers else "PATCH_QA_FAILED")
    bands_present = sorted({band for qa in qa_by_window.values() for band in qa.get("bands_present", "").split(",") if band})
    write_json(
        LOCAL_RUN_DIR / "v1ix_summary.json",
        {
            "stage": STAGE,
            "timestamp": utc_now(),
            "status": status,
            "earthengine_api_installed": availability.get("earthengine_api_installed", False),
            "earthengine_api_version": availability.get("earthengine_api_version", ""),
            "gee_authenticated": availability.get("gee_authenticated", False),
            "gee_available": availability.get("gee_available", False),
            "pre_sentinel_scenes_found": pre_count,
            "post_sentinel_scenes_found": post_count,
            "selected_pre_scene_id_sanitized": selected_by_window.get("PRE_EVENT", {}).get("scene_id_sanitized", ""),
            "selected_post_scene_id_sanitized": selected_by_window.get("POST_EVENT_SURVEY", {}).get("scene_id_sanitized", ""),
            "pre_patch_generated": pre_generated,
            "post_patch_generated": post_generated,
            "bands_present": bands_present,
            "patch_qa_status": "PASS" if all_downloaded and all_qa_pass else ("FAIL" if any(qa_status_by_window.values()) else "NOT_RUN"),
            "can_be_reference_patch_candidate": all_downloaded and all_qa_pass,
            "can_be_operational_ground_truth": False,
            "can_create_training_label": False,
            "can_train_model": False,
            "can_reopen_protocol_b": False,
            "public_versioning_status": "METADATA_ONLY",
            "primary_blocker": "NONE" if all_downloaded and all_qa_pass else status,
            "commit_warranted": all_downloaded and all_qa_pass,
        },
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Clear local v1ix outputs before running.")
    parser.add_argument("--check-gee", action="store_true", help="Check earthengine-api availability.")
    parser.add_argument("--authenticate-check", action="store_true", help="Check ee.Initialize authentication.")
    parser.add_argument("--search-sentinel2", action="store_true", help="Search Sentinel-2 SR Harmonized scenes.")
    parser.add_argument("--export-pre-post-patches", action="store_true", help="Write reproducible pre/post export plans.")
    parser.add_argument("--download-small-patches", action="store_true", help="Try direct small patch download.")
    parser.add_argument("--emit-qa", action="store_true", help="Emit QA outputs.")
    parser.add_argument("--emit-manifest", action="store_true", help="Emit local manifest and readiness outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    prepare_output_dir(args.force)
    if args.export_pre_post_patches:
        write_export_plans()

    availability, ee = check_gee()
    write_json(LOCAL_RUN_DIR / "v1ix_gee_environment_check.json", availability)
    if not availability.get("gee_available") or ee is None:
        if args.export_pre_post_patches:
            write_export_plans()
        write_blocked_outputs("GEE_AUTH_REQUIRED", availability)
    else:
        if args.export_pre_post_patches:
            write_export_plans()
        run_available_flow(args, availability)

    summary = json.loads((LOCAL_RUN_DIR / "v1ix_summary.json").read_text(encoding="utf-8"))
    print("=" * 72)
    print("REV-P v1ix GEE_SENTINEL_PATCH_EXPORT_FOR_OFFICIAL_ANCHOR")
    print("=" * 72)
    print(f"earthengine-api installed: {summary.get('earthengine_api_installed')}")
    print(f"GEE authenticated: {summary.get('gee_authenticated')}")
    print(f"Pre scenes found: {summary.get('pre_sentinel_scenes_found')}")
    print(f"Post scenes found: {summary.get('post_sentinel_scenes_found')}")
    print(f"Selected pre scene: {summary.get('selected_pre_scene_id_sanitized', '')}")
    print(f"Selected post scene: {summary.get('selected_post_scene_id_sanitized', '')}")
    print(f"Pre patch generated: {summary.get('pre_patch_generated')}")
    print(f"Post patch generated: {summary.get('post_patch_generated')}")
    print(f"Bands present: {summary.get('bands_present')}")
    print(f"Patch QA: {summary.get('patch_qa_status')}")
    print(f"Reference patch candidate: {summary.get('can_be_reference_patch_candidate')}")
    print(f"Primary blocker: {summary.get('primary_blocker')}")
    if summary.get("primary_blocker") == "GEE_AUTH_REQUIRED":
        print("Authenticate with: earthengine authenticate")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
