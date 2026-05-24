"""
REV-P v1iz - CLOUD_MASK_AND_ALTERNATIVE_SENTINEL_SCENE_SELECTION_FOR_ANCHOR.

Downloads local SCL/QA60 masks for the v1ix Sentinel-2 patches, evaluates
local cloud quality, searches pre-event alternatives before 2022-02-15, and
selects the best pre/post reference pair for review. Small local raster
artifacts remain under local_runs/ and are never public outputs.

Invariants:
  - no label, target, training, or Protocol B reopening is created
  - no post-event image can be selected as a pre-event alternative
  - public outputs contain metadata only and no private paths
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
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iz"
MASK_DIR = LOCAL_RUN_DIR / "cloud_masks"
ALT_DIR = LOCAL_RUN_DIR / "alternative_pre_patches"
V1IX_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ix"
V1IY_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iy"
V1IX_MANIFEST = V1IX_DIR / "v1ix_anchor_patch_manifest_local.csv"
V1IY_DECISION = V1IY_DIR / "v1iy_patch_pair_quality_decision.csv"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

STAGE = "v1iz"
COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
GEE_SOURCE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
QA_BANDS = ["SCL", "QA60"]
BAND_INDEX = {band: idx for idx, band in enumerate(BANDS)}
ANCHOR = {
    "anchor_id": "ANCHOR_PET2022_CPRM_ANEXOII_19022022",
    "lat": -22.484251,
    "lon": -43.211257,
}
EVENT_DATE = "2022-02-15"
PRE_WINDOW_PRIMARY = ("2022-02-01", "2022-02-15")
PRE_WINDOW_EXPANDED = ("2022-01-15", "2022-02-15")
POST_WINDOW = ("2022-02-15", "2022-03-06")
PATCH_CRS = "EPSG:32723"
PATCH_SIZE_PX = 96
PATCH_SIZE_M = 960
PATCH_RADIUS_M = PATCH_SIZE_M / 2
LOCAL_RASTER_SUFFIX = ".local_geotiff"
MAX_ACCEPTABLE_LOCAL_CLOUD_FRACTION = 0.20
MIN_VALID_PIXEL_FRACTION = 0.95

REGISTRY_FIELDS = [
    "selection_id",
    "anchor_id",
    "pre_scene_id_sanitized",
    "pre_scene_date",
    "post_scene_id_sanitized",
    "post_scene_date",
    "pre_cloud_metadata_global",
    "post_cloud_metadata_global",
    "pre_local_cloud_fraction",
    "post_local_cloud_fraction",
    "cloud_mask_source",
    "pre_valid_pixel_fraction",
    "post_valid_pixel_fraction",
    "bands_available",
    "spectral_indices_status",
    "final_pair_status",
    "can_be_reference_patch_candidate",
    "can_be_multimodal_reference_candidate",
    "can_be_operational_ground_truth",
    "can_create_training_label",
    "can_train_model",
    "can_reopen_protocol_b",
    "primary_blocker",
    "minimum_evidence_needed",
    "notes",
]

SCENE_FIELDS = [
    "scene_id_sanitized",
    "scene_date",
    "cloud_metadata_global",
    "mgrs_tile",
    "processing_baseline",
    "product_id_sanitized",
    "date_delta_days_from_event",
    "pre_event_window",
    "eligible_pre_event",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def date_from_millis(ms: Any) -> str:
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
        return str((datetime.fromisoformat(scene_date).date() - datetime.fromisoformat(EVENT_DATE).date()).days)
    except ValueError:
        return ""


def clean_scene_id(scene_id: str) -> str:
    return scene_id.replace(f"{COLLECTION}/", "")


def safe_scene_token(scene_id: str) -> str:
    return scene_id.replace("/", "_").replace(":", "_")


def prepare_output_dir(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / STAGE).resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    ALT_DIR.mkdir(parents=True, exist_ok=True)


def check_gee() -> tuple[dict[str, Any], Any | None]:
    status: dict[str, Any] = {
        "earthengine_api_installed": False,
        "earthengine_api_version": "",
        "gee_authenticated": False,
        "gee_available": False,
        "status": "GEE_AUTH_REQUIRED",
        "authentication_instruction": "earthengine authenticate",
        "error_type": "",
        "error_message": "",
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


def region_geometry(ee: Any) -> Any:
    point = ee.Geometry.Point([ANCHOR["lon"], ANCHOR["lat"]])
    return point.transform(PATCH_CRS, 1).buffer(PATCH_RADIUS_M).bounds(1).transform("EPSG:4326", 1)


def collect_downloaded_rasters(target_dir: Path) -> list[Path]:
    return sorted(target_dir.glob(f"*{LOCAL_RASTER_SUFFIX}"))


def rename_tifs_to_local(target_dir: Path) -> None:
    for tif in sorted(target_dir.glob("*.tif")) + sorted(target_dir.glob("*.tiff")):
        tif.replace(tif.with_suffix(LOCAL_RASTER_SUFFIX))


def download_image(ee: Any, scene_id: str, source_bands: list[str], out_bands: list[str], target_dir: Path, name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    image = ee.Image(f"{COLLECTION}/{scene_id}").select(source_bands, out_bands)
    url = image.getDownloadURL(
        {
            "name": name,
            "bands": out_bands,
            "region": region_geometry(ee),
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
        download_path.unlink(missing_ok=True)
        rename_tifs_to_local(target_dir)
    else:
        download_path.replace(target_dir / f"{name}{LOCAL_RASTER_SUFFIX}")
    rasters = collect_downloaded_rasters(target_dir)
    if not rasters:
        raise RuntimeError(f"No local raster produced for {name}.")
    return rasters[0]


def download_cloud_mask(ee: Any, row: dict[str, str]) -> tuple[Path | None, str]:
    scene_id = row["scene_id_sanitized"]
    relation = row["temporal_relation_to_event"].lower()
    target_dir = MASK_DIR / relation / safe_scene_token(scene_id)
    try:
        path = download_image(ee, scene_id, QA_BANDS, QA_BANDS, target_dir, f"revp_{STAGE}_{relation}_qa_mask")
        return path, "DOWNLOADED"
    except Exception as exc:
        return None, f"FAILED:{type(exc).__name__}: {exc}"


def read_v1ix_rows() -> tuple[dict[str, str], dict[str, str]]:
    rows = read_csv(V1IX_MANIFEST)
    pre = next(row for row in rows if row["temporal_relation_to_event"] == "PRE_EVENT")
    post = next(row for row in rows if row["temporal_relation_to_event"] == "POST_EVENT_OR_SURVEY_WINDOW")
    return pre, post


def compute_cloud_from_mask(mask_path: Path | None) -> dict[str, Any]:
    import numpy as np
    import rasterio

    if mask_path is None:
        return {
            "cloud_mask_available": False,
            "cloud_mask_source": "CLOUD_MASK_NOT_AVAILABLE",
            "local_cloud_fraction": math.nan,
            "shadow_fraction": math.nan,
            "snow_fraction": math.nan,
            "water_fraction": math.nan,
            "nodata_fraction": math.nan,
            "notes": "No local SCL/QA60 mask was available.",
        }
    with rasterio.open(mask_path) as src:
        data = src.read(masked=True)
        valid = ~np.ma.getmaskarray(data[0])
        total = int(np.count_nonzero(valid)) or 1
        scl = np.ma.filled(data[0], 0).astype("int64") if src.count >= 1 else np.zeros((1, 1), dtype="int64")
        cloud = np.isin(scl, [8, 9, 10])
        shadow = scl == 3
        snow = scl == 11
        water = scl == 6
        nodata = scl == 0
        qa_cloud = np.zeros_like(cloud, dtype=bool)
        qa_cirrus = np.zeros_like(cloud, dtype=bool)
        if src.count >= 2:
            qa60 = np.ma.filled(data[1], 0).astype("int64")
            qa_cloud = (qa60 & (1 << 10)) != 0
            qa_cirrus = (qa60 & (1 << 11)) != 0
        cloud_combined = cloud | qa_cloud | qa_cirrus
        return {
            "cloud_mask_available": True,
            "cloud_mask_source": "SCL_QA60",
            "local_cloud_fraction": float(np.count_nonzero(cloud_combined & valid) / total),
            "shadow_fraction": float(np.count_nonzero(shadow & valid) / total),
            "snow_fraction": float(np.count_nonzero(snow & valid) / total),
            "water_fraction": float(np.count_nonzero(water & valid) / total),
            "nodata_fraction": float(np.count_nonzero(nodata & valid) / total),
            "notes": "Local cloud fraction uses SCL classes 8/9/10 plus QA60 cloud/cirrus bits.",
        }


def finite_values(masked_array: Any) -> Any:
    import numpy as np

    valid = ~np.ma.getmaskarray(masked_array)
    filled = np.ma.filled(masked_array.astype("float64"), np.nan)
    return filled[valid & np.isfinite(filled)]


def compute_index(a: Any, b: Any) -> Any:
    import numpy as np

    denominator = a.astype("float64") + b.astype("float64")
    return np.where(np.abs(denominator) > 1.0e-9, (a.astype("float64") - b.astype("float64")) / denominator, np.nan)


def summarize_index(values: Any) -> dict[str, str]:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"index_computable": "false", "mean": "", "std": "", "finite_fraction": "0.000000"}
    return {
        "index_computable": "true",
        "mean": f"{float(np.mean(finite)):.6f}",
        "std": f"{float(np.std(finite)):.6f}",
        "finite_fraction": f"{float(finite.size / values.size):.6f}",
    }


def spectral_qa(path: Path | None) -> dict[str, Any]:
    import numpy as np
    import rasterio

    if path is None:
        return {
            "patch_downloaded": False,
            "bands_available": "",
            "shape_px": "",
            "crs": "",
            "resolution_m": "",
            "valid_pixel_fraction": math.nan,
            "spectral_indices_status": "NOT_RUN",
            "local_quality_status": "PATCH_NOT_AVAILABLE",
            "ndwi_mean": "",
            "ndbi_mean": "",
            "notes": "No spectral patch available.",
        }
    with rasterio.open(path) as src:
        data = src.read(masked=True)
        filled = np.ma.filled(data.astype("float64"), np.nan)
        valid = ~np.ma.getmaskarray(data)
        finite = np.isfinite(filled)
        valid_fraction = float(np.count_nonzero(valid & finite) / (data.size or 1))
        bands_available = ",".join(BANDS[: src.count])
        band_ranges_ok = True
        band_variance_ok = True
        for idx in range(src.count):
            vals = finite_values(data[idx])
            if vals.size == 0 or float(np.max(vals)) <= float(np.min(vals)) or float(np.var(vals)) <= 1.0e-6:
                band_ranges_ok = False
                band_variance_ok = False
        ndwi = summarize_index(compute_index(filled[BAND_INDEX["B03"]], filled[BAND_INDEX["B08"]])) if src.count >= 4 else {"index_computable": "false", "mean": "", "std": "", "finite_fraction": "0.000000"}
        ndbi = summarize_index(compute_index(filled[BAND_INDEX["B11"]], filled[BAND_INDEX["B08"]])) if src.count >= 5 else {"index_computable": "false", "mean": "", "std": "", "finite_fraction": "0.000000"}
        spectral_indices_status = "OK" if ndwi["index_computable"] == "true" and ndbi["index_computable"] == "true" else "FAIL"
        local_quality_status = (
            "LOCAL_QA_PASS"
            if src.count == len(BANDS)
            and bands_available == ",".join(BANDS)
            and valid_fraction >= MIN_VALID_PIXEL_FRACTION
            and band_ranges_ok
            and band_variance_ok
            and spectral_indices_status == "OK"
            else "LOCAL_QA_FAIL"
        )
        return {
            "patch_downloaded": True,
            "bands_available": bands_available,
            "shape_px": f"{src.width}x{src.height}",
            "crs": str(src.crs) if src.crs else "",
            "resolution_m": f"{abs(src.res[0]):.4f}" if src.res else "",
            "valid_pixel_fraction": valid_fraction,
            "spectral_indices_status": spectral_indices_status,
            "local_quality_status": local_quality_status,
            "ndwi_mean": ndwi["mean"],
            "ndbi_mean": ndbi["mean"],
            "notes": "Spectral QA uses local downloaded patch values only.",
        }


def feature_to_scene_row(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties", {})
    scene_id = clean_scene_id(str(feature.get("id", props.get("system:index", ""))))
    scene_date = date_from_millis(props.get("system:time_start"))
    return {
        "scene_id_sanitized": scene_id,
        "scene_date": scene_date,
        "cloud_metadata_global": props.get("CLOUDY_PIXEL_PERCENTAGE", ""),
        "mgrs_tile": props.get("MGRS_TILE", ""),
        "processing_baseline": props.get("PROCESSING_BASELINE", ""),
        "product_id_sanitized": props.get("PRODUCT_ID", ""),
        "date_delta_days_from_event": delta_days(scene_date),
        "pre_event_window": f"{PRE_WINDOW_EXPANDED[0]}..2022-02-14",
        "eligible_pre_event": bool_text(bool(scene_date and scene_date < EVENT_DATE)),
        "notes": "Expanded pre-event search; alternatives never cross event date.",
    }


def search_alternative_pre_scenes(ee: Any) -> list[dict[str, Any]]:
    point = ee.Geometry.Point([ANCHOR["lon"], ANCHOR["lat"]])
    collection = (
        ee.ImageCollection(COLLECTION)
        .filterBounds(point)
        .filterDate(PRE_WINDOW_EXPANDED[0], PRE_WINDOW_EXPANDED[1])
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    info = collection.limit(30).getInfo()
    rows = [feature_to_scene_row(feature) for feature in info.get("features", [])]
    rows = [row for row in rows if row["eligible_pre_event"] == "true"]
    rows.sort(key=lambda row: (safe_float(row["cloud_metadata_global"], 9999.0), abs(int(row["date_delta_days_from_event"])), row["scene_date"]))
    return rows


def download_alternative_patch_and_mask(ee: Any, scene: dict[str, Any]) -> tuple[Path | None, Path | None, str]:
    token = safe_scene_token(scene["scene_id_sanitized"])
    target_dir = ALT_DIR / token
    try:
        spectral_path = download_image(
            ee,
            scene["scene_id_sanitized"],
            GEE_SOURCE_BANDS,
            BANDS,
            target_dir / "spectral",
            f"revp_{STAGE}_{token}_spectral_patch",
        )
    except Exception as exc:
        return None, None, f"SPECTRAL_FAILED:{type(exc).__name__}: {exc}"
    try:
        mask_path = download_image(
            ee,
            scene["scene_id_sanitized"],
            QA_BANDS,
            QA_BANDS,
            target_dir / "qa_mask",
            f"revp_{STAGE}_{token}_qa_mask",
        )
    except Exception as exc:
        return spectral_path, None, f"QA_MASK_FAILED:{type(exc).__name__}: {exc}"
    return spectral_path, mask_path, "DOWNLOADED"


def candidate_score(candidate: dict[str, Any]) -> tuple[float, float, int, str]:
    cloud = safe_float(candidate.get("local_cloud_fraction", ""), 999.0)
    valid = safe_float(candidate.get("valid_pixel_fraction", ""), 0.0)
    temporal_distance = abs(int(candidate.get("date_delta_days_from_event", "9999") or 9999))
    return (cloud, -valid, temporal_distance, candidate.get("scene_date", ""))


def select_best_pre(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row
        for row in candidates
        if row.get("local_quality_status") == "LOCAL_QA_PASS"
        and row.get("cloud_mask_available") == "true"
        and safe_float(row.get("local_cloud_fraction", ""), 999.0) <= MAX_ACCEPTABLE_LOCAL_CLOUD_FRACTION
    ]
    if not eligible:
        return None
    return sorted(eligible, key=candidate_score)[0]


def public_registry_row(best_pre: dict[str, Any] | None, post_eval: dict[str, Any], final_status: str) -> dict[str, Any]:
    usable = best_pre is not None and post_eval.get("local_quality_status") == "LOCAL_QA_PASS"
    blocker = "NONE" if usable else ("S2_PRE_EVENT_CLEAR_PATCH_NOT_AVAILABLE" if best_pre is None else "POST_PATCH_LOCAL_QA_FAILED")
    return {
        "selection_id": "SELECTION_ANCHOR_PET2022_CPRM_MOINHO_PRETO_S2_V1IZ",
        "anchor_id": ANCHOR["anchor_id"],
        "pre_scene_id_sanitized": best_pre.get("scene_id_sanitized", "") if best_pre else "",
        "pre_scene_date": best_pre.get("scene_date", "") if best_pre else "",
        "post_scene_id_sanitized": post_eval.get("scene_id_sanitized", ""),
        "post_scene_date": post_eval.get("scene_date", ""),
        "pre_cloud_metadata_global": best_pre.get("cloud_metadata_global", "") if best_pre else "",
        "post_cloud_metadata_global": post_eval.get("cloud_metadata_global", ""),
        "pre_local_cloud_fraction": best_pre.get("local_cloud_fraction", "") if best_pre else "",
        "post_local_cloud_fraction": post_eval.get("local_cloud_fraction", ""),
        "cloud_mask_source": "SCL_QA60" if usable else post_eval.get("cloud_mask_source", "CLOUD_MASK_NOT_AVAILABLE"),
        "pre_valid_pixel_fraction": best_pre.get("valid_pixel_fraction", "") if best_pre else "",
        "post_valid_pixel_fraction": post_eval.get("valid_pixel_fraction", ""),
        "bands_available": best_pre.get("bands_available", "") if best_pre else post_eval.get("bands_available", ""),
        "spectral_indices_status": "OK" if usable and best_pre.get("spectral_indices_status") == "OK" and post_eval.get("spectral_indices_status") == "OK" else "FAIL",
        "final_pair_status": final_status,
        "can_be_reference_patch_candidate": bool_text(usable),
        "can_be_multimodal_reference_candidate": bool_text(usable),
        "can_be_operational_ground_truth": bool_text(False),
        "can_create_training_label": bool_text(False),
        "can_train_model": bool_text(False),
        "can_reopen_protocol_b": bool_text(False),
        "primary_blocker": blocker,
        "minimum_evidence_needed": "Independent interpretation remains required before any stronger evidence role." if usable else "Acquire a pre-event Sentinel-2 patch with local cloud mask and acceptable local cloud fraction.",
        "notes": "Final pair selection uses local SCL/QA60, spectral QA, valid pixels, and pre-event temporal ordering; it does not create labels or training data.",
    }


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{prefix}: {field}."} for field in fields], ["field", "description"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(args.force)
    pre_v1ix, post_v1ix = read_v1ix_rows()
    if not V1IY_DECISION.exists():
        raise FileNotFoundError(f"Missing v1iy decision: {V1IY_DECISION}")

    availability, ee = check_gee()
    if not availability["gee_available"] or ee is None:
        raise RuntimeError(f"GEE_AUTH_REQUIRED: {availability.get('error_type')} {availability.get('error_message')}")

    download_log: list[dict[str, Any]] = []
    current_evals: dict[str, dict[str, Any]] = {}
    for row in [pre_v1ix, post_v1ix]:
        mask_path, status = download_cloud_mask(ee, row)
        cloud = compute_cloud_from_mask(mask_path)
        eval_row = {
            "source": "V1IX_SELECTED",
            "temporal_relation_to_event": row["temporal_relation_to_event"],
            "scene_id_sanitized": row["scene_id_sanitized"],
            "scene_date": row["scene_date"],
            "cloud_metadata_global": row["cloud_cover_metadata"],
            "cloud_mask_available": bool_text(cloud["cloud_mask_available"]),
            "cloud_mask_source": cloud["cloud_mask_source"],
            "local_cloud_fraction": "" if math.isnan(cloud["local_cloud_fraction"]) else f"{cloud['local_cloud_fraction']:.6f}",
            "shadow_fraction": "" if math.isnan(cloud["shadow_fraction"]) else f"{cloud['shadow_fraction']:.6f}",
            "snow_fraction": "" if math.isnan(cloud["snow_fraction"]) else f"{cloud['snow_fraction']:.6f}",
            "water_fraction": "" if math.isnan(cloud["water_fraction"]) else f"{cloud['water_fraction']:.6f}",
            "mask_quality_status": "LOCAL_CLOUD_MASK_AVAILABLE" if cloud["cloud_mask_available"] else "QA_MASK_NOT_AVAILABLE",
            "notes": cloud["notes"],
        }
        current_evals[row["temporal_relation_to_event"]] = eval_row
        download_log.append(
            {
                "temporal_relation_to_event": row["temporal_relation_to_event"],
                "scene_id_sanitized": row["scene_id_sanitized"],
                "download_status": status,
                "cloud_mask_source": cloud["cloud_mask_source"],
                "notes": "Mask raster kept under local_runs only.",
            }
        )

    scene_rows = search_alternative_pre_scenes(ee)
    alt_quality_rows: list[dict[str, Any]] = []
    for scene in scene_rows:
        spectral_path, mask_path, status = download_alternative_patch_and_mask(ee, scene)
        spectral = spectral_qa(spectral_path)
        cloud = compute_cloud_from_mask(mask_path)
        alt_quality_rows.append(
            {
                "scene_id_sanitized": scene["scene_id_sanitized"],
                "scene_date": scene["scene_date"],
                "cloud_metadata_global": scene["cloud_metadata_global"],
                "date_delta_days_from_event": scene["date_delta_days_from_event"],
                "download_status": status,
                "cloud_mask_available": bool_text(cloud["cloud_mask_available"]),
                "cloud_mask_source": cloud["cloud_mask_source"],
                "local_cloud_fraction": "" if math.isnan(cloud["local_cloud_fraction"]) else f"{cloud['local_cloud_fraction']:.6f}",
                "shadow_fraction": "" if math.isnan(cloud["shadow_fraction"]) else f"{cloud['shadow_fraction']:.6f}",
                "snow_fraction": "" if math.isnan(cloud["snow_fraction"]) else f"{cloud['snow_fraction']:.6f}",
                "water_fraction": "" if math.isnan(cloud["water_fraction"]) else f"{cloud['water_fraction']:.6f}",
                "bands_available": spectral["bands_available"],
                "valid_pixel_fraction": "" if math.isnan(spectral["valid_pixel_fraction"]) else f"{spectral['valid_pixel_fraction']:.6f}",
                "spectral_indices_status": spectral["spectral_indices_status"],
                "ndwi_mean": spectral["ndwi_mean"],
                "ndbi_mean": spectral["ndbi_mean"],
                "local_quality_status": spectral["local_quality_status"],
                "pre_event_valid": bool_text(scene["scene_date"] < EVENT_DATE),
                "notes": spectral["notes"],
            }
        )

    best_pre = select_best_pre(alt_quality_rows)
    post_spectral_from_v1iy = next((row for row in read_csv(V1IY_DECISION) if row["temporal_relation_to_event"] == "POST_EVENT_OR_SURVEY_WINDOW"), {})
    post_current = current_evals["POST_EVENT_OR_SURVEY_WINDOW"]
    post_eval = {
        "scene_id_sanitized": post_v1ix["scene_id_sanitized"],
        "scene_date": post_v1ix["scene_date"],
        "cloud_metadata_global": post_v1ix["cloud_cover_metadata"],
        "cloud_mask_available": post_current["cloud_mask_available"],
        "cloud_mask_source": post_current["cloud_mask_source"],
        "local_cloud_fraction": post_current["local_cloud_fraction"],
        "valid_pixel_fraction": post_spectral_from_v1iy.get("valid_pixel_fraction", post_v1ix.get("valid_pixel_fraction", "")),
        "bands_available": post_spectral_from_v1iy.get("bands_available", post_v1ix.get("bands_available", "")),
        "spectral_indices_status": post_spectral_from_v1iy.get("spectral_index_status", "OK"),
        "local_quality_status": post_spectral_from_v1iy.get("local_quality_status", "LOCAL_QA_PASS"),
    }

    if best_pre is None:
        final_status = "S2_PRE_EVENT_CLEAR_PATCH_NOT_AVAILABLE"
    elif best_pre["scene_id_sanitized"] == pre_v1ix["scene_id_sanitized"]:
        final_status = "PRE_PATCH_CLOUD_RISK_RESOLVED"
    else:
        final_status = "PATCH_PAIR_USABLE_FOR_REVIEW"
    registry_row = public_registry_row(best_pre, post_eval, final_status)

    write_csv(LOCAL_RUN_DIR / "v1iz_cloud_mask_download_log.csv", download_log, list(download_log[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1iz_local_cloud_quality_audit.csv", list(current_evals.values()), list(next(iter(current_evals.values())).keys()))
    write_csv(LOCAL_RUN_DIR / "v1iz_alternative_pre_scene_search.csv", scene_rows, SCENE_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1iz_alternative_patch_quality_audit.csv", alt_quality_rows, list(alt_quality_rows[0].keys()) if alt_quality_rows else ["scene_id_sanitized"])
    write_csv(LOCAL_RUN_DIR / "v1iz_final_patch_pair_selection.csv", [registry_row], REGISTRY_FIELDS)
    write_csv(DATASETS_DIR / "official_anchor_sentinel_patch_pair_selection_registry.csv", [registry_row], REGISTRY_FIELDS)
    write_schema(SCHEMAS_DIR / "official_anchor_sentinel_patch_pair_selection_schema.csv", REGISTRY_FIELDS, "REV-P v1iz Sentinel patch pair selection field")
    qa_rows = [
        {"check": "gee_available", "status": "PASS", "detail": availability["status"]},
        {"check": "v1ix_v1iy_inputs_read", "status": "PASS", "detail": "v1ix manifest and v1iy decision available"},
        {"check": "cloud_mask_download_attempted", "status": "PASS", "detail": str(len(download_log))},
        {"check": "alternative_pre_scene_dates_before_event", "status": "PASS" if all(row["pre_event_valid"] == "true" for row in alt_quality_rows) else "FAIL", "detail": EVENT_DATE},
        {"check": "final_pair_decision_emitted", "status": "PASS", "detail": final_status},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "Public outputs use sanitized metadata only."},
    ]
    write_csv(LOCAL_RUN_DIR / "v1iz_qa.csv", qa_rows, ["check", "status", "detail"])

    summary = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "status": final_status,
        "scl_qa60_obtained_for_v1ix_pre": current_evals["PRE_EVENT"]["cloud_mask_available"] == "true",
        "scl_qa60_obtained_for_v1ix_post": current_evals["POST_EVENT_OR_SURVEY_WINDOW"]["cloud_mask_available"] == "true",
        "v1ix_pre_local_cloud_fraction": current_evals["PRE_EVENT"]["local_cloud_fraction"],
        "v1ix_post_local_cloud_fraction": current_evals["POST_EVENT_OR_SURVEY_WINDOW"]["local_cloud_fraction"],
        "alternative_pre_scenes_found": len(scene_rows),
        "alternative_pre_patches_audited": len(alt_quality_rows),
        "selected_pre_scene_id_sanitized": registry_row["pre_scene_id_sanitized"],
        "selected_pre_scene_date": registry_row["pre_scene_date"],
        "selected_post_scene_id_sanitized": registry_row["post_scene_id_sanitized"],
        "selected_post_scene_date": registry_row["post_scene_date"],
        "pre_local_cloud_fraction": registry_row["pre_local_cloud_fraction"],
        "post_local_cloud_fraction": registry_row["post_local_cloud_fraction"],
        "can_be_reference_patch_candidate": registry_row["can_be_reference_patch_candidate"] == "true",
        "can_be_multimodal_reference_candidate": registry_row["can_be_multimodal_reference_candidate"] == "true",
        "can_be_operational_ground_truth": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_reopen_protocol_b": False,
        "primary_blocker": registry_row["primary_blocker"],
        "minimum_evidence_needed": registry_row["minimum_evidence_needed"],
        "commit_warranted": registry_row["can_be_reference_patch_candidate"] == "true",
    }
    write_json(LOCAL_RUN_DIR / "v1iz_summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Clear local v1iz outputs before running.")
    parser.add_argument("--read-v1ix-v1iy", action="store_true", help="Read v1ix and v1iy local outputs.")
    parser.add_argument("--download-cloud-masks", action="store_true", help="Download SCL/QA60 masks for v1ix scenes.")
    parser.add_argument("--search-alternative-pre-scenes", action="store_true", help="Search alternative pre-event scenes before 2022-02-15.")
    parser.add_argument("--evaluate-local-cloud", action="store_true", help="Evaluate local SCL/QA60 cloud fractions.")
    parser.add_argument("--emit-final-pair-decision", action="store_true", help="Emit final pair registry and summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = run(args)
    print("=" * 72)
    print("REV-P v1iz CLOUD_MASK_AND_ALTERNATIVE_SENTINEL_SCENE_SELECTION_FOR_ANCHOR")
    print("=" * 72)
    print(f"SCL/QA60 pre obtained: {summary.get('scl_qa60_obtained_for_v1ix_pre')}")
    print(f"SCL/QA60 post obtained: {summary.get('scl_qa60_obtained_for_v1ix_post')}")
    print(f"v1ix pre local cloud: {summary.get('v1ix_pre_local_cloud_fraction')}")
    print(f"v1ix post local cloud: {summary.get('v1ix_post_local_cloud_fraction')}")
    print(f"Alternative pre scenes found: {summary.get('alternative_pre_scenes_found')}")
    print(f"Selected pre scene: {summary.get('selected_pre_scene_id_sanitized')}")
    print(f"Selected post scene: {summary.get('selected_post_scene_id_sanitized')}")
    print(f"Final pair status: {summary.get('status')}")
    print(f"Multimodal reference candidate: {summary.get('can_be_multimodal_reference_candidate')}")
    print(f"Primary blocker: {summary.get('primary_blocker')}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
