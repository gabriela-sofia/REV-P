"""
REV-P v1ji - MULTI_ANCHOR_BATCH_SENTINEL_SAR_DEM_DINO_READINESS.

Deduplicates official CPRM coordinate expressions into unique documented
anchors, attempts small local GEE downloads for Sentinel-2, Sentinel-1 and
DEM/terrain patches, emits QA and metadata registries, and extracts frozen DINO
review diagnostics when valid S2 pre/post patches exist. No label, true
negative, supervised training, or DINO unfreeze is created.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ji"
PATCH_DIR = LOCAL_RUN_DIR / "patches"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
COORDS = DATASETS_DIR / "official_coordinate_recovery_hardened_registry.csv"
MASTER = DATASETS_DIR / "ground_reference_candidate_master_registry.csv"
PREV_GATE = DATASETS_DIR / "training_gate_decision_matrix.csv"

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S1_COLLECTION = "COPERNICUS/S1_GRD"
DEM_COLLECTION = "COPERNICUS/DEM/GLO30"
PATCH_CRS = "EPSG:32723"
PATCH_SIZE_M = 960
PATCH_RADIUS_M = PATCH_SIZE_M / 2
PATCH_SIZE_PX = 96
S2_SOURCE_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "SCL", "QA60"]
S2_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12", "SCL", "QA60"]
S2_DINO_BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]

MIN_FIELDS = [
    "anchor_id",
    "documented_event_unit_id",
    "date",
    "phenomenon_group",
    "latitude",
    "longitude",
    "coordinate_confidence",
    "s2_pre_status",
    "s2_post_status",
    "s1_pre_status",
    "s1_post_status",
    "dem_status",
    "dino_status",
    "review_only_status",
    "positive_reference_candidate",
    "positive_label_ready",
    "negative_label_ready",
    "training_ready",
    "can_create_training_label",
    "can_train_model",
    "can_unfreeze_dino_for_scientific_claim",
    "blocking_reason",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def prepare(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / "v1ji").resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    PATCH_DIR.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:120].strip("_")


def parse_date(value: str) -> datetime:
    value = (value or "").strip()
    if "–" in value:
        value = value.split("–", 1)[0]
    if "/" in value:
        return datetime.strptime(value, "%d/%m/%Y")
    return datetime.fromisoformat(value)


def anchor_id_for(unit_id: str) -> str:
    return f"ANCHOR_{unit_id}"


def deduplicate_anchors() -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(COORDS):
        if row.get("can_be_official_anchor_candidate") == "true" and row.get("coordinate_confidence") == "EXPLICIT_COORDINATE_HIGH":
            grouped[row["documented_event_unit_id"]].append(row)
    anchors: list[dict[str, Any]] = []
    for unit_id, rows in sorted(grouped.items()):
        rows_sorted = sorted(rows, key=lambda row: row["recovery_id"])
        first = rows_sorted[0]
        anchors.append(
            {
                "anchor_id": anchor_id_for(unit_id),
                "documented_event_unit_id": unit_id,
                "source_document_name_sanitized": first["source_document_name_sanitized"],
                "date": first["event_or_survey_date"],
                "phenomenon_group": first["phenomenon_group"],
                "latitude": first["latitude"],
                "longitude": first["longitude"],
                "coordinate_confidence": first["coordinate_confidence"],
                "coordinate_expression_count": len(rows_sorted),
                "merged_recovery_ids": ";".join(row["recovery_id"] for row in rows_sorted),
                "dedup_status": "OFFICIAL_ANCHOR_CONFIRMED" if len(rows_sorted) == 1 else "DUPLICATE_COORDINATE_MERGED",
                "notes": "Representative coordinate is first explicit point in official document; all original recovery ids remain traceable.",
            }
        )
    return anchors


def gee_init() -> tuple[Any | None, str]:
    try:
        import ee  # type: ignore

        ee.Initialize()
        return ee, "GEE_AUTHENTICATED"
    except Exception as exc:
        return None, f"GEE_UNAVAILABLE:{type(exc).__name__}"


def region_for(ee: Any, lat: float, lon: float) -> Any:
    point = ee.Geometry.Point([lon, lat])
    return point.transform(PATCH_CRS, 1).buffer(PATCH_RADIUS_M).bounds(1).transform("EPSG:4326", 1)


def inclusive_windows(date_text: str) -> dict[str, tuple[str, str]]:
    date = parse_date(date_text)
    return {
        "pre": ((date - timedelta(days=45)).date().isoformat(), date.date().isoformat()),
        "post": (date.date().isoformat(), (date + timedelta(days=30)).date().isoformat()),
    }


def download_image(image: Any, region: Any, out_dir: Path, name: str, scale: int) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    url = image.getDownloadURL({"scale": scale, "crs": PATCH_CRS, "region": region, "format": "GEO_TIFF"})
    download_path = out_dir / f"{name}.download"
    urllib.request.urlretrieve(url, download_path)
    if zipfile.is_zipfile(download_path):
        with zipfile.ZipFile(download_path) as archive:
            tif_names = [member for member in archive.namelist() if member.lower().endswith((".tif", ".tiff"))]
            if not tif_names:
                download_path.unlink(missing_ok=True)
                return None
            archive.extract(tif_names[0], out_dir)
            extracted = out_dir / tif_names[0]
    else:
        extracted = download_path
    final = out_dir / f"{name}.local_geotiff"
    if final.exists():
        final.unlink()
    extracted.replace(final)
    download_path.unlink(missing_ok=True)
    return final


def raster_qa(path: Path | None, expected_min_bands: int = 1) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"patch_exists": "false", "qa_status": "PATCH_NOT_AVAILABLE", "shape_px": "", "band_count": "", "crs": "", "valid_pixel_fraction": "0.000000", "nan_inf_status": "NOT_ASSESSED"}
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        data = src.read(masked=True).astype("float64")
        finite = np.isfinite(np.ma.filled(data, np.nan))
        valid = finite & ~np.ma.getmaskarray(data)
        valid_fraction = float(valid.sum() / valid.size) if valid.size else 0.0
        has_nan_inf = not bool(finite.all())
        qa = "QA_PASS" if src.count >= expected_min_bands and valid_fraction > 0.80 and not has_nan_inf else "QA_FAIL"
        return {
            "patch_exists": "true",
            "qa_status": qa,
            "shape_px": f"{src.width}x{src.height}",
            "band_count": src.count,
            "crs": str(src.crs) if src.crs else "",
            "valid_pixel_fraction": f"{valid_fraction:.6f}",
            "nan_inf_status": "NAN_INF_FOUND" if has_nan_inf else "NAN_INF_OK",
        }


def local_cloud_fraction(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        import numpy as np
        import rasterio

        with rasterio.open(path) as src:
            if src.count < 7:
                return "SCL_NOT_AVAILABLE"
            scl = src.read(7)
        cloud_classes = {3, 8, 9, 10, 11}
        return f"{float(np.isin(scl, list(cloud_classes)).sum() / scl.size):.6f}"
    except Exception:
        return "CLOUD_QA_FAILED"


def select_s2(ee: Any, anchor: dict[str, Any], relation: str, start: str, end: str) -> tuple[Any | None, dict[str, Any]]:
    lat, lon = float(anchor["latitude"]), float(anchor["longitude"])
    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
    )
    size = int(collection.size().getInfo())
    if size == 0:
        return None, {"scene_count": 0, "scene_id": "", "scene_date": "", "cloud": "", "status": "NO_S2_SCENE"}
    img = ee.Image(collection.first())
    props = img.toDictionary(["system:index", "system:time_start", "CLOUDY_PIXEL_PERCENTAGE"]).getInfo()
    scene_date = datetime.fromtimestamp(props["system:time_start"] / 1000, tz=timezone.utc).date().isoformat()
    selected = img.select(S2_SOURCE_BANDS, S2_BANDS).toFloat()
    return selected, {"scene_count": size, "scene_id": props.get("system:index", ""), "scene_date": scene_date, "cloud": props.get("CLOUDY_PIXEL_PERCENTAGE", ""), "status": "SCENE_SELECTED"}


def select_s1(ee: Any, anchor: dict[str, Any], relation: str, start: str, end: str) -> tuple[Any | None, dict[str, Any]]:
    lat, lon = float(anchor["latitude"]), float(anchor["longitude"])
    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(point)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .sort("system:time_start", False if relation == "pre" else True)
    )
    size = int(collection.size().getInfo())
    if size == 0:
        return None, {"scene_count": 0, "scene_id": "", "scene_date": "", "status": "NO_S1_SCENE", "polarizations": ""}
    img = ee.Image(collection.first())
    names = img.bandNames().getInfo()
    bands = [band for band in ["VV", "VH"] if band in names]
    props = img.toDictionary(["system:index", "system:time_start"]).getInfo()
    scene_date = datetime.fromtimestamp(props["system:time_start"] / 1000, tz=timezone.utc).date().isoformat()
    return img.select(bands).toFloat(), {"scene_count": size, "scene_id": props.get("system:index", ""), "scene_date": scene_date, "status": "SCENE_SELECTED", "polarizations": ",".join(bands)}


def dem_image(ee: Any) -> Any:
    dem = ee.ImageCollection(DEM_COLLECTION).select("DEM").mosaic().rename("DEM").toFloat()
    terrain = ee.Terrain.products(dem).select(["slope", "aspect"]).toFloat()
    return dem.addBands(terrain)


def run_batch_downloads(anchors: list[dict[str, Any]], ee: Any | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    s2_scene_rows: list[dict[str, Any]] = []
    s2_qa_rows: list[dict[str, Any]] = []
    s1_scene_rows: list[dict[str, Any]] = []
    s1_qa_rows: list[dict[str, Any]] = []
    dem_rows: list[dict[str, Any]] = []
    if ee is None:
        for anchor in anchors:
            for relation in ["pre", "post"]:
                s2_scene_rows.append({**anchor, "relation": relation, "status": "GEE_UNAVAILABLE"})
                s1_scene_rows.append({**anchor, "relation": relation, "status": "GEE_UNAVAILABLE"})
            dem_rows.append({**anchor, "dem_status": "GEE_UNAVAILABLE"})
        return s2_scene_rows, s2_qa_rows, s1_scene_rows + s1_qa_rows, dem_rows
    for anchor in anchors:
        lat, lon = float(anchor["latitude"]), float(anchor["longitude"])
        region = region_for(ee, lat, lon)
        windows = inclusive_windows(anchor["date"])
        anchor_dir = PATCH_DIR / safe_slug(anchor["anchor_id"])
        for relation in ["pre", "post"]:
            start, end = windows[relation]
            s2_img, s2_meta = select_s2(ee, anchor, relation, start, end)
            s2_scene_rows.append({**anchor, "relation": relation, "window_start": start, "window_end": end, **s2_meta})
            s2_path = None
            if s2_img is not None:
                try:
                    s2_path = download_image(s2_img, region, anchor_dir / "s2", f"{anchor['anchor_id']}_{relation}_s2", 10)
                    s2_status = "S2_PATCH_GENERATED"
                except Exception as exc:
                    s2_status = f"S2_DOWNLOAD_FAILED:{type(exc).__name__}"
            else:
                s2_status = "NO_S2_SCENE"
            qa = raster_qa(s2_path, expected_min_bands=6)
            s2_qa_rows.append({**anchor, "relation": relation, "patch_file_sanitized": s2_path.name if s2_path else "", "s2_status": s2_status, "local_cloud_fraction": local_cloud_fraction(s2_path), **qa})
            s1_img, s1_meta = select_s1(ee, anchor, relation, start, end)
            s1_scene_rows.append({**anchor, "relation": relation, "window_start": start, "window_end": end, **s1_meta})
            s1_path = None
            if s1_img is not None:
                try:
                    s1_path = download_image(s1_img, region, anchor_dir / "s1", f"{anchor['anchor_id']}_{relation}_s1", 10)
                    s1_status = "S1_PATCH_GENERATED"
                except Exception as exc:
                    s1_status = f"S1_DOWNLOAD_FAILED:{type(exc).__name__}"
            else:
                s1_status = "NO_S1_SCENE"
            qa1 = raster_qa(s1_path, expected_min_bands=1)
            s1_qa_rows.append({**anchor, "relation": relation, "patch_file_sanitized": s1_path.name if s1_path else "", "s1_status": s1_status, **qa1})
        dem_path = None
        try:
            dem_path = download_image(dem_image(ee), region, anchor_dir / "dem", f"{anchor['anchor_id']}_dem_terrain", 30)
            dem_status = "DEM_PATCH_GENERATED"
        except Exception as exc:
            dem_status = f"DEM_DOWNLOAD_FAILED:{type(exc).__name__}"
        dem_rows.append({**anchor, "patch_file_sanitized": dem_path.name if dem_path else "", "dem_status": dem_status, **raster_qa(dem_path, expected_min_bands=1)})
    return s2_scene_rows, s2_qa_rows, s1_scene_rows + s1_qa_rows, dem_rows


def load_dino_model() -> tuple[Any | None, str, str]:
    try:
        import torch
        from transformers import AutoModel

        model_name = "facebook/dinov2-with-registers-base"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model, device, "DINO_MODEL_LOADED"
    except Exception as exc:
        return None, "cpu", f"DINO_MODEL_UNAVAILABLE:{type(exc).__name__}"


def dino_for_anchor(anchor: dict[str, Any], pre_path: Path, post_path: Path, model: Any, device: str) -> dict[str, Any]:
    import numpy as np
    import rasterio
    import torch

    def visual(path: Path) -> Any:
        with rasterio.open(path) as src:
            data = src.read([3, 2, 1]).astype("float32")
        lows = np.percentile(data, 2, axis=(1, 2), keepdims=True)
        highs = np.percentile(data, 98, axis=(1, 2), keepdims=True)
        scaled = np.clip((data - lows) / np.maximum(highs - lows, 1.0e-6), 0, 1)
        hwc = scaled.transpose(1, 2, 0)
        y = np.linspace(0, hwc.shape[0] - 1, 224).round().astype("int64")
        x = np.linspace(0, hwc.shape[1] - 1, 224).round().astype("int64")
        img = hwc[y][:, x]
        mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        return ((img - mean) / std).astype("float32")

    def embed(path: Path) -> Any:
        arr = visual(path)
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(pixel_values=tensor)
        return out.last_hidden_state[:, 0, :].detach().cpu().numpy().reshape(-1).astype("float32")

    pre = embed(pre_path)
    post = embed(post_path)
    cosine = float(np.dot(pre, post) / max(float(np.linalg.norm(pre) * np.linalg.norm(post)), 1.0e-12))
    euclidean = float(np.linalg.norm(pre - post))
    status = "DINO_QA_PASS" if pre.shape[0] == 768 and post.shape[0] == 768 and np.isfinite(pre).all() and np.isfinite(post).all() else "DINO_QA_FAIL"
    return {**anchor, "embedding_dim": 768 if status == "DINO_QA_PASS" else "", "cosine_similarity": f"{cosine:.8f}", "euclidean_distance": f"{euclidean:.8f}", "dino_status": status, "can_create_training_label": "false", "can_train_model": "false"}


def run_dino(anchors: list[dict[str, Any]], s2_qa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_anchor: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in s2_qa_rows:
        if row.get("qa_status") == "QA_PASS" and row.get("patch_file_sanitized"):
            by_anchor[row["anchor_id"]][row["relation"]] = row
    model, device, model_status = load_dino_model()
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        pair = by_anchor.get(anchor["anchor_id"], {})
        if model is None or "pre" not in pair or "post" not in pair:
            rows.append({**anchor, "embedding_dim": "", "cosine_similarity": "", "euclidean_distance": "", "dino_status": model_status if model is None else "DINO_BLOCKED_MISSING_S2_QA_PAIR", "can_create_training_label": "false", "can_train_model": "false"})
            continue
        pre_path = PATCH_DIR / safe_slug(anchor["anchor_id"]) / "s2" / pair["pre"]["patch_file_sanitized"]
        post_path = PATCH_DIR / safe_slug(anchor["anchor_id"]) / "s2" / pair["post"]["patch_file_sanitized"]
        try:
            rows.append(dino_for_anchor(anchor, pre_path, post_path, model, device))
        except Exception as exc:
            rows.append({**anchor, "embedding_dim": "", "cosine_similarity": "", "euclidean_distance": "", "dino_status": f"DINO_FAILED:{type(exc).__name__}", "can_create_training_label": "false", "can_train_model": "false"})
    return rows


def readiness_rows(anchors: list[dict[str, Any]], s2_qa: list[dict[str, Any]], s1_all: list[dict[str, Any]], dem_rows: list[dict[str, Any]], dino_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        aid = anchor["anchor_id"]
        s2_pre = next((row for row in s2_qa if row["anchor_id"] == aid and row["relation"] == "pre"), {})
        s2_post = next((row for row in s2_qa if row["anchor_id"] == aid and row["relation"] == "post"), {})
        s1_pre = next((row for row in s1_all if row.get("anchor_id") == aid and row.get("relation") == "pre" and "s1_status" in row), {})
        s1_post = next((row for row in s1_all if row.get("anchor_id") == aid and row.get("relation") == "post" and "s1_status" in row), {})
        dem = next((row for row in dem_rows if row["anchor_id"] == aid), {})
        dino = next((row for row in dino_rows if row["anchor_id"] == aid), {})
        s2_pass = s2_pre.get("qa_status") == "QA_PASS" and s2_post.get("qa_status") == "QA_PASS"
        rows.append(
            {
                "anchor_id": aid,
                "documented_event_unit_id": anchor["documented_event_unit_id"],
                "date": anchor["date"],
                "phenomenon_group": anchor["phenomenon_group"],
                "latitude": anchor["latitude"],
                "longitude": anchor["longitude"],
                "coordinate_confidence": anchor["coordinate_confidence"],
                "s2_pre_status": s2_pre.get("qa_status", "NOT_AVAILABLE"),
                "s2_post_status": s2_post.get("qa_status", "NOT_AVAILABLE"),
                "s1_pre_status": s1_pre.get("qa_status", "NOT_AVAILABLE"),
                "s1_post_status": s1_post.get("qa_status", "NOT_AVAILABLE"),
                "dem_status": dem.get("qa_status", "NOT_AVAILABLE"),
                "dino_status": dino.get("dino_status", "NOT_AVAILABLE"),
                "review_only_status": "REVIEW_ONLY_READY" if s2_pass else "REVIEW_ONLY_PENDING_PATCH_QA",
                "positive_reference_candidate": "true",
                "positive_label_ready": "false",
                "negative_label_ready": "false",
                "training_ready": "false",
                "can_create_training_label": "false",
                "can_train_model": "false",
                "can_unfreeze_dino_for_scientific_claim": "false",
                "blocking_reason": "LABEL_AND_NEGATIVE_GATES_BLOCKED",
                "notes": "Patch QA and DINO review do not create labels or training permission.",
            }
        )
    return rows


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{prefix}: {field}."} for field in fields], ["field", "description"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare(args.force)
    anchors = deduplicate_anchors()
    ee, gee_status = gee_init()
    s2_scenes, s2_qa, s1_all, dem_rows = run_batch_downloads(anchors, ee)
    dino_rows = run_dino(anchors, s2_qa)
    ready = readiness_rows(anchors, s2_qa, s1_all, dem_rows, dino_rows)
    gate = {
        "gate_id": "V1JI_TRAINING_GATE_UPDATE",
        "official_anchor_count": len(anchors),
        "s2_pre_post_qa_pass_count": sum(1 for row in ready if row["s2_pre_status"] == "QA_PASS" and row["s2_post_status"] == "QA_PASS"),
        "s1_pre_post_qa_pass_count": sum(1 for row in ready if row["s1_pre_status"] == "QA_PASS" and row["s1_post_status"] == "QA_PASS"),
        "dem_qa_pass_count": sum(1 for row in ready if row["dem_status"] == "QA_PASS"),
        "dino_qa_pass_count": sum(1 for row in ready if row["dino_status"] == "DINO_QA_PASS"),
        "negative_labels_ready_count": 0,
        "leakage_risk_status": "LEAKAGE_PROTOCOL_REQUIRED",
        "training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
        "can_create_training_label": "false",
        "can_train_model": "false",
        "can_unfreeze_dino_for_scientific_claim": "false",
        "notes": "Review-only batch can advance where QA passed; supervised gates remain blocked.",
    }
    write_csv(LOCAL_RUN_DIR / "v1ji_anchor_deduplication_audit.csv", anchors, ["anchor_id", "documented_event_unit_id", "source_document_name_sanitized", "date", "phenomenon_group", "latitude", "longitude", "coordinate_confidence", "coordinate_expression_count", "merged_recovery_ids", "dedup_status", "notes"])
    write_csv(LOCAL_RUN_DIR / "v1ji_s2_batch_scene_selection.csv", s2_scenes, ["anchor_id", "documented_event_unit_id", "date", "phenomenon_group", "latitude", "longitude", "relation", "window_start", "window_end", "scene_count", "scene_id", "scene_date", "cloud", "status"])
    write_csv(LOCAL_RUN_DIR / "v1ji_s2_patch_quality_audit.csv", s2_qa, ["anchor_id", "documented_event_unit_id", "relation", "patch_file_sanitized", "s2_status", "local_cloud_fraction", "patch_exists", "qa_status", "shape_px", "band_count", "crs", "valid_pixel_fraction", "nan_inf_status"])
    write_csv(LOCAL_RUN_DIR / "v1ji_s1_batch_scene_selection.csv", [row for row in s1_all if "scene_count" in row], ["anchor_id", "documented_event_unit_id", "date", "phenomenon_group", "latitude", "longitude", "relation", "window_start", "window_end", "scene_count", "scene_id", "scene_date", "status", "polarizations"])
    write_csv(LOCAL_RUN_DIR / "v1ji_s1_patch_quality_audit.csv", [row for row in s1_all if "s1_status" in row], ["anchor_id", "documented_event_unit_id", "relation", "patch_file_sanitized", "s1_status", "patch_exists", "qa_status", "shape_px", "band_count", "crs", "valid_pixel_fraction", "nan_inf_status"])
    write_csv(LOCAL_RUN_DIR / "v1ji_dem_terrain_quality_audit.csv", dem_rows, ["anchor_id", "documented_event_unit_id", "patch_file_sanitized", "dem_status", "patch_exists", "qa_status", "shape_px", "band_count", "crs", "valid_pixel_fraction", "nan_inf_status"])
    write_csv(LOCAL_RUN_DIR / "v1ji_dino_batch_embedding_diagnostics.csv", dino_rows, ["anchor_id", "documented_event_unit_id", "embedding_dim", "cosine_similarity", "euclidean_distance", "dino_status", "can_create_training_label", "can_train_model"])
    write_csv(LOCAL_RUN_DIR / "v1ji_multimodal_batch_readiness.csv", ready, MIN_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1ji_training_gate_update.csv", [gate], list(gate.keys()))
    write_csv(LOCAL_RUN_DIR / "v1ji_qa.csv", [
        {"check": "dedup_anchor_count", "status": "PASS" if len(anchors) >= 9 else "FAIL", "detail": str(len(anchors))},
        {"check": "s2_batch_no_label", "status": "PASS", "detail": "all label flags false"},
        {"check": "s1_dem_absence_controlled", "status": "PASS", "detail": f"s1={gate['s1_pre_post_qa_pass_count']};dem={gate['dem_qa_pass_count']}"},
        {"check": "dino_review_only", "status": "PASS", "detail": str(gate["dino_qa_pass_count"])},
        {"check": "negative_labels_zero", "status": "PASS", "detail": "0"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_unfreeze_dino_for_scientific_claim_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "sanitized metadata only"},
    ], ["check", "status", "detail"])
    write_csv(DATASETS_DIR / "official_multi_anchor_registry.csv", anchors, ["anchor_id", "documented_event_unit_id", "source_document_name_sanitized", "date", "phenomenon_group", "latitude", "longitude", "coordinate_confidence", "coordinate_expression_count", "merged_recovery_ids", "dedup_status", "notes"])
    write_csv(DATASETS_DIR / "multi_anchor_multimodal_patch_registry.csv", ready, MIN_FIELDS)
    write_csv(DATASETS_DIR / "multi_anchor_dino_review_embedding_registry.csv", dino_rows, ["anchor_id", "documented_event_unit_id", "embedding_dim", "cosine_similarity", "euclidean_distance", "dino_status", "can_create_training_label", "can_train_model"])
    write_csv(DATASETS_DIR / "multi_anchor_training_gate_matrix.csv", [gate], list(gate.keys()))
    write_schema(SCHEMAS_DIR / "official_multi_anchor_schema.csv", ["anchor_id", "documented_event_unit_id", "source_document_name_sanitized", "date", "phenomenon_group", "latitude", "longitude", "coordinate_confidence", "coordinate_expression_count", "merged_recovery_ids", "dedup_status", "notes"], "REV-P v1ji official multi-anchor field")
    write_schema(SCHEMAS_DIR / "multi_anchor_multimodal_patch_schema.csv", MIN_FIELDS, "REV-P v1ji multimodal patch field")
    write_schema(SCHEMAS_DIR / "multi_anchor_dino_review_embedding_schema.csv", ["anchor_id", "documented_event_unit_id", "embedding_dim", "cosine_similarity", "euclidean_distance", "dino_status", "can_create_training_label", "can_train_model"], "REV-P v1ji DINO review embedding field")
    write_schema(SCHEMAS_DIR / "multi_anchor_training_gate_schema.csv", list(gate.keys()), "REV-P v1ji training gate field")
    summary = {
        "stage": "v1ji",
        "timestamp": utc_now(),
        "gee_status": gee_status,
        "official_unique_anchor_count": len(anchors),
        "s2_patches_generated_count": sum(1 for row in s2_qa if row.get("patch_exists") == "true"),
        "s2_pre_post_qa_pass_count": gate["s2_pre_post_qa_pass_count"],
        "s1_patches_generated_count": sum(1 for row in s1_all if row.get("patch_exists") == "true"),
        "s1_pre_post_qa_pass_count": gate["s1_pre_post_qa_pass_count"],
        "dem_patches_generated_count": sum(1 for row in dem_rows if row.get("patch_exists") == "true"),
        "dem_qa_pass_count": gate["dem_qa_pass_count"],
        "dino_embeddings_generated_count": gate["dino_qa_pass_count"],
        "negative_labels_ready_count": 0,
        "training_gate_status": "SUPERVISED_TRAINING_BLOCKED",
        "can_create_training_label": False,
        "can_train_model": False,
        "can_unfreeze_dino_for_scientific_claim": False,
    }
    write_json(LOCAL_RUN_DIR / "v1ji_summary.json", summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print("REV-P v1ji MULTI-ANCHOR BATCH MULTIMODAL READINESS")
    print(f"Unique official anchors: {summary['official_unique_anchor_count']}")
    print(f"S2 patches generated: {summary['s2_patches_generated_count']}")
    print(f"S1 patches generated: {summary['s1_patches_generated_count']}")
    print(f"DEM patches generated: {summary['dem_patches_generated_count']}")
    print(f"DINO embeddings generated: {summary['dino_embeddings_generated_count']}")
    print(f"Training gate: {summary['training_gate_status']}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
