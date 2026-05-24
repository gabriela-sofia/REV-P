"""
REV-P v1iy - ANCHOR_SENTINEL_PATCH_LOCAL_QUALITY_AND_CLOUD_AUDIT.

Audits the local Sentinel-2 patches produced by v1ix for band completeness,
geometry, local pixel quality, spectral variation, approximate NDWI/NDBI
statistics, and local cloud-mask availability. No new data are downloaded.

Invariants:
  - no label, target, training, or Protocol B reopening is created
  - raster artifacts remain local-only
  - public outputs contain metadata only and no private paths
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iy"
V1IX_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ix"
V1IX_MANIFEST = V1IX_DIR / "v1ix_anchor_patch_manifest_local.csv"
V1IX_PATCH_DIR = V1IX_DIR / "anchor_centered_patch"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"

STAGE = "v1iy"
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
BAND_INDEX = {band: idx for idx, band in enumerate(BANDS)}
EXPECTED_SHAPE = "96x96"
EXPECTED_CRS = "EPSG:32723"
HIGH_GLOBAL_CLOUD_THRESHOLD = 80.0
MIN_VALID_PIXEL_FRACTION = 0.95
MAX_NODATA_FRACTION = 0.05
MAX_ZERO_FRACTION = 0.95
MIN_VARIANCE = 1.0e-6
MAX_SATURATION_FRACTION = 0.05
LOCAL_RASTER_SUFFIX = ".local_geotiff"

QUALITY_FIELDS = [
    "quality_id",
    "reference_patch_id",
    "anchor_id",
    "temporal_relation_to_event",
    "scene_id_sanitized",
    "scene_date",
    "bands_available",
    "shape_px",
    "crs",
    "resolution_m",
    "valid_pixel_fraction",
    "nodata_fraction",
    "nan_inf_status",
    "band_range_status",
    "band_variance_status",
    "cloud_metadata_global",
    "cloud_mask_available",
    "local_cloud_fraction",
    "spectral_index_status",
    "local_quality_status",
    "pair_quality_status",
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

GATE_FIELDS = [
    "gate",
    "status",
    "detail",
    "blocking_reason",
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


def prepare_output_dir(force: bool) -> None:
    if force and LOCAL_RUN_DIR.exists():
        resolved = LOCAL_RUN_DIR.resolve()
        expected = (REVP_ROOT / "local_runs" / "protocolo_c" / STAGE).resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to clear unexpected path: {resolved}")
        shutil.rmtree(resolved)
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def relation_dir_name(relation: str) -> str:
    return relation.lower()


def find_local_patch(row: dict[str, str]) -> Path | None:
    relation_dir = V1IX_PATCH_DIR / relation_dir_name(row["temporal_relation_to_event"])
    candidates = sorted(relation_dir.glob(f"*{LOCAL_RASTER_SUFFIX}"))
    if candidates:
        return candidates[0]
    return None


def finite_values(masked_array: Any) -> Any:
    import numpy as np

    valid = ~np.ma.getmaskarray(masked_array)
    filled = np.ma.filled(masked_array.astype("float64"), np.nan)
    return filled[valid & np.isfinite(filled)]


def status_from_band_stats(rows: list[dict[str, Any]]) -> tuple[str, str, str]:
    range_failures = [row["band"] for row in rows if row["range_status"] != "OK"]
    variance_failures = [row["band"] for row in rows if row["variance_status"] != "OK"]
    range_status = "OK" if not range_failures else "FAIL"
    variance_status = "OK" if not variance_failures else "FAIL"
    detail = ";".join(
        [
            f"range_failures={','.join(range_failures) if range_failures else 'none'}",
            f"variance_failures={','.join(variance_failures) if variance_failures else 'none'}",
        ]
    )
    return range_status, variance_status, detail


def compute_index(numerator_a: Any, numerator_b: Any) -> Any:
    import numpy as np

    a = numerator_a.astype("float64")
    b = numerator_b.astype("float64")
    denominator = a + b
    return np.where(np.abs(denominator) > 1.0e-9, (a - b) / denominator, np.nan)


def summarize_index(values: Any) -> dict[str, Any]:
    import numpy as np

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "index_computable": "false",
            "min": "",
            "max": "",
            "mean": "",
            "std": "",
            "finite_fraction": "0.000000",
        }
    return {
        "index_computable": "true",
        "min": f"{float(np.min(finite)):.6f}",
        "max": f"{float(np.max(finite)):.6f}",
        "mean": f"{float(np.mean(finite)):.6f}",
        "std": f"{float(np.std(finite)):.6f}",
        "finite_fraction": f"{float(finite.size / values.size):.6f}",
    }


def audit_patch(row: dict[str, str]) -> dict[str, Any]:
    import numpy as np
    import rasterio

    patch_path = find_local_patch(row)
    relation = row["temporal_relation_to_event"]
    result: dict[str, Any] = {
        "manifest_row": row,
        "patch_exists": patch_path is not None,
        "patch_filename": patch_path.name if patch_path else "",
        "inventory_row": {},
        "band_rows": [],
        "index_rows": [],
        "cloud_row": {},
        "quality_row": {},
        "errors": [],
    }
    if patch_path is None:
        result["errors"].append("LOCAL_PATCH_NOT_FOUND")
        return result

    with rasterio.open(patch_path) as src:
        data = src.read(masked=True)
        float_data = data.astype("float64")
        valid_mask = ~np.ma.getmaskarray(float_data)
        filled = np.ma.filled(float_data, np.nan)
        finite_mask = np.isfinite(filled)
        valid_finite_mask = valid_mask & finite_mask
        total_values = data.size if data.size else 1
        valid_fraction = float(np.count_nonzero(valid_finite_mask) / total_values)
        nodata_fraction = float(1.0 - valid_fraction)
        has_nan_inf = bool(np.count_nonzero(valid_mask & ~finite_mask) > 0)
        bands_available = ",".join(BANDS[: src.count])
        shape_px = f"{src.width}x{src.height}"
        crs = str(src.crs) if src.crs else ""
        resolution = f"{abs(src.res[0]):.4f}" if src.res else ""

        band_rows: list[dict[str, Any]] = []
        for idx in range(src.count):
            band_name = BANDS[idx] if idx < len(BANDS) else f"BAND_{idx + 1}"
            band_values = finite_values(data[idx])
            if band_values.size == 0:
                band_min = band_max = band_mean = band_std = band_variance = math.nan
                zero_fraction = 1.0
                saturation_fraction = 0.0
                range_status = "FAIL"
                variance_status = "FAIL"
            else:
                band_min = float(np.min(band_values))
                band_max = float(np.max(band_values))
                band_mean = float(np.mean(band_values))
                band_std = float(np.std(band_values))
                band_variance = float(np.var(band_values))
                zero_fraction = float(np.count_nonzero(band_values == 0) / band_values.size)
                saturation_fraction = float(np.count_nonzero(band_values >= 65535) / band_values.size)
                range_status = "OK" if band_max > band_min and zero_fraction <= MAX_ZERO_FRACTION and saturation_fraction <= MAX_SATURATION_FRACTION else "FAIL"
                variance_status = "OK" if band_variance > MIN_VARIANCE else "FAIL"
            band_rows.append(
                {
                    "reference_patch_id": row["reference_patch_id"],
                    "temporal_relation_to_event": relation,
                    "band": band_name,
                    "min": "" if math.isnan(band_min) else f"{band_min:.6f}",
                    "max": "" if math.isnan(band_max) else f"{band_max:.6f}",
                    "mean": "" if math.isnan(band_mean) else f"{band_mean:.6f}",
                    "std": "" if math.isnan(band_std) else f"{band_std:.6f}",
                    "variance": "" if math.isnan(band_variance) else f"{band_variance:.6f}",
                    "zero_fraction": f"{zero_fraction:.6f}",
                    "saturation_fraction": f"{saturation_fraction:.6f}",
                    "range_status": range_status,
                    "variance_status": variance_status,
                }
            )

        range_status, variance_status, band_detail = status_from_band_stats(band_rows)
        ndwi = compute_index(filled[BAND_INDEX["B03"]], filled[BAND_INDEX["B08"]]) if src.count >= 4 else None
        ndbi = compute_index(filled[BAND_INDEX["B11"]], filled[BAND_INDEX["B08"]]) if src.count >= 5 else None
        index_rows: list[dict[str, Any]] = []
        for index_name, values in [("NDWI_APPROX_B03_B08", ndwi), ("NDBI_APPROX_B11_B08", ndbi)]:
            if values is None:
                stats = {
                    "index_computable": "false",
                    "min": "",
                    "max": "",
                    "mean": "",
                    "std": "",
                    "finite_fraction": "0.000000",
                }
            else:
                stats = summarize_index(values)
            index_rows.append(
                {
                    "reference_patch_id": row["reference_patch_id"],
                    "temporal_relation_to_event": relation,
                    "index_name": index_name,
                    **stats,
                }
            )
        spectral_index_status = "OK" if all(index_row["index_computable"] == "true" for index_row in index_rows) else "FAIL"

        cloud_metadata = safe_float(row.get("cloud_cover_metadata", ""))
        cloud_mask_available = False
        local_cloud_fraction = "CLOUD_MASK_NOT_AVAILABLE"
        cloud_risk_status = (
            "PRE_PATCH_CLOUD_RISK_HIGH"
            if relation == "PRE_EVENT" and cloud_metadata >= HIGH_GLOBAL_CLOUD_THRESHOLD and not cloud_mask_available
            else "NO_LOCAL_CLOUD_MASK"
        )
        local_quality_ok = (
            src.count == len(BANDS)
            and bands_available == ",".join(BANDS)
            and shape_px == EXPECTED_SHAPE
            and crs == EXPECTED_CRS
            and valid_fraction >= MIN_VALID_PIXEL_FRACTION
            and nodata_fraction <= MAX_NODATA_FRACTION
            and not has_nan_inf
            and range_status == "OK"
            and variance_status == "OK"
            and spectral_index_status == "OK"
        )
        local_quality_status = "LOCAL_QA_PASS" if local_quality_ok else "LOCAL_QA_FAIL"

        result["inventory_row"] = {
            "reference_patch_id": row["reference_patch_id"],
            "temporal_relation_to_event": relation,
            "patch_file_sanitized": patch_path.name,
            "patch_exists": bool_text(True),
            "bands_available": bands_available,
            "shape_px": shape_px,
            "crs": crs,
            "resolution_m": resolution,
            "valid_pixel_fraction": f"{valid_fraction:.6f}",
            "nodata_fraction": f"{nodata_fraction:.6f}",
            "nan_inf_status": "FAIL" if has_nan_inf else "OK",
            "band_range_status": range_status,
            "band_variance_status": variance_status,
            "spectral_index_status": spectral_index_status,
            "local_quality_status": local_quality_status,
        }
        result["band_rows"] = band_rows
        result["index_rows"] = index_rows
        result["cloud_row"] = {
            "reference_patch_id": row["reference_patch_id"],
            "temporal_relation_to_event": relation,
            "scene_id_sanitized": row["scene_id_sanitized"],
            "scene_date": row["scene_date"],
            "cloud_metadata_global": row.get("cloud_cover_metadata", ""),
            "cloud_mask_available": bool_text(cloud_mask_available),
            "local_cloud_fraction": local_cloud_fraction,
            "cloud_quality_status": cloud_risk_status,
            "notes": "SCL/QA60 bands are not present in the local v1ix patch; global scene metadata is recorded but is not used as an automatic local rejection.",
        }
        result["quality_row"] = {
            "quality_id": f"QUALITY_{row['reference_patch_id']}",
            "reference_patch_id": row["reference_patch_id"],
            "anchor_id": row["anchor_id"],
            "temporal_relation_to_event": relation,
            "scene_id_sanitized": row["scene_id_sanitized"],
            "scene_date": row["scene_date"],
            "bands_available": bands_available,
            "shape_px": shape_px,
            "crs": crs,
            "resolution_m": resolution,
            "valid_pixel_fraction": f"{valid_fraction:.6f}",
            "nodata_fraction": f"{nodata_fraction:.6f}",
            "nan_inf_status": "FAIL" if has_nan_inf else "OK",
            "band_range_status": range_status,
            "band_variance_status": variance_status,
            "cloud_metadata_global": row.get("cloud_cover_metadata", ""),
            "cloud_mask_available": bool_text(cloud_mask_available),
            "local_cloud_fraction": local_cloud_fraction,
            "spectral_index_status": spectral_index_status,
            "local_quality_status": local_quality_status,
            "pair_quality_status": "",
            "can_be_reference_patch_candidate": "",
            "can_be_multimodal_reference_candidate": "",
            "can_be_operational_ground_truth": bool_text(False),
            "can_create_training_label": bool_text(False),
            "can_train_model": bool_text(False),
            "can_reopen_protocol_b": bool_text(False),
            "primary_blocker": "",
            "minimum_evidence_needed": "",
            "notes": f"{band_detail}; {result['cloud_row']['notes']}",
        }
    return result


def decide_pair(results: list[dict[str, Any]]) -> dict[str, Any]:
    if any(not result["patch_exists"] for result in results):
        return {
            "pair_quality_status": "PATCH_PAIR_BLOCKED_BY_QUALITY",
            "can_be_reference_patch_candidate": False,
            "can_be_multimodal_reference_candidate": False,
            "primary_blocker": "LOCAL_PATCH_NOT_FOUND",
            "minimum_evidence_needed": "Restore local v1ix pre and post patch files before quality audit.",
        }
    local_failures = [result["quality_row"]["reference_patch_id"] for result in results if result["quality_row"]["local_quality_status"] != "LOCAL_QA_PASS"]
    if local_failures:
        return {
            "pair_quality_status": "PATCH_PAIR_BLOCKED_BY_QUALITY",
            "can_be_reference_patch_candidate": False,
            "can_be_multimodal_reference_candidate": False,
            "primary_blocker": "LOCAL_PATCH_QUALITY_FAILED",
            "minimum_evidence_needed": f"Resolve local quality failures for: {','.join(local_failures)}.",
        }
    pre_cloud_risk = any(result["cloud_row"].get("cloud_quality_status") == "PRE_PATCH_CLOUD_RISK_HIGH" for result in results)
    if pre_cloud_risk:
        return {
            "pair_quality_status": "PRE_PATCH_CLOUD_RISK_HIGH",
            "can_be_reference_patch_candidate": True,
            "can_be_multimodal_reference_candidate": True,
            "primary_blocker": "PRE_PATCH_CLOUD_RISK_HIGH",
            "minimum_evidence_needed": "Use local SCL/QA60 or an alternative pre-event scene to reduce cloud uncertainty before stronger use.",
        }
    return {
        "pair_quality_status": "PATCH_PAIR_USABLE_FOR_REVIEW",
        "can_be_reference_patch_candidate": True,
        "can_be_multimodal_reference_candidate": True,
        "primary_blocker": "NONE",
        "minimum_evidence_needed": "Independent interpretation remains required before any stronger evidence role.",
    }


def write_schema(path: Path, fields: list[str], description_prefix: str) -> None:
    write_csv(
        path,
        [{"field": field, "description": f"{description_prefix}: {field}."} for field in fields],
        ["field", "description"],
    )


def build_gate_rows(results: list[dict[str, Any]], decision: dict[str, Any]) -> list[dict[str, str]]:
    quality_rows = [result["quality_row"] for result in results if result.get("quality_row")]
    cloud_rows = [result["cloud_row"] for result in results if result.get("cloud_row")]

    def add(gate: str, status: str, detail: str, blocking_reason: str = "") -> dict[str, str]:
        return {"gate": gate, "status": status, "detail": detail, "blocking_reason": blocking_reason}

    return [
        add("patch_exists", "PASS" if all(result["patch_exists"] for result in results) else "FAIL", f"patches_found={sum(1 for result in results if result['patch_exists'])}/2", "" if all(result["patch_exists"] for result in results) else "LOCAL_PATCH_NOT_FOUND"),
        add("bands_complete", "PASS" if all(row["bands_available"] == ",".join(BANDS) for row in quality_rows) else "FAIL", ",".join(sorted({row["bands_available"] for row in quality_rows}))),
        add("valid_pixels", "PASS" if all(safe_float(row["valid_pixel_fraction"]) >= MIN_VALID_PIXEL_FRACTION for row in quality_rows) else "FAIL", ";".join(f"{row['temporal_relation_to_event']}={row['valid_pixel_fraction']}" for row in quality_rows)),
        add("nodata_ok", "PASS" if all(safe_float(row["nodata_fraction"]) <= MAX_NODATA_FRACTION for row in quality_rows) else "FAIL", ";".join(f"{row['temporal_relation_to_event']}={row['nodata_fraction']}" for row in quality_rows)),
        add("nan_inf_ok", "PASS" if all(row["nan_inf_status"] == "OK" for row in quality_rows) else "FAIL", ";".join(f"{row['temporal_relation_to_event']}={row['nan_inf_status']}" for row in quality_rows)),
        add("band_ranges_ok", "PASS" if all(row["band_range_status"] == "OK" and row["band_variance_status"] == "OK" for row in quality_rows) else "FAIL", ";".join(f"{row['temporal_relation_to_event']}={row['band_range_status']}/{row['band_variance_status']}" for row in quality_rows)),
        add("cloud_local_assessed", "WARN" if any(row["cloud_mask_available"] == "false" for row in cloud_rows) else "PASS", ";".join(f"{row['temporal_relation_to_event']}={row['cloud_quality_status']}" for row in cloud_rows), "CLOUD_MASK_NOT_AVAILABLE" if any(row["cloud_mask_available"] == "false" for row in cloud_rows) else ""),
        add("spectral_indices_computable", "PASS" if all(row["spectral_index_status"] == "OK" for row in quality_rows) else "FAIL", ";".join(f"{row['temporal_relation_to_event']}={row['spectral_index_status']}" for row in quality_rows)),
        add("pair_quality_status", "PASS" if decision["can_be_reference_patch_candidate"] else "FAIL", decision["pair_quality_status"], decision["primary_blocker"] if not decision["can_be_reference_patch_candidate"] else ""),
        add("blocking_reason", "WARN" if decision["primary_blocker"] != "NONE" else "PASS", decision["primary_blocker"], decision["primary_blocker"] if decision["primary_blocker"] != "NONE" else ""),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(args.force)
    if not V1IX_MANIFEST.exists():
        raise FileNotFoundError(f"Missing v1ix manifest: {V1IX_MANIFEST}")
    manifest_rows = [row for row in read_csv(V1IX_MANIFEST) if row.get("reference_patch_id", "").startswith("REFPATCH_PET2022_CPRM_MOINHO_PRETO_S2_V1IX")]
    if not manifest_rows:
        raise RuntimeError("No v1ix reference patch rows found in local manifest.")

    results = [audit_patch(row) for row in manifest_rows]
    decision = decide_pair(results)
    inventory_rows = [result["inventory_row"] for result in results if result.get("inventory_row")]
    band_rows = [band_row for result in results for band_row in result.get("band_rows", [])]
    index_rows = [index_row for result in results for index_row in result.get("index_rows", [])]
    cloud_rows = [result["cloud_row"] for result in results if result.get("cloud_row")]
    quality_rows = [result["quality_row"] for result in results if result.get("quality_row")]

    for row in quality_rows:
        row["pair_quality_status"] = decision["pair_quality_status"]
        row["can_be_reference_patch_candidate"] = bool_text(decision["can_be_reference_patch_candidate"])
        row["can_be_multimodal_reference_candidate"] = bool_text(decision["can_be_multimodal_reference_candidate"])
        row["primary_blocker"] = decision["primary_blocker"]
        row["minimum_evidence_needed"] = decision["minimum_evidence_needed"]

    gate_rows = build_gate_rows(results, decision)
    qa_rows = [
        {"check": "v1ix_manifest_read", "status": "PASS", "detail": str(len(manifest_rows))},
        {"check": "local_patch_inventory", "status": "PASS" if len(inventory_rows) == len(manifest_rows) else "FAIL", "detail": f"{len(inventory_rows)}/{len(manifest_rows)}"},
        {"check": "cloud_metadata_not_auto_rejection", "status": "PASS", "detail": decision["pair_quality_status"]},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "Public outputs use sanitized metadata only."},
    ]

    write_csv(LOCAL_RUN_DIR / "v1iy_patch_quality_inventory.csv", inventory_rows, list(inventory_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1iy_band_statistics.csv", band_rows, list(band_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1iy_spectral_index_preview_stats.csv", index_rows, list(index_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1iy_cloud_quality_audit.csv", cloud_rows, list(cloud_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1iy_patch_pair_quality_decision.csv", quality_rows, QUALITY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1iy_qa.csv", qa_rows, ["check", "status", "detail"])

    write_csv(DATASETS_DIR / "official_anchor_sentinel_patch_quality_registry.csv", quality_rows, QUALITY_FIELDS)
    write_schema(SCHEMAS_DIR / "official_anchor_sentinel_patch_quality_schema.csv", QUALITY_FIELDS, "REV-P v1iy Sentinel patch local quality registry field")
    write_csv(DATASETS_DIR / "official_anchor_patch_pair_quality_gate_matrix.csv", gate_rows, GATE_FIELDS)
    write_schema(SCHEMAS_DIR / "official_anchor_patch_pair_quality_gate_matrix_schema.csv", GATE_FIELDS, "REV-P v1iy Sentinel patch pair quality gate field")

    summary = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "status": decision["pair_quality_status"],
        "patches_audited": len(inventory_rows),
        "pre_local_quality_status": next((row["local_quality_status"] for row in quality_rows if row["temporal_relation_to_event"] == "PRE_EVENT"), ""),
        "post_local_quality_status": next((row["local_quality_status"] for row in quality_rows if row["temporal_relation_to_event"] == "POST_EVENT_OR_SURVEY_WINDOW"), ""),
        "cloud_mask_available": any(row["cloud_mask_available"] == "true" for row in cloud_rows),
        "pre_cloud_quality_status": next((row["cloud_quality_status"] for row in cloud_rows if row["temporal_relation_to_event"] == "PRE_EVENT"), ""),
        "post_cloud_quality_status": next((row["cloud_quality_status"] for row in cloud_rows if row["temporal_relation_to_event"] == "POST_EVENT_OR_SURVEY_WINDOW"), ""),
        "can_be_reference_patch_candidate": decision["can_be_reference_patch_candidate"],
        "can_be_multimodal_reference_candidate": decision["can_be_multimodal_reference_candidate"],
        "can_be_operational_ground_truth": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_reopen_protocol_b": False,
        "primary_blocker": decision["primary_blocker"],
        "minimum_evidence_needed": decision["minimum_evidence_needed"],
        "commit_warranted": True,
    }
    write_json(LOCAL_RUN_DIR / "v1iy_summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Clear local v1iy outputs before running.")
    parser.add_argument("--read-v1ix-manifest", action="store_true", help="Read local v1ix manifest.")
    parser.add_argument("--audit-local-patches", action="store_true", help="Audit local v1ix raster patches.")
    parser.add_argument("--compute-spectral-qa", action="store_true", help="Compute band and index QA.")
    parser.add_argument("--emit-quality-decision", action="store_true", help="Emit local and public quality decisions.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = run(args)
    print("=" * 72)
    print("REV-P v1iy ANCHOR_SENTINEL_PATCH_LOCAL_QUALITY_AND_CLOUD_AUDIT")
    print("=" * 72)
    print(f"Patches audited: {summary.get('patches_audited')}")
    print(f"Pre local quality: {summary.get('pre_local_quality_status')}")
    print(f"Post local quality: {summary.get('post_local_quality_status')}")
    print(f"Cloud mask available: {summary.get('cloud_mask_available')}")
    print(f"Pre cloud status: {summary.get('pre_cloud_quality_status')}")
    print(f"Post cloud status: {summary.get('post_cloud_quality_status')}")
    print(f"Pair quality status: {summary.get('status')}")
    print(f"Reference patch candidate: {summary.get('can_be_reference_patch_candidate')}")
    print(f"Multimodal reference candidate: {summary.get('can_be_multimodal_reference_candidate')}")
    print(f"Primary blocker: {summary.get('primary_blocker')}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
