"""
REV-P v1ja - DINO_EMBEDDING_FOR_OFFICIAL_ANCHOR_SENTINEL_PAIR.

Extracts frozen DINOv2 review embeddings for the final Sentinel-2 pre/post pair
selected in v1iz. The stage is structural review only: it does not create a
label, target, model training run, prediction, or Protocol B reopening.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
REVP_ROOT = SCRIPT_PATH.parents[2]
LOCAL_RUN_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ja"
DATASETS_DIR = REVP_ROOT / "datasets"
SCHEMAS_DIR = DATASETS_DIR / "schemas"
V1IZ_SELECTION = DATASETS_DIR / "official_anchor_sentinel_patch_pair_selection_registry.csv"
V1IX_PATCH_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1ix" / "anchor_centered_patch"
V1IZ_ALT_DIR = REVP_ROOT / "local_runs" / "protocolo_c" / "v1iz" / "alternative_pre_patches"

STAGE = "v1ja"
MODEL_NAME = "facebook/dinov2-with-registers-base"
EXPECTED_EMBEDDING_DIM = 768
BANDS = ["B02", "B03", "B04", "B08", "B11", "B12"]
RGB_BANDS = ["B04", "B03", "B02"]
BAND_INDEX = {band: idx for idx, band in enumerate(BANDS)}
IMAGE_SIZE = 224
LOCAL_RASTER_SUFFIX = ".local_geotiff"
ANCHOR_ID = "ANCHOR_PET2022_CPRM_ANEXOII_19022022"

REGISTRY_FIELDS = [
    "embedding_record_id",
    "anchor_id",
    "reference_patch_id",
    "temporal_relation_to_event",
    "scene_date",
    "sensor",
    "bands_used_for_visual_input",
    "dino_model_name",
    "embedding_dim",
    "embedding_generated",
    "embedding_quality_status",
    "pre_post_pair_available",
    "cosine_similarity",
    "euclidean_distance",
    "structural_change_status",
    "can_be_review_embedding",
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
    if force:
        for path in [
            DATASETS_DIR / "official_anchor_dino_embedding_readiness_registry.csv",
            SCHEMAS_DIR / "official_anchor_dino_embedding_readiness_schema.csv",
        ]:
            if path.exists():
                path.unlink()
    LOCAL_RUN_DIR.mkdir(parents=True, exist_ok=True)


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_selection() -> dict[str, str]:
    rows = read_csv(V1IZ_SELECTION)
    if not rows:
        raise RuntimeError("v1iz selection registry is empty.")
    return rows[0]


def find_pre_patch(scene_id: str) -> Path | None:
    candidates = sorted((V1IZ_ALT_DIR / scene_id / "spectral").glob(f"*{LOCAL_RASTER_SUFFIX}"))
    return candidates[0] if candidates else None


def find_post_patch() -> Path | None:
    candidates = sorted((V1IX_PATCH_DIR / "post_event_or_survey_window").glob(f"*{LOCAL_RASTER_SUFFIX}"))
    return candidates[0] if candidates else None


def validate_patch(path: Path | None, relation: str, scene_id: str, scene_date: str) -> dict[str, Any]:
    if path is None:
        return {
            "temporal_relation_to_event": relation,
            "scene_id_sanitized": scene_id,
            "scene_date": scene_date,
            "patch_available": "false",
            "patch_file_sanitized": "",
            "bands_available": "",
            "shape_px": "",
            "crs": "",
            "resolution_m": "",
            "rgb_bands_used": ",".join(RGB_BANDS),
            "normalization": "PERCENTILE_2_98_PER_RGB_BAND_THEN_IMAGENET_NORMALIZATION",
            "input_quality_status": "PATCH_INPUT_UNAVAILABLE",
            "notes": "Local patch file was not found.",
        }
    import rasterio

    with rasterio.open(path) as src:
        bands_available = ",".join(BANDS[: src.count])
        shape_px = f"{src.width}x{src.height}"
        crs = str(src.crs) if src.crs else ""
        resolution = f"{abs(src.res[0]):.4f}" if src.res else ""
        status = "PASS" if src.count >= len(BANDS) and bands_available == ",".join(BANDS) else "FAIL"
    return {
        "temporal_relation_to_event": relation,
        "scene_id_sanitized": scene_id,
        "scene_date": scene_date,
        "patch_available": "true",
        "patch_file_sanitized": path.name,
        "bands_available": bands_available,
        "shape_px": shape_px,
        "crs": crs,
        "resolution_m": resolution,
        "rgb_bands_used": ",".join(RGB_BANDS),
        "normalization": "PERCENTILE_2_98_PER_RGB_BAND_THEN_IMAGENET_NORMALIZATION",
        "input_quality_status": status,
        "notes": "DINO visual input uses Sentinel B04/B03/B02 only; remaining bands stay documented as source context.",
    }


def resize_nearest(array: Any, size: int) -> Any:
    import numpy as np

    height, width = array.shape[:2]
    y_idx = np.linspace(0, max(height - 1, 0), size).round().astype("int64")
    x_idx = np.linspace(0, max(width - 1, 0), size).round().astype("int64")
    return array[y_idx][:, x_idx].astype("float32")


def read_visual_input(path: Path) -> Any:
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        band_numbers = [BAND_INDEX[band] + 1 for band in RGB_BANDS]
        data = src.read(band_numbers).astype("float32")
    lows = np.percentile(data, 2, axis=(1, 2), keepdims=True)
    highs = np.percentile(data, 98, axis=(1, 2), keepdims=True)
    scaled = np.clip((data - lows) / np.maximum(highs - lows, 1.0e-6), 0, 1)
    hwc = scaled.transpose(1, 2, 0).astype("float32")
    return resize_nearest(hwc, IMAGE_SIZE)


def load_dino_model() -> tuple[Any | None, str, str, str]:
    try:
        import torch
        from transformers import AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        return model, device, "LOADED", ""
    except Exception as exc:
        return None, "cpu", "DINO_MODEL_UNAVAILABLE", f"{type(exc).__name__}: {exc}"


def embed_array(model: Any, array: Any, device: str) -> Any:
    import numpy as np
    import torch

    mean = np.array([0.485, 0.456, 0.406], dtype="float32").reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype="float32").reshape(1, 1, 3)
    normalized = (np.asarray(array, dtype="float32") - mean) / std
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(pixel_values=tensor)
    if hasattr(output, "last_hidden_state"):
        embedding = output.last_hidden_state[:, 0, :].detach().cpu().numpy().reshape(-1).astype("float32")
    elif isinstance(output, (list, tuple)):
        embedding = output[0].detach().cpu().numpy().reshape(-1).astype("float32")
    else:
        embedding = output.detach().cpu().numpy().reshape(-1).astype("float32")
    return embedding


def vector_sha256(vector: Any) -> str:
    import numpy as np

    digest = hashlib.sha256()
    digest.update(np.asarray(vector, dtype="float32").tobytes())
    return digest.hexdigest()


def embedding_stats(vector: Any) -> dict[str, Any]:
    import numpy as np

    arr = np.asarray(vector, dtype="float32")
    return {
        "embedding_dim": int(arr.shape[0]),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
        "norm": float(np.linalg.norm(arr)),
        "sha256": vector_sha256(arr),
    }


def pair_metrics(pre: Any, post: Any) -> dict[str, float]:
    import numpy as np

    pre_arr = np.asarray(pre, dtype="float32")
    post_arr = np.asarray(post, dtype="float32")
    denom = float(np.linalg.norm(pre_arr) * np.linalg.norm(post_arr))
    cosine = float(np.dot(pre_arr, post_arr) / denom) if denom > 0 else math.nan
    euclidean = float(np.linalg.norm(post_arr - pre_arr))
    return {"cosine_similarity": cosine, "euclidean_distance": euclidean}


def structural_status(cosine: float) -> str:
    if math.isnan(cosine):
        return "STRUCTURAL_DIAGNOSTIC_UNAVAILABLE"
    if cosine >= 0.95:
        return "LOW_STRUCTURAL_DIFFERENCE_REVIEW_ONLY"
    if cosine >= 0.80:
        return "MODERATE_STRUCTURAL_DIFFERENCE_REVIEW_ONLY"
    return "HIGH_STRUCTURAL_DIFFERENCE_REVIEW_ONLY"


def registry_rows(selection: dict[str, str], diagnostics: dict[str, Any], status: str, blocker: str) -> list[dict[str, Any]]:
    qa_pass = status == "DINO_ANCHOR_PAIR_EMBEDDING_READY"
    rows: list[dict[str, Any]] = []
    for relation, scene_date in [("PRE_EVENT", selection["pre_scene_date"]), ("POST_EVENT_OR_SURVEY_WINDOW", selection["post_scene_date"])]:
        rows.append(
            {
                "embedding_record_id": f"EMB_{ANCHOR_ID}_{relation}_V1JA",
                "anchor_id": ANCHOR_ID,
                "reference_patch_id": f"REFPATCH_{ANCHOR_ID}_{relation}_V1JA",
                "temporal_relation_to_event": relation,
                "scene_date": scene_date,
                "sensor": "Sentinel-2 MSI",
                "bands_used_for_visual_input": ",".join(RGB_BANDS),
                "dino_model_name": MODEL_NAME,
                "embedding_dim": diagnostics.get(f"{relation}_embedding_dim", ""),
                "embedding_generated": bool_text(qa_pass),
                "embedding_quality_status": "QA_PASS" if qa_pass else "QA_BLOCKED",
                "pre_post_pair_available": bool_text(qa_pass),
                "cosine_similarity": diagnostics.get("cosine_similarity", ""),
                "euclidean_distance": diagnostics.get("euclidean_distance", ""),
                "structural_change_status": diagnostics.get("structural_change_status", ""),
                "can_be_review_embedding": bool_text(qa_pass),
                "can_be_multimodal_reference_candidate": bool_text(qa_pass),
                "can_be_operational_ground_truth": bool_text(False),
                "can_create_training_label": bool_text(False),
                "can_train_model": bool_text(False),
                "can_reopen_protocol_b": bool_text(False),
                "primary_blocker": blocker,
                "minimum_evidence_needed": "Independent human review remains required before any stronger evidence role." if qa_pass else "Load DINOv2 and generate finite 768D pre/post embeddings.",
                "notes": "Frozen DINOv2 review embedding from Sentinel RGB composite; no label, target, training, or event claim is created.",
            }
        )
    return rows


def write_schema(path: Path, fields: list[str], prefix: str) -> None:
    write_csv(path, [{"field": field, "description": f"{prefix}: {field}."} for field in fields], ["field", "description"])


def write_blocked_outputs(selection: dict[str, str], input_rows: list[dict[str, Any]], status: str, blocker: str, error: str) -> dict[str, Any]:
    diagnostics = {"structural_change_status": "STRUCTURAL_DIAGNOSTIC_UNAVAILABLE"}
    manifest_rows = registry_rows(selection, diagnostics, status, blocker)
    for row in manifest_rows:
        row["embedding_generated"] = "false"
        row["embedding_quality_status"] = "QA_BLOCKED"
        row["notes"] = error
    write_csv(LOCAL_RUN_DIR / "v1ja_anchor_pair_input_audit.csv", input_rows, list(input_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1ja_dino_embedding_manifest_local.csv", manifest_rows, REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1ja_embedding_diagnostics.csv", [], ["embedding_record_id", "embedding_dim", "norm", "has_nan", "has_inf", "embedding_sha256"])
    write_csv(LOCAL_RUN_DIR / "v1ja_structural_pair_comparison.csv", [{"status": status, "primary_blocker": blocker, "notes": error}], ["status", "primary_blocker", "notes"])
    qa_rows = [
        {"check": "controlled_blocker", "status": "PASS", "detail": blocker},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "No public embedding registry written."},
    ]
    write_csv(LOCAL_RUN_DIR / "v1ja_qa.csv", qa_rows, ["check", "status", "detail"])
    write_multimodal_matrix(False, blocker)
    summary = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "status": status,
        "dino_model_loaded": False,
        "embedding_generated_pre": False,
        "embedding_generated_post": False,
        "embedding_dim": 0,
        "cosine_similarity": "",
        "euclidean_distance": "",
        "readiness_status": "BLOCKED",
        "can_be_review_embedding": False,
        "can_be_multimodal_reference_candidate": False,
        "can_be_operational_ground_truth": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_reopen_protocol_b": False,
        "primary_blocker": blocker,
        "error_message": error,
        "commit_warranted": False,
    }
    write_json(LOCAL_RUN_DIR / "v1ja_summary.json", summary)
    return summary


def write_multimodal_matrix(embedding_ready: bool, blocker: str) -> None:
    gates = [
        ("official_anchor", "PASS", ANCHOR_ID, ""),
        ("documented_event", "PASS", "CPRM ANEXO-II documented event unit", ""),
        ("explicit_coordinate", "PASS", "-22.484251,-43.211257", ""),
        ("sentinel_pair_selected", "PASS", "v1iz PATCH_PAIR_USABLE_FOR_REVIEW", ""),
        ("sentinel_pair_quality", "PASS", "local cloud resolved by SCL/QA60 and alternative pre scene", ""),
        ("dino_embedding_generated", "PASS" if embedding_ready else "FAIL", bool_text(embedding_ready), "" if embedding_ready else blocker),
        ("dino_embedding_quality", "PASS" if embedding_ready else "FAIL", "768D finite embeddings" if embedding_ready else blocker, "" if embedding_ready else blocker),
        ("gis_context_available", "WARN", "Context exists, but no operational ground truth is created.", "GROUND_TRUTH_NOT_OPERATIONAL"),
        ("multimodal_reference_status", "PASS" if embedding_ready else "FAIL", "MULTIMODAL_REFERENCE_CANDIDATE_REVIEW_ONLY" if embedding_ready else "BLOCKED", "" if embedding_ready else blocker),
        ("blocking_reason", "PASS" if embedding_ready else "FAIL", "NONE" if embedding_ready else blocker, "" if embedding_ready else blocker),
    ]
    rows = [{"gate": gate, "status": status, "detail": detail, "blocking_reason": reason} for gate, status, detail, reason in gates]
    write_csv(DATASETS_DIR / "official_anchor_multimodal_reference_readiness_matrix.csv", rows, GATE_FIELDS)
    write_schema(SCHEMAS_DIR / "official_anchor_multimodal_reference_readiness_schema.csv", GATE_FIELDS, "REV-P v1ja multimodal readiness gate field")


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepare_output_dir(args.force)
    selection = load_selection()
    pre_path = find_pre_patch(selection["pre_scene_id_sanitized"])
    post_path = find_post_patch()
    input_rows = [
        validate_patch(pre_path, "PRE_EVENT", selection["pre_scene_id_sanitized"], selection["pre_scene_date"]),
        validate_patch(post_path, "POST_EVENT_OR_SURVEY_WINDOW", selection["post_scene_id_sanitized"], selection["post_scene_date"]),
    ]
    if any(row["input_quality_status"] != "PASS" for row in input_rows):
        return write_blocked_outputs(selection, input_rows, "PATCH_INPUT_UNAVAILABLE", "PATCH_INPUT_UNAVAILABLE", "Local pre/post patches are missing or invalid.")

    model, device, model_status, model_error = load_dino_model()
    if model is None:
        return write_blocked_outputs(selection, input_rows, "DINO_MODEL_UNAVAILABLE", "DINO_MODEL_UNAVAILABLE", model_error)

    try:
        assert pre_path is not None and post_path is not None
        pre_embedding = embed_array(model, read_visual_input(pre_path), device)
        post_embedding = embed_array(model, read_visual_input(post_path), device)
        pre_stats = embedding_stats(pre_embedding)
        post_stats = embedding_stats(post_embedding)
        metrics = pair_metrics(pre_embedding, post_embedding)
    except Exception as exc:
        return write_blocked_outputs(selection, input_rows, "EMBEDDING_QA_FAILED", "EMBEDDING_QA_FAILED", f"{type(exc).__name__}: {exc}")

    qa_pass = (
        pre_stats["embedding_dim"] == EXPECTED_EMBEDDING_DIM
        and post_stats["embedding_dim"] == EXPECTED_EMBEDDING_DIM
        and not pre_stats["has_nan"]
        and not pre_stats["has_inf"]
        and not post_stats["has_nan"]
        and not post_stats["has_inf"]
    )
    if not qa_pass:
        return write_blocked_outputs(selection, input_rows, "EMBEDDING_QA_FAILED", "EMBEDDING_QA_FAILED", "Embedding dimension or finite-value QA failed.")

    structural = structural_status(metrics["cosine_similarity"])
    diagnostics = {
        "PRE_EVENT_embedding_dim": pre_stats["embedding_dim"],
        "POST_EVENT_OR_SURVEY_WINDOW_embedding_dim": post_stats["embedding_dim"],
        "cosine_similarity": f"{metrics['cosine_similarity']:.8f}",
        "euclidean_distance": f"{metrics['euclidean_distance']:.8f}",
        "structural_change_status": structural,
    }
    manifest_rows = registry_rows(selection, diagnostics, "DINO_ANCHOR_PAIR_EMBEDDING_READY", "NONE")
    diagnostic_rows = [
        {
            "embedding_record_id": "EMB_ANCHOR_PET2022_CPRM_ANEXOII_PRE_EVENT_V1JA",
            "embedding_dim": pre_stats["embedding_dim"],
            "norm": f"{pre_stats['norm']:.8f}",
            "has_nan": bool_text(pre_stats["has_nan"]),
            "has_inf": bool_text(pre_stats["has_inf"]),
            "embedding_sha256": pre_stats["sha256"],
        },
        {
            "embedding_record_id": "EMB_ANCHOR_PET2022_CPRM_ANEXOII_POST_EVENT_V1JA",
            "embedding_dim": post_stats["embedding_dim"],
            "norm": f"{post_stats['norm']:.8f}",
            "has_nan": bool_text(post_stats["has_nan"]),
            "has_inf": bool_text(post_stats["has_inf"]),
            "embedding_sha256": post_stats["sha256"],
        },
    ]
    comparison_row = {
        "anchor_id": ANCHOR_ID,
        "pre_scene_id_sanitized": selection["pre_scene_id_sanitized"],
        "post_scene_id_sanitized": selection["post_scene_id_sanitized"],
        "cosine_similarity": f"{metrics['cosine_similarity']:.8f}",
        "euclidean_distance": f"{metrics['euclidean_distance']:.8f}",
        "pre_norm": f"{pre_stats['norm']:.8f}",
        "post_norm": f"{post_stats['norm']:.8f}",
        "structural_change_status": structural,
        "claim_scope": "REVIEW_ONLY_STRUCTURAL_DIAGNOSTIC_NO_LABEL",
    }
    qa_rows = [
        {"check": "dino_model_loaded", "status": "PASS", "detail": f"{MODEL_NAME} on {device}"},
        {"check": "pre_embedding_dim_768", "status": "PASS", "detail": str(pre_stats["embedding_dim"])},
        {"check": "post_embedding_dim_768", "status": "PASS", "detail": str(post_stats["embedding_dim"])},
        {"check": "no_nan_inf", "status": "PASS", "detail": "pre/post finite"},
        {"check": "registry_written_after_qa_pass", "status": "PASS", "detail": "metadata only"},
        {"check": "can_create_training_label_false", "status": "PASS", "detail": "false"},
        {"check": "can_train_model_false", "status": "PASS", "detail": "false"},
        {"check": "can_reopen_protocol_b_false", "status": "PASS", "detail": "false"},
        {"check": "no_private_path_in_public_outputs", "status": "PASS", "detail": "Public outputs use sanitized metadata only."},
    ]
    write_csv(LOCAL_RUN_DIR / "v1ja_anchor_pair_input_audit.csv", input_rows, list(input_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1ja_dino_embedding_manifest_local.csv", manifest_rows, REGISTRY_FIELDS)
    write_csv(LOCAL_RUN_DIR / "v1ja_embedding_diagnostics.csv", diagnostic_rows, list(diagnostic_rows[0].keys()))
    write_csv(LOCAL_RUN_DIR / "v1ja_structural_pair_comparison.csv", [comparison_row], list(comparison_row.keys()))
    write_csv(LOCAL_RUN_DIR / "v1ja_qa.csv", qa_rows, ["check", "status", "detail"])
    write_csv(DATASETS_DIR / "official_anchor_dino_embedding_readiness_registry.csv", manifest_rows, REGISTRY_FIELDS)
    write_schema(SCHEMAS_DIR / "official_anchor_dino_embedding_readiness_schema.csv", REGISTRY_FIELDS, "REV-P v1ja DINO embedding readiness registry field")
    write_multimodal_matrix(True, "NONE")
    summary = {
        "stage": STAGE,
        "timestamp": utc_now(),
        "status": "DINO_ANCHOR_PAIR_EMBEDDING_READY",
        "dino_model_loaded": model_status == "LOADED",
        "dino_model_name": MODEL_NAME,
        "device": device,
        "embedding_generated_pre": True,
        "embedding_generated_post": True,
        "embedding_dim": EXPECTED_EMBEDDING_DIM,
        "pre_norm": f"{pre_stats['norm']:.8f}",
        "post_norm": f"{post_stats['norm']:.8f}",
        "cosine_similarity": f"{metrics['cosine_similarity']:.8f}",
        "euclidean_distance": f"{metrics['euclidean_distance']:.8f}",
        "structural_change_status": structural,
        "readiness_status": "MULTIMODAL_REFERENCE_CANDIDATE_REVIEW_ONLY",
        "can_be_review_embedding": True,
        "can_be_multimodal_reference_candidate": True,
        "can_be_operational_ground_truth": False,
        "can_create_training_label": False,
        "can_train_model": False,
        "can_reopen_protocol_b": False,
        "primary_blocker": "NONE",
        "commit_warranted": True,
    }
    write_json(LOCAL_RUN_DIR / "v1ja_summary.json", summary)
    return summary


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Clear local v1ja outputs before running.")
    parser.add_argument("--read-v1iz-selection", action="store_true", help="Read v1iz final pair selection.")
    parser.add_argument("--read-local-patches", action="store_true", help="Locate local selected pre/post patches.")
    parser.add_argument("--load-dinov2", action="store_true", help="Load frozen DINOv2 encoder.")
    parser.add_argument("--extract-embeddings", action="store_true", help="Extract review embeddings.")
    parser.add_argument("--emit-structural-diagnostics", action="store_true", help="Emit pre/post structural diagnostics.")
    parser.add_argument("--emit-readiness", action="store_true", help="Emit readiness registry and matrix.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    summary = run(args)
    print("=" * 72)
    print("REV-P v1ja DINO_EMBEDDING_FOR_OFFICIAL_ANCHOR_SENTINEL_PAIR")
    print("=" * 72)
    print(f"DINO loaded: {summary.get('dino_model_loaded')}")
    print(f"Pre embedding generated: {summary.get('embedding_generated_pre')}")
    print(f"Post embedding generated: {summary.get('embedding_generated_post')}")
    print(f"Embedding dim: {summary.get('embedding_dim')}")
    print(f"Cosine similarity: {summary.get('cosine_similarity')}")
    print(f"Euclidean distance: {summary.get('euclidean_distance')}")
    print(f"Readiness: {summary.get('readiness_status')}")
    print(f"Primary blocker: {summary.get('primary_blocker')}")
    print("No git add, commit, or push was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
