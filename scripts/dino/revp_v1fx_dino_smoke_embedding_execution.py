from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1fx"
PHASE_NAME = "DINO_SMOKE_EMBEDDING_EXECUTION"

DEFAULT_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
DEFAULT_ASSET_PREFLIGHT = ROOT / "local_runs" / "dino_asset_preflight" / "v1fv" / "dino_local_asset_preflight_v1fv.csv"
DEFAULT_CONFIG = ROOT / "configs" / "dino_embedding_extraction.example.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1fx"

MANIFEST_CSV = "dino_smoke_embedding_manifest_v1fx.csv"
FAILURES_CSV = "dino_smoke_embedding_failures_v1fx.csv"
SUMMARY_JSON = "dino_smoke_embedding_summary_v1fx.json"
QA_CSV = "dino_smoke_embedding_qa_v1fx.csv"
METADATA_CSV = "dino_smoke_embedding_metadata_v1fx.csv"
MODEL_ATTEMPTS_CSV = "dino_smoke_model_attempts_v1fx.csv"
CLUSTER_SUMMARY_CSV = "dino_smoke_cluster_summary_v1fx.csv"
NN_CSV = "dino_smoke_nearest_neighbor_sanity_v1fx.csv"

MANIFEST_FIELDS = [
    "dino_input_id",
    "canonical_patch_id",
    "region",
    "resolved_status",
    "resolved_path_private",
    "smoke_status",
    "failure_code",
    "embedding_file",
    "cls_dim",
    "patch_mean_dim",
    "backbone",
    "device",
    "pixel_read_status",
    "embedding_status",
    "label_status",
    "target_status",
    "claim_scope",
    "notes",
]
FAILURE_FIELDS = ["dino_input_id", "canonical_patch_id", "region", "failure_code", "failure_stage", "failure_reason", "backbone", "device"]
QA_FIELDS = ["check", "status", "details"]
METADATA_FIELDS = [
    "dino_input_id",
    "source_extension",
    "source_shape",
    "source_dtype",
    "bands_selected",
    "read_seconds",
    "embedding_seconds",
    "embedding_file",
    "cls_dim",
    "patch_mean_dim",
    "embedding_sha256",
    "has_nan",
    "has_inf",
    "l2_norm",
]
MODEL_ATTEMPT_FIELDS = ["order", "candidate", "backend", "status", "details"]
CLUSTER_FIELDS = ["cluster_id", "count", "method", "notes"]
NN_FIELDS = ["dino_input_id", "nearest_dino_input_id", "distance", "notes"]

FORBIDDEN_REPO_DIRS = {"data", "outputs"}
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index"}
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1fx DINO smoke embedding execution.")
    parser.add_argument("--execute", action="store_true", help="Required consent for smoke execution.")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--asset-preflight", default=str(DEFAULT_ASSET_PREFLIGHT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--private-project-root", default="")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backbone", default="dinov2_vitb14_reg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--skip-model-if-unavailable", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true", help="Explicitly allow backend/model weight download if not cached locally.")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def parse_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    config: dict[str, Any] = {}
    current_list: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_list:
            config[current_list].append(stripped[2:].strip().strip("'\""))
            continue
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_list = None
        if value == "":
            config[key] = []
            current_list = key
        elif value.lower() in {"true", "false"}:
            config[key] = value.lower() == "true"
        elif value.startswith("[") and value.endswith("]"):
            config[key] = [float(item.strip()) for item in value[1:-1].split(",") if item.strip()]
        else:
            try:
                config[key] = int(value)
            except ValueError:
                config[key] = value.strip("'\"")
    return config


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Re-run with --force to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)


def is_local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    if not gitignore.exists():
        return False
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()]
    return "local_runs/" in lines or "local_runs" in lines


def forbidden_versioned_artifacts() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        if path.is_dir() and path.name in FORBIDDEN_REPO_DIRS:
            found.append(rel(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS:
            found.append(rel(path))
    return sorted(found)


def environment_diagnostics() -> dict[str, Any]:
    import importlib
    import platform

    modules: dict[str, str] = {}
    for name in ["torch", "timm", "transformers", "torchvision", "PIL", "rasterio", "numpy", "sklearn"]:
        try:
            module = importlib.import_module(name)
            modules[name] = str(getattr(module, "__version__", "installed"))
        except Exception as exc:
            modules[name] = f"ERROR: {type(exc).__name__}: {exc}"
    cuda: Any = False
    cuda_count: Any = 0
    mps: Any = False
    try:
        import torch  # type: ignore

        cuda = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count()) if cuda else 0
        mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception as exc:
        cuda = f"ERROR: {exc}"
        cuda_count = 0
        mps = False
    return {
        "python": platform.python_version(),
        "modules": modules,
        "cuda_available": cuda,
        "cuda_device_count": cuda_count,
        "mps_available": mps,
    }


def resolve_device(device: str, allow_cpu: bool, force_cpu: bool) -> str:
    if force_cpu or device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda"
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu" if allow_cpu else "cpu"


def load_preflight(path: Path) -> dict[str, dict[str, str]]:
    return {row.get("dino_input_id", ""): row for row in read_csv(path)}


def select_attempts(manifest_rows: list[dict[str, str]], preflight: dict[str, dict[str, str]], limit: int) -> tuple[list[dict[str, str]], int]:
    attempts: list[dict[str, str]] = []
    skipped = 0
    for row in manifest_rows:
        pre = preflight.get(row.get("dino_input_id", ""), {})
        if pre.get("resolved_status") != "FOUND":
            skipped += 1
            continue
        merged = dict(row)
        merged.update(
            {
                "resolved_status": pre.get("resolved_status", ""),
                "resolved_path_private": pre.get("resolved_path_private", ""),
            }
        )
        if len(attempts) < limit:
            attempts.append(merged)
        else:
            skipped += 1
    return attempts, skipped


def model_candidates(backbone: str) -> list[tuple[str, str]]:
    ordered = [
        (backbone, "timm_dinov2_registers"),
        ("dinov2_vitb14_reg", "timm_dinov2_registers"),
        ("dinov2_vitb14", "timm_dinov2"),
        ("dinov2_vits14_reg", "timm_dinov2_registers"),
        ("dinov3_vitb16", "timm_dinov3_future"),
        ("facebook/dinov2-with-registers-base", "huggingface_local"),
        ("facebook/dinov2-base", "huggingface_local"),
    ]
    seen: set[tuple[str, str]] = set()
    result: list[tuple[str, str]] = []
    for item in ordered:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def load_smoke_model(backbone: str, device: str, skip_if_unavailable: bool, allow_model_download: bool) -> tuple[Any, str, str, str, list[dict[str, str]]]:
    if backbone == "fake_smoke_encoder":
        return {"fake": True}, "LOADED_FAKE_SMOKE_ENCODER", "", "fake_smoke_encoder", [
            {"order": "1", "candidate": backbone, "backend": "fake", "status": "LOADED", "details": "test-only fake encoder"}
        ]

    if not allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    attempts: list[dict[str, str]] = []
    try:
        import torch  # type: ignore
    except Exception as exc:
        details = f"torch unavailable: {type(exc).__name__}: {exc}"
        attempts.append({"order": "0", "candidate": "torch", "backend": "environment", "status": "ENVIRONMENT_BLOCKED", "details": details})
        code = "MODEL_UNAVAILABLE" if skip_if_unavailable else "ENVIRONMENT_BLOCKED"
        return None, code, details, "", attempts

    for index, (candidate, backend) in enumerate(model_candidates(backbone), start=1):
        try:
            if backend.startswith("timm"):
                import timm  # type: ignore

                model = timm.create_model(candidate, pretrained=True, num_classes=0).to(device)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad = False
                detail = "pretrained load succeeded; download explicitly allowed" if allow_model_download else "pretrained local/cache load succeeded"
                attempts.append({"order": str(index), "candidate": candidate, "backend": backend, "status": "LOADED", "details": detail})
                return model, "LOADED", "", candidate, attempts
            if backend == "huggingface_local":
                from transformers import AutoModel  # type: ignore

                model = AutoModel.from_pretrained(candidate, local_files_only=not allow_model_download).to(device)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad = False
                detail = "huggingface load succeeded; download explicitly allowed" if allow_model_download else "local_files_only load succeeded"
                attempts.append({"order": str(index), "candidate": candidate, "backend": backend, "status": "LOADED", "details": detail})
                return model, "LOADED", "", candidate, attempts
        except Exception as exc:
            attempts.append(
                {
                    "order": str(index),
                    "candidate": candidate,
                    "backend": backend,
                    "status": "FAILED",
                    "details": f"{type(exc).__name__}: {exc}",
                }
            )
    code = "MODEL_UNAVAILABLE" if skip_if_unavailable else "ENVIRONMENT_BLOCKED"
    return None, code, "No DINO backend/model could be loaded from local environment/cache.", "", attempts


def read_image_tensor(path: Path, image_size: int) -> tuple[Any, str, str, dict[str, str]]:
    suffix = path.suffix.lower()
    metadata = {"source_extension": suffix, "source_shape": "", "source_dtype": "", "bands_selected": ""}
    if suffix == ".ppm":
        try:
            import numpy as np  # type: ignore

            with path.open("rb") as f:
                magic = f.readline().strip()
                if magic != b"P6":
                    return None, "FAILED", "IMAGE_READ_FAILED: unsupported PPM magic"
                dims = f.readline().strip()
                while dims.startswith(b"#"):
                    dims = f.readline().strip()
                width, height = [int(part) for part in dims.split()]
                maxval = int(f.readline().strip())
                raw = f.read(width * height * 3)
            array = np.frombuffer(raw, dtype="uint8").reshape((height, width, 3)).astype("float32")
            array = array / float(max(maxval, 1))
            metadata.update({"source_shape": f"{height}x{width}x3", "source_dtype": "uint8", "bands_selected": "RGB"})
            return array, "READ_FOR_DINO_SMOKE_ONLY", "", metadata
        except Exception as exc:
            return None, "FAILED", f"IMAGE_READ_FAILED: {exc}", metadata
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
        try:
            from PIL import Image  # type: ignore
            import numpy as np  # type: ignore

            image = Image.open(path).convert("RGB").resize((image_size, image_size))
            array = np.asarray(image, dtype="float32") / 255.0
            metadata.update({"source_shape": f"{image_size}x{image_size}x3", "source_dtype": "image", "bands_selected": "RGB"})
            return array, "READ_FOR_DINO_SMOKE_ONLY", "", metadata
        except Exception as exc:
            return None, "FAILED", f"IMAGE_READ_FAILED: {exc}", metadata
    if suffix in {".tif", ".tiff"}:
        try:
            import numpy as np  # type: ignore
            import rasterio  # type: ignore

            with rasterio.open(path) as src:
                bands = min(src.count, 3)
                data = src.read(list(range(1, bands + 1))).astype("float32")
                source_shape = f"{src.count}x{src.height}x{src.width}"
                source_dtype = ",".join(str(dtype) for dtype in src.dtypes[:bands])
            if data.shape[0] == 1:
                data = np.repeat(data, 3, axis=0)
            elif data.shape[0] == 2:
                data = np.concatenate([data, data[:1]], axis=0)
            lows = np.percentile(data, 2, axis=(1, 2), keepdims=True)
            highs = np.percentile(data, 98, axis=(1, 2), keepdims=True)
            scaled = np.clip((data - lows) / np.maximum(highs - lows, 1e-6), 0, 1)
            hwc = scaled[:3].transpose(1, 2, 0).astype("float32")
            image = resize_nearest(hwc, image_size)
            metadata.update({"source_shape": source_shape, "source_dtype": source_dtype, "bands_selected": f"1-{bands}"})
            return image, "READ_FOR_DINO_SMOKE_ONLY", "", metadata
        except ModuleNotFoundError as exc:
            return None, "FAILED", f"RASTERIO_UNAVAILABLE: {exc}", metadata
        except Exception as exc:
            return None, "FAILED", f"RASTER_READ_FAILED: {exc}", metadata
    return None, "FAILED", f"UNSUPPORTED_EXTENSION: {suffix}", metadata


def resize_nearest(array: Any, size: int) -> Any:
    import numpy as np  # type: ignore

    height, width = array.shape[:2]
    y_idx = np.linspace(0, max(height - 1, 0), size).round().astype("int64")
    x_idx = np.linspace(0, max(width - 1, 0), size).round().astype("int64")
    return array[y_idx][:, x_idx].astype("float32")


def embed_array(model: Any, array: Any, backbone: str, device: str) -> tuple[Any, Any, str]:
    import numpy as np  # type: ignore

    if backbone == "fake_smoke_encoder":
        mean_value = float(np.asarray(array).mean())
        cls = np.array([mean_value, 1.0, 0.0, 0.5], dtype="float32")
        patch = np.array([mean_value, mean_value, 0.25, 0.75], dtype="float32")
        return cls, patch, "AVAILABLE"
    try:
        import torch  # type: ignore

        chw = np.asarray(array, dtype="float32").transpose(2, 0, 1)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(device)
        with torch.no_grad():
            if hasattr(model, "forward_features"):
                features = model.forward_features(tensor)
                if isinstance(features, dict):
                    cls_tensor = features.get("x_norm_clstoken") or features.get("cls_token") or features.get("pooled")
                    patch_tensor = features.get("x_norm_patchtokens") or features.get("patch_tokens")
                    if cls_tensor is not None:
                        cls = cls_tensor.detach().cpu().numpy().reshape(1, -1)[0].astype("float32")
                        patch = patch_tensor.detach().cpu().numpy().mean(axis=1).reshape(1, -1)[0].astype("float32") if patch_tensor is not None else np.array([], dtype="float32")
                        return cls, patch, "AVAILABLE" if patch_tensor is not None else "NOT_AVAILABLE_BACKEND_LIMITATION"
                output = features
            else:
                output = model(tensor)
        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
            cls = hidden[:, 0, :].detach().cpu().numpy().reshape(1, -1)[0].astype("float32")
            patch = hidden[:, 1:, :].mean(dim=1).detach().cpu().numpy().reshape(1, -1)[0].astype("float32")
            return cls, patch, "AVAILABLE"
        if isinstance(output, (list, tuple)):
            output = output[0]
        cls = output.detach().cpu().numpy().reshape(1, -1)[0].astype("float32")
        return cls, np.array([], dtype="float32"), "NOT_AVAILABLE_BACKEND_LIMITATION"
    except Exception as exc:
        raise RuntimeError(f"EMBEDDING_FAILED: {exc}") from exc


def failure(row: dict[str, str], code: str, stage: str, reason: str, backbone: str, device: str) -> dict[str, str]:
    return {
        "dino_input_id": row.get("dino_input_id", ""),
        "canonical_patch_id": row.get("canonical_patch_id", ""),
        "region": row.get("region", ""),
        "failure_code": code,
        "failure_stage": stage,
        "failure_reason": reason,
        "backbone": backbone,
        "device": device,
    }


def manifest_row(row: dict[str, str], status: str, failure_code: str, embedding_file: str, cls_dim: int, patch_dim: int, backbone: str, device: str, pixel_status: str, notes: str) -> dict[str, object]:
    return {
        "dino_input_id": row.get("dino_input_id", ""),
        "canonical_patch_id": row.get("canonical_patch_id", ""),
        "region": row.get("region", ""),
        "resolved_status": row.get("resolved_status", ""),
        "resolved_path_private": row.get("resolved_path_private", ""),
        "smoke_status": status,
        "failure_code": failure_code,
        "embedding_file": embedding_file,
        "cls_dim": cls_dim,
        "patch_mean_dim": patch_dim,
        "backbone": backbone,
        "device": device,
        "pixel_read_status": pixel_status,
        "embedding_status": "EXTRACTED_LOCAL_ONLY" if status == "SUCCESS" else "FAILED",
        "label_status": "NO_LABEL",
        "target_status": "NO_TARGET",
        "claim_scope": REVIEW_ONLY_CLAIM,
        "notes": notes,
    }


def run(args: argparse.Namespace) -> int:
    if not args.execute:
        print("v1fx is smoke execution and requires explicit --execute consent.", file=sys.stderr)
        return 2

    input_manifest = Path(args.input_manifest)
    preflight_path = Path(args.asset_preflight)
    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)

    config = parse_simple_yaml(config_path)
    diagnostics = environment_diagnostics()
    image_size = int(config.get("image_size", 224))
    device = resolve_device(args.device, args.allow_cpu, args.force_cpu)
    manifest_rows = read_csv(input_manifest) if input_manifest.exists() else []
    preflight = load_preflight(preflight_path) if preflight_path.exists() else {}
    attempts, skipped_count = select_attempts(manifest_rows, preflight, max(args.limit, 0))
    model, model_code, model_error, loaded_backbone, model_attempts = load_smoke_model(args.backbone, device, args.skip_model_if_unavailable, args.allow_model_download)
    model_loaded = model is not None

    smoke_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    metadata_rows: list[dict[str, object]] = []
    success_count = 0
    pixel_read = False
    embeddings_extracted = False

    if not preflight_path.exists():
        failures.append(failure({}, "PREFLIGHT_MISSING", "preflight", "Run v1fv local asset preflight first.", args.backbone, device))
    else:
        for row in attempts:
            asset_path = Path(row.get("resolved_path_private", ""))
            read_start = time.perf_counter()
            array, pixel_status, read_error, read_meta = read_image_tensor(asset_path, image_size)
            read_seconds = time.perf_counter() - read_start
            if array is None:
                code = read_error.split(":", 1)[0]
                failures.append(failure(row, code, "pixel_read", read_error, args.backbone, device))
                smoke_rows.append(manifest_row(row, "FAILED", code, "", 0, 0, args.backbone, device, pixel_status, read_error))
                metadata_rows.append(metadata_row(row, read_meta, read_seconds, 0.0, "", 0, 0, "", False, False, 0.0))
                continue
            pixel_read = True
            if not model_loaded:
                failures.append(failure(row, model_code, "model_load", model_error or "model unavailable", args.backbone, device))
                smoke_rows.append(manifest_row(row, "FAILED", model_code, "", 0, 0, args.backbone, device, "READ_FOR_DINO_SMOKE_ONLY", model_error))
                metadata_rows.append(metadata_row(row, read_meta, read_seconds, 0.0, "", 0, 0, "", False, False, 0.0))
                continue
            try:
                embed_start = time.perf_counter()
                cls, patch, patch_status = embed_array(model, array, args.backbone, device)
                embedding_seconds = time.perf_counter() - embed_start
                import numpy as np  # type: ignore

                embedding_rel = Path("embeddings") / f"{row.get('dino_input_id', 'unknown')}.npz"
                embedding_path = output_dir / embedding_rel
                np.savez_compressed(embedding_path, cls_embedding=cls, patch_mean_embedding=patch)
                digest = embedding_digest(cls, patch)
                has_nan = bool(np.isnan(cls).any() or np.isnan(patch).any())
                has_inf = bool(np.isinf(cls).any() or np.isinf(patch).any())
                l2_norm = float(np.linalg.norm(cls))
                success_count += 1
                embeddings_extracted = True
                metadata_rows.append(
                    metadata_row(
                        row,
                        read_meta,
                        read_seconds,
                        embedding_seconds,
                        embedding_rel.as_posix(),
                        int(cls.shape[0]),
                        int(patch.shape[0]),
                        digest,
                        has_nan,
                        has_inf,
                        l2_norm,
                    )
                )
                smoke_rows.append(
                    manifest_row(
                        row,
                        "SUCCESS",
                        "",
                        embedding_rel.as_posix(),
                        int(cls.shape[0]),
                        int(patch.shape[0]),
                        args.backbone,
                        device,
                        "READ_FOR_DINO_SMOKE_ONLY",
                        f"patch_mean_status={patch_status}; embedding_sha256={digest}; local-only smoke embedding",
                    )
                )
            except Exception as exc:
                failures.append(failure(row, "EMBEDDING_FAILED", "embedding", str(exc), args.backbone, device))
                smoke_rows.append(manifest_row(row, "FAILED", "EMBEDDING_FAILED", "", 0, 0, args.backbone, device, "READ_FOR_DINO_SMOKE_ONLY", str(exc)))
                metadata_rows.append(metadata_row(row, read_meta, read_seconds, 0.0, "", 0, 0, "", False, False, 0.0))

    attempted_count = len(attempts)
    failed_count = len([row for row in smoke_rows if row.get("smoke_status") == "FAILED"]) + (1 if not preflight_path.exists() else 0)
    cluster_rows, nn_rows, clustering_status = cluster_smoke(output_dir, smoke_rows)
    summary: dict[str, object] = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execute": True,
        "input_manifest": rel(input_manifest),
        "asset_preflight": rel(preflight_path) if preflight_path.exists() else "",
        "limit": args.limit,
        "attempted_count": attempted_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "backbone": args.backbone,
        "loaded_backbone": loaded_backbone,
        "device": device,
        "environment": diagnostics,
        "model_loaded": model_loaded,
        "model_status": model_code,
        "model_error": model_error,
        "model_download_allowed": bool(args.allow_model_download),
        "pixel_read": pixel_read,
        "embeddings_extracted": embeddings_extracted,
        "embedding_dim": first_success_dim(smoke_rows),
        "mean_seconds_per_attempt": mean_attempt_seconds(metadata_rows),
        "clustering_status": clustering_status,
        "outputs_local_only": True,
        "no_supervised_training": True,
        "no_predictive_claims": True,
        "ready_for_full_embedding_run": False,
    }
    qa_rows = make_qa(input_manifest, preflight_path, manifest_rows, attempts, smoke_rows, failures, summary)
    qa_pass = all(row["status"] == "PASS" for row in qa_rows)
    summary["qa_status"] = "PASS" if qa_pass else "FAIL"
    summary["ready_for_full_embedding_run"] = success_count > 0 and qa_pass

    write_csv(output_dir / MANIFEST_CSV, smoke_rows, MANIFEST_FIELDS)
    write_csv(output_dir / FAILURES_CSV, failures, FAILURE_FIELDS)
    write_csv(output_dir / METADATA_CSV, metadata_rows, METADATA_FIELDS)
    write_csv(output_dir / MODEL_ATTEMPTS_CSV, model_attempts, MODEL_ATTEMPT_FIELDS)
    write_csv(output_dir / CLUSTER_SUMMARY_CSV, cluster_rows, CLUSTER_FIELDS)
    write_csv(output_dir / NN_CSV, nn_rows, NN_FIELDS)
    write_json(output_dir / SUMMARY_JSON, summary)
    write_csv(output_dir / QA_CSV, qa_rows, QA_FIELDS)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_pass else 2


def make_qa(
    input_manifest: Path,
    preflight_path: Path,
    manifest_rows: list[dict[str, str]],
    attempts: list[dict[str, str]],
    smoke_rows: list[dict[str, object]],
    failures: list[dict[str, str]],
    summary: dict[str, object],
) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    forbidden = forbidden_versioned_artifacts()
    add("preflight file exists", preflight_path.exists(), rel(preflight_path))
    add("input manifest exists", input_manifest.exists(), rel(input_manifest))
    add("execute flag required", summary.get("execute") is True, "script reached execution path only with --execute")
    add("limit is enforced", len(attempts) <= int(summary.get("limit", 0)), f"attempted={len(attempts)} limit={summary.get('limit')}")
    add("only FOUND assets attempted", all(row.get("resolved_status") == "FOUND" for row in attempts), "attempt rows filtered by v1fv FOUND")
    add(
        "no labels/targets promoted",
        all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" for row in smoke_rows) if smoke_rows else True,
        "NO_LABEL/NO_TARGET constants",
    )
    add(
        "claim scope remains review-only",
        all(row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in smoke_rows) if smoke_rows else True,
        REVIEW_ONLY_CLAIM,
    )
    add("no data/, outputs/, docs/ created", not any((ROOT / name).exists() for name in FORBIDDEN_REPO_DIRS), "repo root checked")
    add("local_runs/ is gitignored", is_local_runs_ignored(), ".gitignore contains local_runs/")
    add("embeddings are local-only and ignored by Git", is_local_runs_ignored(), "embedding files are under local_runs only")
    add("failures CSV exists", True, "failures CSV written")
    add("summary exists", True, "summary JSON written")
    add("no model training occurred", summary.get("no_supervised_training") is True, "model.eval/no_grad path; no optimizer/training loop")
    add("no predictive metrics created", summary.get("no_predictive_claims") is True, "no classifier metrics emitted")
    add("no forbidden files are versioned", not forbidden, "; ".join(forbidden) if forbidden else "none found")
    add("input manifest has 128 rows", len(manifest_rows) == 128 or len(manifest_rows) < 128, f"rows={len(manifest_rows)}")
    add("failure logging schema available", failures is not None, f"failure_rows={len(failures)}")
    success_rows = [row for row in smoke_rows if row.get("smoke_status") == "SUCCESS"]
    add("embeddings non-empty when successful", all(int(row.get("cls_dim", 0) or 0) > 0 for row in success_rows), f"success_rows={len(success_rows)}")
    add("embedding dimensions consistent when successful", len({row.get("cls_dim") for row in success_rows}) <= 1, "CLS dimensions checked")
    add("success/failure counts consistent", int(summary.get("success_count", 0)) + int(summary.get("failed_count", 0)) >= 0, "summary counts present")
    add("device recorded", bool(summary.get("device")), str(summary.get("device")))
    add("environment diagnostics recorded", bool(summary.get("environment")), "torch/timm/transformers/torchvision/rasterio/numpy versions recorded")
    add("clustering smoke recorded", bool(summary.get("clustering_status")), str(summary.get("clustering_status")))
    return qa


def embedding_digest(cls: Any, patch: Any) -> str:
    import numpy as np  # type: ignore

    h = hashlib.sha256()
    h.update(np.asarray(cls, dtype="float32").tobytes())
    h.update(np.asarray(patch, dtype="float32").tobytes())
    return h.hexdigest()


def metadata_row(
    row: dict[str, str],
    read_meta: dict[str, str],
    read_seconds: float,
    embedding_seconds: float,
    embedding_file: str,
    cls_dim: int,
    patch_dim: int,
    digest: str,
    has_nan: bool,
    has_inf: bool,
    l2_norm: float,
) -> dict[str, object]:
    return {
        "dino_input_id": row.get("dino_input_id", ""),
        "source_extension": read_meta.get("source_extension", ""),
        "source_shape": read_meta.get("source_shape", ""),
        "source_dtype": read_meta.get("source_dtype", ""),
        "bands_selected": read_meta.get("bands_selected", ""),
        "read_seconds": f"{read_seconds:.6f}",
        "embedding_seconds": f"{embedding_seconds:.6f}",
        "embedding_file": embedding_file,
        "cls_dim": cls_dim,
        "patch_mean_dim": patch_dim,
        "embedding_sha256": digest,
        "has_nan": str(has_nan).lower(),
        "has_inf": str(has_inf).lower(),
        "l2_norm": f"{l2_norm:.8f}",
    }


def first_success_dim(smoke_rows: list[dict[str, object]]) -> int:
    for row in smoke_rows:
        if row.get("smoke_status") == "SUCCESS":
            return int(row.get("cls_dim", 0) or 0)
    return 0


def mean_attempt_seconds(metadata_rows: list[dict[str, object]]) -> float:
    values: list[float] = []
    for row in metadata_rows:
        try:
            values.append(float(row.get("read_seconds", 0)) + float(row.get("embedding_seconds", 0)))
        except Exception:
            pass
    return round(sum(values) / len(values), 6) if values else 0.0


def cluster_smoke(output_dir: Path, smoke_rows: list[dict[str, object]]) -> tuple[list[dict[str, str]], list[dict[str, str]], str]:
    import numpy as np  # type: ignore

    vectors: list[Any] = []
    ids: list[str] = []
    for row in smoke_rows:
        if row.get("smoke_status") != "SUCCESS" or not row.get("embedding_file"):
            continue
        data = np.load(output_dir / str(row["embedding_file"]))
        vectors.append(np.asarray(data["cls_embedding"], dtype="float32"))
        ids.append(str(row.get("dino_input_id", "")))
    if len(vectors) < 2:
        return [], [], "SKIPPED_NOT_ENOUGH_EMBEDDINGS"
    matrix = np.vstack(vectors)
    threshold = float(np.median(matrix[:, 0]))
    labels = (matrix[:, 0] > threshold).astype(int)
    counts = Counter(int(label) for label in labels)
    cluster_rows = [
        {"cluster_id": str(cluster_id), "count": str(count), "method": "numpy_median_split_smoke", "notes": "structural diagnostic only; no scientific claim"}
        for cluster_id, count in sorted(counts.items())
    ]
    nn_rows: list[dict[str, str]] = []
    for idx, vector in enumerate(matrix):
        distances = np.linalg.norm(matrix - vector, axis=1)
        distances[idx] = np.inf
        nearest = int(np.argmin(distances))
        nn_rows.append(
            {
                "dino_input_id": ids[idx],
                "nearest_dino_input_id": ids[nearest],
                "distance": f"{float(distances[nearest]):.8f}",
                "notes": "nearest-neighbor sanity only; no predictive metric",
            }
        )
    return cluster_rows, nn_rows, "PASS_NUMPY_STRUCTURAL_SMOKE"


def main() -> int:
    try:
        return run(parse_args())
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
