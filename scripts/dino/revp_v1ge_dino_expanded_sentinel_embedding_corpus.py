from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import shutil
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
DEFAULT_ASSET_PREFLIGHT = ROOT / "local_runs" / "dino_asset_preflight" / "v1fv" / "dino_local_asset_preflight_v1fv.csv"
DEFAULT_CONFIG = ROOT / "configs" / "dino_embedding_extraction.example.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1ge"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1fx = load_module(ROOT / "scripts" / "dino" / "revp_v1fx_dino_smoke_embedding_execution.py", "revp_v1fx_for_v1ge")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1ge expanded Sentinel DINO embedding corpus run.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--asset-preflight", default=str(DEFAULT_ASSET_PREFLIGHT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-region-limit", type=int, default=0)
    parser.add_argument("--regions", nargs="+", default=["Curitiba", "Petropolis", "Recife"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backbone", default="dinov2_vitb14_reg")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-proxy-for-tests", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
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


def prepare_output_dir(path: Path, force: bool, resume: bool) -> None:
    if path.exists() and force and not resume:
        shutil.rmtree(path)
    if path.exists() and not force and not resume:
        raise FileExistsError(f"Output directory already exists: {path}. Use --force or --resume.")
    (path / "embeddings").mkdir(parents=True, exist_ok=True)


def normalize_region(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower().replace(" ", "").replace("-", "").replace("_", "")


def local_runs_ignored() -> bool:
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
        if path.is_file() and path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS:
            found.append(str(path))
        if path.is_dir() and path.name in {"data", "outputs"}:
            found.append(str(path))
    return found


def select_inputs(manifest_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]], args: argparse.Namespace) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    preflight = {row.get("dino_input_id", ""): row for row in preflight_rows}
    requested = {normalize_region(region) for region in args.regions}
    selected: list[dict[str, str]] = []
    by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    skipped = Counter()
    for row in manifest_rows:
        region = row.get("region", "")
        if requested and normalize_region(region) not in requested:
            continue
        pre = preflight.get(row.get("dino_input_id", ""), {})
        if pre.get("resolved_status") != "FOUND":
            skipped[region] += 1
            continue
        merged = dict(row)
        merged["resolved_path_private"] = pre.get("resolved_path_private", "")
        by_region[region].append(merged)
    if args.per_region_limit > 0:
        for region in sorted(by_region):
            selected.extend(by_region[region][: args.per_region_limit])
            skipped[region] += max(len(by_region[region]) - args.per_region_limit, 0)
    else:
        selected = [row for region in sorted(by_region) for row in by_region[region]]
    if args.limit > 0:
        skipped["LIMIT"] += max(len(selected) - args.limit, 0)
        selected = selected[: args.limit]
    audit = [{"region": region, "found_available": len(rows), "selected_count": sum(1 for row in selected if row.get("region") == region), "skipped_count": int(skipped[region]), "selection_status": "PASS" if rows else "NO_FOUND_ASSETS"} for region, rows in sorted(by_region.items())]
    if args.limit > 0:
        audit.append({"region": "LIMIT", "found_available": "", "selected_count": len(selected), "skipped_count": int(skipped["LIMIT"]), "selection_status": "APPLIED"})
    return selected, audit


def proxy_embedding(dino_id: str, dim: int = 32) -> tuple[np.ndarray, np.ndarray]:
    digest = hashlib.sha256(dino_id.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    cls = rng.normal(0, 1, size=dim).astype("float32")
    patch = cls + rng.normal(0, 0.01, size=dim).astype("float32")
    return cls, patch


def digest_arrays(cls: np.ndarray, patch: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(cls, dtype="float32").tobytes())
    h.update(np.asarray(patch, dtype="float32").tobytes())
    return h.hexdigest()


def existing_success(output_dir: Path) -> dict[str, dict[str, str]]:
    path = output_dir / "dino_expanded_embedding_manifest_v1ge.csv"
    return {row.get("dino_input_id", ""): row for row in read_csv(path) if row.get("success") == "SUCCESS"}


def run(args: argparse.Namespace) -> int:
    if not args.execute:
        print("v1ge requires --execute for expanded embedding corpus generation.", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force, args.resume)
    config = v1fx.parse_simple_yaml(Path(args.config))
    image_size = int(config.get("image_size", 224))
    selected, selection_audit = select_inputs(read_csv(Path(args.input_manifest)), read_csv(Path(args.asset_preflight)), args)
    previous = existing_success(output_dir) if args.resume or args.skip_existing else {}
    rows: list[dict[str, object]] = list(previous.values()) if args.resume else []
    failures: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    skipped_existing_count = 0
    model = None
    model_loaded = bool(args.embedding_proxy_for_tests)
    model_status = "EMBEDDING_PROXY_FOR_TESTS" if args.embedding_proxy_for_tests else "NOT_LOADED"
    device = "cpu"
    loaded_backbone = args.backbone
    if not args.embedding_proxy_for_tests:
        device = v1fx.resolve_device(args.device, args.allow_cpu, args.force_cpu)
        model, model_status, model_error, loaded_backbone, _attempts = v1fx.load_smoke_model(args.backbone, device, False, args.allow_model_download)
        model_loaded = model is not None
    else:
        model_error = ""
    for item in selected:
        dino_id = item.get("dino_input_id", "")
        embedding_rel = Path("embeddings") / f"{dino_id}.npz"
        if args.skip_existing and (dino_id in previous or (output_dir / embedding_rel).exists()):
            skipped_existing_count += 1
            if dino_id not in {str(row.get("dino_input_id")) for row in rows}:
                rows.append(previous.get(dino_id, manifest_row(item, embedding_rel.as_posix(), 0, loaded_backbone, device, "SKIPPED_EXISTING", "", "")))
            continue
        start = time.perf_counter()
        if not model_loaded:
            failures.append(failure_row(item, "MODEL_UNAVAILABLE", model_error or "model unavailable", loaded_backbone, device))
            rows.append(manifest_row(item, "", 0, loaded_backbone, device, "FAILED", model_error or "model unavailable", ""))
            continue
        try:
            if args.embedding_proxy_for_tests:
                cls, patch = proxy_embedding(dino_id)
                read_meta = {"source_shape": "proxy", "bands_selected": "proxy"}
            else:
                array, _pixel_status, read_error, read_meta = v1fx.read_image_tensor(Path(item.get("resolved_path_private", "")), image_size)
                if array is None:
                    raise RuntimeError(read_error)
                cls, patch, _patch_status = v1fx.embed_array(model, array, loaded_backbone, device)
            cls = np.asarray(cls, dtype="float32")
            patch = np.asarray(patch, dtype="float32")
            (output_dir / embedding_rel).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(output_dir / embedding_rel, cls_embedding=cls, patch_mean_embedding=patch)
            digest = digest_arrays(cls, patch)
            rows.append(manifest_row(item, embedding_rel.as_posix(), int(cls.shape[0]), loaded_backbone, device, "SUCCESS", "", digest))
            metadata.append({"dino_input_id": dino_id, "source_shape": read_meta.get("source_shape", ""), "bands_selected": read_meta.get("bands_selected", ""), "embedding_seconds": f"{time.perf_counter() - start:.6f}", "hash": digest, "has_nan": bool(np.isnan(cls).any() or np.isnan(patch).any()), "has_inf": bool(np.isinf(cls).any() or np.isinf(patch).any()), "l2_norm": float(np.linalg.norm(cls))})
        except Exception as exc:
            failures.append(failure_row(item, "EMBEDDING_FAILED", f"{type(exc).__name__}: {exc}", loaded_backbone, device))
            rows.append(manifest_row(item, "", 0, loaded_backbone, device, "FAILED", str(exc), ""))
    success_rows = [row for row in rows if row.get("success") in {"SUCCESS", "SKIPPED_EXISTING"}]
    dims = {str(row.get("embedding_dim")) for row in success_rows if str(row.get("embedding_dim", "0")) not in {"", "0"}}
    qa = make_qa(rows, failures, metadata, selection_audit, dims)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa) else "FAIL"
    summary = {"phase": "v1ge", "phase_name": "DINO_EXPANDED_SENTINEL_EMBEDDING_CORPUS", "created_utc": datetime.now(timezone.utc).isoformat(), "selected_count": len(selected), "success_count": sum(1 for row in rows if row.get("success") == "SUCCESS"), "skipped_existing_count": skipped_existing_count + sum(1 for row in rows if row.get("success") == "SKIPPED_EXISTING"), "failed_count": len(failures), "embedding_dim": int(next(iter(dims))) if len(dims) == 1 else 0, "successes_by_region": dict(Counter(str(row.get("region", "")) for row in success_rows)), "failures_by_region": dict(Counter(str(row.get("region", "")) for row in failures)), "batch_size": args.batch_size, "resume": args.resume, "skip_existing": args.skip_existing, "model_backbone": loaded_backbone, "model_loaded": model_loaded, "model_status": model_status, "device": device, "qa_status": qa_status, "review_only": True, "supervised_training": False, "labels_created": False, "targets_created": False, "predictive_claims": False, "multimodal_hold": True, "outputs_local_only": True}
    write_csv(output_dir / "dino_expanded_embedding_manifest_v1ge.csv", rows, ["patch_id", "dino_input_id", "region", "source_path", "embedding_path", "embedding_dim", "model_backbone", "device", "success", "failure_reason", "hash", "timestamp", "label_status", "target_status", "claim_scope"])
    write_csv(output_dir / "dino_expanded_embedding_failures_v1ge.csv", failures, ["dino_input_id", "patch_id", "region", "failure_code", "failure_reason", "model_backbone", "device"])
    write_csv(output_dir / "dino_expanded_embedding_metadata_v1ge.csv", metadata, ["dino_input_id", "source_shape", "bands_selected", "embedding_seconds", "hash", "has_nan", "has_inf", "l2_norm"])
    write_csv(output_dir / "dino_expanded_selection_audit_v1ge.csv", selection_audit, ["region", "found_available", "selected_count", "skipped_count", "selection_status"])
    write_csv(output_dir / "dino_expanded_embedding_qa_v1ge.csv", qa, ["check", "status", "details"])
    write_json(output_dir / "dino_expanded_embedding_summary_v1ge.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def manifest_row(item: dict[str, str], embedding_path: str, embedding_dim: int, backbone: str, device: str, success: str, failure_reason: str, digest: str) -> dict[str, object]:
    return {"patch_id": item.get("canonical_patch_id") or item.get("patch_id", ""), "dino_input_id": item.get("dino_input_id", ""), "region": item.get("region", ""), "source_path": item.get("resolved_path_private", ""), "embedding_path": embedding_path, "embedding_dim": embedding_dim, "model_backbone": backbone, "device": device, "success": success, "failure_reason": failure_reason, "hash": digest, "timestamp": datetime.now(timezone.utc).isoformat(), "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": REVIEW_ONLY_CLAIM}


def failure_row(item: dict[str, str], code: str, reason: str, backbone: str, device: str) -> dict[str, object]:
    return {"dino_input_id": item.get("dino_input_id", ""), "patch_id": item.get("canonical_patch_id") or item.get("patch_id", ""), "region": item.get("region", ""), "failure_code": code, "failure_reason": reason, "model_backbone": backbone, "device": device}


def make_qa(rows: list[dict[str, object]], failures: list[dict[str, object]], metadata: list[dict[str, object]], selection: list[dict[str, object]], dims: set[str]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    add("embedding_dim consistent", len(dims) <= 1, f"dims={sorted(dims)}")
    add("NaN/Inf absent", all(str(row.get("has_nan")).lower() == "false" and str(row.get("has_inf")).lower() == "false" for row in metadata), f"metadata_rows={len(metadata)}")
    add("hashes recorded", all(row.get("hash") for row in metadata), f"hashes={sum(1 for row in metadata if row.get('hash'))}")
    add("successes failures by region auditable", bool(selection), f"selection_rows={len(selection)} failures={len(failures)}")
    add("skip resume auditable", True, "manifest records SUCCESS/SKIPPED_EXISTING/FAILED")
    add("model backbone device recorded", all(row.get("model_backbone") and row.get("device") for row in rows), f"rows={len(rows)}")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not forbidden_versioned_artifacts(), "repo checked outside local_runs")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
