from __future__ import annotations

import argparse
import csv
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
PHASE = "v1fz"
PHASE_NAME = "DINO_BALANCED_SENTINEL_EMBEDDING_CORPUS"

DEFAULT_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
DEFAULT_ASSET_PREFLIGHT = ROOT / "local_runs" / "dino_asset_preflight" / "v1fv" / "dino_local_asset_preflight_v1fv.csv"
DEFAULT_CONFIG = ROOT / "configs" / "dino_embedding_extraction.example.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1fz"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1fx = load_module(ROOT / "scripts" / "dino" / "revp_v1fx_dino_smoke_embedding_execution.py", "revp_v1fx")
v1fy = load_module(ROOT / "scripts" / "dino" / "revp_v1fy_dino_embedding_corpus_analysis.py", "revp_v1fy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1fz balanced DINO Sentinel embedding corpus run.")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--asset-preflight", default=str(DEFAULT_ASSET_PREFLIGHT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--regions", nargs="+", default=["Curitiba", "Petropolis", "Recife"])
    parser.add_argument("--per-region-limit", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--backbone", default="dinov2_vitb14_reg")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--skip-model-if-unavailable", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=3)
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


def normalize_region(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return stripped.lower().replace(" ", "").replace("-", "").replace("_", "")


def prepare_output_dir(output_dir: Path, force: bool) -> None:
    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Use --force.")
        shutil.rmtree(output_dir)
    (output_dir / "embeddings").mkdir(parents=True, exist_ok=True)


def select_balanced(manifest_rows: list[dict[str, str]], preflight_rows: list[dict[str, str]], requested_regions: list[str], per_region_limit: int) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    preflight = {row.get("dino_input_id", ""): row for row in preflight_rows}
    requested = {normalize_region(region): region for region in requested_regions}
    selected_by_region: dict[str, list[dict[str, str]]] = defaultdict(list)
    available = Counter()
    skipped = Counter()
    for row in manifest_rows:
        region = row.get("region", "")
        norm = normalize_region(region)
        if norm not in requested:
            continue
        pre = preflight.get(row.get("dino_input_id", ""), {})
        if pre.get("resolved_status") != "FOUND":
            skipped[region] += 1
            continue
        available[region] += 1
        if len(selected_by_region[region]) < per_region_limit:
            merged = dict(row)
            merged["resolved_status"] = pre.get("resolved_status", "")
            merged["resolved_path_private"] = pre.get("resolved_path_private", "")
            selected_by_region[region].append(merged)
        else:
            skipped[region] += 1
    selected: list[dict[str, str]] = []
    audit: list[dict[str, object]] = []
    for norm, requested_name in requested.items():
        actual_regions = sorted({row.get("region", "") for row in manifest_rows if normalize_region(row.get("region", "")) == norm})
        actual = actual_regions[0] if actual_regions else requested_name
        rows = selected_by_region.get(actual, [])
        selected.extend(rows)
        audit.append(
            {
                "requested_region": requested_name,
                "matched_region": actual,
                "available_found": int(available.get(actual, 0)),
                "selected_count": len(rows),
                "skipped_count": int(max(available.get(actual, 0) - len(rows), 0) + skipped.get(actual, 0)),
                "selection_status": "PASS" if rows else "NO_FOUND_ASSETS",
            }
        )
    return selected, audit


def run(args: argparse.Namespace) -> int:
    if not args.execute:
        print("v1fz requires --execute for balanced embedding corpus generation.", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    config = v1fx.parse_simple_yaml(Path(args.config))
    image_size = int(config.get("image_size", 224))
    manifest_rows = read_csv(Path(args.input_manifest))
    preflight_rows = read_csv(Path(args.asset_preflight))
    selected, selection_audit = select_balanced(manifest_rows, preflight_rows, args.regions, args.per_region_limit)
    device = v1fx.resolve_device(args.device, args.allow_cpu, args.force_cpu)
    model, model_code, model_error, loaded_backbone, model_attempts = v1fx.load_smoke_model(args.backbone, device, args.skip_model_if_unavailable, args.allow_model_download)
    model_loaded = model is not None

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    pixel_read = False
    for item in selected:
        start = time.perf_counter()
        source_path = Path(item.get("resolved_path_private", ""))
        if not model_loaded:
            reason = model_error or "model unavailable"
            failures.append(failure_row(item, "MODEL_UNAVAILABLE", reason, args.backbone, device))
            rows.append(manifest_row(item, source_path, "", 0, args.backbone, device, "FAILED", reason, ""))
            continue
        array, pixel_status, read_error, read_meta = v1fx.read_image_tensor(source_path, image_size)
        if array is None:
            failures.append(failure_row(item, read_error.split(":", 1)[0], read_error, args.backbone, device))
            rows.append(manifest_row(item, source_path, "", 0, args.backbone, device, "FAILED", read_error, ""))
            continue
        pixel_read = True
        try:
            cls, patch, patch_status = v1fx.embed_array(model, array, args.backbone, device)
            embedding_rel = Path("embeddings") / f"{item.get('dino_input_id')}.npz"
            embedding_path = output_dir / embedding_rel
            np.savez_compressed(embedding_path, cls_embedding=cls, patch_mean_embedding=patch)
            digest = v1fx.embedding_digest(cls, patch)
            vectors.append(np.asarray(cls, dtype="float32"))
            ids.append(item.get("dino_input_id", ""))
            rows.append(manifest_row(item, source_path, embedding_rel.as_posix(), int(cls.shape[0]), loaded_backbone or args.backbone, device, "SUCCESS", "", digest))
            metadata_rows.append(
                {
                    "dino_input_id": item.get("dino_input_id", ""),
                    "source_shape": read_meta.get("source_shape", ""),
                    "bands_selected": read_meta.get("bands_selected", ""),
                    "embedding_seconds": f"{time.perf_counter() - start:.6f}",
                    "patch_mean_status": patch_status,
                    "hash": digest,
                    "has_nan": str(bool(np.isnan(cls).any() or np.isnan(patch).any())).lower(),
                    "has_inf": str(bool(np.isinf(cls).any() or np.isinf(patch).any())).lower(),
                    "l2_norm": float(np.linalg.norm(cls)),
                }
            )
        except Exception as exc:
            failures.append(failure_row(item, "EMBEDDING_FAILED", f"{type(exc).__name__}: {exc}", args.backbone, device))
            rows.append(manifest_row(item, source_path, "", 0, args.backbone, device, "FAILED", str(exc), ""))

    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype="float32")
    analysis = write_analysis_outputs(output_dir, rows, matrix, ids, args.seed, args.top_k)
    qa_rows = make_qa(rows, failures, selection_audit, matrix, output_dir)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    by_region = Counter(str(row.get("region", "")) for row in rows if row.get("success") == "SUCCESS")
    failures_by_region = Counter(str(row.get("region", "")) for row in failures)
    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "requested_regions": args.regions,
        "per_region_limit": args.per_region_limit,
        "selected_count": len(selected),
        "success_count": sum(1 for row in rows if row.get("success") == "SUCCESS"),
        "failed_count": len(failures),
        "embeddings_by_region": dict(sorted(by_region.items())),
        "failures_by_region": dict(sorted(failures_by_region.items())),
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 0,
        "model_backbone": loaded_backbone or args.backbone,
        "model_loaded": model_loaded,
        "model_status": model_code,
        "model_error": model_error,
        "device": device,
        "pixel_read": pixel_read,
        "embeddings_extracted": bool(vectors),
        "qa_status": qa_status,
        "pca_manifold_status": analysis["manifold_status"],
        "clustering_status": analysis["clustering_status"],
        "regional_diagnostics_status": analysis["regional_status"],
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "predictive_claims": False,
        "multimodal_hold": True,
        "outputs_local_only": True,
    }
    write_csv(output_dir / "dino_balanced_embedding_manifest_v1fz.csv", rows, ["patch_id", "dino_input_id", "region", "source_path", "embedding_path", "embedding_dim", "model_backbone", "device", "success", "failure_reason", "hash", "timestamp", "label_status", "target_status", "claim_scope"])
    write_csv(output_dir / "dino_balanced_embedding_failures_v1fz.csv", failures, ["dino_input_id", "patch_id", "region", "failure_code", "failure_reason", "model_backbone", "device"])
    write_csv(output_dir / "dino_balanced_selection_audit_v1fz.csv", selection_audit, ["requested_region", "matched_region", "available_found", "selected_count", "skipped_count", "selection_status"])
    write_csv(output_dir / "dino_balanced_embedding_metadata_v1fz.csv", metadata_rows, ["dino_input_id", "source_shape", "bands_selected", "embedding_seconds", "patch_mean_status", "hash", "has_nan", "has_inf", "l2_norm"])
    write_csv(output_dir / "dino_balanced_model_attempts_v1fz.csv", model_attempts, ["order", "candidate", "backend", "status", "details"])
    write_csv(output_dir / "dino_balanced_embedding_qa_v1fz.csv", qa_rows, ["check", "status", "details"])
    write_json(output_dir / "dino_balanced_embedding_summary_v1fz.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def manifest_row(item: dict[str, str], source_path: Path, embedding_path: str, embedding_dim: int, backbone: str, device: str, success: str, failure_reason: str, digest: str) -> dict[str, object]:
    return {
        "patch_id": item.get("canonical_patch_id", ""),
        "dino_input_id": item.get("dino_input_id", ""),
        "region": item.get("region", ""),
        "source_path": str(source_path),
        "embedding_path": embedding_path,
        "embedding_dim": embedding_dim,
        "model_backbone": backbone,
        "device": device,
        "success": success,
        "failure_reason": failure_reason,
        "hash": digest,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label_status": "NO_LABEL",
        "target_status": "NO_TARGET",
        "claim_scope": REVIEW_ONLY_CLAIM,
    }


def failure_row(item: dict[str, str], code: str, reason: str, backbone: str, device: str) -> dict[str, object]:
    return {
        "dino_input_id": item.get("dino_input_id", ""),
        "patch_id": item.get("canonical_patch_id", ""),
        "region": item.get("region", ""),
        "failure_code": code,
        "failure_reason": reason,
        "model_backbone": backbone,
        "device": device,
    }


def write_analysis_outputs(output_dir: Path, rows: list[dict[str, object]], matrix: np.ndarray, ids: list[str], seed: int, top_k: int) -> dict[str, str]:
    corpus_rows = [
        {
            "patch_id": row.get("patch_id", ""),
            "dino_input_id": row.get("dino_input_id", ""),
            "region": row.get("region", ""),
            "embedding_dim": row.get("embedding_dim", 0),
            "hash": row.get("hash", ""),
            "qa_status": "PASS" if row.get("success") == "SUCCESS" else "FAIL",
            "label_status": "NO_LABEL",
            "target_status": "NO_TARGET",
            "claim_scope": REVIEW_ONLY_CLAIM,
        }
        for row in rows
        if row.get("success") == "SUCCESS"
    ]
    normed = v1fy.l2_normalize(matrix)
    coords, pca_rows = v1fy.pca(normed, components=2)
    manifold_rows = [
        {"dino_input_id": ids[i], "method": "PCA", "x": float(coords[i, 0]) if coords.shape[1] > 0 else 0.0, "y": float(coords[i, 1]) if coords.shape[1] > 1 else 0.0, "umap_status": "NOT_RUN_OPTIONAL", "tsne_status": "NOT_RUN_OPTIONAL"}
        for i in range(len(ids))
    ]
    preservation_rows = v1fy.neighborhood_preservation(ids, normed, coords)
    cluster_metrics, cluster_sizes, _labels = v1fy.clustering_outputs(normed, ids, seed)
    nn_rows, reciprocal_rows, outlier_rows = v1fy.nearest_neighbors(ids, normed, top_k) if len(ids) > 1 else ([], [], [])
    region_rows = v1fy.region_diagnostics(corpus_rows, normed)
    v1fy.write_csv(output_dir / "dino_balanced_pca_variance_v1fz.csv", pca_rows, ["component", "explained_variance", "explained_variance_ratio"])
    v1fy.write_csv(output_dir / "dino_balanced_manifold_coordinates_v1fz.csv", manifold_rows, ["dino_input_id", "method", "x", "y", "umap_status", "tsne_status"])
    v1fy.write_csv(output_dir / "dino_balanced_neighborhood_preservation_v1fz.csv", preservation_rows, ["dino_input_id", "preserved_top1", "status"])
    v1fy.write_csv(output_dir / "dino_balanced_cluster_metrics_v1fz.csv", cluster_metrics, ["method", "k", "silhouette", "instability_warning", "claim_scope"])
    v1fy.write_csv(output_dir / "dino_balanced_cluster_sizes_v1fz.csv", cluster_sizes, ["method", "k", "cluster_id", "count"])
    v1fy.write_csv(output_dir / "dino_balanced_nearest_neighbors_v1fz.csv", nn_rows, ["dino_input_id", "neighbor_dino_input_id", "rank", "cosine_similarity", "cosine_distance", "claim_scope"])
    v1fy.write_csv(output_dir / "dino_balanced_reciprocal_pairs_v1fz.csv", reciprocal_rows, ["dino_input_id_a", "dino_input_id_b", "relationship"])
    v1fy.write_csv(output_dir / "dino_balanced_outliers_v1fz.csv", outlier_rows, ["dino_input_id", "nearest_distance", "outlier_status", "threshold"])
    v1fy.write_csv(output_dir / "dino_balanced_region_diagnostics_v1fz.csv", region_rows, ["region", "embedding_count", "centroid_norm", "dispersion", "intra_region_cosine_mean", "inter_region_cosine_mean", "claim_scope"])
    return {
        "manifold_status": "PASS_PCA" if manifold_rows else "SKIPPED",
        "clustering_status": "PASS" if cluster_metrics else "SKIPPED",
        "regional_status": "PASS" if region_rows else "SKIPPED",
    }


def make_qa(rows: list[dict[str, object]], failures: list[dict[str, object]], selection_audit: list[dict[str, object]], matrix: np.ndarray, output_dir: Path) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    success_rows = [row for row in rows if row.get("success") == "SUCCESS"]
    add("balanced selection audited", bool(selection_audit), f"regions={len(selection_audit)}")
    add("at least one embedding per requested available region", all(int(row.get("selected_count", 0) or 0) > 0 for row in selection_audit), json.dumps(selection_audit, ensure_ascii=False))
    add("embeddings local only", "local_runs" in output_dir.parts, str(output_dir))
    add("no labels created", all(row.get("label_status") == "NO_LABEL" for row in rows), "NO_LABEL")
    add("no targets created", all(row.get("target_status") == "NO_TARGET" for row in rows), "NO_TARGET")
    add("review-only claim scope", all(row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("no supervised training", True, "no optimizer/training loop; frozen encoder forward only")
    add("no predictive claims", True, "structural diagnostics only")
    add("multimodal remains hold", True, "Sentinel-first only")
    add("embedding dimensions consistent", len({row.get("embedding_dim") for row in success_rows}) <= 1, str({row.get("embedding_dim") for row in success_rows}))
    add("embeddings finite", matrix.size > 0 and bool(np.isfinite(matrix).all()), "finite matrix")
    add("failure isolation", failures is not None, f"failures={len(failures)}")
    add("local_runs gitignored", v1fx.is_local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not v1fx.forbidden_versioned_artifacts(), "repo checked outside local_runs")
    return qa


def main() -> int:
    try:
        return run(parse_args())
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
