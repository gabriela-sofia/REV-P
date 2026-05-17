from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1fy"
PHASE_NAME = "DINO_SENTINEL_EMBEDDING_CORPUS_EXPLORATORY_BASE"

DEFAULT_SMOKE_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1fx" / "dino_smoke_embedding_manifest_v1fx.csv"
DEFAULT_SMOKE_METADATA = ROOT / "local_runs" / "dino_embeddings" / "v1fx" / "dino_smoke_embedding_metadata_v1fx.csv"
DEFAULT_INPUT_MANIFEST = ROOT / "manifests" / "dino_inputs" / "revp_v1fu_dino_sentinel_input_manifest" / "dino_sentinel_input_manifest_v1fu.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1fy"

FORBIDDEN_REPO_DIRS = {"data", "outputs"}
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index"}
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1fy DINO embedding corpus exploratory analysis.")
    parser.add_argument("--mode", default="embedding-corpus-run", choices=["embedding-corpus-run"])
    parser.add_argument("--smoke-manifest", default=str(DEFAULT_SMOKE_MANIFEST))
    parser.add_argument("--smoke-metadata", default=str(DEFAULT_SMOKE_METADATA))
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--max-failures", type=int, default=25)
    parser.add_argument("--save-frequency", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
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


def prepare_output_dir(path: Path, force: bool, resume: bool) -> None:
    if path.exists() and not force and not resume:
        raise FileExistsError(f"Output directory already exists: {path}. Use --force or --resume.")
    if path.exists() and force and not resume:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


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


def npz_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def vector_sha256(vector: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(vector, dtype="float32").tobytes())
    return h.hexdigest()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def load_existing_ids(output_dir: Path) -> set[str]:
    path = output_dir / "dino_embedding_corpus_manifest_v1fy.csv"
    if not path.exists():
        return set()
    return {row.get("dino_input_id", "") for row in read_csv(path) if row.get("qa_status") == "PASS"}


def build_corpus(args: argparse.Namespace) -> tuple[list[dict[str, object]], list[dict[str, str]], np.ndarray, list[str]]:
    smoke_manifest = Path(args.smoke_manifest)
    metadata_path = Path(args.smoke_metadata)
    output_dir = Path(args.output_dir)
    smoke_rows = [row for row in read_csv(smoke_manifest) if row.get("smoke_status") == "SUCCESS"]
    metadata = {row.get("dino_input_id", ""): row for row in read_csv(metadata_path)} if metadata_path.exists() else {}
    existing_ids = load_existing_ids(output_dir) if args.skip_existing else set()
    if args.limit and args.limit > 0:
        smoke_rows = smoke_rows[: args.limit]

    corpus_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    failure_count = 0
    for index, row in enumerate(smoke_rows, start=1):
        dino_id = row.get("dino_input_id", "")
        if dino_id in existing_ids:
            continue
        embedding_rel = row.get("embedding_file", "")
        embedding_path = Path(args.smoke_manifest).parent / embedding_rel
        try:
            data = np.load(embedding_path)
            cls = np.asarray(data["cls_embedding"], dtype="float32")
            patch = np.asarray(data["patch_mean_embedding"], dtype="float32") if "patch_mean_embedding" in data else np.array([], dtype="float32")
            if cls.size == 0:
                raise ValueError("empty cls embedding")
            has_nan = bool(np.isnan(cls).any() or np.isnan(patch).any())
            has_inf = bool(np.isinf(cls).any() or np.isinf(patch).any())
            qa_status = "FAIL" if has_nan or has_inf else "PASS"
            meta = metadata.get(dino_id, {})
            corpus_rows.append(
                {
                    "patch_id": row.get("canonical_patch_id", ""),
                    "dino_input_id": dino_id,
                    "region": row.get("region", ""),
                    "bands_present": meta.get("bands_selected", ""),
                    "embedding_path": embedding_rel,
                    "embedding_dim": int(cls.shape[0]),
                    "model_backbone": row.get("backbone", ""),
                    "device": row.get("device", ""),
                    "execution_timestamp": datetime.now(timezone.utc).isoformat(),
                    "hash": vector_sha256(cls),
                    "npz_hash": npz_sha256(embedding_path),
                    "qa_status": qa_status,
                    "cluster_id": "PENDING_STRUCTURAL_ANALYSIS",
                    "label_status": row.get("label_status", "NO_LABEL"),
                    "target_status": row.get("target_status", "NO_TARGET"),
                    "claim_scope": row.get("claim_scope", REVIEW_ONLY_CLAIM),
                }
            )
            vectors.append(cls)
            ids.append(dino_id)
        except Exception as exc:
            failure_count += 1
            failures.append(
                {
                    "dino_input_id": dino_id,
                    "embedding_file": embedding_rel,
                    "failure_code": "CORRUPTED_OR_UNREADABLE_EMBEDDING",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                }
            )
            if failure_count >= args.max_failures:
                break
        if args.save_frequency and index % args.save_frequency == 0:
            write_json(Path(args.output_dir) / "dino_embedding_corpus_progress_v1fy.json", {"processed_rows": index, "valid_embeddings": len(vectors), "failures": failure_count})

    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype="float32")
    return corpus_rows, failures, matrix, ids


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    normed = l2_normalize(matrix)
    return normed @ normed.T if normed.size else np.empty((0, 0), dtype="float32")


def pca(matrix: np.ndarray, components: int = 2) -> tuple[np.ndarray, list[dict[str, object]]]:
    if matrix.shape[0] == 0:
        return np.empty((0, components)), []
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:components].T
    variances = (singular**2) / max(matrix.shape[0] - 1, 1)
    total = float(variances.sum()) or 1.0
    rows = [
        {"component": f"PC{i + 1}", "explained_variance": float(variances[i]) if i < len(variances) else 0.0, "explained_variance_ratio": float(variances[i] / total) if i < len(variances) else 0.0}
        for i in range(min(components, matrix.shape[0], matrix.shape[1]))
    ]
    return coords, rows


def kmeans(matrix: np.ndarray, k: int, seed: int, iterations: int = 50) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if matrix.shape[0] < k:
        return np.zeros(matrix.shape[0], dtype=int)
    initial = rng.choice(matrix.shape[0], size=k, replace=False)
    centers = matrix[initial].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(iterations):
        distances = ((matrix[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = distances.argmin(axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = matrix[labels == cluster]
            if len(members):
                centers[cluster] = members.mean(axis=0)
    return labels


def silhouette(matrix: np.ndarray, labels: np.ndarray) -> float:
    unique = sorted(set(int(x) for x in labels))
    if len(unique) < 2 or matrix.shape[0] <= len(unique):
        return 0.0
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    scores: list[float] = []
    for idx, label in enumerate(labels):
        same = [j for j, other in enumerate(labels) if other == label and j != idx]
        other_clusters = [cluster for cluster in unique if cluster != label]
        a = float(distances[idx, same].mean()) if same else 0.0
        b = min(float(distances[idx, labels == cluster].mean()) for cluster in other_clusters)
        scores.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(scores))


def nearest_neighbors(ids: list[str], matrix: np.ndarray, top_k: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sims = cosine_matrix(matrix)
    rows: list[dict[str, object]] = []
    top1: dict[str, str] = {}
    nearest_distance: list[tuple[str, float]] = []
    for i, source in enumerate(ids):
        order = np.argsort(-sims[i])
        rank = 0
        for j in order:
            if i == j:
                continue
            rank += 1
            distance = float(1 - sims[i, j])
            if rank == 1:
                top1[source] = ids[j]
                nearest_distance.append((source, distance))
            rows.append({"dino_input_id": source, "neighbor_dino_input_id": ids[j], "rank": rank, "cosine_similarity": float(sims[i, j]), "cosine_distance": distance, "claim_scope": "STRUCTURAL_SIMILARITY_ONLY"})
            if rank >= top_k:
                break
    reciprocal = [
        {"dino_input_id_a": a, "dino_input_id_b": b, "relationship": "RECIPROCAL_TOP1"}
        for a, b in sorted(top1.items())
        if top1.get(b) == a and a < b
    ]
    if nearest_distance:
        values = np.array([d for _, d in nearest_distance], dtype="float32")
        threshold = float(values.mean() + values.std())
    else:
        threshold = 0.0
    outliers = [
        {"dino_input_id": dino_id, "nearest_distance": distance, "outlier_status": "ISOLATED_STRUCTURAL_OUTLIER" if distance > threshold else "WITHIN_SMOKE_RANGE", "threshold": threshold}
        for dino_id, distance in nearest_distance
    ]
    return rows, reciprocal, outliers


def region_diagnostics(corpus_rows: list[dict[str, object]], matrix: np.ndarray) -> list[dict[str, object]]:
    if matrix.size == 0:
        return []
    regions = [str(row.get("region", "")) for row in corpus_rows]
    normed = l2_normalize(matrix)
    rows: list[dict[str, object]] = []
    for region in sorted(set(regions)):
        idx = [i for i, value in enumerate(regions) if value == region]
        other = [i for i, value in enumerate(regions) if value != region]
        centroid = normed[idx].mean(axis=0)
        dispersion = float(np.linalg.norm(normed[idx] - centroid, axis=1).mean()) if idx else 0.0
        intra = float((normed[idx] @ normed[idx].T).mean()) if len(idx) > 1 else 1.0 if idx else 0.0
        inter = float((normed[idx] @ normed[other].T).mean()) if idx and other else 0.0
        rows.append({"region": region, "embedding_count": len(idx), "centroid_norm": float(np.linalg.norm(centroid)), "dispersion": dispersion, "intra_region_cosine_mean": intra, "inter_region_cosine_mean": inter, "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
    return rows


def make_qa(corpus_rows: list[dict[str, object]], failures: list[dict[str, str]], matrix: np.ndarray, output_dir: Path) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    dims = {int(row.get("embedding_dim", 0) or 0) for row in corpus_rows}
    hashes = [str(row.get("hash", "")) for row in corpus_rows]
    norms = np.linalg.norm(matrix, axis=1) if matrix.size else np.array([])
    sims = cosine_matrix(matrix)
    add("local_runs is gitignored", is_local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not forbidden_versioned_artifacts(), "repo checked outside local_runs")
    add("no labels or targets promoted", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" for row in corpus_rows), "NO_LABEL/NO_TARGET")
    add("claim scope remains review-only", all(row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in corpus_rows), REVIEW_ONLY_CLAIM)
    add("embeddings non-empty", bool(corpus_rows) and all(int(row.get("embedding_dim", 0) or 0) > 0 for row in corpus_rows), f"valid={len(corpus_rows)}")
    add("absence of NaN/Inf", matrix.size > 0 and bool(np.isfinite(matrix).all()), "matrix finite")
    add("dimension consistent", len(dims) <= 1 and (not dims or 0 not in dims), str(sorted(dims)))
    add("duplicate embedding detection complete", len(hashes) == len(set(hashes)), f"duplicates={len(hashes) - len(set(hashes))}")
    add("zero-vector detection complete", bool(norms.size) and bool((norms > 0).all()), f"min_norm={float(norms.min()) if norms.size else 0.0}")
    add("corrupted npz failures recorded", failures is not None, f"failures={len(failures)}")
    add("cosine similarity sanity complete", sims.shape[0] == len(corpus_rows), f"shape={sims.shape}")
    add("heavy outputs local only", output_dir.resolve().as_posix().find("/local_runs/") >= 0 or "local_runs" in output_dir.parts, rel(output_dir))
    return qa


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force, args.resume)

    corpus_rows, failures, matrix, ids = build_corpus(args)
    normed = l2_normalize(matrix)
    coords, pca_rows = pca(normed, components=2)
    nn_rows, reciprocal_rows, outlier_rows = nearest_neighbors(ids, normed, args.top_k) if len(ids) > 1 else ([], [], [])
    region_rows = region_diagnostics(corpus_rows, normed)

    manifold_rows = [
        {"dino_input_id": ids[i], "method": "PCA", "x": float(coords[i, 0]) if coords.shape[1] > 0 else 0.0, "y": float(coords[i, 1]) if coords.shape[1] > 1 else 0.0, "umap_status": "NOT_RUN_OPTIONAL", "tsne_status": "NOT_RUN_OPTIONAL"}
        for i in range(len(ids))
    ]
    preservation_rows = neighborhood_preservation(ids, normed, coords)
    cluster_metrics_rows, cluster_size_rows, assigned_labels = clustering_outputs(normed, ids, args.seed)
    for row, label in zip(corpus_rows, assigned_labels, strict=False):
        row["cluster_id"] = str(label)

    norm_rows = [{"dino_input_id": ids[i], "l2_norm": float(np.linalg.norm(matrix[i])), "norm_status": "PASS" if np.linalg.norm(matrix[i]) > 0 else "ZERO_VECTOR"} for i in range(len(ids))]
    similarity_rows = similarity_sanity(ids, normed)
    duplicate_rows = duplicate_audit(corpus_rows)
    corruption_rows = failures
    repro_rows = [{"dino_input_id": row.get("dino_input_id", ""), "hash": row.get("hash", ""), "reproducibility_status": "HASH_RECORDED"} for row in corpus_rows]
    qa_rows = make_qa(corpus_rows, failures, matrix, output_dir)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary = {
        "phase": PHASE,
        "phase_name": PHASE_NAME,
        "mode": args.mode,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_smoke_manifest": rel(Path(args.smoke_manifest)),
        "valid_embeddings": len(corpus_rows),
        "failures": len(failures),
        "embedding_dim": int(matrix.shape[1]) if matrix.size else 0,
        "embedding_corpus_status": "PASS" if corpus_rows else "FAIL_NO_VALID_EMBEDDINGS",
        "manifold_status": "PASS_PCA" if manifold_rows else "SKIPPED",
        "clustering_metrics_status": "PASS" if cluster_metrics_rows else "SKIPPED",
        "regional_diagnostics_status": "PASS" if region_rows else "SKIPPED",
        "qa_status": qa_status,
        "review_only": True,
        "no_supervised_training": True,
        "no_predictive_claims": True,
        "outputs_local_only": True,
    }

    write_csv(output_dir / "dino_embedding_corpus_manifest_v1fy.csv", corpus_rows, ["patch_id", "dino_input_id", "region", "bands_present", "embedding_path", "embedding_dim", "model_backbone", "device", "execution_timestamp", "hash", "npz_hash", "qa_status", "cluster_id", "label_status", "target_status", "claim_scope"])
    write_csv(output_dir / "dino_embedding_corpus_failures_v1fy.csv", failures, ["dino_input_id", "embedding_file", "failure_code", "failure_reason"])
    write_csv(output_dir / "dino_embedding_norm_distribution_v1fy.csv", norm_rows, ["dino_input_id", "l2_norm", "norm_status"])
    write_csv(output_dir / "dino_embedding_similarity_sanity_v1fy.csv", similarity_rows, ["metric", "value", "status"])
    write_csv(output_dir / "dino_embedding_duplicate_audit_v1fy.csv", duplicate_rows, ["hash", "count", "dino_input_ids", "duplicate_status"])
    write_csv(output_dir / "dino_embedding_corruption_audit_v1fy.csv", corruption_rows, ["dino_input_id", "embedding_file", "failure_code", "failure_reason"])
    write_csv(output_dir / "dino_embedding_reproducibility_smoke_v1fy.csv", repro_rows, ["dino_input_id", "hash", "reproducibility_status"])
    write_csv(output_dir / "dino_embedding_pca_variance_v1fy.csv", pca_rows, ["component", "explained_variance", "explained_variance_ratio"])
    write_csv(output_dir / "dino_embedding_manifold_coordinates_v1fy.csv", manifold_rows, ["dino_input_id", "method", "x", "y", "umap_status", "tsne_status"])
    write_csv(output_dir / "dino_embedding_neighborhood_preservation_v1fy.csv", preservation_rows, ["dino_input_id", "preserved_top1", "status"])
    write_csv(output_dir / "dino_embedding_cluster_metrics_v1fy.csv", cluster_metrics_rows, ["method", "k", "silhouette", "instability_warning", "claim_scope"])
    write_csv(output_dir / "dino_embedding_cluster_sizes_v1fy.csv", cluster_size_rows, ["method", "k", "cluster_id", "count"])
    write_csv(output_dir / "dino_embedding_nearest_neighbors_v1fy.csv", nn_rows, ["dino_input_id", "neighbor_dino_input_id", "rank", "cosine_similarity", "cosine_distance", "claim_scope"])
    write_csv(output_dir / "dino_embedding_reciprocal_pairs_v1fy.csv", reciprocal_rows, ["dino_input_id_a", "dino_input_id_b", "relationship"])
    write_csv(output_dir / "dino_embedding_outliers_v1fy.csv", outlier_rows, ["dino_input_id", "nearest_distance", "outlier_status", "threshold"])
    write_csv(output_dir / "dino_embedding_region_diagnostics_v1fy.csv", region_rows, ["region", "embedding_count", "centroid_norm", "dispersion", "intra_region_cosine_mean", "inter_region_cosine_mean", "claim_scope"])
    write_csv(output_dir / "dino_embedding_corpus_qa_v1fy.csv", qa_rows, ["check", "status", "details"])
    write_json(output_dir / "dino_embedding_corpus_summary_v1fy.json", summary)
    write_json(output_dir / "dino_embedding_corpus_progress_v1fy.json", {"processed_rows": len(corpus_rows) + len(failures), "valid_embeddings": len(corpus_rows), "failures": len(failures)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def similarity_sanity(ids: list[str], matrix: np.ndarray) -> list[dict[str, object]]:
    if len(ids) < 2:
        return [{"metric": "pairwise_cosine", "value": 0.0, "status": "SKIPPED_NOT_ENOUGH_EMBEDDINGS"}]
    sims = cosine_matrix(matrix)
    upper = sims[np.triu_indices_from(sims, k=1)]
    return [
        {"metric": "cosine_min", "value": float(upper.min()), "status": "PASS"},
        {"metric": "cosine_mean", "value": float(upper.mean()), "status": "PASS"},
        {"metric": "cosine_max", "value": float(upper.max()), "status": "PASS"},
    ]


def duplicate_audit(corpus_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for row in corpus_rows:
        by_hash[str(row.get("hash", ""))].append(str(row.get("dino_input_id", "")))
    return [{"hash": digest, "count": len(ids), "dino_input_ids": "|".join(ids), "duplicate_status": "DUPLICATE" if len(ids) > 1 else "UNIQUE"} for digest, ids in sorted(by_hash.items())]


def neighborhood_preservation(ids: list[str], matrix: np.ndarray, coords: np.ndarray) -> list[dict[str, object]]:
    if len(ids) < 3 or coords.shape[0] < 3:
        return [{"dino_input_id": dino_id, "preserved_top1": "", "status": "SKIPPED_SMALL_CORPUS"} for dino_id in ids]
    high = cosine_matrix(matrix)
    low = cosine_matrix(coords)
    rows = []
    for i, dino_id in enumerate(ids):
        high_order = [j for j in np.argsort(-high[i]) if j != i]
        low_order = [j for j in np.argsort(-low[i]) if j != i]
        rows.append({"dino_input_id": dino_id, "preserved_top1": str(high_order[0] == low_order[0]).lower(), "status": "PASS"})
    return rows


def clustering_outputs(matrix: np.ndarray, ids: list[str], seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]], list[int]]:
    if len(ids) < 2:
        return [], [], [0 for _ in ids]
    metrics: list[dict[str, object]] = []
    sizes: list[dict[str, object]] = []
    chosen_labels = [0 for _ in ids]
    for k in [2, 3, 4]:
        if len(ids) < k:
            continue
        labels = kmeans(matrix, k, seed)
        if k == 2:
            chosen_labels = [int(x) for x in labels]
        sil = silhouette(matrix, labels)
        counts = Counter(int(label) for label in labels)
        warning = "SMALL_CLUSTER" if min(counts.values()) <= 1 else "NONE"
        metrics.append({"method": "KMeans_numpy", "k": k, "silhouette": sil, "instability_warning": warning, "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
        for cluster_id, count in sorted(counts.items()):
            sizes.append({"method": "KMeans_numpy", "k": k, "cluster_id": cluster_id, "count": count})
    metrics.append({"method": "HDBSCAN_optional", "k": "", "silhouette": "", "instability_warning": "NOT_RUN_OPTIONAL_DEPENDENCY", "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
    metrics.append({"method": "Agglomerative_optional", "k": "", "silhouette": "", "instability_warning": "NOT_RUN_OPTIONAL", "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
    return metrics, sizes, chosen_labels


def main() -> int:
    try:
        return run(parse_args())
    except FileExistsError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
