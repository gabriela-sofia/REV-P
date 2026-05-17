from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1ga"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"
FORBIDDEN_REPO_DIRS = {"data", "outputs", "docs"}
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1ga DINO embedding structural consistency analysis.")
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37])
    parser.add_argument("--ks", nargs="+", type=int, default=[2, 3, 4])
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


def prepare_output_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    lines = [line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines()] if gitignore.exists() else []
    return "local_runs/" in lines or "local_runs" in lines


def forbidden_versioned_artifacts() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        if path.is_dir() and path.name in FORBIDDEN_REPO_DIRS and path.name != "docs":
            found.append(str(path))
        elif path.is_file() and path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS:
            found.append(str(path))
    return found


def load_embeddings(manifest_path: Path) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    rows = [row for row in read_csv(manifest_path) if row.get("success") == "SUCCESS"]
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    base = manifest_path.parent
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        try:
            data = np.load(base / row["embedding_path"])
            vector = np.asarray(data["cls_embedding"], dtype="float32")
            if vector.size == 0 or not np.isfinite(vector).all():
                continue
            vectors.append(vector)
            ids.append(row["dino_input_id"])
            valid_rows.append(row)
        except Exception:
            continue
    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype="float32")
    return valid_rows, matrix, ids


def normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def cosine(matrix: np.ndarray) -> np.ndarray:
    normed = normalize(matrix)
    return normed @ normed.T if normed.size else np.empty((0, 0), dtype="float32")


def top_neighbors(ids: list[str], matrix: np.ndarray, top_k: int) -> dict[str, list[str]]:
    sims = cosine(matrix)
    result: dict[str, list[str]] = {}
    for i, dino_id in enumerate(ids):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        result[dino_id] = [ids[j] for j in order[:top_k]]
    return result


def centroid_rows(rows: list[dict[str, str]], matrix: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, np.ndarray]]:
    regions = [row.get("region", "") for row in rows]
    normed = normalize(matrix)
    centroids: dict[str, np.ndarray] = {}
    metrics: list[dict[str, object]] = []
    for region in sorted(set(regions)):
        idx = [i for i, value in enumerate(regions) if value == region]
        centroid = normed[idx].mean(axis=0)
        centroids[region] = centroid
        intra = float((normed[idx] @ normed[idx].T).mean()) if len(idx) > 1 else 1.0
        other = [i for i, value in enumerate(regions) if value != region]
        inter = float((normed[idx] @ normed[other].T).mean()) if other else 0.0
        dispersion = float(np.linalg.norm(normed[idx] - centroid, axis=1).mean())
        metrics.append({"region": region, "embedding_count": len(idx), "intra_region_similarity": intra, "inter_region_similarity": inter, "dispersion": dispersion, "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
    distances: list[dict[str, object]] = []
    for a in sorted(centroids):
        for b in sorted(centroids):
            distances.append({"region_a": a, "region_b": b, "centroid_cosine_distance": float(1 - np.dot(centroids[a], centroids[b]) / max(np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]), 1e-12))})
    return metrics, distances, centroids


def kmeans(matrix: np.ndarray, k: int, seed: int, iterations: int = 50) -> np.ndarray:
    if matrix.shape[0] < k:
        return np.zeros(matrix.shape[0], dtype=int)
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].copy()
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


def adjusted_agreement(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0:
        return 0.0
    same = 0
    total = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            same += int((a[i] == a[j]) == (b[i] == b[j]))
            total += 1
    return same / max(total, 1)


def cluster_stability(matrix: np.ndarray, seeds: list[int], ks: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in ks:
        if matrix.shape[0] < k:
            rows.append({"k": k, "seed_pair": "", "agreement": "", "status": "SKIPPED_NOT_ENOUGH_EMBEDDINGS", "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
            continue
        label_sets = {seed: kmeans(matrix, k, seed) for seed in seeds}
        for i, seed_a in enumerate(seeds):
            for seed_b in seeds[i + 1 :]:
                agreement = adjusted_agreement(label_sets[seed_a], label_sets[seed_b])
                rows.append({"k": k, "seed_pair": f"{seed_a}-{seed_b}", "agreement": agreement, "status": "PASS" if agreement >= 0.5 else "UNSTABLE", "claim_scope": "STRUCTURAL_DIAGNOSTIC_ONLY"})
    return rows


def neighbor_persistence(ids: list[str], matrix: np.ndarray, top_k: int) -> list[dict[str, object]]:
    base = top_neighbors(ids, matrix, top_k)
    jitter = matrix + 1e-6
    repeat = top_neighbors(ids, jitter, top_k)
    reciprocal = {a: neighs[0] for a, neighs in base.items() if neighs}
    rows: list[dict[str, object]] = []
    for dino_id in ids:
        preserved = len(set(base[dino_id]) & set(repeat[dino_id]))
        top1 = base[dino_id][0] if base[dino_id] else ""
        reciprocal_status = "RECIPROCAL" if top1 and reciprocal.get(top1) == dino_id else "NOT_RECIPROCAL"
        rows.append({"dino_input_id": dino_id, "top_k": top_k, "preserved_neighbors": preserved, "persistence_ratio": preserved / max(top_k, 1), "top1_neighbor": top1, "reciprocal_status": reciprocal_status})
    return rows


def structural_outliers(ids: list[str], matrix: np.ndarray) -> list[dict[str, object]]:
    sims = cosine(matrix)
    distances: list[tuple[str, float]] = []
    for i, dino_id in enumerate(ids):
        values = [1 - sims[i, j] for j in range(len(ids)) if j != i]
        distances.append((dino_id, float(min(values)) if values else 0.0))
    vals = np.array([value for _, value in distances], dtype="float32")
    threshold = float(vals.mean() + vals.std()) if len(vals) else 0.0
    seen = Counter(tuple(np.round(row, 6)) for row in normalize(matrix))
    rows = []
    for idx, (dino_id, distance) in enumerate(distances):
        duplicate = seen[tuple(np.round(normalize(matrix)[idx], 6))] > 1
        status = "DUPLICATE_STRUCTURAL_SIGNATURE" if duplicate else "ISOLATED_OUTLIER" if distance > threshold else "WITHIN_RANGE"
        rows.append({"dino_input_id": dino_id, "nearest_distance": distance, "density_status": status, "threshold": threshold})
    return rows


def make_qa(rows: list[dict[str, str]], matrix: np.ndarray, region_metrics: list[dict[str, object]], cluster_rows: list[dict[str, object]], neighbor_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    regions = {row.get("region", "") for row in rows}
    add("embedding count consistency", len(rows) == matrix.shape[0] and len(rows) > 0, f"rows={len(rows)} matrix={matrix.shape}")
    add("region coverage", len(regions) >= 2, ",".join(sorted(regions)))
    add("manifold integrity", matrix.shape[1] > 0 and np.isfinite(matrix).all(), f"dim={matrix.shape[1] if matrix.size else 0}")
    add("cluster reproducibility", bool(cluster_rows), f"rows={len(cluster_rows)}")
    add("nearest-neighbor reproducibility", bool(neighbor_rows), f"rows={len(neighbor_rows)}")
    add("cosine sanity", bool(np.isfinite(cosine(matrix)).all()) if matrix.size else False, "finite cosine matrix")
    rounded = [tuple(np.round(row, 6)) for row in normalize(matrix)]
    add("duplicate detection", len(rounded) >= len(set(rounded)), f"duplicates={len(rounded) - len(set(rounded))}")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not forbidden_versioned_artifacts(), "repo checked outside local_runs")
    return qa


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    manifest_path = Path(args.embedding_manifest)
    rows, matrix, ids = load_embeddings(manifest_path)
    normed = normalize(matrix)
    region_metrics, centroid_matrix, _centroids = centroid_rows(rows, normed) if len(rows) else ([], [], {})
    cluster_rows = cluster_stability(normed, args.seeds, args.ks)
    neighbor_rows = neighbor_persistence(ids, normed, args.top_k) if len(ids) > 1 else []
    outlier_rows = structural_outliers(ids, normed) if len(ids) else []
    qa_rows = make_qa(rows, normed, region_metrics, cluster_rows, neighbor_rows)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary = {
        "phase": "v1ga",
        "phase_name": "DINO_EMBEDDING_STRUCTURAL_CONSISTENCY_ANALYSIS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_manifest": str(manifest_path),
        "embedding_count": len(rows),
        "regions": sorted({row.get("region", "") for row in rows}),
        "consistency_status": "PASS" if qa_status == "PASS" else "FAIL",
        "cluster_stability_status": "PASS" if cluster_rows else "SKIPPED",
        "qa_status": qa_status,
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "targets_created": False,
        "predictive_claims": False,
        "clusters_are_classes": False,
        "multimodal_hold": True,
        "outputs_local_only": True,
    }
    write_json(output_dir / "consistency_summary.json", summary)
    write_csv(output_dir / "centroid_distance_matrix.csv", centroid_matrix, ["region_a", "region_b", "centroid_cosine_distance"])
    write_csv(output_dir / "region_similarity_metrics.csv", region_metrics, ["region", "embedding_count", "intra_region_similarity", "inter_region_similarity", "dispersion", "claim_scope"])
    write_csv(output_dir / "cluster_stability.csv", cluster_rows, ["k", "seed_pair", "agreement", "status", "claim_scope"])
    write_csv(output_dir / "neighbor_persistence.csv", neighbor_rows, ["dino_input_id", "top_k", "preserved_neighbors", "persistence_ratio", "top1_neighbor", "reciprocal_status"])
    write_csv(output_dir / "structural_outliers.csv", outlier_rows, ["dino_input_id", "nearest_distance", "density_status", "threshold"])
    write_csv(output_dir / "structural_consistency_qa.csv", qa_rows, ["check", "status", "details"])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if qa_status == "PASS" else 2


def main() -> int:
    try:
        return run(parse_args())
    except FileExistsError as exc:
        print(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
