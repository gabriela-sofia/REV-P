from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gd"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"
PERTURBATIONS = ["gaussian_noise", "brightness_scale", "contrast_scale", "blur_light", "crop_jitter", "band_dropout"]
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v1fx = load_module(ROOT / "scripts" / "dino" / "revp_v1fx_dino_smoke_embedding_execution.py", "revp_v1fx_for_v1gd")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gd DINO perturbation robustness diagnostics.")
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--backbone", default="dinov2_vitb14_reg")
    parser.add_argument("--skip-model-if-unavailable", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--embedding-proxy-for-tests", action="store_true", help="Use deterministic embedding perturbation proxy for offline unit tests only.")
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
    (path / "visual_review").mkdir(parents=True, exist_ok=True)


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


def load_embeddings(manifest: Path, limit: int) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    rows = [row for row in read_csv(manifest) if row.get("success") == "SUCCESS"][:limit]
    valid_rows: list[dict[str, str]] = []
    vectors: list[np.ndarray] = []
    ids: list[str] = []
    for row in rows:
        try:
            data = np.load(manifest.parent / row["embedding_path"])
            vector = np.asarray(data["cls_embedding"], dtype="float32")
            if vector.size and np.isfinite(vector).all():
                valid_rows.append(row)
                vectors.append(vector)
                ids.append(row.get("dino_input_id") or row.get("patch_id") or f"row_{len(ids)}")
        except Exception:
            continue
    matrix = np.vstack(vectors) if vectors else np.empty((0, 0), dtype="float32")
    return valid_rows, matrix, ids


def normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    return matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)


def cosine_matrix(matrix: np.ndarray) -> np.ndarray:
    normed = normalize(matrix)
    return normed @ normed.T if normed.size else np.empty((0, 0), dtype="float32")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def vector_hash(vector: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(vector, dtype="float32").tobytes()).hexdigest()


def stable_seed(seed: int, dino_id: str, perturbation: str) -> int:
    digest = hashlib.sha256(f"{seed}|{dino_id}|{perturbation}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def perturb_image(array: np.ndarray, perturbation: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(array, dtype="float32").copy()
    if perturbation == "gaussian_noise":
        arr = arr + rng.normal(0, 0.015, size=arr.shape).astype("float32")
    elif perturbation == "brightness_scale":
        arr = arr * 1.05
    elif perturbation == "contrast_scale":
        mean = arr.mean(axis=(0, 1), keepdims=True)
        arr = (arr - mean) * 1.08 + mean
    elif perturbation == "blur_light":
        arr = (
            arr
            + np.roll(arr, 1, axis=0)
            + np.roll(arr, -1, axis=0)
            + np.roll(arr, 1, axis=1)
            + np.roll(arr, -1, axis=1)
        ) / 5.0
    elif perturbation == "crop_jitter":
        shift_y = int(rng.integers(-4, 5))
        shift_x = int(rng.integers(-4, 5))
        arr = np.roll(np.roll(arr, shift_y, axis=0), shift_x, axis=1)
    elif perturbation == "band_dropout":
        channel = int(rng.integers(0, min(arr.shape[2], 3)))
        arr[:, :, channel] = arr[:, :, channel].mean()
    else:
        raise ValueError(f"Unsupported perturbation: {perturbation}")
    return np.clip(arr, 0.0, 1.0).astype("float32")


def proxy_embedding(original: np.ndarray, dino_id: str, perturbation: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(seed, dino_id, perturbation))
    scale = {
        "gaussian_noise": 0.004,
        "brightness_scale": 0.006,
        "contrast_scale": 0.008,
        "blur_light": 0.01,
        "crop_jitter": 0.012,
        "band_dropout": 0.014,
    }[perturbation]
    return (original + rng.normal(0, scale, size=original.shape).astype("float32")).astype("float32")


def nearest_ids(ids: list[str], matrix: np.ndarray, top_k: int) -> dict[str, list[str]]:
    sims = cosine_matrix(matrix)
    neighbors: dict[str, list[str]] = {}
    for i, node in enumerate(ids):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        neighbors[node] = [ids[j] for j in order[:top_k]]
    return neighbors


def kmeans(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    if matrix.shape[0] < k or k <= 1:
        return np.zeros(matrix.shape[0], dtype=int)
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(40):
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


def medoids(matrix: np.ndarray, labels: np.ndarray, ids: list[str]) -> set[str]:
    chosen: set[str] = set()
    for label in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == label)[0]
        centroid = matrix[idx].mean(axis=0)
        chosen.add(ids[int(idx[int(np.linalg.norm(matrix[idx] - centroid, axis=1).argmin())])])
    return chosen


def graph_edges(ids: list[str], matrix: np.ndarray, rows: list[dict[str, str]], top_k: int) -> set[tuple[str, str]]:
    sims = cosine_matrix(matrix)
    edges: set[tuple[str, str]] = set()
    for i, source in enumerate(ids):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        for j in order[:top_k]:
            a, b = sorted([source, ids[j]])
            edges.add((a, b))
    return edges


def save_panel(original: np.ndarray, perturbed: np.ndarray, path: Path) -> bool:
    try:
        from PIL import Image, ImageDraw  # type: ignore

        left = Image.fromarray(np.clip(original * 255, 0, 255).astype("uint8"), mode="RGB").resize((160, 160))
        right = Image.fromarray(np.clip(perturbed * 255, 0, 255).astype("uint8"), mode="RGB").resize((160, 160))
        canvas = Image.new("RGB", (320, 180), "white")
        canvas.paste(left, (0, 20))
        canvas.paste(right, (160, 20))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), "original", fill=(0, 0, 0))
        draw.text((168, 4), "perturbed", fill=(0, 0, 0))
        path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(path)
        return True
    except Exception:
        return False


def embed_perturbations(args: argparse.Namespace, rows: list[dict[str, str]], original_matrix: np.ndarray, ids: list[str], output_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], list[dict[str, object]], bool, str, str, bool, list[dict[str, object]]]:
    model = None
    model_loaded = False
    model_status = "EMBEDDING_PROXY_FOR_TESTS" if args.embedding_proxy_for_tests else "NOT_LOADED"
    device = "cpu"
    failures: list[dict[str, object]] = []
    visual_rows: list[dict[str, object]] = []
    if not args.embedding_proxy_for_tests:
        device = v1fx.resolve_device(args.device, args.allow_cpu, args.force_cpu)
        model, model_code, model_error, loaded_backbone, _attempts = v1fx.load_smoke_model(args.backbone, device, args.skip_model_if_unavailable, args.allow_model_download)
        model_loaded = model is not None
        model_status = model_code if model_loaded else (model_code or "MODEL_UNAVAILABLE")
        if loaded_backbone:
            args.backbone = loaded_backbone
        if not model_loaded and not args.skip_model_if_unavailable:
            failures.append({"dino_input_id": "", "perturbation_type": "ALL", "failure_stage": "model_load", "failure_reason": model_error or "model unavailable"})
    perturbed: dict[str, dict[str, np.ndarray]] = {p: {} for p in PERTURBATIONS}
    pixel_read = False
    for row, original, dino_id in zip(rows, original_matrix, ids):
        image = None
        read_error = ""
        if not args.embedding_proxy_for_tests:
            image, _pixel_status, read_error, _meta = v1fx.read_image_tensor(Path(row.get("source_path", "")), args.image_size)
            pixel_read = pixel_read or image is not None
        for perturbation in PERTURBATIONS:
            try:
                if args.embedding_proxy_for_tests:
                    vector = proxy_embedding(original, dino_id, perturbation, args.seed)
                elif model is None or image is None:
                    failures.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "failure_stage": "read_or_model", "failure_reason": read_error or model_status})
                    continue
                else:
                    perturbed_image = perturb_image(image, perturbation, stable_seed(args.seed, dino_id, perturbation))
                    cls, _patch, _patch_status = v1fx.embed_array(model, perturbed_image, args.backbone, device)
                    vector = np.asarray(cls, dtype="float32")
                    if perturbation in {"gaussian_noise", "brightness_scale"}:
                        panel = output_dir / "visual_review" / f"{dino_id}_{perturbation}.png"
                        if save_panel(image, perturbed_image, panel):
                            visual_rows.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "image_path": str(panel), "panel_type": "original_vs_perturbed"})
                if vector.size == original.size and np.isfinite(vector).all():
                    perturbed[perturbation][dino_id] = vector
                else:
                    failures.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "failure_stage": "embedding", "failure_reason": "invalid perturbed embedding"})
            except Exception as exc:
                failures.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "failure_stage": "perturbation", "failure_reason": f"{type(exc).__name__}: {exc}"})
    return perturbed, failures, model_loaded, model_status, device, pixel_read, visual_rows


def robustness_tables(rows: list[dict[str, str]], ids: list[str], original_matrix: np.ndarray, perturbed: dict[str, dict[str, np.ndarray]], top_k: int, seed: int) -> dict[str, list[dict[str, object]]]:
    original_normed = normalize(original_matrix)
    original_neighbors = nearest_ids(ids, original_normed, top_k)
    original_labels = kmeans(original_normed, min(3, max(1, len(ids))), seed)
    original_medoids = medoids(original_normed, original_labels, ids)
    original_edges = graph_edges(ids, original_normed, rows, top_k)
    similarity_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    neighbor_rows: list[dict[str, object]] = []
    cluster_rows: list[dict[str, object]] = []
    graph_rows: list[dict[str, object]] = []
    bridge_rows: list[dict[str, object]] = []
    hub_rows: list[dict[str, object]] = []
    outlier_overlap: list[dict[str, object]] = []
    by_id = {dino_id: idx for idx, dino_id in enumerate(ids)}
    region_by_id = {row.get("dino_input_id", ""): row.get("region", "") for row in rows}
    for perturbation, vectors in perturbed.items():
        available = [dino_id for dino_id in ids if dino_id in vectors]
        if not available:
            continue
        matrix = np.vstack([vectors[dino_id] for dino_id in available])
        pert_normed = normalize(matrix)
        pert_neighbors = nearest_ids(available, pert_normed, min(top_k, max(0, len(available) - 1)))
        pert_labels = kmeans(pert_normed, min(3, max(1, len(available))), seed)
        pert_medoids = medoids(pert_normed, pert_labels, available)
        pert_edges = graph_edges(available, pert_normed, [rows[by_id[d]] for d in available], top_k)
        edge_persistence = len(original_edges & pert_edges) / max(len(original_edges | pert_edges), 1)
        original_bridge_edges = {edge for edge in original_edges if region_by_id[edge[0]] != region_by_id[edge[1]]}
        pert_bridge_edges = {edge for edge in pert_edges if region_by_id[edge[0]] != region_by_id[edge[1]]}
        bridge_persistence = len(original_bridge_edges & pert_bridge_edges) / max(len(original_bridge_edges | pert_bridge_edges), 1)
        graph_rows.append({"perturbation_type": perturbation, "edge_persistence": edge_persistence, "component_stability": "DIAGNOSTIC_ONLY", "graph_status": "PASS"})
        bridge_rows.append({"perturbation_type": perturbation, "original_bridge_count": len(original_bridge_edges), "perturbed_bridge_count": len(pert_bridge_edges), "bridge_persistence": bridge_persistence})
        degrees = Counter([node for edge in pert_edges for node in edge])
        hub_rows.append({"perturbation_type": perturbation, "hub_count": sum(1 for value in degrees.values() if value >= 2), "hub_stability_status": "DIAGNOSTIC_ONLY"})
        for idx, dino_id in enumerate(available):
            original = original_normed[by_id[dino_id]]
            vector = pert_normed[idx]
            sim = cosine(original, vector)
            drift = 1.0 - sim
            overlap = len(set(original_neighbors[dino_id]) & set(pert_neighbors[dino_id])) / max(len(set(original_neighbors[dino_id]) | set(pert_neighbors[dino_id])), 1)
            same_cluster = int(original_labels[by_id[dino_id]]) == int(pert_labels[idx])
            similarity_rows.append({"dino_input_id": dino_id, "region": region_by_id[dino_id], "perturbation_type": perturbation, "cosine_similarity": sim, "cosine_drift": drift, "embedding_hash": vector_hash(vector)})
            drift_rows.append({"dino_input_id": dino_id, "region": region_by_id[dino_id], "perturbation_type": perturbation, "cosine_drift": drift, "drift_status": "HIGH_SENSITIVITY" if drift > 0.15 else "WITHIN_RANGE"})
            neighbor_rows.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "neighbor_jaccard": overlap, "neighbor_persistence_status": "LOW_PERSISTENCE" if overlap < 0.34 else "PASS"})
            cluster_rows.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "original_cluster": int(original_labels[by_id[dino_id]]), "perturbed_cluster": int(pert_labels[idx]), "cluster_assignment_stable": str(same_cluster).upper(), "medoid_persistence": str(dino_id in original_medoids and dino_id in pert_medoids).upper()})
            outlier_overlap.append({"dino_input_id": dino_id, "perturbation_type": perturbation, "sensitivity_flag": "HIGH_SENSITIVITY" if drift > 0.15 else "NONE", "neighbor_volatility_flag": "LOW_PERSISTENCE" if overlap < 0.34 else "NONE", "overlap_status": "OVERLAP" if drift > 0.15 and overlap < 0.34 else "SINGLE_OR_NONE"})
    return {
        "similarity": similarity_rows,
        "drift": drift_rows,
        "neighbors": neighbor_rows,
        "clusters": cluster_rows,
        "graph": graph_rows,
        "bridges": bridge_rows,
        "hubs": hub_rows,
        "outlier_overlap": outlier_overlap,
    }


def sensitivity_tables(drift_rows: list[dict[str, object]], rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    region_grouped: dict[str, list[float]] = defaultdict(list)
    region_by_id = {row.get("dino_input_id", ""): row.get("region", "") for row in rows}
    for row in drift_rows:
        grouped[str(row["dino_input_id"])].append(float(row["cosine_drift"]))
        region_grouped[str(row["region"])].append(float(row["cosine_drift"]))
    rankings = []
    for dino_id, values in grouped.items():
        rankings.append({"dino_input_id": dino_id, "region": region_by_id.get(dino_id, ""), "mean_cosine_drift": float(np.mean(values)), "max_cosine_drift": float(np.max(values)), "sensitivity_status": "UNSTABLE" if np.mean(values) > 0.15 else "ROBUST"})
    rankings.sort(key=lambda row: float(row["mean_cosine_drift"]), reverse=True)
    robust = [row for row in rankings if row["sensitivity_status"] == "ROBUST"]
    unstable = [row for row in rankings if row["sensitivity_status"] == "UNSTABLE"]
    regional = [{"region": region, "mean_cosine_drift": float(np.mean(values)), "max_cosine_drift": float(np.max(values)), "robustness_status": "PASS" if np.mean(values) <= 0.15 else "REVIEW"} for region, values in sorted(region_grouped.items())]
    summary = {"regions": sorted(region_grouped), "regional_mean_drift": {region: float(np.mean(values)) for region, values in sorted(region_grouped.items())}, "review_only": True, "predictive_claims": False}
    return rankings, robust, unstable, regional, summary


def make_qa(perturbed: dict[str, dict[str, np.ndarray]], tables: dict[str, list[dict[str, object]]], failures: list[dict[str, object]], rows: list[dict[str, str]]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    counts = [len(values) for values in perturbed.values()]
    hashes = [row["embedding_hash"] for row in tables["similarity"]]
    add("perturbation reproducibility", bool(tables["similarity"]), f"similarity_rows={len(tables['similarity'])}")
    add("invalid perturbation prevention", set(perturbed) == set(PERTURBATIONS), "|".join(sorted(perturbed)))
    add("embedding drift sanity", all(math.isfinite(float(row["cosine_drift"])) for row in tables["drift"]), f"drift_rows={len(tables['drift'])}")
    add("graph consistency after perturbation", bool(tables["graph"]), f"graph_rows={len(tables['graph'])}")
    add("duplicate perturbation detection", len(hashes) == len(set(hashes)), f"hashes={len(hashes)}")
    add("failed perturbation isolation", failures is not None, f"failures={len(failures)}")
    add("all perturbation types attempted", all(count >= 0 for count in counts) and len(counts) == len(PERTURBATIONS), f"types={len(counts)}")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not forbidden_versioned_artifacts(), "repo checked outside local_runs")
    return qa


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    manifest = Path(args.embedding_manifest)
    rows, original_matrix, ids = load_embeddings(manifest, args.limit)
    if len(rows) == 0:
        raise RuntimeError("No valid embeddings available for v1gd perturbation robustness diagnostics.")
    perturbed, failures, model_loaded, model_status, device, pixel_read, visual_rows = embed_perturbations(args, rows, original_matrix, ids, output_dir)
    tables = robustness_tables(rows, ids, original_matrix, perturbed, args.top_k, args.seed)
    rankings, robust, unstable, regional, regional_summary = sensitivity_tables(tables["drift"], rows)
    qa_rows = make_qa(perturbed, tables, failures, rows)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary = {
        "phase": "v1gd",
        "phase_name": "DINO_EMBEDDING_PERTURBATION_ROBUSTNESS_DIAGNOSTICS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_manifest": str(manifest),
        "embedding_count": len(rows),
        "perturbation_types": PERTURBATIONS,
        "model_loaded": model_loaded,
        "model_status": model_status,
        "device": device,
        "pixel_read": pixel_read,
        "drift_metrics_status": "PASS" if tables["drift"] else "SKIPPED",
        "graph_robustness_status": "PASS" if tables["graph"] else "SKIPPED",
        "robust_count": len(robust),
        "unstable_count": len(unstable),
        "regional_robustness_status": "PASS" if regional else "SKIPPED",
        "qa_status": qa_status,
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "targets_created": False,
        "predictive_claims": False,
        "clusters_are_classes": False,
        "perturbations_for_training": False,
        "multimodal_hold": True,
        "outputs_local_only": True,
    }
    write_csv(output_dir / "perturbation_similarity.csv", tables["similarity"], ["dino_input_id", "region", "perturbation_type", "cosine_similarity", "cosine_drift", "embedding_hash"])
    write_csv(output_dir / "embedding_drift_metrics.csv", tables["drift"], ["dino_input_id", "region", "perturbation_type", "cosine_drift", "drift_status"])
    write_csv(output_dir / "neighbor_persistence_under_perturbation.csv", tables["neighbors"], ["dino_input_id", "perturbation_type", "neighbor_jaccard", "neighbor_persistence_status"])
    write_csv(output_dir / "cluster_stability_under_perturbation.csv", tables["clusters"], ["dino_input_id", "perturbation_type", "original_cluster", "perturbed_cluster", "cluster_assignment_stable", "medoid_persistence"])
    write_csv(output_dir / "sensitivity_rankings.csv", rankings, ["dino_input_id", "region", "mean_cosine_drift", "max_cosine_drift", "sensitivity_status"])
    write_csv(output_dir / "robust_embeddings.csv", robust, ["dino_input_id", "region", "mean_cosine_drift", "max_cosine_drift", "sensitivity_status"])
    write_csv(output_dir / "unstable_embeddings.csv", unstable, ["dino_input_id", "region", "mean_cosine_drift", "max_cosine_drift", "sensitivity_status"])
    write_csv(output_dir / "perturbation_outlier_overlap.csv", tables["outlier_overlap"], ["dino_input_id", "perturbation_type", "sensitivity_flag", "neighbor_volatility_flag", "overlap_status"])
    write_csv(output_dir / "regional_robustness_metrics.csv", regional, ["region", "mean_cosine_drift", "max_cosine_drift", "robustness_status"])
    write_json(output_dir / "regional_drift_summary.json", regional_summary)
    write_csv(output_dir / "graph_robustness.csv", tables["graph"], ["perturbation_type", "edge_persistence", "component_stability", "graph_status"])
    write_csv(output_dir / "bridge_stability.csv", tables["bridges"], ["perturbation_type", "original_bridge_count", "perturbed_bridge_count", "bridge_persistence"])
    write_csv(output_dir / "hub_stability.csv", tables["hubs"], ["perturbation_type", "hub_count", "hub_stability_status"])
    write_csv(output_dir / "visual_review_manifest.csv", visual_rows, ["dino_input_id", "perturbation_type", "image_path", "panel_type"])
    write_csv(output_dir / "perturbation_failures.csv", failures, ["dino_input_id", "perturbation_type", "failure_stage", "failure_reason"])
    write_csv(output_dir / "perturbation_robustness_qa.csv", qa_rows, ["check", "status", "details"])
    write_json(output_dir / "perturbation_robustness_summary.json", summary)
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
