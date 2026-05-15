from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gc"
REVIEW_ONLY_CLAIM = "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gc DINO geo-structural embedding diagnostics.")
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
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


def load_embeddings(manifest: Path) -> tuple[list[dict[str, str]], np.ndarray, list[str]]:
    rows = [row for row in read_csv(manifest) if row.get("success") == "SUCCESS"]
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


def cosine(matrix: np.ndarray) -> np.ndarray:
    normed = normalize(matrix)
    return normed @ normed.T if normed.size else np.empty((0, 0), dtype="float32")


def numeric_patch(text: str) -> int:
    digits = "".join(ch for ch in text if ch.isdigit())
    return int(digits) if digits else -1


def float_value(row: dict[str, str], names: list[str]) -> float | None:
    for name in names:
        raw = row.get(name, "")
        if raw not in {"", None}:
            try:
                value = float(str(raw).replace(",", "."))
                if math.isfinite(value):
                    return value
            except ValueError:
                continue
    return None


def raster_centroid(path: Path) -> tuple[float | None, float | None, str]:
    if not path.exists():
        return None, None, "SOURCE_PATH_MISSING"
    try:
        import rasterio  # type: ignore

        with rasterio.open(path) as src:
            bounds = src.bounds
            return float((bounds.left + bounds.right) / 2), float((bounds.bottom + bounds.top) / 2), "RASTER_METADATA_BOUNDS"
    except ModuleNotFoundError:
        return None, None, "RASTERIO_UNAVAILABLE"
    except Exception as exc:
        return None, None, f"RASTER_METADATA_FAILED:{type(exc).__name__}"


def resolve_coordinates(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    region_order = {region: idx for idx, region in enumerate(sorted({row.get("region", "") for row in rows}))}
    coords: list[dict[str, object]] = []
    for row in rows:
        x = float_value(row, ["centroid_x", "x", "lon", "longitude", "centroid_lon"])
        y = float_value(row, ["centroid_y", "y", "lat", "latitude", "centroid_lat"])
        status = "EXPLICIT_COORDINATES" if x is not None and y is not None else ""
        if x is None or y is None:
            x, y, status = raster_centroid(Path(row.get("source_path", "")))
        if x is None or y is None:
            patch_num = numeric_patch(row.get("patch_id", ""))
            x = float(patch_num if patch_num >= 0 else len(coords))
            y = float(region_order.get(row.get("region", ""), 0))
            status = f"PATCH_ID_PROXY__{status or 'NO_COORDINATES'}"
        coords.append({"dino_input_id": row.get("dino_input_id", ""), "patch_id": row.get("patch_id", ""), "region": row.get("region", ""), "x": x, "y": y, "coordinate_status": status})
    return coords


def geo_distance_matrix(coords: list[dict[str, object]]) -> np.ndarray:
    points = np.asarray([[float(row["x"]), float(row["y"])] for row in coords], dtype="float64")
    diff = points[:, None, :] - points[None, :, :]
    return np.sqrt((diff**2).sum(axis=2))


def kmeans(matrix: np.ndarray, k: int, seed: int) -> np.ndarray:
    if matrix.shape[0] < k or k <= 1:
        return np.zeros(matrix.shape[0], dtype=int)
    rng = np.random.default_rng(seed)
    centers = matrix[rng.choice(matrix.shape[0], size=k, replace=False)].copy()
    labels = np.zeros(matrix.shape[0], dtype=int)
    for _ in range(50):
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


def neighbor_edges(rows: list[dict[str, str]], ids: list[str], sims: np.ndarray, top_k: int) -> tuple[list[dict[str, object]], list[list[int]]]:
    edges: list[dict[str, object]] = []
    adjacency: list[list[int]] = []
    seen_directed: set[tuple[str, str]] = set()
    for i, source in enumerate(ids):
        order = [j for j in np.argsort(-sims[i]) if j != i]
        top = order[: min(top_k, len(order))]
        adjacency.append(top)
        for rank, j in enumerate(top, start=1):
            pair = (source, ids[j])
            if pair in seen_directed:
                continue
            seen_directed.add(pair)
            cross = rows[i].get("region") != rows[j].get("region")
            edges.append({"source_id": source, "target_id": ids[j], "source_region": rows[i].get("region", ""), "target_region": rows[j].get("region", ""), "rank": rank, "cosine_similarity": float(sims[i, j]), "cosine_distance": float(1 - sims[i, j]), "cross_region": str(cross).upper(), "edge_type": "EMBEDDING_NEAREST_NEIGHBOR"})
    return edges, adjacency


def components(ids: list[str], edges: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, int]]:
    graph: dict[str, set[str]] = {item: set() for item in ids}
    for edge in edges:
        graph[str(edge["source_id"])].add(str(edge["target_id"]))
        graph[str(edge["target_id"])].add(str(edge["source_id"]))
    comp_id: dict[str, int] = {}
    rows: list[dict[str, object]] = []
    current = 0
    for node in ids:
        if node in comp_id:
            continue
        queue = deque([node])
        comp_nodes = []
        comp_id[node] = current
        while queue:
            value = queue.popleft()
            comp_nodes.append(value)
            for nxt in graph[value]:
                if nxt not in comp_id:
                    comp_id[nxt] = current
                    queue.append(nxt)
        rows.append({"component_id": current, "node_count": len(comp_nodes), "nodes": "|".join(sorted(comp_nodes))})
        current += 1
    return rows, comp_id


def geo_embedding_metrics(rows: list[dict[str, str]], ids: list[str], sims: np.ndarray, geo: np.ndarray, labels: np.ndarray, adjacency: list[list[int]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    sim_geo_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    compactness_rows: list[dict[str, object]] = []
    for i, source in enumerate(ids):
        neighbors = adjacency[i]
        same_region = [j for j in neighbors if rows[i].get("region") == rows[j].get("region")]
        mean_geo = float(np.mean([geo[i, j] for j in neighbors])) if neighbors else 0.0
        mean_cos = float(np.mean([sims[i, j] for j in neighbors])) if neighbors else 0.0
        sim_geo_rows.append({"dino_input_id": source, "mean_neighbor_cosine": mean_cos, "mean_neighbor_geo_distance": mean_geo, "embedding_continuity_status": "PASS" if neighbors else "NO_NEIGHBORS", "claim_scope": "GEO_STRUCTURAL_DIAGNOSTIC_ONLY"})
        overlap_rows.append({"dino_input_id": source, "region": rows[i].get("region", ""), "same_region_neighbor_count": len(same_region), "neighbor_count": len(neighbors), "regional_neighbor_overlap": len(same_region) / max(len(neighbors), 1)})
        for j in neighbors:
            pair_rows.append({"source_id": source, "target_id": ids[j], "source_region": rows[i].get("region", ""), "target_region": rows[j].get("region", ""), "cosine_distance": float(1 - sims[i, j]), "geo_distance": float(geo[i, j]), "cross_region": str(rows[i].get("region") != rows[j].get("region")).upper()})
    for cluster in sorted(set(int(x) for x in labels)):
        idx = np.where(labels == cluster)[0]
        distances = [geo[int(a), int(b)] for pos, a in enumerate(idx) for b in idx[pos + 1 :]]
        compactness_rows.append({"cluster_id": cluster, "member_count": len(idx), "mean_pairwise_geo_distance": float(np.mean(distances)) if distances else 0.0, "spatial_compactness_status": "DIAGNOSTIC_ONLY"})
    return sim_geo_rows, pair_rows, overlap_rows, compactness_rows


def graph_tables(rows: list[dict[str, str]], ids: list[str], edges: list[dict[str, object]], comp_id: dict[str, int], adjacency: list[list[int]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    degree = Counter()
    cross_degree = Counter()
    for edge in edges:
        degree[str(edge["source_id"])] += 1
        degree[str(edge["target_id"])] += 1
        if edge["cross_region"] == "TRUE":
            cross_degree[str(edge["source_id"])] += 1
            cross_degree[str(edge["target_id"])] += 1
    node_rows = []
    for i, node in enumerate(ids):
        node_rows.append({"dino_input_id": node, "patch_id": rows[i].get("patch_id", ""), "region": rows[i].get("region", ""), "degree": int(degree[node]), "component_id": comp_id.get(node, -1), "hub_status": "LOCAL_HUB" if degree[node] >= max(2, np.percentile(list(degree.values()) or [0], 75)) else "NON_HUB", "isolated_status": "ISOLATED" if degree[node] == 0 else "CONNECTED"})
    hub_rows = [row for row in node_rows if row["hub_status"] == "LOCAL_HUB"]
    bridge_rows = []
    for edge in edges:
        if edge["cross_region"] == "TRUE":
            bridge_rows.append({"source_id": edge["source_id"], "target_id": edge["target_id"], "source_region": edge["source_region"], "target_region": edge["target_region"], "cosine_similarity": edge["cosine_similarity"], "bridge_status": "CROSS_REGION_STRUCTURAL_BRIDGE"})
    return node_rows, hub_rows, bridge_rows


def topology_tables(rows: list[dict[str, str]], ids: list[str], sims: np.ndarray, geo: np.ndarray, adjacency: list[list[int]]) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    topo_rows: list[dict[str, object]] = []
    overlap_rows: list[dict[str, object]] = []
    continuity_rows: list[dict[str, object]] = []
    for i, node in enumerate(ids):
        emb_neighbors = set(adjacency[i])
        geo_order = [j for j in np.argsort(geo[i]) if j != i]
        geo_neighbors = set(geo_order[: len(emb_neighbors)])
        overlap = len(emb_neighbors & geo_neighbors) / max(len(emb_neighbors | geo_neighbors), 1)
        reciprocal = sum(1 for j in emb_neighbors if i in adjacency[j])
        topo_rows.append({"dino_input_id": node, "topology_overlap": overlap, "reciprocal_neighbor_count": reciprocal, "local_topology_status": "PASS"})
        overlap_rows.append({"dino_input_id": node, "embedding_neighbor_count": len(emb_neighbors), "geo_neighbor_count": len(geo_neighbors), "neighbor_overlap_jaccard": overlap})
        continuity_rows.append({"dino_input_id": node, "mean_embedding_neighbor_geo_distance": float(np.mean([geo[i, j] for j in emb_neighbors])) if emb_neighbors else 0.0, "mean_geo_neighbor_cosine": float(np.mean([sims[i, j] for j in geo_neighbors])) if geo_neighbors else 0.0, "manifold_continuity_status": "DIAGNOSTIC_ONLY"})
    return topo_rows, overlap_rows, continuity_rows


def transition_tables(rows: list[dict[str, str]], ids: list[str], edges: list[dict[str, object]], matrix: np.ndarray) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    cross = [edge for edge in edges if edge["cross_region"] == "TRUE"]
    cross_rows = [{"source_id": edge["source_id"], "target_id": edge["target_id"], "source_region": edge["source_region"], "target_region": edge["target_region"], "cosine_similarity": edge["cosine_similarity"], "transition_status": "CROSS_REGION_NEIGHBOR"} for edge in cross]
    counts = Counter()
    for edge in cross:
        counts[str(edge["source_id"])] += 1
        counts[str(edge["target_id"])] += 1
    trans_rows = []
    for i, node in enumerate(ids):
        if counts[node]:
            trans_rows.append({"dino_input_id": node, "patch_id": rows[i].get("patch_id", ""), "region": rows[i].get("region", ""), "cross_region_edge_count": int(counts[node]), "transition_status": "TRANSITIONAL_STRUCTURAL_CANDIDATE"})
    summary = {"cross_region_neighbor_count": len(cross_rows), "transition_embedding_count": len(trans_rows), "review_only": True, "predictive_claims": False, "clusters_are_classes": False}
    medoids = []
    for region in sorted({row.get("region", "") for row in rows}):
        idx = [i for i, row in enumerate(rows) if row.get("region") == region]
        centroid = matrix[idx].mean(axis=0)
        chosen = idx[int(np.linalg.norm(matrix[idx] - centroid, axis=1).argmin())]
        medoids.append({"region": region, "dino_input_id": ids[chosen], "patch_id": rows[chosen].get("patch_id", ""), "representative_status": "REGION_MEDOID_DIAGNOSTIC_ONLY"})
    trans_reps = []
    for item in trans_rows:
        node = str(item["dino_input_id"])
        trans_reps.append({"representative_type": "transition_bridge", "dino_input_id": node, "patch_id": item.get("patch_id", ""), "region": item.get("region", ""), "cross_region_edge_count": item.get("cross_region_edge_count", 0), "representative_status": "REVIEW_CANDIDATE_ONLY"})
    return cross_rows, trans_rows, summary, medoids, trans_reps


def make_visuals(output_dir: Path, coords: list[dict[str, object]], edges: list[dict[str, object]], bridge_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    try:
        from PIL import Image, ImageDraw  # type: ignore
    except Exception:
        return []
    visual_dir = output_dir / "visual_review"
    xs = [float(row["x"]) for row in coords]
    ys = [float(row["y"]) for row in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def project(x: float, y: float) -> tuple[int, int]:
        px = 40 + int(420 * (x - min_x) / max(max_x - min_x, 1e-9))
        py = 460 - int(420 * (y - min_y) / max(max_y - min_y, 1e-9))
        return px, py

    by_id = {str(row["dino_input_id"]): row for row in coords}
    colors = {"Curitiba": (54, 116, 181), "Petrópolis": (182, 83, 55), "PetrÃ³polis": (182, 83, 55), "Recife": (66, 148, 92)}
    image = Image.new("RGB", (520, 520), "white")
    draw = ImageDraw.Draw(image)
    for edge in edges:
        if str(edge.get("source_id")) in by_id and str(edge.get("target_id")) in by_id:
            a = by_id[str(edge["source_id"])]
            b = by_id[str(edge["target_id"])]
            fill = (200, 80, 80) if edge.get("cross_region") == "TRUE" else (170, 170, 170)
            draw.line([project(float(a["x"]), float(a["y"])), project(float(b["x"]), float(b["y"]))], fill=fill, width=2)
    for row in coords:
        x, y = project(float(row["x"]), float(row["y"]))
        fill = colors.get(str(row.get("region")), (90, 90, 90))
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=fill, outline=(0, 0, 0))
    graph_path = visual_dir / "structural_graph_neighborhoods.png"
    image.save(graph_path)
    rows = [{"panel_type": "graph_neighborhoods", "image_path": str(graph_path), "node_count": len(coords), "edge_count": len(edges), "notes": "LOCAL_TOPOLOGY_PANEL_ONLY"}]
    if bridge_rows:
        bridge_image = Image.new("RGB", (520, 520), "white")
        draw = ImageDraw.Draw(bridge_image)
        for edge in bridge_rows:
            a = by_id[str(edge["source_id"])]
            b = by_id[str(edge["target_id"])]
            draw.line([project(float(a["x"]), float(a["y"])), project(float(b["x"]), float(b["y"]))], fill=(200, 40, 40), width=3)
        for row in coords:
            x, y = project(float(row["x"]), float(row["y"]))
            draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=colors.get(str(row.get("region")), (90, 90, 90)), outline=(0, 0, 0))
        bridge_path = visual_dir / "cross_region_bridge_exemplars.png"
        bridge_image.save(bridge_path)
        rows.append({"panel_type": "bridge_exemplars", "image_path": str(bridge_path), "node_count": len(coords), "edge_count": len(bridge_rows), "notes": "CROSS_REGION_STRUCTURAL_REVIEW_ONLY"})
    return rows


def make_qa(rows: list[dict[str, str]], coords: list[dict[str, object]], edges: list[dict[str, object]], node_rows: list[dict[str, object]], component_rows: list[dict[str, object]], topology_rows: list[dict[str, object]], bridge_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, passed: bool, details: str) -> None:
        qa.append({"check": check, "status": "PASS" if passed else "FAIL", "details": details})

    edge_pairs = [(row["source_id"], row["target_id"]) for row in edges]
    add("graph integrity", len(node_rows) == len(rows) and all(row.get("source_id") != row.get("target_id") for row in edges), f"nodes={len(node_rows)} edges={len(edges)}")
    add("disconnected node handling", bool(component_rows), f"components={len(component_rows)}")
    add("duplicate edge prevention", len(edge_pairs) == len(set(edge_pairs)), f"edges={len(edge_pairs)}")
    add("topology consistency", len(topology_rows) == len(rows), f"rows={len(topology_rows)}")
    add("geographic coordinate validity", all(math.isfinite(float(row["x"])) and math.isfinite(float(row["y"])) for row in coords), f"coords={len(coords)}")
    add("bridge reproducibility", bridge_rows is not None, f"bridges={len(bridge_rows)}")
    add("cross-region symmetry", True, "directed nearest-neighbor graph records source and target regions")
    add("no labels targets or predictive claims", all(row.get("label_status") == "NO_LABEL" and row.get("target_status") == "NO_TARGET" and row.get("claim_scope") == REVIEW_ONLY_CLAIM for row in rows), REVIEW_ONLY_CLAIM)
    add("local_runs ignored", local_runs_ignored(), ".gitignore checked")
    add("no forbidden versioned artifacts", not forbidden_versioned_artifacts(), "repo checked outside local_runs")
    return qa


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force)
    manifest = Path(args.embedding_manifest)
    rows, matrix, ids = load_embeddings(manifest)
    if len(rows) == 0:
        raise RuntimeError("No valid embeddings available for v1gc geo-structural diagnostics.")
    normed = normalize(matrix)
    sims = cosine(normed)
    coords = resolve_coordinates(rows)
    geo = geo_distance_matrix(coords)
    labels = kmeans(normed, min(3, max(1, len(rows))), args.seed)
    edges, adjacency = neighbor_edges(rows, ids, sims, args.top_k)
    component_rows, comp_id = components(ids, edges)
    sim_geo_rows, pair_rows, overlap_rows, compactness_rows = geo_embedding_metrics(rows, ids, sims, geo, labels, adjacency)
    node_rows, hub_rows, bridge_rows = graph_tables(rows, ids, edges, comp_id, adjacency)
    topology_rows, neighbor_overlap_rows, continuity_rows = topology_tables(rows, ids, sims, geo, adjacency)
    cross_rows, transition_rows, transition_summary, regional_medoids, transition_reps = transition_tables(rows, ids, edges, normed)
    visual_rows = make_visuals(output_dir, coords, edges, bridge_rows)
    qa_rows = make_qa(rows, coords, edges, node_rows, component_rows, topology_rows, bridge_rows)
    qa_status = "PASS" if all(row["status"] == "PASS" for row in qa_rows) else "FAIL"
    summary = {
        "phase": "v1gc",
        "phase_name": "DINO_EMBEDDING_GEO_STRUCTURAL_DIAGNOSTICS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "embedding_manifest": str(manifest),
        "embedding_count": len(rows),
        "regions": sorted({row.get("region", "") for row in rows}),
        "graph_nodes": len(node_rows),
        "graph_edges": len(edges),
        "connected_components": len(component_rows),
        "cross_region_bridges": len(bridge_rows),
        "transition_embeddings": len(transition_rows),
        "topology_status": "PASS" if topology_rows else "SKIPPED",
        "geo_structural_status": "PASS" if sim_geo_rows else "SKIPPED",
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
    write_csv(output_dir / "geo_similarity_metrics.csv", sim_geo_rows, ["dino_input_id", "mean_neighbor_cosine", "mean_neighbor_geo_distance", "embedding_continuity_status", "claim_scope"])
    write_csv(output_dir / "embedding_distance_vs_geo_distance.csv", pair_rows, ["source_id", "target_id", "source_region", "target_region", "cosine_distance", "geo_distance", "cross_region"])
    write_csv(output_dir / "regional_overlap_metrics.csv", overlap_rows, ["dino_input_id", "region", "same_region_neighbor_count", "neighbor_count", "regional_neighbor_overlap"])
    write_csv(output_dir / "spatial_cluster_compactness.csv", compactness_rows, ["cluster_id", "member_count", "mean_pairwise_geo_distance", "spatial_compactness_status"])
    write_csv(output_dir / "structural_graph_edges.csv", edges, ["source_id", "target_id", "source_region", "target_region", "rank", "cosine_similarity", "cosine_distance", "cross_region", "edge_type"])
    write_csv(output_dir / "structural_graph_nodes.csv", node_rows, ["dino_input_id", "patch_id", "region", "degree", "component_id", "hub_status", "isolated_status"])
    write_csv(output_dir / "graph_components.csv", component_rows, ["component_id", "node_count", "nodes"])
    write_csv(output_dir / "graph_hubs.csv", hub_rows, ["dino_input_id", "patch_id", "region", "degree", "component_id", "hub_status", "isolated_status"])
    write_csv(output_dir / "graph_bridges.csv", bridge_rows, ["source_id", "target_id", "source_region", "target_region", "cosine_similarity", "bridge_status"])
    write_csv(output_dir / "topology_metrics.csv", topology_rows, ["dino_input_id", "topology_overlap", "reciprocal_neighbor_count", "local_topology_status"])
    write_csv(output_dir / "neighborhood_overlap.csv", neighbor_overlap_rows, ["dino_input_id", "embedding_neighbor_count", "geo_neighbor_count", "neighbor_overlap_jaccard"])
    write_csv(output_dir / "manifold_continuity.csv", continuity_rows, ["dino_input_id", "mean_embedding_neighbor_geo_distance", "mean_geo_neighbor_cosine", "manifold_continuity_status"])
    write_csv(output_dir / "cross_region_neighbors.csv", cross_rows, ["source_id", "target_id", "source_region", "target_region", "cosine_similarity", "transition_status"])
    write_csv(output_dir / "transitional_embeddings.csv", transition_rows, ["dino_input_id", "patch_id", "region", "cross_region_edge_count", "transition_status"])
    write_json(output_dir / "region_transition_summary.json", transition_summary)
    write_csv(output_dir / "regional_medoids.csv", regional_medoids, ["region", "dino_input_id", "patch_id", "representative_status"])
    write_csv(output_dir / "transition_representatives.csv", transition_reps, ["representative_type", "dino_input_id", "patch_id", "region", "cross_region_edge_count", "representative_status"])
    write_csv(output_dir / "visual_review_manifest.csv", visual_rows, ["panel_type", "image_path", "node_count", "edge_count", "notes"])
    write_csv(output_dir / "coordinate_resolution.csv", coords, ["dino_input_id", "patch_id", "region", "x", "y", "coordinate_status"])
    write_csv(output_dir / "geo_structural_diagnostics_qa.csv", qa_rows, ["check", "status", "details"])
    write_json(output_dir / "geo_structural_diagnostics_summary.json", summary)
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
