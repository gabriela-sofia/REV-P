"""REV-P v1gu: Embedding structural evidence package.

Extracts structural evidence from DINO embeddings (v1fx/v1fz outputs).
When embeddings are unavailable, emits an explicit blocker document
instead of producing silent empty outputs.

Field mapping (v1fu manifest):
  canonical_patch_id  -> patch identifier (authoritative)
  region              -> Curitiba | Petrópolis | Recife
  asset_path_reference -> relative path to TIF (not to embedding)

Allowed claims: structural coherence, stability, exploratory similarity
Forbidden: prediction, classification, real-world risk, ground truth
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PHASE = "v1gu"
PHASE_NAME = "EMBEDDING_STRUCTURAL_EVIDENCE_PACKAGE"

# Primary manifest: 128 patches with canonical_patch_id + region
DEFAULT_INPUT_MANIFEST = (
    ROOT / "manifests" / "dino_inputs"
    / "revp_v1fu_dino_sentinel_input_manifest"
    / "dino_sentinel_input_manifest_v1fu.csv"
)

# Embedding search paths (v1fx smoke run, v1fz balanced corpus)
EMBEDDING_SEARCH_DIRS = [
    ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "embeddings",
    ROOT / "local_runs" / "dino_embeddings" / "v1fx" / "embeddings",
    ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "embeddings",
    ROOT / "local_runs" / "dino_embeddings" / "v1ge",
    ROOT / "local_runs" / "dino_embeddings" / "v1fx",
    ROOT / "local_runs" / "dino_embeddings" / "v1fz",
    ROOT / "local_runs" / "dino_embeddings",
]

# Corpus execution manifests: maps canonical_patch_id -> embedding_path
# Each manifest has: patch_id (= canonical_patch_id), embedding_path, success
EMBEDDING_CORPUS_MANIFESTS = [
    ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv",
    ROOT / "local_runs" / "dino_embeddings" / "v1fx" / "dino_smoke_embedding_manifest_v1fx.csv",
    ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv",
]

# NPZ key priority order (v1ge/v1fz write cls_embedding + patch_mean_embedding)
NPZ_EMBEDDING_KEYS = ["cls_embedding", "patch_mean_embedding", "embedding", "embeddings", "features", "arr_0"]

DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gu"

METHODOLOGICAL_GUARDRAILS: dict[str, Any] = {
    "review_only": True,
    "supervised_training": False,
    "labels_created": False,
    "targets_created": False,
    "predictive_claims": False,
    "clustering_is_class": False,
    "multimodal_execution_enabled": False,
    "dino_is_classifier": False,
    "gis_is_ground_truth": False,
}

ALLOWED_CLAIMS = [
    "structural coherence",
    "embedding stability",
    "exploratory similarity",
    "intra-region neighborhood rate",
    "outlier identification",
    "medoid detection",
]

FORBIDDEN_CLAIMS = [
    "vulnerability prediction",
    "flood susceptibility classification",
    "ground truth validation",
    "model performance",
    "predictive accuracy",
]

SIMILARITY_METRIC = "cosine"

# Field names in v1fu manifest
FIELD_PATCH_ID = "canonical_patch_id"
FIELD_REGION = "region"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REV-P v1gu embedding structural evidence package."
    )
    parser.add_argument("--mode", default="evidence-extraction-run",
                        choices=["evidence-extraction-run"])
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
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
        raise FileExistsError(
            f"Output directory already exists: {path}. Use --force or --resume."
        )
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
        if path.is_file() and path.suffix.lower() in {".npy", ".npz", ".tif", ".tiff"}:
            found.append(rel(path))
    return found


def find_npz_for_patch(
    patch_id: str,
    corpus_index: dict[str, Path] | None = None,
) -> Path | None:
    """Search for a .npz file for patch_id.

    Priority:
    1. Corpus manifest index (v1ge/v1fx/v1fz manifests, authoritative)
    2. Filesystem scan of EMBEDDING_SEARCH_DIRS by filename stem
    """
    if corpus_index and patch_id in corpus_index:
        return corpus_index[patch_id]
    stems_to_try = [
        patch_id,
        patch_id.lower(),
        patch_id.replace("_", "-"),
    ]
    for search_dir in EMBEDDING_SEARCH_DIRS:
        if not search_dir.exists():
            continue
        for npz in search_dir.rglob("*.npz"):
            for stem in stems_to_try:
                if stem in npz.stem:
                    return npz
    return None


def load_corpus_manifest_index() -> tuple[dict[str, Path], list[dict[str, Any]]]:
    """
    Build a corpus index from v1ge/v1fx/v1fz execution manifests.

    Each manifest CSV has: patch_id (= canonical_patch_id), embedding_path (relative),
    success (SUCCESS | FAILED | SKIPPED_EXISTING).

    Returns:
        index: canonical_patch_id -> absolute npz Path (only files that exist on disk)
        audit: per-manifest audit records for blocker document
    """
    index: dict[str, Path] = {}
    audit: list[dict[str, Any]] = []

    for manifest_path in EMBEDDING_CORPUS_MANIFESTS:
        record: dict[str, Any] = {
            "manifest": rel(manifest_path),
            "exists": manifest_path.exists(),
            "n_rows": 0,
            "n_success_rows": 0,
            "n_npz_resolved": 0,
        }
        if not manifest_path.exists():
            audit.append(record)
            continue

        rows = read_csv(manifest_path)
        record["n_rows"] = len(rows)
        output_dir = manifest_path.parent

        for row in rows:
            if row.get("success") not in ("SUCCESS", "SKIPPED_EXISTING"):
                continue
            record["n_success_rows"] += 1
            pid = (row.get("patch_id") or row.get("canonical_patch_id") or "").strip()
            emb_rel = row.get("embedding_path", "").strip()
            if not pid or not emb_rel:
                continue
            full_path = (output_dir / emb_rel).resolve()
            if full_path.exists() and full_path.suffix == ".npz":
                index[pid] = full_path
                record["n_npz_resolved"] += 1

        audit.append(record)

    return index, audit


def load_embeddings_from_manifest(
    manifest: list[dict[str, str]],
    corpus_index: dict[str, Path] | None = None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    """
    Attempt to load embeddings for each patch in the manifest.
    Returns (embeddings dict, list of patch_ids with missing embeddings).
    Uses canonical_patch_id as the authoritative patch identifier.

    Tries corpus manifest index first (v1ge/v1fx/v1fz), then filesystem scan.
    NPZ key priority: cls_embedding > patch_mean_embedding > embedding > arr_0
    """
    embeddings: dict[str, np.ndarray] = {}
    missing: list[str] = []

    for row in manifest:
        patch_id = row.get(FIELD_PATCH_ID, "").strip()
        if not patch_id:
            continue

        npz_path = find_npz_for_patch(patch_id, corpus_index)
        if npz_path is None:
            missing.append(patch_id)
            continue

        try:
            with np.load(npz_path) as data:
                arr = None
                for key in NPZ_EMBEDDING_KEYS:
                    if key in data:
                        arr = data[key]
                        break
                if arr is not None:
                    embeddings[patch_id] = arr.flatten() if arr.ndim > 1 else arr
                else:
                    missing.append(patch_id)
        except Exception:
            missing.append(patch_id)

    return embeddings, missing


def build_blocker_document(
    manifest: list[dict[str, str]],
    missing: list[str],
    output_dir: Path,
    corpus_audit: list[dict[str, Any]] | None = None,
) -> None:
    """
    Generate an explicit blocker document when embeddings are unavailable.
    Includes full audit trail of corpus manifests checked and paths searched.
    """
    patch_ids = [r.get(FIELD_PATCH_ID, "") for r in manifest if r.get(FIELD_PATCH_ID)]
    regions_count: dict[str, int] = defaultdict(int)
    for r in manifest:
        region = r.get(FIELD_REGION, "Unknown")
        if r.get(FIELD_PATCH_ID):
            regions_count[region] += 1

    # Diagnosis: which corpus manifests were found vs missing
    manifests_found = [a["manifest"] for a in (corpus_audit or []) if a["exists"]]
    manifests_missing = [a["manifest"] for a in (corpus_audit or []) if not a["exists"]]
    total_npz_resolved = sum(a.get("n_npz_resolved", 0) for a in (corpus_audit or []))

    if manifests_missing and not manifests_found:
        blocker_code = "CORPUS_MANIFESTS_NOT_FOUND_EMBEDDINGS_NOT_EXTRACTED"
    elif manifests_found and total_npz_resolved == 0:
        blocker_code = "CORPUS_MANIFESTS_EXIST_BUT_NO_NPZ_ON_DISK"
    else:
        blocker_code = "NO_NPZ_EMBEDDINGS_FOUND"

    blocker_json: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "status": "BLOCKED",
        "blocker_code": blocker_code,
        "n_patches_in_manifest": len(patch_ids),
        "n_embeddings_found": len(patch_ids) - len(missing),
        "n_embeddings_missing": len(missing),
        "patches_by_region": dict(regions_count),
        "corpus_manifest_audit": corpus_audit or [],
        "corpus_manifests_checked": [rel(m) for m in EMBEDDING_CORPUS_MANIFESTS],
        "corpus_manifests_found": manifests_found,
        "corpus_manifests_missing": manifests_missing,
        "n_npz_resolved_from_manifests": total_npz_resolved,
        "search_dirs_checked": [rel(d) for d in EMBEDDING_SEARCH_DIRS],
        "blocker_explanation": (
            "v1gu searched corpus execution manifests (v1ge, v1fx, v1fz) and "
            "EMBEDDING_SEARCH_DIRS for .npz embedding files. "
            f"Corpus manifests found: {len(manifests_found)}, "
            f"NPZ files resolved: {total_npz_resolved}. "
            "The v1fu manifest marks pixel_read_status as "
            "NOT_READ__FUTURE_DINO_ENCODING_ONLY for all 128 patches — "
            "embedding extraction (v1ge --execute) has not been run yet. "
            "Similarity matrix, neighbor analysis, centroids, and medoid "
            "detection require actual .npz embedding vectors."
        ),
        "what_is_needed_to_unblock": [
            "Execute v1ge expanded corpus: python scripts/dino/revp_v1ge_dino_expanded_sentinel_embedding_corpus.py --execute --force",
            "OR execute v1fx smoke run: python scripts/dino/revp_v1fx_dino_smoke_embedding_execution.py --execute --force",
            "Verify .npz files appear in local_runs/dino_embeddings/v1ge/embeddings/",
            "Re-run v1gu: python scripts/dino/revp_v1gu_embedding_structural_evidence_package.py --force",
        ],
        "methodological_guardrails": METHODOLOGICAL_GUARDRAILS,
        "allowed_claims_when_unblocked": ALLOWED_CLAIMS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
    }
    write_json(output_dir / "embedding_structural_evidence_blocker_v1gu.json", blocker_json)

    # Emit the patch status CSV so downstream scripts have the patch list
    rows: list[dict[str, object]] = [
        {
            "canonical_patch_id": pid,
            "region": next(
                (r[FIELD_REGION] for r in manifest if r.get(FIELD_PATCH_ID) == pid), ""
            ),
            "embedding_status": "MISSING",
            "blocker": "NO_NPZ_FILE_FOUND_IN_SEARCH_DIRS",
        }
        for pid in sorted(missing)
    ]
    write_csv(
        output_dir / "embedding_patch_status_v1gu.csv",
        rows,
        ["canonical_patch_id", "region", "embedding_status", "blocker"],
    )

    print(f"[{PHASE}] BLOCKED: {len(missing)} patches missing embeddings")
    print(f"[{PHASE}] Blocker document: {rel(output_dir / 'embedding_structural_evidence_blocker_v1gu.json')}")
    print(f"[{PHASE}] Patch status: {rel(output_dir / 'embedding_patch_status_v1gu.csv')}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_matrix(
    embeddings: dict[str, np.ndarray],
) -> tuple[dict[str, list[float]], list[str]]:
    patch_ids = sorted(embeddings.keys())
    matrix: dict[str, list[float]] = {}
    for i, pid_i in enumerate(patch_ids):
        row: list[float] = []
        for j, pid_j in enumerate(patch_ids):
            row.append(1.0 if i == j else cosine_similarity(embeddings[pid_i], embeddings[pid_j]))
        matrix[pid_i] = row
    return matrix, patch_ids


def top_k_neighbors(
    embeddings: dict[str, np.ndarray],
    patch_ids_order: list[str],
    k: int,
) -> dict[str, list[dict[str, Any]]]:
    neighbors: dict[str, list[dict[str, Any]]] = {}
    for pid in patch_ids_order:
        sims = [
            (other_pid, cosine_similarity(embeddings[pid], embeddings[other_pid]))
            for other_pid in patch_ids_order
            if other_pid != pid
        ]
        sims.sort(key=lambda x: x[1], reverse=True)
        neighbors[pid] = [
            {"patch_id": other_pid, "similarity": sim, "rank": idx + 1}
            for idx, (other_pid, sim) in enumerate(sims[:k])
        ]
    return neighbors


def intra_inter_region_rate(
    manifest: list[dict[str, str]],
    patch_ids_order: list[str],
    neighbors_dict: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    # Build region lookup from manifest
    region_map = {
        r[FIELD_PATCH_ID]: r[FIELD_REGION]
        for r in manifest
        if r.get(FIELD_PATCH_ID)
    }

    total_neighbors = 0
    intra_region = 0
    regions: dict[str, list[str]] = defaultdict(list)
    for pid in patch_ids_order:
        regions[region_map.get(pid, "Unknown")].append(pid)

    for pid in patch_ids_order:
        region = region_map.get(pid, "Unknown")
        for nb in neighbors_dict.get(pid, []):
            neighbor_region = region_map.get(nb["patch_id"], "Unknown")
            total_neighbors += 1
            if region == neighbor_region:
                intra_region += 1

    intra_rate = intra_region / total_neighbors if total_neighbors > 0 else 0.0
    return {
        "total_neighbor_edges": total_neighbors,
        "intra_region_edges": intra_region,
        "inter_region_edges": total_neighbors - intra_region,
        "intra_region_rate": intra_rate,
        "inter_region_rate": 1.0 - intra_rate,
        "regions": {k: v for k, v in regions.items()},
    }


def regional_centroids(
    embeddings: dict[str, np.ndarray],
    manifest: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    region_map = {
        r[FIELD_PATCH_ID]: r[FIELD_REGION]
        for r in manifest
        if r.get(FIELD_PATCH_ID)
    }
    by_region: dict[str, list[np.ndarray]] = defaultdict(list)
    by_region_pids: dict[str, list[str]] = defaultdict(list)
    for pid, emb in embeddings.items():
        region = region_map.get(pid, "Unknown")
        by_region[region].append(emb)
        by_region_pids[region].append(pid)

    centroids: dict[str, dict[str, Any]] = {}
    for region, embs in by_region.items():
        centroid = np.mean(embs, axis=0)
        centroids[region] = {
            "centroid_norm": float(np.linalg.norm(centroid)),
            "n_patches": len(embs),
            "patches": by_region_pids[region],
        }
    return centroids


def medoids_and_outliers(
    embeddings: dict[str, np.ndarray],
    manifest: list[dict[str, str]],
) -> dict[str, Any]:
    region_map = {
        r[FIELD_PATCH_ID]: r[FIELD_REGION]
        for r in manifest
        if r.get(FIELD_PATCH_ID)
    }
    by_region: dict[str, list[str]] = defaultdict(list)
    for pid in embeddings:
        by_region[region_map.get(pid, "Unknown")].append(pid)

    result: dict[str, Any] = {}
    for region, patches in by_region.items():
        if len(patches) == 1:
            result[region] = {
                "medoid": patches[0],
                "outliers": [],
                "info": "single patch region",
            }
            continue
        centroid = np.mean([embeddings[p] for p in patches], axis=0)
        dists = sorted([(p, float(np.linalg.norm(embeddings[p] - centroid))) for p in patches],
                       key=lambda x: x[1])
        threshold = float(np.percentile([d[1] for d in dists], 75))
        result[region] = {
            "medoid": dists[0][0],
            "n_patches": len(patches),
            "outliers": [p for p, d in dists if d > threshold],
            "medoid_distance_to_centroid": dists[0][1],
            "outlier_count": sum(1 for _, d in dists if d > threshold),
        }
    return result


def run_evidence_extraction(args: argparse.Namespace) -> int:
    print(f"[{PHASE}] Starting embedding structural evidence extraction...")

    if not is_local_runs_ignored():
        print("[!] WARNING: local_runs/ not in .gitignore")
    forbidden = forbidden_versioned_artifacts()
    if forbidden:
        print(f"[!] ERROR: forbidden artifacts: {forbidden}")
        return 1

    manifest_path = Path(args.input_manifest)
    if not manifest_path.exists():
        print(f"[!] ERROR: manifest not found: {manifest_path}")
        return 1

    manifest = read_csv(manifest_path)
    print(f"[{PHASE}] Manifest: {len(manifest)} entries ({rel(manifest_path)})")

    output_dir = Path(args.output_dir)
    prepare_output_dir(output_dir, args.force, args.resume)

    # Step 1: build corpus index from v1ge/v1fx/v1fz manifests
    corpus_index, corpus_audit = load_corpus_manifest_index()
    n_indexed = len(corpus_index)
    print(f"[{PHASE}] Corpus manifest index: {n_indexed} patches with resolved .npz")
    for a in corpus_audit:
        status = "found" if a["exists"] else "NOT FOUND"
        print(f"  {a['manifest']}: {status} | success_rows={a.get('n_success_rows', 0)} | npz_resolved={a.get('n_npz_resolved', 0)}")

    # Step 2: load embeddings using manifest index + filesystem fallback
    embeddings, missing = load_embeddings_from_manifest(manifest, corpus_index)
    print(f"[{PHASE}] Embeddings loaded: {len(embeddings)} / {len(manifest)}")
    print(f"[{PHASE}] Missing: {len(missing)}")

    if len(embeddings) < 2:
        build_blocker_document(manifest, missing, output_dir, corpus_audit)
        print(f"[{PHASE}] Output: {rel(output_dir)}")
        # Return 0: blocker is valid state, not a crash
        return 0

    # Enough embeddings — compute structural evidence
    sim_matrix, patch_ids_order = similarity_matrix(embeddings)
    neighbors = top_k_neighbors(embeddings, patch_ids_order, args.top_k)
    intra_inter = intra_inter_region_rate(manifest, patch_ids_order, neighbors)
    centroids = regional_centroids(embeddings, manifest)
    medoids = medoids_and_outliers(embeddings, manifest)

    # Export neighbors
    neighbor_rows: list[dict[str, object]] = [
        {"patch_id": pid, "neighbor_patch_id": nb["patch_id"],
         "similarity": round(nb["similarity"], 6), "rank": nb["rank"]}
        for pid, nbs in neighbors.items()
        for nb in nbs
    ]
    write_csv(
        output_dir / "embedding_neighbors_v1gu.csv",
        neighbor_rows,
        ["patch_id", "neighbor_patch_id", "similarity", "rank"],
    )

    write_json(output_dir / "embedding_similarity_matrix_v1gu.json", {
        "metric": SIMILARITY_METRIC,
        "n_patches": len(patch_ids_order),
        "patch_ids": patch_ids_order,
        "matrix": {pid: [round(v, 4) for v in vals] for pid, vals in sim_matrix.items()},
    })

    write_json(output_dir / "embedding_regional_summary_v1gu.json", {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": PHASE,
        "n_embeddings": len(embeddings),
        "corpus_manifest_audit": corpus_audit,
        "centroids": centroids,
        "medoids_and_outliers": medoids,
        "intra_inter_region_analysis": intra_inter,
        "methodological_guardrails": METHODOLOGICAL_GUARDRAILS,
        "allowed_claims": ALLOWED_CLAIMS,
        "forbidden_claims": FORBIDDEN_CLAIMS,
    })

    print(f"[{PHASE}] Evidence extraction complete — {len(embeddings)} embeddings")
    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(run_evidence_extraction(args))
