from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gn"
DEFAULT_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1ge" / "dino_expanded_embedding_manifest_v1ge.csv"
FALLBACK_MANIFEST = ROOT / "local_runs" / "dino_embeddings" / "v1fz" / "dino_balanced_embedding_manifest_v1fz.csv"
EXPECTED_UPSTREAM = ["v1fw", "v1fx", "v1fy", "v1fz", "v1ga", "v1gb", "v1gc", "v1gd", "v1ge", "v1gf", "v1gg", "v1gh", "v1gi", "v1gj", "v1gk"]
FORBIDDEN_VERSIONED_EXTENSIONS = {".npy", ".npz", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".index", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="REV-P v1gn DINO execution health monitor.")
    parser.add_argument("--embedding-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--force", action="store_true")
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


def prepare(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {path}. Use --force.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def local_runs_ignored() -> bool:
    gitignore = ROOT / ".gitignore"
    return gitignore.exists() and any(line.strip() in {"local_runs", "local_runs/"} for line in gitignore.read_text(encoding="utf-8").splitlines())


def forbidden_versioned_artifacts() -> list[str]:
    found: list[str] = []
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or "local_runs" in path.parts:
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_VERSIONED_EXTENSIONS:
            found.append(path.as_posix())
        if path.is_dir() and path.name in {"data", "outputs"}:
            found.append(path.as_posix())
    return found


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def run(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    prepare(output_dir, args.force)
    manifest_path = Path(args.embedding_manifest)
    if not manifest_path.exists() and FALLBACK_MANIFEST.exists():
        manifest_path = FALLBACK_MANIFEST
    rows = [row for row in read_csv(manifest_path) if row.get("success") in {"SUCCESS", "SKIPPED_EXISTING"}]
    missing_dependencies = upstream_dependencies(manifest_path)
    manifest_integrity, corrupted, dims, hashes = inspect_manifest(manifest_path, rows)
    duplicates = duplicate_embeddings(hashes)
    qa = make_qa(rows, missing_dependencies, corrupted, manifest_integrity, dims, duplicates)
    warning_count = sum(1 for row in qa if row["status"] == "WARN")
    fail_count = sum(1 for row in qa if row["status"] == "FAIL")
    health = "DEGRADED" if fail_count else ("WARNING" if warning_count or missing_dependencies or corrupted else "HEALTHY")
    report = [
        {"health_dimension": "embedding_availability", "status": "PASS" if rows else "FAIL", "details": f"embeddings={len(rows)}"},
        {"health_dimension": "manifest_integrity", "status": "PASS" if all(row["status"] == "PASS" for row in manifest_integrity) else "FAIL", "details": f"rows={len(manifest_integrity)}"},
        {"health_dimension": "corrupted_embeddings", "status": "PASS" if not corrupted else "FAIL", "details": f"corrupted={len(corrupted)}"},
        {"health_dimension": "embedding_dim_consistency", "status": "PASS" if len(dims) <= 1 else "FAIL", "details": "|".join(sorted(dims))},
        {"health_dimension": "regional_consistency", "status": "PASS" if len({row.get('region', '') for row in rows}) >= 1 else "FAIL", "details": json.dumps(dict(Counter(row.get('region', '') for row in rows)), ensure_ascii=False)},
        {"health_dimension": "structural_duplicates", "status": "WARN" if duplicates else "PASS", "details": f"duplicate_hashes={len(duplicates)}"},
        {"health_dimension": "upstream_outputs", "status": "WARN" if missing_dependencies else "PASS", "details": "|".join(row["dependency"] for row in missing_dependencies) if missing_dependencies else "none"},
    ]
    summary = {
        "phase": "v1gn",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "embeddings_checked": len(rows),
        "corrupted_count": len(corrupted),
        "missing_dependency_count": len(missing_dependencies),
        "duplicate_hash_count": len(duplicates),
        "embedding_dims": sorted(dims),
        "operational_health_status": health,
        "qa_status": "PASS" if fail_count == 0 else "FAIL",
        "review_only": True,
        "supervised_training": False,
        "labels_created": False,
        "targets_created": False,
        "predictive_claims": False,
        "multimodal_execution_enabled": False,
    }
    write_csv(output_dir / "execution_health_report.csv", report, ["health_dimension", "status", "details"])
    write_csv(output_dir / "missing_dependencies.csv", missing_dependencies, ["dependency", "path", "status", "details"])
    write_csv(output_dir / "corrupted_embeddings.csv", corrupted, ["dino_input_id", "embedding_path", "failure_reason"])
    write_csv(output_dir / "manifest_integrity.csv", manifest_integrity, ["dino_input_id", "region", "embedding_path", "embedding_dim", "status", "details"])
    write_csv(output_dir / "health_monitor_qa.csv", qa, ["check", "status", "details"])
    write_json(output_dir / "health_monitor_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if fail_count == 0 else 2


def upstream_dependencies(manifest_path: Path) -> list[dict[str, object]]:
    missing = []
    base = ROOT / "local_runs" / "dino_embeddings"
    for phase in EXPECTED_UPSTREAM:
        path = base / phase
        if phase == "v1gn":
            continue
        if not path.exists():
            missing.append({"dependency": phase, "path": str(path), "status": "MISSING", "details": "local output directory not present"})
    if not manifest_path.exists():
        missing.append({"dependency": "embedding_manifest", "path": str(manifest_path), "status": "MISSING", "details": "embedding manifest missing"})
    return missing


def inspect_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]], set[str], dict[str, str]]:
    base = manifest_path.parent
    integrity = []
    corrupted = []
    dims: set[str] = set()
    hashes: dict[str, str] = {}
    for row in rows:
        rel = row.get("embedding_path", "")
        path = base / rel
        dim = row.get("embedding_dim", "")
        if dim:
            dims.add(dim)
        if not rel or not path.exists():
            integrity.append({"dino_input_id": row.get("dino_input_id", ""), "region": row.get("region", ""), "embedding_path": rel, "embedding_dim": dim, "status": "FAIL", "details": "embedding file missing"})
            corrupted.append({"dino_input_id": row.get("dino_input_id", ""), "embedding_path": rel, "failure_reason": "missing file"})
            continue
        try:
            data = np.load(path)
            cls = np.asarray(data["cls_embedding"], dtype="float32")
            if cls.size == 0 or not np.isfinite(cls).all():
                raise ValueError("empty or non-finite cls_embedding")
            if dim and int(float(dim)) != int(cls.shape[0]):
                raise ValueError(f"embedding_dim mismatch manifest={dim} actual={cls.shape[0]}")
            file_hash = sha256(path)
            hashes[row.get("dino_input_id", "")] = file_hash
            integrity.append({"dino_input_id": row.get("dino_input_id", ""), "region": row.get("region", ""), "embedding_path": rel, "embedding_dim": dim, "status": "PASS", "details": f"sha256={file_hash}"})
        except Exception as exc:
            integrity.append({"dino_input_id": row.get("dino_input_id", ""), "region": row.get("region", ""), "embedding_path": rel, "embedding_dim": dim, "status": "FAIL", "details": str(exc)})
            corrupted.append({"dino_input_id": row.get("dino_input_id", ""), "embedding_path": rel, "failure_reason": str(exc)})
    return integrity, corrupted, dims, hashes


def duplicate_embeddings(hashes: dict[str, str]) -> dict[str, list[str]]:
    by_hash: dict[str, list[str]] = {}
    for dino_id, digest in hashes.items():
        by_hash.setdefault(digest, []).append(dino_id)
    return {digest: ids for digest, ids in by_hash.items() if len(ids) > 1}


def make_qa(rows: list[dict[str, str]], missing: list[dict[str, object]], corrupted: list[dict[str, object]], integrity: list[dict[str, object]], dims: set[str], duplicates: dict[str, list[str]]) -> list[dict[str, str]]:
    qa: list[dict[str, str]] = []

    def add(check: str, status: str, details: str) -> None:
        qa.append({"check": check, "status": status, "details": details})

    add("embeddings available", "PASS" if rows else "FAIL", f"rows={len(rows)}")
    add("manifest consistency", "PASS" if integrity and all(row["status"] == "PASS" for row in integrity) else "FAIL", f"rows={len(integrity)}")
    add("corrupted embedding handling", "PASS" if not corrupted else "FAIL", f"corrupted={len(corrupted)}")
    add("embedding_dim consistency", "PASS" if len(dims) <= 1 else "FAIL", f"dims={sorted(dims)}")
    add("upstream missing detection", "WARN" if missing else "PASS", f"missing={len(missing)}")
    add("duplicate structural detection", "WARN" if duplicates else "PASS", f"duplicates={len(duplicates)}")
    add("local_runs isolation", "PASS" if local_runs_ignored() else "FAIL", ".gitignore checked")
    add("forbidden artifact detection", "PASS" if not forbidden_versioned_artifacts() else "FAIL", "repo checked outside local_runs")
    return qa


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
