from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1fy_dino_embedding_corpus_analysis.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "local_runs" / "dino_embeddings" / "v1fx"
    embeddings = root / "embeddings"
    embeddings.mkdir(parents=True)
    vectors = {
        "DINO_A": np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
        "DINO_B": np.array([0.9, 0.1, 0.0, 0.0], dtype="float32"),
        "DINO_C": np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
        "DINO_D": np.array([0.0, 0.0, 1.0, 0.0], dtype="float32"),
    }
    manifest_rows = []
    metadata_rows = []
    for idx, (dino_id, vector) in enumerate(vectors.items(), start=1):
        rel = f"embeddings/{dino_id}.npz"
        np.savez_compressed(root / rel, cls_embedding=vector, patch_mean_embedding=vector)
        manifest_rows.append(
            {
                "dino_input_id": dino_id,
                "canonical_patch_id": f"PATCH_{idx}",
                "region": ["Curitiba", "Petrópolis", "Recife", "Curitiba"][idx - 1],
                "embedding_file": rel,
                "backbone": "fake",
                "device": "cpu",
                "label_status": "NO_LABEL",
                "target_status": "NO_TARGET",
                "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
                "smoke_status": "SUCCESS",
            }
        )
        metadata_rows.append({"dino_input_id": dino_id, "bands_selected": "1-3"})
    np.savez_compressed(root / "embeddings" / "CORRUPT.npz", broken=np.array([1]))
    manifest_rows.append(
        {
            "dino_input_id": "DINO_CORRUPT",
            "canonical_patch_id": "PATCH_X",
            "region": "Recife",
            "embedding_file": "embeddings/CORRUPT.npz",
            "backbone": "fake",
            "device": "cpu",
            "label_status": "NO_LABEL",
            "target_status": "NO_TARGET",
            "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
            "smoke_status": "SUCCESS",
        }
    )
    manifest = root / "manifest.csv"
    metadata = root / "metadata.csv"
    write_csv(manifest, manifest_rows, ["dino_input_id", "canonical_patch_id", "region", "embedding_file", "backbone", "device", "label_status", "target_status", "claim_scope", "smoke_status"])
    write_csv(metadata, metadata_rows, ["dino_input_id", "bands_selected"])
    return manifest, metadata, tmp_path / "local_runs" / "dino_embeddings" / "v1fy"


def test_v1fy_corpus_outputs_pca_clustering_neighbors_and_failures(tmp_path: Path) -> None:
    manifest, metadata, output_dir = build_fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "embedding-corpus-run",
            "--smoke-manifest",
            str(manifest),
            "--smoke-metadata",
            str(metadata),
            "--output-dir",
            str(output_dir),
            "--force",
            "--top-k",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    summary = json.loads((output_dir / "dino_embedding_corpus_summary_v1fy.json").read_text(encoding="utf-8"))
    assert summary["valid_embeddings"] == 4
    assert summary["failures"] == 1
    assert summary["embedding_dim"] == 4
    assert summary["qa_status"] == "PASS"

    assert read_csv(output_dir / "dino_embedding_pca_variance_v1fy.csv")
    assert read_csv(output_dir / "dino_embedding_manifold_coordinates_v1fy.csv")
    assert read_csv(output_dir / "dino_embedding_cluster_metrics_v1fy.csv")
    assert read_csv(output_dir / "dino_embedding_nearest_neighbors_v1fy.csv")
    assert read_csv(output_dir / "dino_embedding_reciprocal_pairs_v1fy.csv")
    assert read_csv(output_dir / "dino_embedding_region_diagnostics_v1fy.csv")
    failures = read_csv(output_dir / "dino_embedding_corruption_audit_v1fy.csv")
    assert failures[0]["failure_code"] == "CORRUPTED_OR_UNREADABLE_EMBEDDING"


def test_v1fy_resume_skip_existing_and_refuse_overwrite(tmp_path: Path) -> None:
    manifest, metadata, output_dir = build_fixture(tmp_path)
    first = subprocess.run([sys.executable, str(SCRIPT), "--smoke-manifest", str(manifest), "--smoke-metadata", str(metadata), "--output-dir", str(output_dir), "--force"], cwd=ROOT, check=False)
    assert first.returncode == 0
    refused = subprocess.run([sys.executable, str(SCRIPT), "--smoke-manifest", str(manifest), "--smoke-metadata", str(metadata), "--output-dir", str(output_dir)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert refused.returncode == 2
    assert "Use --force or --resume" in refused.stderr
    resumed = subprocess.run([sys.executable, str(SCRIPT), "--smoke-manifest", str(manifest), "--smoke-metadata", str(metadata), "--output-dir", str(output_dir), "--resume", "--skip-existing"], cwd=ROOT, check=False)
    assert resumed.returncode == 2
