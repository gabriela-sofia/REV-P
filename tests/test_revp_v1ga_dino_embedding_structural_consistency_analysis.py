from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1ga_dino_embedding_structural_consistency_analysis.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fixture(tmp_path: Path, include_corrupt: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "local_runs" / "dino_embeddings" / "v1fz"
    emb = root / "embeddings"
    emb.mkdir(parents=True)
    vectors = [
        ("D1", "Curitiba", np.array([1.0, 0.0, 0.0], dtype="float32")),
        ("D2", "Curitiba", np.array([0.9, 0.1, 0.0], dtype="float32")),
        ("D3", "Recife", np.array([0.0, 1.0, 0.0], dtype="float32")),
        ("D4", "Recife", np.array([0.0, 0.9, 0.1], dtype="float32")),
        ("D5", "Petrópolis", np.array([0.0, 0.0, 1.0], dtype="float32")),
        ("D6", "Petrópolis", np.array([0.1, 0.0, 0.9], dtype="float32")),
    ]
    rows = []
    for dino_id, region, vector in vectors:
        rel = f"embeddings/{dino_id}.npz"
        np.savez_compressed(root / rel, cls_embedding=vector)
        rows.append({"dino_input_id": dino_id, "region": region, "embedding_path": rel, "success": "SUCCESS", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"})
    if include_corrupt:
        np.savez_compressed(root / "embeddings/BAD.npz", broken=np.array([1]))
        rows.append({"dino_input_id": "BAD", "region": "Recife", "embedding_path": "embeddings/BAD.npz", "success": "SUCCESS", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"})
    manifest = root / "manifest.csv"
    write_csv(manifest, rows, ["dino_input_id", "region", "embedding_path", "success", "label_status", "target_status", "claim_scope"])
    return manifest, tmp_path / "local_runs" / "dino_embeddings" / "v1ga"


def test_v1ga_structural_consistency_outputs(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), "--embedding-manifest", str(manifest), "--output-dir", str(output_dir), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    summary = json.loads((output_dir / "consistency_summary.json").read_text(encoding="utf-8"))
    assert summary["embedding_count"] == 6
    assert summary["qa_status"] == "PASS"
    assert summary["review_only"] is True
    assert summary["supervised_training"] is False
    assert summary["predictive_claims"] is False
    assert summary["multimodal_hold"] is True
    assert read_csv(output_dir / "centroid_distance_matrix.csv")
    assert read_csv(output_dir / "region_similarity_metrics.csv")
    assert read_csv(output_dir / "cluster_stability.csv")
    assert read_csv(output_dir / "neighbor_persistence.csv")
    assert read_csv(output_dir / "structural_outliers.csv")


def test_v1ga_empty_or_corrupt_input_fails_cleanly(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path, include_corrupt=True)
    result = subprocess.run([sys.executable, str(SCRIPT), "--embedding-manifest", str(manifest), "--output-dir", str(output_dir), "--force"], cwd=ROOT, check=False)
    assert result.returncode == 0
    summary = json.loads((output_dir / "consistency_summary.json").read_text(encoding="utf-8"))
    assert summary["embedding_count"] == 6
