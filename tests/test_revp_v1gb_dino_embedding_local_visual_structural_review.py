from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1gb_dino_embedding_local_visual_structural_review.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def ppm(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"P6\n10 10\n255\n" + bytes(color) * 100)


def fixture(tmp_path: Path, missing_one: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "local_runs" / "dino_embeddings" / "v1fz"
    emb = root / "embeddings"
    src = tmp_path / "PROJETO"
    emb.mkdir(parents=True)
    rows = []
    regions = ["Curitiba", "Curitiba", "Recife", "Recife", "Petrópolis", "Petrópolis"]
    vectors = np.eye(6, dtype="float32")[:, :4]
    for idx, region in enumerate(regions):
        dino_id = f"DINO_{idx}"
        image = src / region / f"{dino_id}.ppm"
        if not (missing_one and idx == 0):
            ppm(image, (20 + idx, 40 + idx, 60 + idx))
        rel = f"embeddings/{dino_id}.npz"
        np.savez_compressed(root / rel, cls_embedding=vectors[idx])
        rows.append({"patch_id": f"PATCH_{idx}", "dino_input_id": dino_id, "region": region, "source_path": str(image), "embedding_path": rel, "embedding_dim": "4", "model_backbone": "fake", "device": "cpu", "success": "SUCCESS", "failure_reason": "", "hash": str(idx), "timestamp": "now", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"})
    manifest = root / "manifest.csv"
    write_csv(manifest, rows, ["patch_id", "dino_input_id", "region", "source_path", "embedding_path", "embedding_dim", "model_backbone", "device", "success", "failure_reason", "hash", "timestamp", "label_status", "target_status", "claim_scope"])
    return manifest, tmp_path / "local_runs" / "dino_embeddings" / "v1gb"


def test_v1gb_visual_manifest_medoids_spatial_and_multiscale(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path)
    result = subprocess.run([sys.executable, str(SCRIPT), "--embedding-manifest", str(manifest), "--output-dir", str(output_dir), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    summary = json.loads((output_dir / "visual_structural_review_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["visual_panels"] > 0
    assert summary["medoids"] > 0
    assert summary["spatial_consistency_status"] == "PASS"
    assert summary["multiscale_status"] == "PASS"
    assert summary["review_only"] is True
    assert summary["supervised_training"] is False
    assert summary["predictive_claims"] is False
    assert read_csv(output_dir / "visual_review_manifest.csv")
    assert read_csv(output_dir / "cluster_medoids.csv")
    assert read_csv(output_dir / "outlier_taxonomy.csv")
    assert read_csv(output_dir / "spatial_similarity_metrics.csv")
    assert read_csv(output_dir / "multiscale_similarity.csv")
    assert list((output_dir / "visual_review").glob("*.png"))


def test_v1gb_missing_image_handling(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path, missing_one=True)
    result = subprocess.run([sys.executable, str(SCRIPT), "--embedding-manifest", str(manifest), "--output-dir", str(output_dir), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0
    missing = read_csv(output_dir / "missing_visual_sources.csv")
    assert missing
