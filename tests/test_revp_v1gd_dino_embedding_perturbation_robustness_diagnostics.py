from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1gd_dino_embedding_perturbation_robustness_diagnostics.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fixture(tmp_path: Path, empty: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "local_runs" / "dino_embeddings" / "v1fz"
    emb = root / "embeddings"
    emb.mkdir(parents=True)
    rows = []
    regions = ["Curitiba", "Curitiba", "Recife", "Recife", "Petrópolis", "Petrópolis"]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.04, 0.96, 0.0, 0.0],
            [0.45, 0.45, 0.2, 0.0],
            [0.46, 0.43, 0.22, 0.0],
        ],
        dtype="float32",
    )
    if not empty:
        for idx, region in enumerate(regions):
            dino_id = f"DINO_{idx}"
            rel = f"embeddings/{dino_id}.npz"
            np.savez_compressed(root / rel, cls_embedding=vectors[idx])
            rows.append(
                {
                    "patch_id": f"PATCH_{idx}",
                    "dino_input_id": dino_id,
                    "region": region,
                    "source_path": str(tmp_path / "private" / f"{dino_id}.tif"),
                    "embedding_path": rel,
                    "embedding_dim": "4",
                    "model_backbone": "fake",
                    "device": "cpu",
                    "success": "SUCCESS",
                    "failure_reason": "",
                    "hash": str(idx),
                    "timestamp": "now",
                    "label_status": "NO_LABEL",
                    "target_status": "NO_TARGET",
                    "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
                }
            )
    manifest = root / "manifest.csv"
    write_csv(
        manifest,
        rows,
        [
            "patch_id",
            "dino_input_id",
            "region",
            "source_path",
            "embedding_path",
            "embedding_dim",
            "model_backbone",
            "device",
            "success",
            "failure_reason",
            "hash",
            "timestamp",
            "label_status",
            "target_status",
            "claim_scope",
        ],
    )
    return manifest, tmp_path / "local_runs" / "dino_embeddings" / "v1gd"


def run_script(manifest: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--embedding-manifest",
            str(manifest),
            "--output-dir",
            str(output_dir),
            "--force",
            "--embedding-proxy-for-tests",
            "--top-k",
            "2",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v1gd_perturbation_generation_drift_graph_and_reproducibility(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path)
    result = run_script(manifest, output_dir)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((output_dir / "perturbation_robustness_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["embedding_count"] == 6
    assert len(summary["perturbation_types"]) == 6
    assert summary["drift_metrics_status"] == "PASS"
    assert summary["graph_robustness_status"] == "PASS"
    assert summary["regional_robustness_status"] == "PASS"
    assert summary["review_only"] is True
    assert summary["supervised_training"] is False
    assert summary["perturbations_for_training"] is False
    assert read_csv(output_dir / "perturbation_similarity.csv")
    assert read_csv(output_dir / "embedding_drift_metrics.csv")
    assert read_csv(output_dir / "neighbor_persistence_under_perturbation.csv")
    assert read_csv(output_dir / "graph_robustness.csv")
    first = (output_dir / "perturbation_similarity.csv").read_text(encoding="utf-8")
    second_dir = tmp_path / "local_runs" / "dino_embeddings" / "v1gd_second"
    result2 = run_script(manifest, second_dir)
    assert result2.returncode == 0
    assert first == (second_dir / "perturbation_similarity.csv").read_text(encoding="utf-8")


def test_v1gd_empty_embedding_handling(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path, empty=True)
    result = run_script(manifest, output_dir)
    assert result.returncode != 0
    assert "No valid embeddings" in result.stderr + result.stdout
