from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1gc_dino_embedding_geo_structural_diagnostics.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fixture(tmp_path: Path, malformed_coordinate: bool = False) -> tuple[Path, Path]:
    root = tmp_path / "local_runs" / "dino_embeddings" / "v1fz"
    emb = root / "embeddings"
    emb.mkdir(parents=True)
    rows = []
    regions = ["Curitiba", "Curitiba", "Recife", "Recife", "Petrópolis", "Petrópolis"]
    coords = [(0, 0), (1, 0), (10, 0), (11, 0), (5, 5), (6, 5)]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.95, 0.05, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.05, 0.95, 0.0, 0.0],
            [0.5, 0.5, 0.1, 0.0],
            [0.5, 0.45, 0.15, 0.0],
        ],
        dtype="float32",
    )
    for idx, region in enumerate(regions):
        dino_id = f"DINO_{idx}"
        rel = f"embeddings/{dino_id}.npz"
        np.savez_compressed(root / rel, cls_embedding=vectors[idx])
        x, y = coords[idx]
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
                "centroid_x": "bad" if malformed_coordinate and idx == 0 else str(x),
                "centroid_y": str(y),
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
            "centroid_x",
            "centroid_y",
        ],
    )
    return manifest, tmp_path / "local_runs" / "dino_embeddings" / "v1gc"


def run_script(manifest: Path, output_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--embedding-manifest", str(manifest), "--output-dir", str(output_dir), "--force", "--top-k", "2"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_v1gc_graph_topology_bridge_and_geo_outputs(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path)
    result = run_script(manifest, output_dir)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((output_dir / "geo_structural_diagnostics_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["graph_nodes"] == 6
    assert summary["graph_edges"] > 0
    assert summary["connected_components"] >= 1
    assert summary["topology_status"] == "PASS"
    assert summary["geo_structural_status"] == "PASS"
    assert summary["review_only"] is True
    assert summary["supervised_training"] is False
    assert summary["predictive_claims"] is False
    assert read_csv(output_dir / "structural_graph_edges.csv")
    assert read_csv(output_dir / "structural_graph_nodes.csv")
    assert read_csv(output_dir / "topology_metrics.csv")
    assert read_csv(output_dir / "embedding_distance_vs_geo_distance.csv")
    assert read_csv(output_dir / "regional_medoids.csv")
    assert (output_dir / "visual_review" / "structural_graph_neighborhoods.png").exists()


def test_v1gc_malformed_coordinates_fall_back_without_failure(tmp_path: Path) -> None:
    manifest, output_dir = fixture(tmp_path, malformed_coordinate=True)
    result = run_script(manifest, output_dir)
    assert result.returncode == 0, result.stderr + result.stdout
    coords = read_csv(output_dir / "coordinate_resolution.csv")
    assert coords[0]["coordinate_status"].startswith("PATCH_ID_PROXY")
    qa = read_csv(output_dir / "geo_structural_diagnostics_qa.csv")
    assert {row["status"] for row in qa} == {"PASS"}
