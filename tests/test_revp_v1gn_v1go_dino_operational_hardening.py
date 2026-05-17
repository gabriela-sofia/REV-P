from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
V1GN = ROOT / "scripts" / "dino" / "revp_v1gn_dino_execution_health_monitor.py"
V1GO = ROOT / "scripts" / "dino" / "revp_v1go_dino_pipeline_orchestrator.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fixture_manifest(tmp_path: Path, corrupted: bool = False) -> Path:
    root = tmp_path / "local_runs" / "dino_embeddings" / "v1ge"
    emb = root / "embeddings"
    emb.mkdir(parents=True)
    rows = []
    for idx in range(3):
        dino_id = f"DINO_{idx}"
        rel = f"embeddings/{dino_id}.npz"
        path = root / rel
        if corrupted and idx == 1:
            path.write_text("not an npz", encoding="utf-8")
        else:
            np.savez_compressed(path, cls_embedding=np.ones(4, dtype="float32") * (idx + 1))
        rows.append({"patch_id": f"PATCH_{idx}", "dino_input_id": dino_id, "region": "Curitiba", "embedding_path": rel, "embedding_dim": "4", "success": "SUCCESS", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"})
    manifest = root / "manifest.csv"
    write_csv(manifest, rows, ["patch_id", "dino_input_id", "region", "embedding_path", "embedding_dim", "success", "label_status", "target_status", "claim_scope"])
    return manifest


def test_v1gn_health_monitor_healthy_fixture(tmp_path: Path) -> None:
    manifest = fixture_manifest(tmp_path)
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1gn"
    result = subprocess.run([sys.executable, str(V1GN), "--embedding-manifest", str(manifest), "--output-dir", str(out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    summary = json.loads((out / "health_monitor_summary.json").read_text(encoding="utf-8"))
    assert summary["qa_status"] == "PASS"
    assert summary["operational_health_status"] in {"HEALTHY", "WARNING"}
    assert read_csv(out / "manifest_integrity.csv")


def test_v1gn_corrupted_embedding_handling(tmp_path: Path) -> None:
    manifest = fixture_manifest(tmp_path, corrupted=True)
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1gn"
    result = subprocess.run([sys.executable, str(V1GN), "--embedding-manifest", str(manifest), "--output-dir", str(out), "--force"], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode != 0
    corrupted = read_csv(out / "corrupted_embeddings.csv")
    assert corrupted
    summary = json.loads((out / "health_monitor_summary.json").read_text(encoding="utf-8"))
    assert summary["operational_health_status"] == "DEGRADED"


def test_v1go_orchestrator_dry_run_and_dependency_graph(tmp_path: Path) -> None:
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1go"
    result = subprocess.run([sys.executable, str(V1GO), "--stage", "v1fx", "--dry-run", "--output-dir", str(out)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr + result.stdout
    registry = json.loads((out / "pipeline_execution_registry.json").read_text(encoding="utf-8"))
    assert registry["multimodal_execution_enabled"] is False
    graph = read_csv(out / "pipeline_dependency_graph.csv")
    assert graph
    report = read_csv(out / "pipeline_validation_report.csv")
    assert any(row["execution_status"] == "DRY_RUN" for row in report)


def test_v1go_invalid_stage_and_validate_only(tmp_path: Path) -> None:
    out = tmp_path / "local_runs" / "dino_embeddings" / "v1go"
    invalid = subprocess.run([sys.executable, str(V1GO), "--stage", "bad_stage", "--dry-run", "--output-dir", str(out)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert invalid.returncode != 0
    valid = subprocess.run([sys.executable, str(V1GO), "--stage", "all", "--validate-only", "--output-dir", str(out)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert valid.returncode == 0, valid.stderr + valid.stdout
    rows = read_csv(out / "pipeline_validation_report.csv")
    assert rows
    assert {row["cycle_detected"] for row in rows} == {"false"}
    assert {row["multimodal_disabled"] for row in rows} == {"true"}
