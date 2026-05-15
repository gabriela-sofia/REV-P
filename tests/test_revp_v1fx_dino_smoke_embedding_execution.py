from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1fx_dino_smoke_embedding_execution.py"
CONFIG = ROOT / "configs" / "dino_embedding_extraction.example.yaml"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    private_root = tmp_path / "PROJETO"
    image_dir = private_root / "images"
    image_dir.mkdir(parents=True)
    for name, color in [("found_1.ppm", (10, 20, 30)), ("found_2.ppm", (40, 50, 60))]:
        pixels = bytes(color) * (12 * 12)
        (image_dir / name).write_bytes(b"P6\n12 12\n255\n" + pixels)
    manifest = tmp_path / "manifest.csv"
    fields = [
        "dino_input_id",
        "canonical_patch_id",
        "region",
        "asset_path_reference",
        "encoder_mode",
        "label_status",
        "target_status",
        "claim_scope",
    ]
    write_csv(
        manifest,
        [
            {"dino_input_id": "DINO_TEST_001", "canonical_patch_id": "TEST_001", "region": "Test", "asset_path_reference": "images/found_1.ppm", "encoder_mode": "frozen_encoder", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"},
            {"dino_input_id": "DINO_TEST_002", "canonical_patch_id": "TEST_002", "region": "Test", "asset_path_reference": "images/found_2.ppm", "encoder_mode": "frozen_encoder", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"},
            {"dino_input_id": "DINO_TEST_003", "canonical_patch_id": "TEST_003", "region": "Test", "asset_path_reference": "images/missing.ppm", "encoder_mode": "frozen_encoder", "label_status": "NO_LABEL", "target_status": "NO_TARGET", "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"},
        ],
        fields,
    )
    preflight = tmp_path / "preflight.csv"
    write_csv(
        preflight,
        [
            {"dino_input_id": "DINO_TEST_001", "resolved_status": "FOUND", "resolved_path_private": str(image_dir / "found_1.ppm")},
            {"dino_input_id": "DINO_TEST_002", "resolved_status": "FOUND", "resolved_path_private": str(image_dir / "found_2.ppm")},
            {"dino_input_id": "DINO_TEST_003", "resolved_status": "MISSING", "resolved_path_private": ""},
        ],
        ["dino_input_id", "resolved_status", "resolved_path_private"],
    )
    return manifest, preflight, private_root


def test_v1fx_requires_execute(tmp_path: Path) -> None:
    manifest, preflight, _private_root = build_inputs(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-manifest",
            str(manifest),
            "--asset-preflight",
            str(preflight),
            "--output-dir",
            str(tmp_path / "local_runs" / "v1fx"),
            "--force",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 2
    assert "requires explicit --execute" in result.stderr


def test_v1fx_fake_encoder_smoke_respects_limit_and_writes_local_npz(tmp_path: Path) -> None:
    manifest, preflight, _private_root = build_inputs(tmp_path)
    output_dir = tmp_path / "local_runs" / "dino_embeddings" / "v1fx"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--execute",
            "--input-manifest",
            str(manifest),
            "--asset-preflight",
            str(preflight),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--limit",
            "1",
            "--backbone",
            "fake_smoke_encoder",
            "--allow-cpu",
            "--force",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0

    summary = json.loads((output_dir / "dino_smoke_embedding_summary_v1fx.json").read_text(encoding="utf-8"))
    assert summary["attempted_count"] == 1
    assert summary["success_count"] == 1
    assert summary["failed_count"] == 0
    assert summary["skipped_count"] == 2
    assert summary["model_loaded"] is True
    assert summary["pixel_read"] is True
    assert summary["embeddings_extracted"] is True

    rows = read_csv(output_dir / "dino_smoke_embedding_manifest_v1fx.csv")
    assert len(rows) == 1
    assert rows[0]["smoke_status"] == "SUCCESS"
    assert rows[0]["pixel_read_status"] == "READ_FOR_DINO_SMOKE_ONLY"
    assert rows[0]["embedding_status"] == "EXTRACTED_LOCAL_ONLY"
    assert rows[0]["label_status"] == "NO_LABEL"
    assert rows[0]["target_status"] == "NO_TARGET"
    assert rows[0]["claim_scope"] == "REVIEW_ONLY_NO_PREDICTIVE_CLAIM"
    npz_files = list((output_dir / "embeddings").glob("*.npz"))
    assert len(npz_files) == 1
    assert str(npz_files[0]).startswith(str(tmp_path))

    for dirname in ("data", "outputs", "docs"):
        assert not (ROOT / dirname).exists()
    assert "local_runs/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_v1fx_fake_encoder_records_metadata_and_clustering(tmp_path: Path) -> None:
    manifest, preflight, _private_root = build_inputs(tmp_path)
    output_dir = tmp_path / "local_runs" / "dino_embeddings" / "v1fx_cluster"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--execute",
            "--input-manifest",
            str(manifest),
            "--asset-preflight",
            str(preflight),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--backbone",
            "fake_smoke_encoder",
            "--allow-cpu",
            "--force",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0

    summary = json.loads((output_dir / "dino_smoke_embedding_summary_v1fx.json").read_text(encoding="utf-8"))
    assert summary["success_count"] == 2
    assert summary["embedding_dim"] == 4
    assert summary["clustering_status"] == "PASS_NUMPY_STRUCTURAL_SMOKE"
    assert summary["environment"]["modules"]["numpy"]

    metadata = read_csv(output_dir / "dino_smoke_embedding_metadata_v1fx.csv")
    assert len(metadata) == 2
    assert all(row["embedding_sha256"] for row in metadata)
    assert {row["has_nan"] for row in metadata} == {"false"}
    assert {row["has_inf"] for row in metadata} == {"false"}

    model_attempts = read_csv(output_dir / "dino_smoke_model_attempts_v1fx.csv")
    assert model_attempts[0]["status"] == "LOADED"

    clusters = read_csv(output_dir / "dino_smoke_cluster_summary_v1fx.csv")
    nn = read_csv(output_dir / "dino_smoke_nearest_neighbor_sanity_v1fx.csv")
    assert clusters
    assert len(nn) == 2
