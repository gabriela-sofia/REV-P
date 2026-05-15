from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1fv_dino_local_asset_preflight.py"


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_v1fv_resolves_found_missing_and_ambiguous_without_repo_outputs(tmp_path: Path) -> None:
    private_root = tmp_path / "PROJETO"
    (private_root / "data" / "sentinel").mkdir(parents=True)
    (private_root / "other_a").mkdir()
    (private_root / "other_b").mkdir()
    (private_root / "data" / "sentinel" / "patch_found.tif").write_text("dummy", encoding="utf-8")
    (private_root / "other_a" / "patch_ambiguous.tif").write_text("dummy", encoding="utf-8")
    (private_root / "other_b" / "patch_ambiguous.tif").write_text("dummy", encoding="utf-8")

    fields = [
        "dino_input_id",
        "canonical_patch_id",
        "region",
        "source_asset_id",
        "source_manifest",
        "asset_path_reference",
        "modality",
        "eligibility_status",
    ]
    input_manifest = tmp_path / "input.csv"
    write_csv(
        input_manifest,
        [
            {
                "dino_input_id": "DINO_TEST_001",
                "canonical_patch_id": "TEST_001",
                "region": "Test",
                "source_asset_id": "asset_1",
                "source_manifest": "test",
                "asset_path_reference": "data/sentinel/patch_found.tif",
                "modality": "sentinel_raster_path_only",
                "eligibility_status": "READY_SENTINEL_FIRST_REVIEW_ONLY",
            },
            {
                "dino_input_id": "DINO_TEST_002",
                "canonical_patch_id": "TEST_002",
                "region": "Test",
                "source_asset_id": "asset_2",
                "source_manifest": "test",
                "asset_path_reference": "data/sentinel/patch_missing.tif",
                "modality": "sentinel_raster_path_only",
                "eligibility_status": "READY_SENTINEL_FIRST_REVIEW_ONLY",
            },
            {
                "dino_input_id": "DINO_TEST_003",
                "canonical_patch_id": "TEST_003",
                "region": "Test",
                "source_asset_id": "asset_3",
                "source_manifest": "test",
                "asset_path_reference": "data/sentinel/patch_ambiguous.tif",
                "modality": "sentinel_raster_path_only",
                "eligibility_status": "READY_SENTINEL_FIRST_REVIEW_ONLY",
            },
        ],
        fields,
    )
    output_dir = tmp_path / "local_runs" / "dino_asset_preflight" / "v1fv"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--private-project-root",
            str(private_root),
            "--input-manifest",
            str(input_manifest),
            "--output-dir",
            str(output_dir),
            "--force",
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2

    csv_path = output_dir / "dino_local_asset_preflight_v1fv.csv"
    summary_path = output_dir / "dino_local_asset_preflight_summary_v1fv.json"
    qa_path = output_dir / "dino_local_asset_preflight_qa_v1fv.csv"
    assert csv_path.exists()
    assert summary_path.exists()
    assert qa_path.exists()

    rows = read_csv(csv_path)
    statuses = {row["dino_input_id"]: row["resolved_status"] for row in rows}
    assert statuses == {
        "DINO_TEST_001": "FOUND",
        "DINO_TEST_002": "MISSING",
        "DINO_TEST_003": "AMBIGUOUS",
    }
    assert {row["pixel_read_status"] for row in rows} == {"NOT_READ__PREFLIGHT_ONLY"}
    assert {row["embedding_status"] for row in rows} == {"NOT_EXTRACTED"}
    assert rows[0]["future_pixel_read_allowed"] == "YES"
    assert rows[0]["resolved_path_private"].endswith("patch_found.tif")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_inputs"] == 3
    assert summary["found_count"] == 1
    assert summary["missing_count"] == 1
    assert summary["ambiguous_count"] == 1
    assert summary["pixel_read"] is False
    assert summary["embeddings_extracted"] is False
    assert summary["ready_for_v1fw"] is True

    for dirname in ("data", "outputs"):
        assert not (ROOT / dirname).exists()
    assert "local_runs/" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_v1fv_refuses_existing_output_without_force(tmp_path: Path) -> None:
    private_root = tmp_path / "PROJETO"
    private_root.mkdir()
    input_manifest = tmp_path / "input.csv"
    write_csv(
        input_manifest,
        [
            {
                "dino_input_id": "DINO_TEST_001",
                "canonical_patch_id": "TEST_001",
                "region": "Test",
                "source_asset_id": "asset_1",
                "source_manifest": "test",
                "asset_path_reference": "missing.tif",
                "modality": "sentinel_raster_path_only",
                "eligibility_status": "READY_SENTINEL_FIRST_REVIEW_ONLY",
            }
        ],
        [
            "dino_input_id",
            "canonical_patch_id",
            "region",
            "source_asset_id",
            "source_manifest",
            "asset_path_reference",
            "modality",
            "eligibility_status",
        ],
    )
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--private-project-root",
            str(private_root),
            "--input-manifest",
            str(input_manifest),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Re-run with --force" in result.stderr
