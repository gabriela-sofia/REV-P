from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dino" / "revp_v1fw_dino_embedding_extraction_scaffold.py"
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


def test_v1fw_dry_run_writes_plan_summary_qa_schema_without_embeddings(tmp_path: Path) -> None:
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
    rows = [
        {
            "dino_input_id": "DINO_TEST_001",
            "canonical_patch_id": "TEST_001",
            "region": "Test",
            "asset_path_reference": "data/sentinel/patch_found.tif",
            "encoder_mode": "frozen_encoder",
            "label_status": "NO_LABEL",
            "target_status": "NO_TARGET",
            "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
        },
        {
            "dino_input_id": "DINO_TEST_002",
            "canonical_patch_id": "TEST_002",
            "region": "Test",
            "asset_path_reference": "data/sentinel/patch_missing.tif",
            "encoder_mode": "frozen_encoder",
            "label_status": "NO_LABEL",
            "target_status": "NO_TARGET",
            "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
        },
        {
            "dino_input_id": "DINO_TEST_003",
            "canonical_patch_id": "TEST_003",
            "region": "Test",
            "asset_path_reference": "data/sentinel/patch_ambiguous.tif",
            "encoder_mode": "frozen_encoder",
            "label_status": "NO_LABEL",
            "target_status": "NO_TARGET",
            "claim_scope": "REVIEW_ONLY_NO_PREDICTIVE_CLAIM",
        },
    ]
    input_manifest = tmp_path / "input.csv"
    write_csv(input_manifest, rows, fields)
    preflight = tmp_path / "preflight.csv"
    write_csv(
        preflight,
        [
            {"dino_input_id": "DINO_TEST_001", "resolved_status": "FOUND"},
            {"dino_input_id": "DINO_TEST_002", "resolved_status": "MISSING"},
            {"dino_input_id": "DINO_TEST_003", "resolved_status": "AMBIGUOUS"},
        ],
        ["dino_input_id", "resolved_status"],
    )
    output_dir = tmp_path / "local_runs" / "dino_embeddings" / "v1fw"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-manifest",
            str(input_manifest),
            "--asset-preflight",
            str(preflight),
            "--config",
            str(CONFIG),
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

    plan = output_dir / "dino_embedding_extraction_plan_v1fw.csv"
    summary_path = output_dir / "dino_embedding_extraction_summary_v1fw.json"
    qa_path = output_dir / "dino_embedding_extraction_qa_v1fw.csv"
    schema_path = output_dir / "dino_embedding_output_schema_v1fw.csv"
    for path in (plan, summary_path, qa_path, schema_path):
        assert path.exists(), path

    assert not (output_dir / "embeddings").exists()
    assert list(output_dir.rglob("*.npz")) == []

    plan_rows = read_csv(plan)
    assert {row["label_status"] for row in plan_rows} == {"NO_LABEL"}
    assert {row["target_status"] for row in plan_rows} == {"NO_TARGET"}
    assert {row["claim_scope"] for row in plan_rows} == {"REVIEW_ONLY_NO_PREDICTIVE_CLAIM"}
    assert {row["pixel_read_status"] for row in plan_rows} == {"NOT_READ__DRY_RUN_ONLY"}
    assert sum(1 for row in plan_rows if row["planned_status"] == "BLOCKED_BY_PREFLIGHT") == 2

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["total_inputs"] == 3
    assert summary["planned_count"] == 1
    assert summary["blocked_count"] == 2
    assert summary["pixel_read"] is False
    assert summary["embeddings_extracted"] is False
    assert summary["model_loaded"] is False

    schema_columns = {row["column"] for row in read_csv(schema_path)}
    assert {"cls_embedding", "patch_mean_embedding"}.issubset(schema_columns)
    for dirname in ("data", "outputs", "docs"):
        assert not (ROOT / dirname).exists()
    assert "local_runs/" in (ROOT / ".gitignore").read_text(encoding="utf-8")
