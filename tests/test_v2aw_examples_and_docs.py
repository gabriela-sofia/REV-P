"""v2aw - synthetic examples, schemas and documentation tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_four_v2aw_schemas_exist_and_describe_required_fields():
    schema_dir = ROOT / "datasets" / "schemas"
    paths = sorted(schema_dir.glob("v2aw_*.schema.json"))
    assert len(paths) == 4
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["type"] == "object"
        assert schema["required"]
        assert schema["properties"]


def test_examples_are_explicitly_synthetic_and_use_no_real_patch_ids():
    example_dir = ROOT / "datasets" / "examples" / "v2aw_geometry_intake"
    files = sorted(path for path in example_dir.iterdir() if path.is_file())
    assert len(files) == 5
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "SYNTHETIC EXAMPLE" in combined
    assert "EXAMPLE_PATCH_" in combined
    with open(ROOT / "datasets" / "v2av_patch_boundary_recovery_queue.csv",
              encoding="utf-8-sig", newline="") as handle:
        real_patch_ids = {row["patch_id"] for row in csv.DictReader(handle) if row["patch_id"]}
    assert real_patch_ids
    assert all(patch_id not in combined for patch_id in real_patch_ids)


def test_docs_state_geometry_and_guardrail_requirements():
    text = (ROOT / "docs" / "v2aw_geometry_source_intake_instructions.md").read_text(
        encoding="utf-8")
    for required in ("55", "CRS", "bbox", "WKT", "GeoJSON", "NAO invente geometria",
                     "NAO cria label", "ground truth final", "treina modelo"):
        assert required in text


def test_public_summary_matches_expected_blocked_real_state():
    summary = json.loads((ROOT / "outputs_public" / "execution_reports" /
                          "v2aw_geometry_source_intake_summary.json").read_text(encoding="utf-8"))
    assert summary["total_priority_patches"] == 55
    assert summary["patch_sources_provided"] == 0
    assert summary["patch_sources_valid"] == 0
    assert summary["event_point_anchors_seeded"] == 9
    assert summary["ready_for_v2av_count"] == 0
    assert summary["ready_for_v2au_count"] == 0
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
