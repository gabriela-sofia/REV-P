"""v2ax - documentation, synthetic example and schema tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_docs_exist_and_contain_guardrails():
    names = [
        "v2ax_recife_geometry_intake_workflow.md", "v2ax_manual_digitization_protocol.md",
        "v2ax_geometry_quality_checklist.md", "v2ax_operator_handoff.md",
    ]
    combined = ""
    for name in names:
        path = ROOT / "docs" / name
        assert path.is_file()
        combined += path.read_text(encoding="utf-8")
    for term in ("CRS", "ponto", "proveniencia", "licenca", "label", "ground truth"):
        assert term.lower() in combined.lower()


def test_six_schemas_are_valid_json_with_required_columns():
    paths = sorted((ROOT / "datasets" / "schemas").glob("v2ax_*.schema.json"))
    assert len(paths) == 6
    for path in paths:
        schema = json.loads(path.read_text(encoding="utf-8"))
        assert schema["type"] == "object"
        assert schema["required"]
        assert set(schema["required"]) == set(schema["properties"])


def test_examples_are_synthetic_and_include_expected_blockers():
    directory = ROOT / "datasets" / "examples" / "v2ax_recife_geometry_pack"
    files = sorted(path for path in directory.iterdir() if path.is_file())
    assert len(files) == 7
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)
    assert "SYNTHETIC_PATCH_" in combined
    assert "POINT(1 2)" in combined
    assert "UNKNOWN" in combined
    with open(ROOT / "datasets" / "v2av_patch_boundary_recovery_queue.csv",
              encoding="utf-8-sig", newline="") as handle:
        real_ids = {row["patch_id"] for row in csv.DictReader(handle)}
    assert all(patch_id not in combined for patch_id in real_ids)
