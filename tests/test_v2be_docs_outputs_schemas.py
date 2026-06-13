"""v2be artifact completeness tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_outputs_and_schemas_exist():
    assert len(list((ROOT / "docs").glob("v2be_*.md"))) == 3
    assert len(list((ROOT / "datasets/schemas").glob("v2be_*.schema.json"))) == 10
    for path in (ROOT / "datasets/schemas").glob("v2be_*.schema.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["type"] == "object"
    for path in (
        ROOT / "outputs_public/execution_reports/v2be_tp1_patch_boundary_integration_report.md",
        ROOT / "outputs_public/execution_reports/v2be_tp1_patch_boundary_integration_summary.json",
        ROOT / "outputs_public/logs_summary/v2be_tp1_patch_boundary_integration.txt",
    ):
        assert path.is_file()
