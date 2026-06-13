"""v2bg artifact completeness tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_docs_outputs_schemas_and_digitization_package_exist():
    assert len(list((ROOT / "docs").glob("v2bg_*.md"))) == 4
    assert len(list((ROOT / "datasets/schemas").glob("v2bg_*.schema.json"))) == 14
    for path in (ROOT / "datasets/schemas").glob("v2bg_*.schema.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["type"] == "object"
    base = ROOT / "datasets/external_sources/recife_minimal_tp/event_polygon_REC_2022_05_24_30/charter758"
    assert (base / "README.md").is_file()
    assert len(list((base / "digitization").iterdir())) == 3
