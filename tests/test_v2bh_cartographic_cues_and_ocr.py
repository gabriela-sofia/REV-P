"""v2bh cartographic cue and OCR audit tests."""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    with (ROOT / "datasets" / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cues_are_inventoried_and_ocr_is_optional():
    cues = read("v2bh_cartographic_cue_registry.csv")
    assert {"coordinate_text", "grid_tick", "map_frame", "scar_symbol_legend", "north_arrow", "scale_bar"} <= {r["cue_type"] for r in cues}
    assert any(r["cue_text_or_description"] == "Projected coordinate system WGS 84 / UTM ZONE 25S" for r in cues)
    ocr = read("v2bh_ocr_text_extraction_audit.csv")[0]
    assert ocr["ocr_available"] in {"true", "false"}
    assert ocr["coordinate_like_tokens"]
