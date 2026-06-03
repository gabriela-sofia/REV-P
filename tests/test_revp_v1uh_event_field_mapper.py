"""Tests for v1uh — Event Field Mapper."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uh_event_field_mapper.py")

MAPPING_COLUMNS = [
    "mapping_id", "candidate_id", "event_id", "asset_id",
    "canonical_field", "source_field", "source_field_confidence",
    "mapping_status", "ambiguity_reason", "requires_human_review",
]


def _make_inputs(tmp_path, candidates, assets):
    cand_path = os.path.join(tmp_path, "candidates.csv")
    cand_cols = ["candidate_id", "event_id", "asset_id", "candidate_class",
                 "has_geometry", "can_be_ground_reference_candidate"]
    with open(cand_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols)
        writer.writeheader()
        writer.writerows(candidates)

    asset_path = os.path.join(tmp_path, "assets.csv")
    asset_cols = ["asset_id", "columns_detected", "asset_type"]
    with open(asset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=asset_cols)
        writer.writeheader()
        writer.writerows(assets)

    return cand_path, asset_path


def _run(tmp_path, candidates, assets):
    cand_path, asset_path = _make_inputs(tmp_path, candidates, assets)
    out = os.path.join(tmp_path, "mappings.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--candidates", cand_path, "--assets", asset_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestEventFieldMapper:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [], [])
        assert len(rows) == 0

    def test_maps_standard_fields(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "TABLE_WITH_COORDINATES_CANDIDATE",
                      "has_geometry": "false",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"asset_id": "A1",
                      "columns_detected": "data_ocorrencia|tipo|bairro|latitude|longitude|fonte",
                      "asset_type": "tabular"}])
        canonical_fields = {r["canonical_field"] for r in rows}
        assert "event_date" in canonical_fields
        assert "latitude" in canonical_fields
        assert "longitude" in canonical_fields

    def test_high_confidence_for_exact_match(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "TABLE_WITH_COORDINATES_CANDIDATE",
                      "has_geometry": "false",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"asset_id": "A1",
                      "columns_detected": "latitude|longitude",
                      "asset_type": "tabular"}])
        lat_rows = [r for r in rows if r["canonical_field"] == "latitude"]
        assert lat_rows[0]["source_field_confidence"] == "HIGH"

    def test_no_mapping_without_columns(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "DOCUMENT_ONLY",
                      "has_geometry": "false",
                      "can_be_ground_reference_candidate": "false"}],
                    [{"asset_id": "A1", "columns_detected": "",
                      "asset_type": "document"}])
        assert len(rows) == 0

    def test_map_fields_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uh_event_field_mapper import map_fields
        result = map_fields(
            {"candidate_id": "C1"},
            {"columns_detected": "data|fenomeno|bairro|lat|lon"})
        canonical = {m["canonical_field"] for m in result}
        assert "event_date" in canonical
        assert "latitude" in canonical

    def test_columns_present(self, tmp_path):
        rows = _run(str(tmp_path),
                    [{"candidate_id": "C1", "event_id": "E1", "asset_id": "A1",
                      "candidate_class": "TABLE_WITH_COORDINATES_CANDIDATE",
                      "has_geometry": "false",
                      "can_be_ground_reference_candidate": "true"}],
                    [{"asset_id": "A1",
                      "columns_detected": "latitude|longitude",
                      "asset_type": "tabular"}])
        if rows:
            for col in MAPPING_COLUMNS:
                assert col in rows[0], f"Missing column: {col}"
