"""Tests for v1ui — Observed Geometry Extractor."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_observed_geometry_extractor.py")


def _make_inventory(tmp_path, items):
    path = os.path.join(tmp_path, "inventory.csv")
    cols = ["inventory_id", "artifact_id", "event_id", "source_id",
            "asset_type", "extension", "has_geometry", "geometry_type",
            "crs", "feature_count", "columns_detected", "internal_path",
            "event_term_detected", "hazard_term_detected", "locality_term_detected",
            "inventory_status"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for item in items:
            row = {c: "" for c in cols}
            row.update(item)
            writer.writerow(row)
    return path


def _run(tmp_path, items):
    inv_path = _make_inventory(tmp_path, items)
    out = os.path.join(tmp_path, "extractions.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--inventory", inv_path, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestObservedGeometryExtractor:
    def test_empty_input(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_geojson_becomes_candidate(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "inventory_id": "I1", "artifact_id": "A1", "event_id": "E1",
            "asset_type": "geospatial_vector", "has_geometry": "true",
            "extension": ".geojson", "internal_path": "ocorrencias.geojson",
            "event_term_detected": "true",
        }])
        assert rows[0]["geometry_candidate_class"] == "OBSERVED_OCCURRENCE_POLYGONS_CANDIDATE"
        assert rows[0]["can_be_observed_geometry_candidate"] == "true"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_csv_with_coords_becomes_points_candidate(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "inventory_id": "I2", "artifact_id": "A2", "event_id": "E1",
            "asset_type": "tabular", "has_geometry": "false",
            "columns_detected": "data|tipo|bairro|latitude|longitude",
            "internal_path": "ocorrencias.csv", "event_term_detected": "true",
        }])
        assert rows[0]["geometry_candidate_class"] == "OBSERVED_OCCURRENCE_POINTS_CANDIDATE"
        assert rows[0]["can_be_observed_geometry_candidate"] == "true"
        assert rows[0]["can_create_ground_reference"] == "false"

    def test_susceptibility_not_observed(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "inventory_id": "I3", "artifact_id": "A3", "event_id": "E1",
            "asset_type": "tabular", "has_geometry": "false",
            "columns_detected": "id|classe_suscetibilidade|area",
            "internal_path": "suscetibilidade_cprm.csv",
        }])
        assert rows[0]["geometry_candidate_class"] == "SUSCEPTIBILITY_CONTEXT"
        assert rows[0]["can_be_observed_geometry_candidate"] == "false"
        assert rows[0]["is_modeled_product"] == "true"

    def test_pdf_document_only(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "inventory_id": "I4", "artifact_id": "A4", "event_id": "E1",
            "asset_type": "document", "has_geometry": "false",
            "internal_path": "relatorio.pdf",
        }])
        assert rows[0]["geometry_candidate_class"] == "DOCUMENT_ONLY"
        assert rows[0]["can_be_observed_geometry_candidate"] == "false"

    def test_static_map_not_geometry(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "inventory_id": "I5", "artifact_id": "A5", "event_id": "E1",
            "asset_type": "static_map", "has_geometry": "false",
            "internal_path": "mapa.png",
        }])
        assert rows[0]["geometry_candidate_class"] == "STATIC_MAP_ONLY"

    def test_no_ground_reference_ever(self, tmp_path):
        rows = _run(str(tmp_path), [
            {"inventory_id": "I1", "artifact_id": "A1", "event_id": "E1",
             "asset_type": "geospatial_vector", "has_geometry": "true",
             "internal_path": "x.geojson"},
            {"inventory_id": "I2", "artifact_id": "A2", "event_id": "E1",
             "asset_type": "tabular", "has_geometry": "false",
             "columns_detected": "lat|lon|data", "internal_path": "y.csv"},
        ])
        for r in rows:
            assert r["can_create_ground_reference"] == "false"
            assert r["can_create_training_label"] == "false"
