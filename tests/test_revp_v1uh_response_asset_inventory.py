"""Tests for v1uh — Response Asset Inventory."""

import csv
import os
import shutil
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uh_response_asset_inventory.py")
FIXTURES = os.path.join("tests", "fixtures", "v1uh")

ASSET_COLUMNS = [
    "asset_id", "response_id", "event_id", "institution",
    "container_type", "internal_path", "asset_type", "extension",
    "file_size_bytes", "sha256", "has_geometry", "geometry_type",
    "crs", "feature_count", "has_prj", "has_attribute_table",
    "columns_detected", "pdf_pages", "text_extract_status",
    "inventory_status", "notes",
]


def _setup_intake(tmp_path, files):
    """Create a fake response registry + staging with given files."""
    staging = os.path.join(tmp_path, "staging")
    os.makedirs(staging, exist_ok=True)
    responses = []
    for i, src in enumerate(files):
        fname = os.path.basename(src)
        shutil.copy2(src, staging)
        responses.append({
            "response_id": f"RESP_v1uh_{i:04d}",
            "intake_status": "ACCEPTED",
            "original_filename": fname,
            "event_id": "", "institution": "",
        })

    reg_path = os.path.join(tmp_path, "v1uh_formal_response_registry.csv")
    with open(reg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "response_id", "intake_status", "original_filename",
            "event_id", "institution"])
        writer.writeheader()
        writer.writerows(responses)
    return reg_path, staging


def _run(tmp_path, files):
    reg_path, staging = _setup_intake(tmp_path, files)
    out = os.path.join(tmp_path, "v1uh_response_asset_inventory.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--responses", reg_path, "--staging", staging, "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    return out


class TestResponseAssetInventory:
    def test_empty_responses(self, tmp_path):
        reg = os.path.join(str(tmp_path), "empty.csv")
        with open(reg, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["response_id", "intake_status", "original_filename"])
            writer.writeheader()
        out = os.path.join(str(tmp_path), "out.csv")
        result = subprocess.run(
            [sys.executable, SCRIPT, "--responses", reg,
             "--staging", str(tmp_path), "--out", out],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_csv_inventoried(self, tmp_path):
        out = _run(str(tmp_path),
                   [os.path.join(FIXTURES, "synthetic_occurrences.csv")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["asset_type"] == "tabular"
        assert rows[0]["inventory_status"] == "INVENTORIED"

    def test_csv_columns_detected(self, tmp_path):
        out = _run(str(tmp_path),
                   [os.path.join(FIXTURES, "synthetic_occurrences.csv")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        cols = rows[0].get("columns_detected", "")
        assert "data_ocorrencia" in cols
        assert "latitude" in cols

    def test_zip_inventoried_without_extraction(self, tmp_path):
        out = _run(str(tmp_path),
                   [os.path.join(FIXTURES, "synthetic_package.zip")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 2
        types = {r["asset_type"] for r in rows}
        assert "tabular" in types or "document" in types

    def test_geojson_has_geometry_flag(self, tmp_path):
        out = _run(str(tmp_path),
                   [os.path.join(FIXTURES, "synthetic_geometry.geojson")])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["has_geometry"] == "true"
        assert rows[0]["asset_type"] == "geospatial_vector"

    def test_columns_present(self, tmp_path):
        out = _run(str(tmp_path),
                   [os.path.join(FIXTURES, "synthetic_occurrences.csv")])
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in ASSET_COLUMNS:
            assert col in cols

    def test_asset_ids_unique(self, tmp_path):
        out = _run(str(tmp_path), [
            os.path.join(FIXTURES, "synthetic_occurrences.csv"),
            os.path.join(FIXTURES, "synthetic_geometry.geojson"),
        ])
        with open(out, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        ids = [r["asset_id"] for r in rows]
        assert len(ids) == len(set(ids))
