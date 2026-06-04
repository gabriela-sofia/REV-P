"""Tests for v1ui — Public Artifact Inventory."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1ui_public_artifact_inventory.py")


def _make_downloads(tmp_path, entries):
    path = os.path.join(tmp_path, "downloads.csv")
    cols = ["artifact_id", "event_id", "source_id", "url", "extension",
            "download_status", "sha256", "file_size_bytes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(entries)
    return path


def _run(tmp_path, entries):
    dl_path = _make_downloads(tmp_path, entries)
    out = os.path.join(tmp_path, "inventory.csv")
    result = subprocess.run(
        [sys.executable, SCRIPT, "--downloads", dl_path,
         "--raw-dir", os.path.join(tmp_path, "raw"), "--out", out],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestPublicArtifactInventory:
    def test_empty_downloads(self, tmp_path):
        rows = _run(str(tmp_path), [])
        assert len(rows) == 0

    def test_no_crash_with_missing_dir(self, tmp_path):
        rows = _run(str(tmp_path), [{
            "artifact_id": "A1", "event_id": "E1", "source_id": "S1",
            "url": "https://x.gov.br/a.csv", "extension": ".csv",
            "download_status": "DOWNLOADED", "sha256": "abc", "file_size_bytes": "100",
        }])
        assert len(rows) == 0

    def test_inventory_columns(self, tmp_path):
        dl_path = _make_downloads(str(tmp_path), [])
        out = os.path.join(str(tmp_path), "inv.csv")
        subprocess.run(
            [sys.executable, SCRIPT, "--downloads", dl_path,
             "--raw-dir", str(tmp_path), "--out", out],
            capture_output=True, text=True, timeout=60,
        )
        with open(out, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        expected = ["inventory_id", "artifact_id", "asset_type", "has_geometry",
                    "inventory_status"]
        for c in expected:
            assert c in cols

    def test_detect_terms_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ui_public_artifact_inventory import detect_terms
        ev, hz, loc = detect_terms("ocorrencias_petropolis_inundacao_2022.csv")
        assert ev is True
        assert hz is True
        assert loc is True

    def test_detect_asset_type_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1ui_public_artifact_inventory import detect_asset_type
        assert detect_asset_type(".geojson") == "geospatial_vector"
        assert detect_asset_type(".pdf") == "document"
        assert detect_asset_type(".csv") == "tabular"
        assert detect_asset_type(".png") == "static_map"
