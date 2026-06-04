"""Tests for v1uj — RIGeo Bitstream Resolver."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_rigeo_bitstream_resolver.py")
ITEM = os.path.join("tests", "fixtures", "v1uj", "synthetic_rigeo_item.html")


def _run(tmp_path, extra=None):
    out = os.path.join(tmp_path, "rigeo.csv")
    cmd = [sys.executable, SCRIPT, "--out", out] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestRigeoBitstreamResolver:
    def test_dry_run_no_web(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 1
        assert all(r["blocking_reason"] == "DRY_RUN" for r in rows)

    def test_fixture_extracts_bitstreams(self, tmp_path):
        rows = _run(str(tmp_path), ["--item-fixture", ITEM])
        assert rows
        assert any(r["bitstream_class"] == "technical_report" for r in rows)
        assert any(r["bitstream_class"] == "attachment_package" for r in rows)

    def test_fixture_geodata_candidate(self, tmp_path):
        rows = _run(str(tmp_path), ["--item-fixture", ITEM])
        assert any(r["is_geodata_candidate"] == "true" for r in rows)

    def test_classify_bitstream_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_rigeo_bitstream_resolver import classify_bitstream
        cfg = {"bitstream_classes": {
            "technical_report": [".pdf"],
            "attachment_package": [".zip"],
            "geodata": [".shp", ".geojson"],
        }}
        assert classify_bitstream(".pdf", cfg) == "technical_report"
        assert classify_bitstream(".zip", cfg) == "attachment_package"
        assert classify_bitstream(".geojson", cfg) == "geodata"
        assert classify_bitstream(".xyz", cfg) == "other"
