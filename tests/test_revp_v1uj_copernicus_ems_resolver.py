"""Tests for v1uj — Copernicus EMS Resolver."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_copernicus_ems_resolver.py")
FIXTURE = os.path.join("tests", "fixtures", "v1uj", "synthetic_copernicus_activation.html")


def _run(tmp_path, extra=None):
    out = os.path.join(tmp_path, "ems.csv")
    cmd = [sys.executable, SCRIPT, "--out", out] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestCopernicusEmsResolver:
    def test_dry_run_no_web(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 1
        assert all(r["blocking_reason"] == "DRY_RUN" for r in rows)

    def test_fixture_parses_products(self, tmp_path):
        rows = _run(str(tmp_path), ["--html-fixture", FIXTURE])
        assert any(r["is_vector_candidate"] == "true" for r in rows)
        assert any(r["product_type"] == "map_pdf" for r in rows)

    def test_quicklook_blocked(self, tmp_path):
        rows = _run(str(tmp_path), ["--html-fixture", FIXTURE])
        ql = [r for r in rows if r["product_type"] == "quicklook"]
        assert ql, "expected a quicklook product"
        assert all(r["download_allowed"] == "false" for r in ql)
        assert all(r["blocking_reason"] == "quicklook_not_ground_truth" for r in ql)

    def test_classify_product_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_copernicus_ems_resolver import classify_product, is_vector
        ptype, ext = classify_product("EMSR564_AOI01_DEL_PRODUCT_v1.zip",
                                      "https://x/EMSR564_AOI01_DEL_PRODUCT_v1.zip")
        assert ptype == "delineation"
        assert is_vector(ptype, ext) is True
        ptype2, ext2 = classify_product("quicklook", "https://x/ql.png")
        assert ptype2 == "quicklook"
        assert is_vector(ptype2, ext2) is False
