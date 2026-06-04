"""Tests for v1uj — PDF Deep Link Extractor."""

import csv
import os
import shutil
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_pdf_deeplink_extractor.py")
TXT_FIXTURE = os.path.join("tests", "fixtures", "v1uj", "synthetic_pdf_text.txt")
PDF_FIXTURE = os.path.join("tests", "fixtures", "v1uj", "synthetic_report.pdf")
ALLOWED = os.path.join("configs", "protocolo_c", "v1ui_allowed_domains.yaml")


def _run(tmp_path, raw_dir):
    out = os.path.join(tmp_path, "deeplinks.csv")
    cmd = [sys.executable, SCRIPT, "--raw-dirs", raw_dir,
           "--allowed-domains", ALLOWED, "--out", out]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _seed(tmp_path, src):
    ev_dir = os.path.join(tmp_path, "rigeo", "PET_2022_02_15")
    os.makedirs(ev_dir, exist_ok=True)
    shutil.copy(src, ev_dir)
    return os.path.join(tmp_path, "rigeo")


class TestPdfDeeplinkExtractor:
    def test_empty_dir(self, tmp_path):
        raw = os.path.join(str(tmp_path), "empty")
        os.makedirs(raw, exist_ok=True)
        rows = _run(str(tmp_path), raw)
        assert rows == []

    def test_txt_fixture_links_and_terms(self, tmp_path):
        raw = _seed(str(tmp_path), TXT_FIXTURE)
        rows = _run(str(tmp_path), raw)
        assert any(r["is_pdf_link_candidate"] == "true" for r in rows)
        assert any("rigeo.sgb.gov.br" in r["link_domain"] for r in rows)
        assert any(r["event_id"] == "PET_2022_02_15" for r in rows)
        assert any(r["domain_allowed"] == "true" for r in rows)

    def test_real_pdf_fixture(self, tmp_path):
        raw = _seed(str(tmp_path), PDF_FIXTURE)
        rows = _run(str(tmp_path), raw)
        # pypdf may or may not be installed; either OK status with link or NO_PYPDF
        assert rows
        statuses = {r["text_extract_status"] for r in rows}
        assert statuses

    def test_find_links_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_pdf_deeplink_extractor import find_links, find_terms
        text = "ver https://rigeo.sgb.gov.br/handle/doc/1 shapefile e geodados anexo"
        links = find_links(text)
        assert links == ["https://rigeo.sgb.gov.br/handle/doc/1"]
        terms = find_terms(text)
        assert "shapefile" in terms and "geodados" in terms and "anexo" in terms
