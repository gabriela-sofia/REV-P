"""Tests for v1uf — INMET ZIP Selective Extractor."""

import csv
import hashlib
import os
import sys
import zipfile

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uf_inmet_zip_selective_extractor.py")
MANIFEST = os.path.join("datasets", "protocolo_c", "v1uf_large_download_manifest.csv")
BINDING = os.path.join("configs", "protocolo_c", "v1uf_station_target_binding.yaml")
STATIONS = os.path.join("datasets", "protocolo_c", "v1ue_station_candidate_registry.csv")

ASSET_COLUMNS = [
    "asset_id", "event_id", "source_id", "year", "station_candidate_id",
    "station_code", "station_name", "zip_sha256", "internal_zip_path",
    "extracted_local_path_hash", "file_sha256", "file_size_bytes",
    "extraction_status", "reason", "has_datetime_column",
    "has_precipitation_column", "has_temperature_column",
    "has_quality_flags", "notes",
]


def _import(name):
    sys.path.insert(0, os.path.abspath("."))
    import importlib
    return importlib.import_module(f"scripts.protocolo_c.{name}")


class TestSelectiveExtractor:
    def test_runs_and_required_columns(self, tmp_path):
        out = str(tmp_path / "out")
        import subprocess
        result = subprocess.run(
            [sys.executable, SCRIPT, "--manifest", MANIFEST, "--binding", BINDING,
             "--stations", STATIONS, "--out-dir", out, "--local-only-dir", "local_only/protocolo_c"],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        reg = os.path.join(out, "v1uf_station_series_asset_registry.csv")
        assert os.path.exists(reg)
        with open(reg, "r", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames
        for col in ASSET_COLUMNS:
            assert col in cols

    def test_matches_station_function(self):
        mod = _import("revp_v1uf_inmet_zip_selective_extractor")
        # Should match by code
        assert mod.matches_station(
            "2022/INMET_SE_RJ_A610_PICO DO COUTO_01-01-2022_A_31-12-2022.CSV",
            ["A610"], ["PETROPOLIS"], "RJ", "2022") is True
        # Should NOT match a different station
        assert mod.matches_station(
            "2022/INMET_SE_RJ_A621_OTHER_01-01-2022.CSV",
            ["A610"], ["PETROPOLIS"], "RJ", "2022") is False
        # Should NOT match wrong year
        assert mod.matches_station(
            "2021/INMET_SE_RJ_A610_PICO DO COUTO_01-01-2021.CSV",
            ["A610"], ["PETROPOLIS"], "RJ", "2022") is False

    def test_selective_not_extract_all(self, tmp_path):
        """ZIP with many stations should extract only the matching one."""
        mod = _import("revp_v1uf_inmet_zip_selective_extractor")
        zpath = tmp_path / "year.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("2022/INMET_SE_RJ_A610_PICO DO COUTO_01-01-2022.CSV", "Data;Hora;PRECIPITACAO\n2022/01/01;0000;1,0\n")
            zf.writestr("2022/INMET_SE_SP_A701_SAOPAULO_01-01-2022.CSV", "x")
            zf.writestr("2022/INMET_NE_BA_A402_SALVADOR_01-01-2022.CSV", "x")
        with zipfile.ZipFile(zpath, "r") as zf:
            names = zf.namelist()
            matched = [n for n in names if mod.matches_station(n, ["A610"], ["PICO"], "RJ", "2022")]
        assert len(matched) == 1
        assert "A610" in matched[0]

    def test_extracted_file_gets_hash(self, tmp_path):
        mod = _import("revp_v1uf_inmet_zip_selective_extractor")
        zpath = tmp_path / "z.zip"
        content = "Data;Hora UTC;PRECIPITACAO TOTAL (mm)\n2022/01/01;0000 UTC;1,4\n"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("2022/INMET_SE_RJ_A610_TEST_2022.CSV", content)
        staging = str(tmp_path / "staging")
        with zipfile.ZipFile(zpath, "r") as zf:
            extracted = mod.safe_extract_member(zf, "2022/INMET_SE_RJ_A610_TEST_2022.CSV", staging)
        assert os.path.exists(extracted)
        h = mod.sha256_file(extracted)
        assert h == hashlib.sha256(content.encode("utf-8")).hexdigest()

    def test_path_traversal_blocked(self, tmp_path):
        """Path traversal entries must return empty (no extraction outside staging)."""
        mod = _import("revp_v1uf_inmet_zip_selective_extractor")
        zpath = tmp_path / "evil.zip"
        with zipfile.ZipFile(zpath, "w") as zf:
            zf.writestr("safe.CSV", "ok")
        staging = str(tmp_path / "staging")
        with zipfile.ZipFile(zpath, "r") as zf:
            safe_result = mod.safe_extract_member(zf, "safe.CSV", staging)
            unsafe_result = mod.safe_extract_member(zf, "../escape.CSV", staging)
        assert os.path.exists(safe_result)
        assert unsafe_result == ""
