"""Tests for v1uj — S2iD / dados.gov.br Resolver."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_s2id_dadosgov_resolver.py")
CSV_FIXTURE = os.path.join("tests", "fixtures", "v1uj", "synthetic_s2id_dadosgov.csv")


def _run(tmp_path, extra=None):
    out = os.path.join(tmp_path, "s2id.csv")
    cmd = [sys.executable, SCRIPT, "--out", out] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestS2idDadosgovResolver:
    def test_dry_run_no_web(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 1
        assert any(r["blocking_reason"] == "DRY_RUN" for r in rows)

    def test_csv_fixture_coords_candidate(self, tmp_path):
        rows = _run(str(tmp_path), ["--csv-fixture", CSV_FIXTURE])
        cand = [r for r in rows if r["record_class"] == "table_with_coordinates_candidate"]
        assert cand
        assert all(r["is_geometry_of_occurrence"] == "false" for r in cand)

    def test_municipality_never_geometry(self, tmp_path):
        rows = _run(str(tmp_path), ["--csv-fixture", CSV_FIXTURE])
        assert all(r["is_geometry_of_occurrence"] == "false" for r in rows)

    def test_classify_table_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_s2id_dadosgov_resolver import classify_table
        # municipio + cobrade, sem coordenadas -> disaster_event_registry
        cl = classify_table(["municipio", "cobrade", "data_registro"],
                            [["Recife", "12100", "2022-05-25"]])
        assert cl["record_class"] in ("disaster_event_registry", "recognition_record")
        assert cl["has_coord_cols"] is False
        # coordenadas preenchidas -> candidate
        cl2 = classify_table(["municipio", "latitude", "longitude"],
                             [["Recife", "-8.05", "-34.88"]])
        assert cl2["record_class"] == "table_with_coordinates_candidate"
        assert cl2["has_coord_cols"] is True
