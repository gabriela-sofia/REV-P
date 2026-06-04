"""Tests for v1uj — GeoSGB ArcGIS REST Resolver."""

import csv
import os
import subprocess
import sys

SCRIPT = os.path.join("scripts", "protocolo_c", "revp_v1uj_geosgb_rest_resolver.py")
SERVICES = os.path.join("tests", "fixtures", "v1uj", "synthetic_arcgis_services.json")
LAYERS = os.path.join("tests", "fixtures", "v1uj", "synthetic_arcgis_service_layers.json")


def _run(tmp_path, extra=None):
    out = os.path.join(tmp_path, "geosgb.csv")
    cmd = [sys.executable, SCRIPT, "--out", out] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"STDERR: {result.stderr}"
    with open(out, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestGeosgbRestResolver:
    def test_dry_run_no_web(self, tmp_path):
        rows = _run(str(tmp_path))
        assert len(rows) >= 1
        assert all(r["blocking_reason"] == "DRY_RUN" for r in rows)

    def test_fixture_lists_layers(self, tmp_path):
        rows = _run(str(tmp_path), ["--services-fixture", SERVICES,
                                    "--layers-fixture", LAYERS])
        layer_rows = [r for r in rows if r["layer_id"] != ""]
        assert layer_rows
        assert any(r["geometry_type"] == "esriGeometryPoint" for r in layer_rows)
        assert any(r["spatial_reference"] == "4326" for r in layer_rows)

    def test_susceptibility_is_contextual(self, tmp_path):
        rows = _run(str(tmp_path), ["--services-fixture", SERVICES,
                                    "--layers-fixture", LAYERS])
        ctx = [r for r in rows if r["is_contextual_layer"] == "true"]
        assert ctx, "expected a contextual (susceptibility) layer"
        assert all(r["is_observed_occurrence_candidate"] == "false" for r in ctx)

    def test_classify_layer_unit(self):
        sys.path.insert(0, os.path.abspath("."))
        from scripts.protocolo_c.revp_v1uj_geosgb_rest_resolver import classify_layer
        cfg = {
            "relevance_terms": {
                "observed_occurrence": ["ocorrencia", "deslizamento"],
                "contextual_only": ["suscetibilidade"],
            },
            "institution_terms": ["CPRM"],
        }
        obs = classify_layer("Pontos de Ocorrencia Petropolis", cfg,
                             {"PET_2022_02_15": ["Petropolis"]})
        assert obs["is_observed"] is True
        assert obs["event_id"] == "PET_2022_02_15"
        ctx = classify_layer("Suscetibilidade Movimento Massa", cfg, {})
        assert ctx["is_contextual"] is True
        assert ctx["is_observed"] is False
