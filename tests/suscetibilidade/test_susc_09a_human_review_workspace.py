"""
SUSC-09A human review workspace tests.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace"
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_09a_human_review_workspace_schema_v1.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_human_review_workspace_report.md"
CASES = ROOT / "datasets" / "suscetibilidade" / "susc_08_spatiotemporal_review_cases_v1.csv"

FORBIDDEN = {"approved", "strong_candidate", "ground_truth", "training_ready"}


def _csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_workspace_artifacts_exist():
    for p in (SCHEMA, REPORT, WS / "SUSC_09A_review_index.csv",
              WS / "SUSC_09A_review_index.md", WS / "SUSC_09A_workspace_manifest.json",
              WS / "forms" / "SUSC_09A_human_review_form_prefill.csv",
              WS / "geojson" / "SUSC_09A_review_cases.geojson"):
        assert p.exists(), p


def test_dossier_and_svg_per_case():
    n = len(_csv(CASES))
    assert len(list((WS / "case_dossiers").glob("*.md"))) == n
    assert len(list((WS / "maps_svg").glob("*.svg"))) == n


def test_index_governance_and_pre_review():
    for r in _csv(WS / "SUSC_09A_review_index.csv"):
        assert r["can_be_ground_truth"] == "false"
        assert r["allowed_for_training"] == "false"
        assert r["review_only"] == "true"
        assert r["machine_pre_review"] not in FORBIDDEN
        assert r["requires_human_review"] == "true"


def test_form_ground_truth_false():
    form = _csv(WS / "forms" / "SUSC_09A_human_review_form_prefill.csv")
    assert form
    for r in form:
        assert r["approved_for_ground_truth"] == "false"


def test_geojson_layers_valid_and_not_gt():
    for layer in ("SUSC_09A_review_cases.geojson", "SUSC_09A_patch_bbox_layer.geojson",
                  "SUSC_09A_event_geometry_layer.geojson"):
        obj = json.loads((WS / "geojson" / layer).read_text(encoding="utf-8"))
        assert obj["type"] == "FeatureCollection"
        assert obj["properties"]["can_be_ground_truth"] is False


def test_report_mandatory_statement():
    assert "O SUSC-09A não executa nem substitui a revisão humana" in REPORT.read_text(encoding="utf-8")


def test_svg_is_svg():
    sample = next((WS / "maps_svg").glob("*.svg"))
    assert sample.read_text(encoding="utf-8").startswith("<svg")
