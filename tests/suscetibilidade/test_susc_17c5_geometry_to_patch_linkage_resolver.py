"""Tests for SUSC-17C5 Geometry-to-Patch Linkage Resolver."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
OUT = ROOT / "outputs_public" / "data" / "susc_17c5_geometry_to_patch_linkage_resolver"
REPORTS = ROOT / "outputs_public" / "reports"
SCHEMAS = ROOT / "schemas" / "suscetibilidade"

EXPECTED = [
    OUT / "susc_17c5_preflight_registry_inventory.csv",
    OUT / "susc_17c5_normalized_geometry.csv",
    OUT / "susc_17c5_patch_links.csv",
    OUT / "susc_17c5_patch_link_blockers.csv",
    OUT / "reference_patch_link_qa_queue.csv",
    OUT / "susc_17c5_candidate_patch_links.geojson",
    OUT / "susc_17c5_geometry_to_patch_linkage_summary.json",
    REPORTS / "SUSC_17C5_GEOMETRY_TO_PATCH_LINKAGE_RESOLVER_REPORT.md",
    SCHEMAS / "susc_17c5_geometry_resolver_schema_v1.json",
    SCHEMAS / "susc_17c5_patch_link_schema_v1.json",
    SCHEMAS / "susc_17c5_patch_link_qa_schema_v1.json",
]


def _load_common():
    path = SCRIPTS / "susc_17c5_geometry_to_patch_linkage_common.py"
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("s17c5_geometry_common", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_script(name: str):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / name)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )


def test_build_validator_e_determinismo():
    result = run_script("build_susc_17c5_geometry_to_patch_linkage_resolver.py")
    assert result.returncode == 0, result.stderr + result.stdout
    first = {path: path.read_bytes() for path in EXPECTED}
    result = run_script("build_susc_17c5_geometry_to_patch_linkage_resolver.py")
    assert result.returncode == 0, result.stderr + result.stdout
    assert {path: path.read_bytes() for path in EXPECTED} == first
    result = run_script("validate_susc_17c5_geometry_to_patch_linkage_resolver.py")
    assert result.returncode == 0, result.stderr + result.stdout


def test_geometry_classification_point_polygon_bbox():
    s17c5 = _load_common()
    patch = (0.0, 0.0, 10.0, 10.0)
    assert s17c5.classify_patch_link("point", None, (5.0, 5.0), None, patch)[0] == "point_within_patch"
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [5.0, 5.0],
            [15.0, 5.0],
            [15.0, 15.0],
            [5.0, 15.0],
            [5.0, 5.0],
        ]],
    }
    link_class, overlap, ratio, distance = s17c5.classify_patch_link("polygon", polygon, None, s17c5.bbox_of_geometry(polygon), patch)
    assert link_class == "exact_polygon_overlap"
    assert overlap > 0
    assert ratio > 0
    assert distance == 0
    assert s17c5.classify_patch_link("bbox", None, None, (5.0, 5.0, 12.0, 12.0), patch)[0] == "bbox_overlap"


def test_texto_bairro_e_endereco_ficam_unresolved():
    geometries = read_csv(OUT / "susc_17c5_normalized_geometry.csv")
    text_rows = [row for row in geometries if row["source_type"] == "official_address_resolved"]
    assert text_rows
    assert {row["geometry_type"] for row in text_rows} == {"unresolved_geometry"}
    assert {row["has_geometry_object"] for row in text_rows} == {"false"}
    assert {row["geometry_resolution_status"] for row in text_rows} == {"blocked_text_only"}


def test_same_region_only_nao_e_elegivel():
    links = read_csv(OUT / "susc_17c5_patch_links.csv")
    same_region = [row for row in links if row["link_class"] == "same_region_only"]
    assert same_region
    assert {row["eligible_for_evaluation"] for row in same_region} == {"false"}
    assert {row["strong_link_candidate"] for row in same_region} == {"false"}


def test_alert_risk_ground_truth_trainable_proibidos_sinteticos():
    s17c5 = _load_common()
    geometries = read_csv(OUT / "susc_17c5_normalized_geometry.csv")
    links = read_csv(OUT / "susc_17c5_patch_links.csv")
    qa = read_csv(OUT / "reference_patch_link_qa_queue.csv")

    bad_links = [dict(row) for row in links]
    bad_links[0]["ground_truth"] = "true"
    assert any("ground_truth_true_forbidden" in err for err in s17c5.validate_output_rows(geometries, bad_links, qa))

    bad_links = [dict(row) for row in links]
    bad_links[0]["eligible_for_training"] = "true"
    assert any("eligible_for_training_true_forbidden" in err for err in s17c5.validate_output_rows(geometries, bad_links, qa))

    bad_links = [dict(row) for row in links]
    bad_links[0]["score_v7_allowed"] = "true"
    assert any("score_v7_allowed_true_forbidden" in err for err in s17c5.validate_output_rows(geometries, bad_links, qa))

    bad_geoms = [dict(row) for row in geometries]
    bad_links = [dict(row) for row in links]
    bad_geoms[0]["source_type"] = "alert_only"
    bad_links[0]["source_type"] = "alert_only"
    bad_links[0]["strong_link_candidate"] = "true"
    assert any("alert_or_risk_promoted_to_strong" in err for err in s17c5.validate_output_rows(bad_geoms, bad_links, qa))


def test_qa_accepted_automatico_falha():
    s17c5 = _load_common()
    geometries = read_csv(OUT / "susc_17c5_normalized_geometry.csv")
    links = read_csv(OUT / "susc_17c5_patch_links.csv")
    qa = read_csv(OUT / "reference_patch_link_qa_queue.csv")
    assert qa
    bad_qa = [dict(row) for row in qa]
    bad_qa[0]["qa_status"] = "accepted"
    assert any("qa_accepted_automatic_forbidden" in err for err in s17c5.validate_output_rows(geometries, links, bad_qa))


def test_resumo_relatorio_e_guardrails():
    summary = json.loads((OUT / "susc_17c5_geometry_to_patch_linkage_summary.json").read_text(encoding="utf-8"))
    report = (REPORTS / "SUSC_17C5_GEOMETRY_TO_PATCH_LINKAGE_RESOLVER_REPORT.md").read_text(encoding="utf-8")
    assert summary["geometries_resolved_count"] >= 1
    assert summary["strong_link_candidate_count"] >= 1
    assert summary["qa_queue_count"] >= 1
    assert summary["eligible_for_training_count"] == 0
    assert summary["ground_truth_true_count"] == 0
    assert summary["score_v7_allowed_true_count"] == 0
    assert summary["eligible_for_17b_now"] is False
    assert "17B continua **bloqueado**" in report
    geojson = json.loads((OUT / "susc_17c5_candidate_patch_links.geojson").read_text(encoding="utf-8"))
    assert geojson["metadata"]["ground_truth"] is False
    assert geojson["metadata"]["eligible_for_training"] is False
