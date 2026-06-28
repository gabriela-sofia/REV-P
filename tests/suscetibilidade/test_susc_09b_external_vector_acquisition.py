"""
SUSC-09B external vector acquisition tests.
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schemas" / "suscetibilidade" / "susc_09b_external_vector_acquisition_schema_v1.json"
REGISTRY = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_source_registry_v1.csv"
DL_MANIFEST = ROOT / "manifests" / "suscetibilidade" / "susc_09b_external_download_manifest_v1.csv"
PARSED = ROOT / "datasets" / "suscetibilidade" / "susc_09b_external_geometries_parsed_v1.csv"
CANDIDATES = ROOT / "datasets" / "suscetibilidade" / "susc_09b_event_geometry_candidates_v1.csv"
OVERLAY = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_overlay_candidates.csv"
LIMITS = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_limitations.json"
REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09B_external_vector_acquisition_report.md"
DL_DIR = ROOT / "datasets" / "suscetibilidade" / "external_event_geometry_sources_susc09b"

RASTER_EXTS = {".tif", ".tiff", ".geotiff", ".jp2", ".img", ".nc", ".hdf", ".grib"}


def _csv(p):
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_artifacts_exist():
    for p in (SCHEMA, REGISTRY, DL_MANIFEST, PARSED, CANDIDATES, OVERLAY, LIMITS, REPORT):
        assert p.exists(), p


def test_registry_official_sources():
    reg = _csv(REGISTRY)
    assert reg
    insts = {r["target_institution"] for r in reg}
    assert any("GeoCuritiba" in i for i in insts)
    assert any("APAC" in i for i in insts)
    assert any("CPRM" in i for i in insts)


def test_download_manifest_has_url_and_status():
    for r in _csv(DL_MANIFEST):
        assert r["url_or_local_path"]
        assert r["download_status"]
        if r["download_status"] == "downloaded":
            assert r["sha256"]


def test_no_raster_downloaded():
    for r in _csv(DL_MANIFEST):
        if r["download_status"] == "downloaded":
            assert r["content_ext"].lower() not in RASTER_EXTS


def test_no_raster_in_download_dir():
    if DL_DIR.exists():
        assert not [p for p in DL_DIR.rglob("*") if p.suffix.lower() in RASTER_EXTS]


def test_governance_preserved():
    for r in _csv(PARSED) + _csv(CANDIDATES):
        assert r["can_be_ground_truth"] == "false"
        assert r["allowed_for_training"] == "false"
        assert r["review_only"] == "true"
    for r in _csv(OVERLAY):
        assert r["can_be_used_as_ground_truth"] == "false"
        assert r["allowed_for_training"] == "false"


def test_candidate_linkable_requires_geometry():
    for r in _csv(CANDIDATES):
        if r["can_link_to_patch"] == "true":
            assert r["bbox"]
