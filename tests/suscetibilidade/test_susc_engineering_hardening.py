"""
SUSC engineering-hardening tests — shared utility layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "suscetibilidade"
sys.path.insert(0, str(SCRIPTS))

ALIAS_REPORT = ROOT / "outputs_public" / "suscetibilidade" / "SUSC_09A_alias_resolution_report.csv"
MODULES = ["susc_io", "susc_governance", "susc_aliases", "susc_geometry",
           "susc_downloads", "susc_common"]


def test_modules_exist_and_import():
    for m in MODULES:
        assert (SCRIPTS / f"{m}.py").exists(), m
        __import__(m)


def test_governance_constants():
    import susc_governance as g
    assert g.ALLOWED_FOR_TRAINING is False
    assert g.CAN_BE_GROUND_TRUTH is False
    assert g.REVIEW_ONLY is True


def test_governance_violation_detection():
    import susc_governance as g
    assert g.row_violates_governance({"can_be_ground_truth": "true"})
    assert g.row_violates_governance({"allowed_for_training": "true"})
    assert not g.row_violates_governance(
        {"can_be_ground_truth": "false", "allowed_for_training": "false", "review_only": "true"})


def test_aliases_resolve():
    import susc_aliases as a
    assert a.resolve("flow_acc_p75") == "flow_acc_log_p75"
    assert a.resolve("rain_persistence") == "rain_persistence_index"
    assert a.resolve("s1_vv_minus_vh") == "s1_vv_minus_vh_mean_clean"
    assert ALIAS_REPORT.exists()


def test_geometry_helpers():
    import susc_geometry as geo
    assert geo.point_in_bbox(-34.95, -8.13, [-35.14, -8.23, -34.94, -8.01])
    assert geo.bbox_intersect([0, 0, 2, 2], [1, 1, 3, 3])
    assert not geo.bbox_intersect([0, 0, 1, 1], [2, 2, 3, 3])
    assert geo.region_of_coord(-43.2, -22.5) == "petropolis"
    assert geo.in_brazil(-43.2, -22.5)
    assert geo.haversine_m(-43.2, -22.5, -43.2, -22.5) == pytest.approx(0.0, abs=1e-6)


def test_downloader_blocks_raster():
    import susc_downloads as dl
    r = dl.safe_download("https://example.org/big_scene.tif", SCRIPTS / "_tmp_should_not_exist")
    assert r["download_status"] == "blocked_raster"
    assert not (SCRIPTS / "_tmp_should_not_exist").exists()


def test_downloader_refuses_non_http():
    import susc_downloads as dl
    r = dl.safe_download("s3://bucket/x.csv", SCRIPTS / "_tmp2")
    assert r["download_status"] in {"not_attempted_no_direct_url", "blocked_unknown_extension"}


def test_io_repo_containment():
    import susc_io as io
    assert io.is_within_repo(ROOT / "datasets")
    assert not io.is_within_repo("/etc/passwd")


def test_matrix_unchanged_and_no_model():
    import susc_common as common
    assert common.matrix_sha256_ok()
    assert common.find_model_artifacts() == []
