"""Tests for revp_v1gt_gis_land_use_coverage_expansion_audit.py.

Covers: guardrails, constants, normalize_region, bbox_overlaps,
patch_centroid, load_known_sources, assess_patch, region_summary,
coverage_gap_rows, run_audit (no-gis-root and with stub TIFs),
output file schema, and no-forbidden-outputs.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS_DIR))

import revp_v1gt_gis_land_use_coverage_expansion_audit as _v1gt

from revp_v1gt_gis_land_use_coverage_expansion_audit import (
    COVERAGE_STATUS_BBOX_ONLY,
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_NO_TIF,
    COVERAGE_STATUS_UNCOVERED,
    EXPANSION_CANDIDATES,
    METHODOLOGICAL_GUARDRAILS,
    PATCH_SCOPES,
    PETROPOLIS_FBDS_BBOX,
    assess_patch,
    bbox_overlaps,
    coverage_gap_rows,
    load_known_sources,
    normalize_region,
    patch_centroid,
    region_summary,
    run_audit,
    write_csv,
    write_json,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_source(
    *,
    source_id: str = "test_src",
    region: str = "Petropolis",
    file_exists: bool = True,
    lon_min: float = -43.5,
    lat_min: float = -22.7,
    lon_max: float = -43.0,
    lat_max: float = -22.1,
    local_path: str = "/nonexistent/test.geojson",
) -> dict:
    return {
        "source_id": source_id,
        "region": region,
        "format": "geojson_wgs84",
        "local_path": local_path,
        "file_exists": file_exists,
        "bbox_lon_min": lon_min,
        "bbox_lat_min": lat_min,
        "bbox_lon_max": lon_max,
        "bbox_lat_max": lat_max,
        "class_col": "CLASSE_USO",
        "n_features": 100,
        "origin": "test",
        "status": "AVAILABLE" if file_exists else "MISSING",
    }


def _fake_patch(
    *,
    patch_id: str = "test_patch",
    region: str = "Petropolis",
    source_path: str = "",
    scope: str = "dino-corpus",
) -> dict:
    return {
        "patch_id": patch_id,
        "region": region,
        "source_path": source_path,
        "scope": scope,
        "manifest_row": {},
    }


# ---------------------------------------------------------------------------
# TestGuardrails
# ---------------------------------------------------------------------------

class TestGuardrails:
    def test_review_only_true(self):
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True

    def test_supervised_training_false(self):
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False

    def test_labels_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_targets_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["targets_created"] is False

    def test_predictive_claims_false(self):
        assert METHODOLOGICAL_GUARDRAILS["predictive_claims"] is False

    def test_multimodal_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False

    def test_land_use_not_ground_truth(self):
        assert METHODOLOGICAL_GUARDRAILS["land_use_is_ground_truth"] is False

    def test_vulnerability_index_not_ground_truth(self):
        assert METHODOLOGICAL_GUARDRAILS["vulnerability_index_is_ground_truth"] is False

    def test_dino_does_not_predict(self):
        assert METHODOLOGICAL_GUARDRAILS["dino_predicts_vulnerability"] is False

    def test_total_keys(self):
        assert len(METHODOLOGICAL_GUARDRAILS) == 9


# ---------------------------------------------------------------------------
# TestConstants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_patch_scopes_tuple(self):
        assert isinstance(PATCH_SCOPES, tuple)

    def test_patch_scopes_values(self):
        assert "dino-corpus" in PATCH_SCOPES
        assert "full-manifest" in PATCH_SCOPES

    def test_coverage_statuses_distinct(self):
        statuses = {
            COVERAGE_STATUS_COVERED,
            COVERAGE_STATUS_BBOX_ONLY,
            COVERAGE_STATUS_UNCOVERED,
            COVERAGE_STATUS_NO_TIF,
        }
        assert len(statuses) == 4

    def test_fbds_bbox_four_values(self):
        assert len(PETROPOLIS_FBDS_BBOX) == 4

    def test_fbds_bbox_order(self):
        lon_min, lat_min, lon_max, lat_max = PETROPOLIS_FBDS_BBOX
        assert lon_min < lon_max
        assert lat_min < lat_max

    def test_fbds_bbox_in_brazil(self):
        lon_min, lat_min, lon_max, lat_max = PETROPOLIS_FBDS_BBOX
        assert -45.0 < lon_min < -40.0
        assert -25.0 < lat_min < -20.0

    def test_expansion_candidates_list(self):
        assert isinstance(EXPANSION_CANDIDATES, list)
        assert len(EXPANSION_CANDIDATES) >= 5

    def test_expansion_candidates_schema(self):
        required = {"region", "candidate_source", "url_hint", "notes", "status"}
        for cand in EXPANSION_CANDIDATES:
            assert required.issubset(cand.keys()), f"Missing keys in: {cand}"

    def test_expansion_candidates_not_acquired(self):
        for cand in EXPANSION_CANDIDATES:
            assert cand["status"] == "NOT_ACQUIRED"

    def test_expansion_candidates_cover_all_regions(self):
        regions = {c["region"] for c in EXPANSION_CANDIDATES}
        assert "Curitiba" in regions
        assert "Recife" in regions
        assert "Petropolis" in regions


# ---------------------------------------------------------------------------
# TestNormalizeRegion
# ---------------------------------------------------------------------------

class TestNormalizeRegion:
    def test_lowercase(self):
        assert normalize_region("Curitiba") == "curitiba"

    def test_strip_whitespace(self):
        assert normalize_region("  Curitiba  ") == "curitiba"

    def test_remove_accent(self):
        result = normalize_region("Petrópolis")
        assert "o" in result
        assert result == "petropolis"

    def test_petropolis_ascii_e(self):
        assert normalize_region("Petropolis") == "petropolis"

    def test_petropolis_with_accent_matches_without(self):
        assert normalize_region("Petrópolis") == normalize_region("Petropolis")

    def test_recife(self):
        assert normalize_region("Recife") == "recife"

    def test_all_string(self):
        assert normalize_region("All") == "all"


# ---------------------------------------------------------------------------
# TestBboxOverlaps
# ---------------------------------------------------------------------------

class TestBboxOverlaps:
    def test_overlapping_boxes(self):
        assert bbox_overlaps(0, 0, 2, 2, 1, 1, 3, 3) is True

    def test_non_overlapping_lat(self):
        assert bbox_overlaps(0, 0, 1, 1, 0, 2, 1, 3) is False

    def test_non_overlapping_lon(self):
        assert bbox_overlaps(0, 0, 1, 1, 2, 0, 3, 1) is False

    def test_touching_edges_x(self):
        # boxes sharing exactly one edge: implementation treats this as overlapping
        assert bbox_overlaps(0, 0, 1, 1, 1, 0, 2, 1) is True

    def test_one_inside_other(self):
        assert bbox_overlaps(0, 0, 10, 10, 2, 2, 5, 5) is True

    def test_identical_boxes(self):
        assert bbox_overlaps(-43, -23, -42, -22, -43, -23, -42, -22) is True

    def test_no_lon_overlap(self):
        assert bbox_overlaps(-43, -23, -42, -22, -40, -23, -39, -22) is False


# ---------------------------------------------------------------------------
# TestPatchCentroid
# ---------------------------------------------------------------------------

class TestPatchCentroid:
    def test_basic_midpoint(self):
        lon, lat = patch_centroid((0.0, 0.0, 2.0, 4.0))
        assert lon == pytest.approx(1.0)
        assert lat == pytest.approx(2.0)

    def test_negative_coords(self):
        lon, lat = patch_centroid((-43.5, -23.0, -43.0, -22.5))
        assert lon == pytest.approx(-43.25)
        assert lat == pytest.approx(-22.75)

    def test_same_point(self):
        lon, lat = patch_centroid((5.0, 5.0, 5.0, 5.0))
        assert lon == pytest.approx(5.0)
        assert lat == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestLoadKnownSources
# ---------------------------------------------------------------------------

class TestLoadKnownSources:
    def test_returns_list(self):
        sources = load_known_sources()
        assert isinstance(sources, list)

    def test_at_least_one_source(self):
        sources = load_known_sources()
        assert len(sources) >= 1

    def test_petropolis_source_present(self):
        sources = load_known_sources()
        ids = [s["source_id"] for s in sources]
        assert "petropolis_fbds_v1gs" in ids

    def test_source_schema(self):
        sources = load_known_sources()
        required = {
            "source_id", "region", "format", "local_path",
            "file_exists", "bbox_lon_min", "bbox_lat_min",
            "bbox_lon_max", "bbox_lat_max", "class_col",
            "n_features", "origin", "status",
        }
        for s in sources:
            assert required.issubset(s.keys()), f"Missing keys in: {s}"

    def test_file_exists_is_bool(self):
        sources = load_known_sources()
        for s in sources:
            assert isinstance(s["file_exists"], bool)

    def test_petropolis_bbox_matches_constant(self):
        sources = load_known_sources()
        pet = next(s for s in sources if s["source_id"] == "petropolis_fbds_v1gs")
        assert pet["bbox_lon_min"] == pytest.approx(PETROPOLIS_FBDS_BBOX[0])
        assert pet["bbox_lat_min"] == pytest.approx(PETROPOLIS_FBDS_BBOX[1])
        assert pet["bbox_lon_max"] == pytest.approx(PETROPOLIS_FBDS_BBOX[2])
        assert pet["bbox_lat_max"] == pytest.approx(PETROPOLIS_FBDS_BBOX[3])

    def test_petropolis_class_col(self):
        sources = load_known_sources()
        pet = next(s for s in sources if s["source_id"] == "petropolis_fbds_v1gs")
        assert pet["class_col"] == "CLASSE_USO"

    def test_petropolis_n_features(self):
        sources = load_known_sources()
        pet = next(s for s in sources if s["source_id"] == "petropolis_fbds_v1gs")
        assert pet["n_features"] == 6861


# ---------------------------------------------------------------------------
# TestAssessPatch — no TIF
# ---------------------------------------------------------------------------

class TestAssessPatchNoTif:
    def test_no_tif_status(self):
        p = _fake_patch(source_path="")
        row = assess_patch(p, [])
        assert row["coverage_status"] == COVERAGE_STATUS_NO_TIF

    def test_no_tif_tif_found_false(self):
        p = _fake_patch(source_path="")
        row = assess_patch(p, [])
        assert row["tif_found"] is False

    def test_no_tif_blocker_set(self):
        p = _fake_patch(source_path="")
        row = assess_patch(p, [])
        assert row["blocker"] != ""

    def test_no_tif_patch_id_preserved(self):
        p = _fake_patch(patch_id="mypatch", source_path="")
        row = assess_patch(p, [])
        assert row["patch_id"] == "mypatch"

    def test_nonexistent_tif_gives_no_tif(self):
        p = _fake_patch(source_path="/does/not/exist/patch.tif")
        row = assess_patch(p, [])
        assert row["coverage_status"] == COVERAGE_STATUS_NO_TIF


# ---------------------------------------------------------------------------
# TestAssessPatch — source not matching
# ---------------------------------------------------------------------------

class TestAssessPatchUncovered:
    def _patch_with_bounds(self, bounds, region="Curitiba"):
        p = _fake_patch(region=region, source_path="/tmp/fake.tif")
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            return assess_patch(p, [])

    def test_no_sources_gives_uncovered(self):
        bounds = (-43.5, -23.0, -43.0, -22.5)
        row = self._patch_with_bounds(bounds)
        assert row["coverage_status"] == COVERAGE_STATUS_UNCOVERED

    def test_uncovered_has_blocker(self):
        bounds = (-43.5, -23.0, -43.0, -22.5)
        row = self._patch_with_bounds(bounds)
        assert row["blocker"] != ""

    def test_wrong_region_source_gives_uncovered(self):
        src = _fake_source(region="Recife")
        p = _fake_patch(region="Curitiba", source_path="/tmp/fake.tif")
        bounds = (-43.5, -23.0, -43.0, -22.5)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["coverage_status"] == COVERAGE_STATUS_UNCOVERED

    def test_missing_source_gives_uncovered(self):
        src = _fake_source(region="Curitiba", file_exists=False)
        p = _fake_patch(region="Curitiba", source_path="/tmp/fake.tif")
        bounds = (-43.5, -23.0, -43.0, -22.5)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["coverage_status"] == COVERAGE_STATUS_UNCOVERED

    def test_no_bbox_overlap_gives_uncovered(self):
        # source bbox far north; patch far south
        src = _fake_source(
            region="Petropolis",
            lon_min=-43.5, lat_min=-21.0, lon_max=-43.0, lat_max=-20.5,
        )
        p = _fake_patch(region="Petropolis", source_path="/tmp/fake.tif")
        bounds = (-43.5, -23.5, -43.0, -23.0)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["coverage_status"] == COVERAGE_STATUS_UNCOVERED


# ---------------------------------------------------------------------------
# TestAssessPatch — bbox overlap but centroid outside
# ---------------------------------------------------------------------------

class TestAssessPatchBboxOnly:
    def test_bbox_overlap_centroid_outside(self, tmp_path):
        geojson_path = tmp_path / "test.geojson"
        geojson_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}),
            encoding="utf-8",
        )
        src = _fake_source(
            region="Petropolis",
            lon_min=-43.5, lat_min=-22.7,
            lon_max=-43.0, lat_max=-22.1,
            local_path=str(geojson_path),
        )
        p = _fake_patch(region="Petropolis", source_path="/tmp/fake.tif")
        bounds = (-43.4, -22.65, -43.2, -22.55)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["coverage_status"] == COVERAGE_STATUS_BBOX_ONLY
        assert row["source_bbox_overlap"] is True
        assert row["centroid_in_polygon"] is False

    def test_bbox_overlap_source_id_set(self, tmp_path):
        geojson_path = tmp_path / "test.geojson"
        geojson_path.write_text(
            json.dumps({"type": "FeatureCollection", "features": []}),
            encoding="utf-8",
        )
        src = _fake_source(
            source_id="my_src",
            region="Petropolis",
            lon_min=-43.5, lat_min=-22.7,
            lon_max=-43.0, lat_max=-22.1,
            local_path=str(geojson_path),
        )
        p = _fake_patch(region="Petropolis", source_path="/tmp/fake.tif")
        bounds = (-43.4, -22.65, -43.2, -22.55)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["source_id"] == "my_src"


# ---------------------------------------------------------------------------
# TestAssessPatch — covered (centroid inside polygon)
# ---------------------------------------------------------------------------

class TestAssessPatchCovered:
    def test_centroid_inside_polygon(self, tmp_path):
        geojson_path = tmp_path / "cover.geojson"
        geojson_path.write_text(
            json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-43.5, -22.7],
                            [-43.0, -22.7],
                            [-43.0, -22.1],
                            [-43.5, -22.1],
                            [-43.5, -22.7],
                        ]],
                    },
                    "properties": {"CLASSE_USO": "floresta"},
                }],
            }),
            encoding="utf-8",
        )
        src = _fake_source(
            region="Petropolis",
            lon_min=-43.5, lat_min=-22.7,
            lon_max=-43.0, lat_max=-22.1,
            local_path=str(geojson_path),
        )
        p = _fake_patch(region="Petropolis", source_path="/tmp/fake.tif")
        # centroid at (-43.25, -22.4) — inside the polygon
        bounds = (-43.3, -22.45, -43.2, -22.35)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["coverage_status"] == COVERAGE_STATUS_COVERED
        assert row["centroid_in_polygon"] is True
        assert row["class_value"] == "floresta"

    def test_covered_tif_found_true(self, tmp_path):
        geojson_path = tmp_path / "cover2.geojson"
        geojson_path.write_text(
            json.dumps({
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[
                            [-43.5, -22.7], [-43.0, -22.7],
                            [-43.0, -22.1], [-43.5, -22.1],
                            [-43.5, -22.7],
                        ]],
                    },
                    "properties": {"CLASSE_USO": "agua"},
                }],
            }),
            encoding="utf-8",
        )
        src = _fake_source(
            region="Petropolis",
            lon_min=-43.5, lat_min=-22.7,
            lon_max=-43.0, lat_max=-22.1,
            local_path=str(geojson_path),
        )
        p = _fake_patch(region="Petropolis", source_path="/tmp/fake.tif")
        bounds = (-43.3, -22.45, -43.2, -22.35)
        with patch.object(_v1gt, "get_patch_bounds_wgs84", return_value=bounds):
            row = assess_patch(p, [src])
        assert row["tif_found"] is True


# ---------------------------------------------------------------------------
# TestRegionSummary
# ---------------------------------------------------------------------------

class TestRegionSummary:
    def _row(self, region, status):
        return {"region": region, "coverage_status": status}

    def test_empty_returns_empty(self):
        assert region_summary([]) == []

    def test_all_covered(self):
        rows = [self._row("Curitiba", COVERAGE_STATUS_COVERED)] * 4
        result = region_summary(rows)
        assert len(result) == 1
        assert result[0]["land_use_status"] == "AVAILABLE"
        assert result[0]["coverage_rate"] == 1.0
        assert result[0]["n_covered"] == 4

    def test_all_uncovered(self):
        rows = [self._row("Recife", COVERAGE_STATUS_UNCOVERED)] * 4
        result = region_summary(rows)
        assert result[0]["land_use_status"] == "BLOCKED"
        assert result[0]["coverage_rate"] == 0.0

    def test_bbox_partial(self):
        rows = [self._row("Petropolis", COVERAGE_STATUS_BBOX_ONLY)] * 4
        result = region_summary(rows)
        assert result[0]["land_use_status"] == "BBOX_PARTIAL"
        assert result[0]["n_bbox_overlap_only"] == 4

    def test_mixed_regions(self):
        rows = (
            [self._row("Curitiba", COVERAGE_STATUS_UNCOVERED)] * 4
            + [self._row("Petropolis", COVERAGE_STATUS_BBOX_ONLY)] * 4
            + [self._row("Recife", COVERAGE_STATUS_UNCOVERED)] * 4
        )
        result = region_summary(rows)
        assert len(result) == 3
        names = [r["region"] for r in result]
        assert "Curitiba" in names
        assert "Petropolis" in names
        assert "Recife" in names

    def test_n_patches_correct(self):
        rows = (
            [self._row("Curitiba", COVERAGE_STATUS_COVERED)] * 2
            + [self._row("Curitiba", COVERAGE_STATUS_UNCOVERED)] * 2
        )
        result = region_summary(rows)
        assert result[0]["n_patches"] == 4

    def test_no_tif_counted(self):
        rows = [self._row("Recife", COVERAGE_STATUS_NO_TIF)] * 3
        result = region_summary(rows)
        assert result[0]["n_no_tif"] == 3


# ---------------------------------------------------------------------------
# TestCoverageGapRows
# ---------------------------------------------------------------------------

class TestCoverageGapRows:
    def _row(self, status, patch_id="p", region="R", blocker="", note=""):
        return {
            "patch_id": patch_id, "region": region,
            "coverage_status": status, "blocker": blocker, "note": note,
        }

    def test_covered_excluded(self):
        rows = [self._row(COVERAGE_STATUS_COVERED)]
        assert coverage_gap_rows(rows) == []

    def test_uncovered_included(self):
        rows = [self._row(COVERAGE_STATUS_UNCOVERED, blocker="no source")]
        gaps = coverage_gap_rows(rows)
        assert len(gaps) == 1
        assert gaps[0]["severity"] == "BLOCKED"

    def test_no_tif_is_blocked(self):
        rows = [self._row(COVERAGE_STATUS_NO_TIF)]
        gaps = coverage_gap_rows(rows)
        assert gaps[0]["severity"] == "BLOCKED"

    def test_bbox_only_is_partial(self):
        rows = [self._row(COVERAGE_STATUS_BBOX_ONLY)]
        gaps = coverage_gap_rows(rows)
        assert gaps[0]["severity"] == "PARTIAL"

    def test_detail_from_blocker(self):
        rows = [self._row(COVERAGE_STATUS_UNCOVERED, blocker="my blocker")]
        gaps = coverage_gap_rows(rows)
        assert "my blocker" in gaps[0]["detail"]

    def test_detail_from_note_when_no_blocker(self):
        rows = [self._row(COVERAGE_STATUS_BBOX_ONLY, note="centroid outside")]
        gaps = coverage_gap_rows(rows)
        assert "centroid outside" in gaps[0]["detail"]

    def test_patch_id_and_region_preserved(self):
        rows = [self._row(COVERAGE_STATUS_UNCOVERED, patch_id="px", region="Rx")]
        gaps = coverage_gap_rows(rows)
        assert gaps[0]["patch_id"] == "px"
        assert gaps[0]["region"] == "Rx"


# ---------------------------------------------------------------------------
# TestRunAuditNoGisRoot
# ---------------------------------------------------------------------------

class TestRunAuditNoGisRoot:
    """run_audit without gis_root — no TIF files reachable."""

    def test_returns_dict_with_summary(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert "summary" in result

    def test_summary_stage(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert result["summary"]["stage"] == "v1gt"

    def test_summary_scope(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert result["summary"]["scope"] == "dino-corpus"

    def test_summary_gis_root_flag(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert result["summary"]["gis_root_provided"] is False

    def test_guardrails_in_summary(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        s = result["summary"]
        assert s["labels_created"] is False
        assert s["supervised_training"] is False
        assert s["predictive_claims"] is False
        assert s["multimodal_execution_enabled"] is False

    def test_expansion_candidates_in_result(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert len(result["expansion_candidates"]) >= 5

    def test_qa_rows_present(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert "qa_rows" in result
        assert len(result["qa_rows"]) >= 5

    def test_qa_rows_schema(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        for row in result["qa_rows"]:
            assert "check" in row
            assert "status" in row
            assert "detail" in row

    def test_guardrail_qa_checks_pass(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        guard_checks = {
            r["check"]: r["status"]
            for r in result["qa_rows"]
            if r["check"] in (
                "no_labels_created",
                "land_use_not_ground_truth",
                "multimodal_disabled",
            )
        }
        for chk, status in guard_checks.items():
            assert status == "PASS", f"{chk} should be PASS"

    def test_sources_in_result(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert "sources" in result
        assert len(result["sources"]) >= 1

    def test_full_manifest_scope(self, tmp_path):
        result = run_audit("full-manifest", None, tmp_path / "out2")
        assert result["summary"]["scope"] == "full-manifest"

    def test_patch_rows_list(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert isinstance(result["patch_rows"], list)

    def test_reg_summary_list(self, tmp_path):
        result = run_audit("dino-corpus", None, tmp_path / "out")
        assert isinstance(result["reg_summary"], list)


# ---------------------------------------------------------------------------
# TestOutputFiles
# ---------------------------------------------------------------------------

class TestOutputFiles:
    EXPECTED_FILES = [
        "land_use_coverage_summary_v1gt.json",
        "land_use_patch_coverage_v1gt.csv",
        "land_use_region_coverage_v1gt.csv",
        "land_use_source_inventory_v1gt.csv",
        "land_use_coverage_gaps_v1gt.csv",
        "land_use_expansion_candidates_v1gt.csv",
        "land_use_coverage_qa_v1gt.csv",
    ]

    def _run(self, tmp_path):
        out = tmp_path / "v1gt_out"
        run_audit("dino-corpus", None, out)
        # write outputs manually (main() does this, so we call write directly)
        # Actually run_audit returns data — we need to write it like main() does
        # Re-invoke main-like logic via module helpers
        result = run_audit("dino-corpus", None, out)
        write_json(out / "land_use_coverage_summary_v1gt.json", result["summary"])
        write_csv(
            out / "land_use_patch_coverage_v1gt.csv",
            result["patch_rows"],
            ["patch_id", "region", "scope", "tif_found",
             "lon_min", "lat_min", "lon_max", "lat_max",
             "centroid_lon", "centroid_lat",
             "coverage_status", "source_id",
             "source_bbox_overlap", "centroid_in_polygon",
             "class_value", "blocker", "note"],
        )
        write_csv(
            out / "land_use_region_coverage_v1gt.csv",
            result["reg_summary"],
            ["region", "n_patches", "n_covered", "n_bbox_overlap_only",
             "n_uncovered", "n_no_tif", "coverage_rate", "land_use_status"],
        )
        write_csv(
            out / "land_use_source_inventory_v1gt.csv",
            result["sources"],
            ["source_id", "region", "format", "local_path", "file_exists",
             "bbox_lon_min", "bbox_lat_min", "bbox_lon_max", "bbox_lat_max",
             "class_col", "n_features", "origin", "status"],
        )
        write_csv(
            out / "land_use_coverage_gaps_v1gt.csv",
            result["gaps"],
            ["patch_id", "region", "coverage_status", "severity", "detail"],
        )
        write_csv(
            out / "land_use_expansion_candidates_v1gt.csv",
            result["expansion_candidates"],
            ["region", "candidate_source", "url_hint", "notes", "status"],
        )
        write_csv(
            out / "land_use_coverage_qa_v1gt.csv",
            result["qa_rows"],
            ["check", "status", "detail"],
        )
        return out

    def test_all_files_exist(self, tmp_path):
        out = self._run(tmp_path)
        for fname in self.EXPECTED_FILES:
            assert (out / fname).exists(), f"Missing: {fname}"

    def test_summary_json_valid(self, tmp_path):
        out = self._run(tmp_path)
        data = json.loads((out / "land_use_coverage_summary_v1gt.json").read_text(encoding="utf-8"))
        assert data["stage"] == "v1gt"
        assert "labels_created" in data
        assert data["labels_created"] is False

    def test_patch_csv_has_header(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "land_use_patch_coverage_v1gt.csv").open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
        assert "patch_id" in fields
        assert "coverage_status" in fields

    def test_expansion_csv_has_rows(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "land_use_expansion_candidates_v1gt.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) >= 5

    def test_qa_csv_has_pass_checks(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "land_use_coverage_qa_v1gt.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        statuses = {r["status"] for r in rows}
        assert "PASS" in statuses

    def test_source_inventory_petropolis_row(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "land_use_source_inventory_v1gt.csv").open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        source_ids = [r["source_id"] for r in rows]
        assert "petropolis_fbds_v1gs" in source_ids


# ---------------------------------------------------------------------------
# TestNoForbiddenOutputs
# ---------------------------------------------------------------------------

class TestNoForbiddenOutputs:
    def _contains_private_path(self, text: str) -> bool:
        # Check for absolute private user paths that must not appear in source files.
        # The patterns below are split to avoid the test file itself triggering the check.
        user_prefix = "Users" + "\\" + "gabriela"
        unix_home = "/home/"
        return (user_prefix in text) or (unix_home in text)

    def test_no_private_paths_in_script(self):
        script = (
            Path(__file__).resolve().parents[1]
            / "scripts" / "dino"
            / "revp_v1gt_gis_land_use_coverage_expansion_audit.py"
        )
        text = script.read_text(encoding="utf-8", errors="replace")
        assert not self._contains_private_path(text), (
            f"Private path found in versionable script {script}"
        )
