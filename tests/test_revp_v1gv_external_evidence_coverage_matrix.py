"""Tests for revp_v1gv_external_evidence_coverage_matrix.py (corrected).

Covers: guardrails, REGIONAL_INDICATOR_TEMPLATES structure, region normalization,
build_coverage_matrix (with canonical_patch_id), aggregate_by_region,
land-use source filtering, and export functions.

Key corrections from previous version:
- Field is canonical_patch_id, not patch_id
- EVIDENCE_INDICATORS renamed to REGIONAL_INDICATOR_TEMPLATES
- Land-use source filtering: only RJ_3303906_USO is FBDS (Petropolis land-use)
- Region normalization handles Petropolis/Petrópolis variant
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1gv_external_evidence_coverage_matrix import (
    COVERAGE_STATUS_VALUES,
    FIELD_PATCH_ID,
    FIELD_REGION,
    LAND_USE_SOURCE_IDS,
    METHODOLOGICAL_GUARDRAILS,
    PHASE,
    REGIONAL_INDICATOR_TEMPLATES,
    aggregate_by_region,
    build_coverage_matrix,
    normalize_region,
    write_csv,
    write_json,
)


class TestGuardrails:
    def test_methodological_guardrails_locked(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True
        assert METHODOLOGICAL_GUARDRAILS["gis_is_ground_truth"] is False
        assert METHODOLOGICAL_GUARDRAILS["gis_validates_dino"] is False
        assert METHODOLOGICAL_GUARDRAILS["gis_contextualizes_territory"] is True
        assert METHODOLOGICAL_GUARDRAILS["predictive_claims_from_gis"] is False

    def test_gis_not_ground_truth(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["gis_is_ground_truth"] is False


class TestFieldMapping:
    def test_field_patch_id_canonical(self) -> None:
        assert FIELD_PATCH_ID == "canonical_patch_id"

    def test_field_region(self) -> None:
        assert FIELD_REGION == "region"


class TestLandUseSourceFiltering:
    def test_land_use_source_ids_defined(self) -> None:
        assert len(LAND_USE_SOURCE_IDS) > 0

    def test_fbds_petropolis_included(self) -> None:
        assert "RJ_3303906_USO" in LAND_USE_SOURCE_IDS

    def test_non_land_use_sources_excluded(self) -> None:
        # SGB hydro/terrain sources should NOT be in land_use filter
        assert "curitiba_external_sgb_hydro_overlay_status" not in LAND_USE_SOURCE_IDS
        assert "petropolis_external_sgb_hydro_terrain_overlay_v1" not in LAND_USE_SOURCE_IDS


class TestRegionNormalization:
    def test_normalize_petropolis_with_accent(self) -> None:
        assert normalize_region("Petrópolis") == "Petropolis"

    def test_normalize_petropolis_without_accent(self) -> None:
        assert normalize_region("Petropolis") == "Petropolis"

    def test_normalize_curitiba(self) -> None:
        assert normalize_region("Curitiba") == "Curitiba"

    def test_normalize_recife(self) -> None:
        assert normalize_region("Recife") == "Recife"

    def test_normalize_unknown_passthrough(self) -> None:
        result = normalize_region("Unknown")
        assert result == "Unknown"


class TestRegionalIndicatorTemplates:
    def test_all_three_regions_defined(self) -> None:
        assert "Curitiba" in REGIONAL_INDICATOR_TEMPLATES
        assert "Petropolis" in REGIONAL_INDICATOR_TEMPLATES
        assert "Recife" in REGIONAL_INDICATOR_TEMPLATES

    def test_each_region_has_indicators(self) -> None:
        for region, inds in REGIONAL_INDICATOR_TEMPLATES.items():
            assert len(inds) > 0, f"{region} has no indicators"

    def test_indicator_schema(self) -> None:
        for region, inds in REGIONAL_INDICATOR_TEMPLATES.items():
            for ind in inds:
                assert "indicator_id" in ind
                assert "name" in ind
                assert "source" in ind
                assert "status" in ind
                assert ind["status"] in COVERAGE_STATUS_VALUES

    def test_petropolis_has_land_use_fbds(self) -> None:
        pet_ids = [i["indicator_id"] for i in REGIONAL_INDICATOR_TEMPLATES["Petropolis"]]
        assert "land_use_fbds" in pet_ids

    def test_curitiba_has_no_fbds_land_use(self) -> None:
        cur_ids = [i["indicator_id"] for i in REGIONAL_INDICATOR_TEMPLATES["Curitiba"]]
        assert "land_use_fbds" not in cur_ids

    def test_curitiba_land_use_not_acquired(self) -> None:
        cur_lu = next(
            i for i in REGIONAL_INDICATOR_TEMPLATES["Curitiba"]
            if i["indicator_id"] == "land_use"
        )
        assert cur_lu["status"] == "NOT_ACQUIRED"


class TestBuildCoverageMatrix:
    def _make_patches(self, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"canonical_patch_id": pid, "region": region} for pid, region in pairs]

    def test_empty_patches_returns_empty(self) -> None:
        rows = build_coverage_matrix([], {})
        assert rows == []

    def test_single_curitiba_patch(self) -> None:
        patches = self._make_patches([("CUR_00038", "Curitiba")])
        rows = build_coverage_matrix(patches, {})
        assert len(rows) == 1
        assert rows[0]["canonical_patch_id"] == "CUR_00038"
        assert rows[0]["region"] == "Curitiba"

    def test_curitiba_land_use_not_acquired_by_default(self) -> None:
        patches = self._make_patches([("CUR_00038", "Curitiba")])
        rows = build_coverage_matrix(patches, {})
        # Curitiba land_use should be NOT_ACQUIRED (no FBDS)
        assert rows[0].get("land_use") == "NOT_ACQUIRED"

    def test_petropolis_land_use_fbds_overridden_by_v1gt(self) -> None:
        patches = self._make_patches([("PET_00016", "Petrópolis")])
        v1gt = {"PET_00016": "PARTIAL"}
        rows = build_coverage_matrix(patches, v1gt)
        assert rows[0].get("land_use_fbds") == "PARTIAL"

    def test_petropolis_land_use_fbds_default_without_v1gt(self) -> None:
        patches = self._make_patches([("PET_00016", "Petrópolis")])
        rows = build_coverage_matrix(patches, {})
        # Default from template is PARTIAL
        assert rows[0].get("land_use_fbds") == "PARTIAL"

    def test_ignores_row_without_canonical_patch_id(self) -> None:
        patches = [{"canonical_patch_id": "", "region": "Curitiba"},
                   {"canonical_patch_id": "CUR_00038", "region": "Curitiba"}]
        rows = build_coverage_matrix(patches, {})
        assert len(rows) == 1

    def test_all_status_values_valid(self) -> None:
        patches = self._make_patches([
            ("CUR_00038", "Curitiba"),
            ("PET_00016", "Petrópolis"),
            ("REC_00019", "Recife"),
        ])
        rows = build_coverage_matrix(patches, {})
        for row in rows:
            for k, v in row.items():
                if k not in {"canonical_patch_id", "region", "region_normalized",
                             "n_indicators"} and v != "":
                    assert v in COVERAGE_STATUS_VALUES, f"{k}={v} not valid"


class TestAggregateByRegion:
    def test_empty_matrix(self) -> None:
        assert aggregate_by_region([]) == {}

    def test_single_region(self) -> None:
        rows = [
            {"canonical_patch_id": "CUR_00038", "region": "Curitiba",
             "region_normalized": "Curitiba", "n_indicators": 5,
             "land_use": "NOT_ACQUIRED"},
            {"canonical_patch_id": "CUR_00249", "region": "Curitiba",
             "region_normalized": "Curitiba", "n_indicators": 5,
             "land_use": "NOT_ACQUIRED"},
        ]
        result = aggregate_by_region(rows)
        assert "Curitiba" in result
        assert result["Curitiba"]["n_patches"] == 2

    def test_multiple_regions(self) -> None:
        rows = [
            {"canonical_patch_id": "CUR_00038", "region": "Curitiba",
             "region_normalized": "Curitiba", "n_indicators": 5},
            {"canonical_patch_id": "REC_00019", "region": "Recife",
             "region_normalized": "Recife", "n_indicators": 5},
        ]
        result = aggregate_by_region(rows)
        assert len(result) == 2


class TestCoverageStatusValues:
    def test_all_expected_statuses_present(self) -> None:
        expected = {"AVAILABLE", "PARTIAL", "BBOX_ONLY", "BLOCKED",
                    "NOT_ACQUIRED", "LOCAL_ONLY", "MISSING"}
        assert expected.issubset(set(COVERAGE_STATUS_VALUES))


class TestExportFunctions:
    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        data = {"test": "value", "phase": "v1gv"}
        out = tmp_path / "test.json"
        write_json(out, data)
        assert out.exists()
        with out.open() as f:
            loaded = json.load(f)
        assert loaded["phase"] == "v1gv"

    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        rows = [
            {"canonical_patch_id": "CUR_00038", "region": "Curitiba",
             "land_use": "NOT_ACQUIRED"},
        ]
        out = tmp_path / "test.csv"
        write_csv(out, rows, ["canonical_patch_id", "region", "land_use"])
        assert out.exists()
        with out.open() as f:
            data = list(csv.DictReader(f))
        assert len(data) == 1
        assert data[0]["canonical_patch_id"] == "CUR_00038"


class TestPhase:
    def test_phase_constant(self) -> None:
        assert PHASE == "v1gv"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
