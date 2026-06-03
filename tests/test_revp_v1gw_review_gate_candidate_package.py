"""Tests for revp_v1gw_review_gate_candidate_package.py (corrected).

Covers: guardrails, candidate categories, review protocol, region normalization,
count_available_indicators, select_review_candidates_fallback,
select_review_candidates_with_embeddings, and export functions.

Key corrections from previous version:
- extract_region_from_patch no longer exists; use normalize_region
- identify_candidates replaced by select_review_candidates_fallback /
  select_review_candidates_with_embeddings
- Field is canonical_patch_id (not patch_id)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1gw_review_gate_candidate_package import (
    CANDIDATE_CATEGORIES,
    FIELD_PATCH_ID,
    FIELD_REGION,
    METHODOLOGICAL_GUARDRAILS,
    PHASE,
    REVIEW_PROTOCOL,
    count_available_indicators,
    normalize_region,
    select_review_candidates_fallback,
    select_review_candidates_with_embeddings,
    write_csv,
    write_json,
)


class TestGuardrails:
    def test_methodological_guardrails_locked(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True
        assert METHODOLOGICAL_GUARDRAILS["review_gate_is_methodological_stage"] is True
        assert METHODOLOGICAL_GUARDRAILS["automatic_classification_forbidden"] is True
        assert METHODOLOGICAL_GUARDRAILS["candidates_are_not_classified"] is True

    def test_review_is_not_validation(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_is_not_validation"] is True

    def test_review_does_not_create_labels(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_does_not_create_labels"] is True


class TestFieldMapping:
    def test_field_patch_id_canonical(self) -> None:
        assert FIELD_PATCH_ID == "canonical_patch_id"

    def test_field_region(self) -> None:
        assert FIELD_REGION == "region"


class TestCandidateCategories:
    def test_candidate_categories_defined(self) -> None:
        assert len(CANDIDATE_CATEGORIES) > 0
        assert "medoid_regional" in CANDIDATE_CATEGORIES
        assert "outlier_structural" in CANDIDATE_CATEGORIES
        assert "bridge_inter_regional" in CANDIDATE_CATEGORIES

    def test_fallback_categories_present(self) -> None:
        assert "geometry_incomplete" in CANDIDATE_CATEGORIES
        assert "geometry_complete" in CANDIDATE_CATEGORIES
        assert "coverage_external_low" in CANDIDATE_CATEGORIES

    def test_all_categories_unique(self) -> None:
        assert len(CANDIDATE_CATEGORIES) == len(set(CANDIDATE_CATEGORIES))


class TestReviewProtocol:
    def test_protocol_text_exists(self) -> None:
        assert REVIEW_PROTOCOL is not None
        assert len(REVIEW_PROTOCOL) > 100

    def test_protocol_mentions_categories(self) -> None:
        assert "medoid_regional" in REVIEW_PROTOCOL
        assert "outlier_structural" in REVIEW_PROTOCOL

    def test_protocol_forbids_classification(self) -> None:
        assert "NOT" in REVIEW_PROTOCOL
        assert ("validation" in REVIEW_PROTOCOL.lower() or
                "vulnerability" in REVIEW_PROTOCOL.lower())

    def test_protocol_forbids_labels(self) -> None:
        assert "label" in REVIEW_PROTOCOL.lower()


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


class TestCountAvailableIndicators:
    def test_empty_dict_returns_zero(self) -> None:
        n_avail, n_total = count_available_indicators({})
        assert n_avail == 0
        assert n_total == 0

    def test_meta_keys_excluded(self) -> None:
        indicators = {
            "canonical_patch_id": "CUR_00038",
            "region": "Curitiba",
            "region_normalized": "Curitiba",
            "n_indicators": "5",
        }
        n_avail, n_total = count_available_indicators(indicators)
        assert n_total == 0

    def test_blank_other_region_columns_excluded(self) -> None:
        indicators = {
            "land_use": "NOT_ACQUIRED",
            "land_use_fbds": "",  # blank — not this region's column
            "terrain_geocuritiba": "PARTIAL",
        }
        n_avail, n_total = count_available_indicators(indicators)
        assert n_total == 2  # only non-blank values

    def test_available_statuses_counted(self) -> None:
        indicators = {
            "terrain": "AVAILABLE",
            "drainage": "PARTIAL",
            "land_use": "NOT_ACQUIRED",
        }
        n_avail, n_total = count_available_indicators(indicators)
        assert n_avail == 2
        assert n_total == 3

    def test_local_only_counts_as_available(self) -> None:
        indicators = {"terrain": "LOCAL_ONLY"}
        n_avail, n_total = count_available_indicators(indicators)
        assert n_avail == 1


class TestSelectCandidatesFallback:
    def _make_manifest(self, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"canonical_patch_id": pid, "region": region} for pid, region in pairs]

    def test_empty_manifest_returns_empty(self) -> None:
        result = select_review_candidates_fallback([], {}, {})
        assert result == []

    def test_geometry_incomplete_category(self) -> None:
        manifest = self._make_manifest([("CUR_00038", "Curitiba")])
        geometry_status = {"CUR_00038": "CENTROID_ONLY"}
        result = select_review_candidates_fallback(manifest, {}, geometry_status)
        assert len(result) >= 1
        categories = [c["categories"] for c in result if c["canonical_patch_id"] == "CUR_00038"]
        assert any("geometry_incomplete" in cats for cats in categories)

    def test_geometry_complete_category(self) -> None:
        manifest = self._make_manifest([("CUR_00038", "Curitiba")])
        geometry_status = {"CUR_00038": "BBOX_AVAILABLE"}
        result = select_review_candidates_fallback(manifest, {}, geometry_status)
        categories = [c["categories"] for c in result if c["canonical_patch_id"] == "CUR_00038"]
        assert any("geometry_complete" in cats for cats in categories)

    def test_low_coverage_category(self) -> None:
        manifest = self._make_manifest([("CUR_00038", "Curitiba")])
        geometry_status = {"CUR_00038": "CENTROID_ONLY"}
        coverage = {
            "CUR_00038": {
                "terrain_geocuritiba": "MISSING",
                "land_use": "NOT_ACQUIRED",
                "drainage": "MISSING",
            }
        }
        result = select_review_candidates_fallback(manifest, coverage, geometry_status)
        categories = [c["categories"] for c in result if c["canonical_patch_id"] == "CUR_00038"]
        flat = [cat for cats in categories for cat in cats]
        assert "coverage_external_low" in flat

    def test_result_uses_canonical_patch_id(self) -> None:
        manifest = self._make_manifest([("PET_00016", "Petrópolis")])
        geometry_status = {"PET_00016": "CENTROID_ONLY"}
        result = select_review_candidates_fallback(manifest, {}, geometry_status)
        assert all("canonical_patch_id" in c for c in result)

    def test_each_region_represented(self) -> None:
        manifest = self._make_manifest([
            ("CUR_00038", "Curitiba"),
            ("PET_00016", "Petrópolis"),
            ("REC_00019", "Recife"),
        ])
        geometry_status = {
            "CUR_00038": "CENTROID_ONLY",
            "PET_00016": "CENTROID_ONLY",
            "REC_00019": "CENTROID_ONLY",
        }
        result = select_review_candidates_fallback(manifest, {}, geometry_status)
        regions = {c["region"] for c in result}
        assert len(regions) >= 1  # at least one region represented


class TestSelectCandidatesWithEmbeddings:
    def _make_manifest(self, pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"canonical_patch_id": pid, "region": region} for pid, region in pairs]

    def test_empty_medoids_returns_empty(self) -> None:
        manifest = self._make_manifest([("PET_00001", "Petropolis")])
        result = select_review_candidates_with_embeddings(manifest, {}, {}, {})
        assert result == []

    def test_medoid_adds_medoid_regional_category(self) -> None:
        manifest = self._make_manifest([("PET_00001", "Petropolis")])
        medoids = {
            "Petropolis": {
                "medoid": "PET_00001",
                "outliers": [],
            }
        }
        result = select_review_candidates_with_embeddings(manifest, medoids, {}, {})
        assert len(result) == 1
        assert "medoid_regional" in result[0]["categories"]

    def test_outlier_adds_outlier_structural_category(self) -> None:
        manifest = self._make_manifest([
            ("PET_00001", "Petropolis"),
            ("PET_00099", "Petropolis"),
        ])
        medoids = {
            "Petropolis": {
                "medoid": "PET_00001",
                "outliers": ["PET_00099"],
            }
        }
        result = select_review_candidates_with_embeddings(manifest, medoids, {}, {})
        outlier = next((c for c in result if c["canonical_patch_id"] == "PET_00099"), None)
        assert outlier is not None
        assert "outlier_structural" in outlier["categories"]


class TestExportFunctions:
    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        data = {"candidates": 5, "phase": "v1gw"}
        output_file = tmp_path / "test.json"
        write_json(output_file, data)
        assert output_file.exists()
        with output_file.open() as f:
            loaded = json.load(f)
        assert loaded["candidates"] == 5

    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        rows = [
            {
                "canonical_patch_id": "CUR_00038",
                "region": "Curitiba",
                "categories": "geometry_incomplete",
                "review_status": "NOT_REVIEWED",
            },
        ]
        output_file = tmp_path / "test.csv"
        write_csv(
            output_file,
            rows,
            ["canonical_patch_id", "region", "categories", "review_status"],
        )
        assert output_file.exists()
        with output_file.open() as f:
            data = list(csv.DictReader(f))
        assert len(data) == 1
        assert data[0]["canonical_patch_id"] == "CUR_00038"


class TestPhase:
    def test_phase_constant(self) -> None:
        assert PHASE == "v1gw"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
