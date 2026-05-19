"""Tests for revp_v1gx_tcc_figures_and_tables_export_plan.py.

Covers: guardrails, status values, figures/tables inventory,
metadata structure, no overcommit principle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1gx_tcc_figures_and_tables_export_plan import (
    METHODOLOGICAL_GUARDRAILS,
    PHASE,
    STATUS_VALUES,
    TCC_FIGURES,
    TCC_TABLES,
)


class TestGuardrails:
    def test_methodological_guardrails_locked(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True
        assert METHODOLOGICAL_GUARDRAILS["readiness_claims_honest"] is True
        assert METHODOLOGICAL_GUARDRAILS["no_overcommit"] is True
        assert METHODOLOGICAL_GUARDRAILS["blocked_artifacts_documented"] is True
        assert METHODOLOGICAL_GUARDRAILS["data_dependencies_clear"] is True


class TestStatusValues:
    def test_status_values_defined(self) -> None:
        assert "READY" in STATUS_VALUES
        assert "NEEDS_LOCAL_OUTPUT" in STATUS_VALUES
        assert "NEEDS_MANUAL_REVIEW" in STATUS_VALUES
        assert "BLOCKED" in STATUS_VALUES

    def test_status_values_count(self) -> None:
        assert len(STATUS_VALUES) == 4


class TestTCCFigures:
    def test_figures_not_empty(self) -> None:
        assert len(TCC_FIGURES) > 0

    def test_figures_have_required_fields(self) -> None:
        for fig in TCC_FIGURES:
            assert "figure_id" in fig
            assert "title" in fig
            assert "section" in fig
            assert "source_files" in fig
            assert "format" in fig
            assert "status" in fig
            assert "notes" in fig

    def test_figures_valid_status(self) -> None:
        for fig in TCC_FIGURES:
            assert fig["status"] in STATUS_VALUES

    def test_figures_unique_ids(self) -> None:
        ids = [f["figure_id"] for f in TCC_FIGURES]
        assert len(ids) == len(set(ids))

    def test_figures_have_sections(self) -> None:
        sections = [f["section"] for f in TCC_FIGURES if f["section"]]
        assert len(sections) > 0


class TestTCCTables:
    def test_tables_not_empty(self) -> None:
        assert len(TCC_TABLES) > 0

    def test_tables_have_required_fields(self) -> None:
        for tbl in TCC_TABLES:
            assert "table_id" in tbl
            assert "title" in tbl
            assert "section" in tbl
            assert "source_files" in tbl
            assert "format" in tbl
            assert "status" in tbl
            assert "notes" in tbl

    def test_tables_valid_status(self) -> None:
        for tbl in TCC_TABLES:
            assert tbl["status"] in STATUS_VALUES

    def test_tables_unique_ids(self) -> None:
        ids = [t["table_id"] for t in TCC_TABLES]
        assert len(ids) == len(set(ids))

    def test_tables_have_sections(self) -> None:
        sections = [t["section"] for t in TCC_TABLES if t["section"]]
        assert len(sections) > 0


class TestNoOvercommit:
    def test_figures_blocked_are_documented(self) -> None:
        for fig in TCC_FIGURES:
            if fig["status"] == "BLOCKED":
                assert fig["notes"] != ""

    def test_tables_blocked_are_documented(self) -> None:
        for tbl in TCC_TABLES:
            if tbl["status"] == "BLOCKED":
                assert tbl["notes"] != ""

    def test_figures_needs_local_have_sources(self) -> None:
        for fig in TCC_FIGURES:
            if fig["status"] == "NEEDS_LOCAL_OUTPUT":
                assert fig["source_files"] != ""

    def test_tables_needs_local_have_sources(self) -> None:
        for tbl in TCC_TABLES:
            if tbl["status"] == "NEEDS_LOCAL_OUTPUT":
                assert tbl["source_files"] != ""


class TestTCCOrganization:
    def test_figures_and_tables_map_to_sections(self) -> None:
        artifacts = TCC_FIGURES + TCC_TABLES
        sections = [a["section"] for a in artifacts if a["section"]]
        # Check that sections are documented
        assert len(sections) > 0
        for section in sections:
            assert "." in section  # Likely structured (e.g., "3.1")

    def test_all_artifacts_are_either_figures_or_tables(self) -> None:
        all_ids = {f["figure_id"] for f in TCC_FIGURES} | {t["table_id"] for t in TCC_TABLES}
        assert len(all_ids) == len(TCC_FIGURES) + len(TCC_TABLES)


class TestPhase:
    def test_phase_constant(self) -> None:
        assert PHASE == "v1gx"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
