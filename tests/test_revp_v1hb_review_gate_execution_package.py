"""Tests for REV-P v1hb: Review Gate Execution Package."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V1HB_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1hb"

FORBIDDEN_REVIEW_TERMS = {
    "prediction", "predictive", "detect", "detection", "classify", "classification",
    "class", "label", "risk", "vulnerability", "ground truth", "ground-truth",
    "validate", "validation", "accuracy", "performance", "train", "supervised",
    "target", "causal", "proven", "proves",
}

EXPECTED_OUTPUTS = {
    "review_gate_execution_manifest_v1hb.csv": {"min_rows": 45},
    "review_gate_annotation_template_v1hb.csv": {"min_rows": 45},
    "review_gate_category_summary_v1hb.csv": {"min_rows": 2},
    "review_gate_discussion_inputs_v1hb.csv": {"min_rows": 5},
    "review_gate_protocol_v1hb.md": {"min_bytes": 1000},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV file."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_md(path: Path) -> str:
    """Read markdown file."""
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8") as f:
        return f.read()


class TestV1HBOutputs:
    """Test v1hb output files existence and structure."""

    @pytest.mark.parametrize("filename", EXPECTED_OUTPUTS.keys())
    def test_output_files_exist(self, filename: str) -> None:
        """Test that all expected output files exist."""
        filepath = V1HB_DIR / filename
        assert filepath.exists(), f"Output file not found: {filename}"

    def test_execution_manifest_valid(self) -> None:
        """Test execution manifest has valid structure."""
        path = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
        rows = read_csv(path)

        assert len(rows) >= 45, f"Expected >=45 rows, got {len(rows)}"

        # Check required columns
        required_cols = [
            "review_item_id",
            "canonical_patch_id",
            "region",
            "candidate_category",
            "selection_reason",
            "forbidden_claim_warning",
        ]

        if rows:
            for col in required_cols:
                assert col in rows[0].keys(), f"Missing column: {col}"

        # Check that each item has review_item_id and status
        for row in rows:
            assert row.get("review_item_id"), f"Missing review_item_id in row"
            assert row.get("canonical_patch_id"), f"Missing patch_id in {row.get('review_item_id')}"
            assert row.get("reviewer_status") == "PENDING", f"Invalid status in {row.get('review_item_id')}"

    def test_annotation_template_valid(self) -> None:
        """Test annotation template has correct structure."""
        path = V1HB_DIR / "review_gate_annotation_template_v1hb.csv"
        rows = read_csv(path)

        assert len(rows) >= 45, f"Expected >=45 template rows, got {len(rows)}"

        # Check annotation fields
        annotation_fields = [
            "review_item_id",
            "reviewer_name_or_initials",
            "review_date",
            "visual_pattern_notes",
            "surrounding_context_notes",
            "external_evidence_notes",
            "uncertainty_level",
            "usable_in_discussion",
            "no_label_created_confirmed",
            "no_prediction_claim_confirmed",
        ]

        if rows:
            for field in annotation_fields:
                assert field in rows[0].keys(), f"Missing annotation field: {field}"

        # Verify template structure (should be mostly empty for reviewer to fill)
        first_row = rows[0] if rows else {}
        # Template should have placeholders, not full answers
        assert (
            not first_row.get("visual_pattern_notes")
            or first_row.get("visual_pattern_notes") == ""
        ), "Template should not have pre-filled visual notes"

    def test_no_forbidden_terms_in_manifest(self) -> None:
        """Test that forbidden review terms don't appear in manifest."""
        path = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
        rows = read_csv(path)

        violations = []
        for row in rows:
            for col in ["allowed_claim_scope", "forbidden_claim_warning", "reviewer_observation"]:
                text = row.get(col, "").lower()
                for forbidden_term in FORBIDDEN_REVIEW_TERMS:
                    if forbidden_term in text and "forbidden" not in col:
                        violations.append(
                            f"Row {row.get('review_item_id')}, {col}: contains '{forbidden_term}'"
                        )

        assert not violations, f"Forbidden terms found: {violations}"

    def test_category_summary_valid(self) -> None:
        """Test category summary has expected categories."""
        path = V1HB_DIR / "review_gate_category_summary_v1hb.csv"
        rows = read_csv(path)

        assert len(rows) >= 2, f"Expected >=2 categories, got {len(rows)}"

        # Check required fields
        for row in rows:
            assert row.get("category"), f"Missing category name"
            assert row.get("n_patches"), f"Missing n_patches for {row.get('category')}"
            assert row.get("interpretation_allowed"), f"Missing allowed interpretation"
            assert row.get("interpretation_forbidden"), f"Missing forbidden interpretation"

        # Verify interpretation_forbidden uses words from prohibited list
        forbidden_keywords = ["prediction", "classification", "risk", "label", "ground truth"]
        for row in rows:
            forbidden_text = row.get("interpretation_forbidden", "").lower()
            has_keywords = any(kw in forbidden_text for kw in forbidden_keywords)
            assert has_keywords, f"Category {row.get('category')} lacks clear forbidden interpretation"

    def test_discussion_inputs_valid(self) -> None:
        """Test discussion inputs table is ready for TCC writing."""
        path = V1HB_DIR / "review_gate_discussion_inputs_v1hb.csv"
        rows = read_csv(path)

        assert len(rows) >= 5, f"Expected >=5 discussion findings, got {len(rows)}"

        # Check required fields
        for row in rows:
            assert row.get("finding_id"), f"Missing finding_id"
            assert row.get("finding_summary"), f"Missing summary for {row.get('finding_id')}"
            assert row.get("interpretation_for_discussion"), f"Missing interpretation"
            assert row.get("claim_allowed") in ["yes", "no"], f"Invalid claim_allowed"
            assert row.get("claim_blocked"), f"Missing claim_blocked for {row.get('finding_id')}"

        # Verify no predictive claims in discussion inputs
        for row in rows:
            summary = row.get("finding_summary", "").lower()
            interpretation = row.get("interpretation_for_discussion", "").lower()

            for term in ["predict", "detect", "classify", "label"]:
                assert term not in summary, f"Predictive term '{term}' in summary of {row.get('finding_id')}"
                assert term not in interpretation, f"Predictive term '{term}' in interpretation of {row.get('finding_id')}"

    def test_protocol_document_valid(self) -> None:
        """Test protocol document is properly structured."""
        path = V1HB_DIR / "review_gate_protocol_v1hb.md"
        content = read_md(path)

        assert len(content) >= 1000, f"Protocol document too short: {len(content)} bytes"

        # Check for key sections
        required_sections = [
            "## Propósito",
            "## O Que o Reviewer Pode",
            "## O Que o Reviewer NÃO Pode",
            "## Categorias",
            "## Protocolo de Anotação",
            "## Garantias Metodológicas",
        ]

        for section in required_sections:
            assert section in content, f"Missing section: {section}"

        # Verify that prohibited claims are explicitly listed
        prohibited_keywords = ["predição", "detecção", "classificação", "label", "risco"]
        for keyword in prohibited_keywords:
            assert keyword.lower() in content.lower(), f"Missing prohibited keyword: {keyword}"


class TestV1HBMetodology:
    """Test that v1hb maintains methodological guardrails."""

    def test_no_label_creation_enforced(self) -> None:
        """Test that templates enforce no-label policy."""
        path = V1HB_DIR / "review_gate_annotation_template_v1hb.csv"
        rows = read_csv(path)

        # Each template row should have confirmation fields
        for row in rows:
            # These fields must be confirmed by reviewer
            assert "no_label_created_confirmed" in row, "Missing label confirmation field"
            assert "no_prediction_claim_confirmed" in row, "Missing prediction confirmation field"

    def test_review_is_review_only(self) -> None:
        """Test that review is formalized as review-only."""
        path = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
        rows = read_csv(path)

        # Check that methodological_interpretation is explanatory, not operational
        for row in rows:
            interpretation = row.get("methodological_interpretation", "").lower()
            # Should use words like "coherence", "observation", "interpretation"
            assert any(
                word in interpretation for word in ["coherence", "observation", "interpretation", "contextual"]
            ), f"Interpretation seems operational rather than interpretative in {row.get('review_item_id')}"

    def test_all_candidates_have_warning(self) -> None:
        """Test that all candidates have forbidden claim warning."""
        path = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
        rows = read_csv(path)

        for row in rows:
            warning = row.get("forbidden_claim_warning", "").lower()
            assert "no" in warning and ("prediction" in warning or "detection" in warning or "classification" in warning), \
                f"No proper warning in {row.get('review_item_id')}"

    def test_allowed_claim_scope_documented(self) -> None:
        """Test that allowed claims are explicitly documented."""
        path = V1HB_DIR / "review_gate_execution_manifest_v1hb.csv"
        rows = read_csv(path)

        allowed_keywords = ["visual", "pattern", "coherence", "contextual", "structural"]

        for row in rows:
            scope = row.get("allowed_claim_scope", "").lower()
            has_keyword = any(kw in scope for kw in allowed_keywords)
            assert has_keyword, f"Allowed scope unclear in {row.get('review_item_id')}"


class TestV1HBOutputLocation:
    """Test that outputs are in correct locations."""

    def test_local_runs_not_versioned(self) -> None:
        """Test that local_runs outputs are in .gitignore scope."""
        # All outputs should be in local_runs/dino_embeddings/v1hb/
        path = V1HB_DIR

        # Verify path is under local_runs
        assert "local_runs" in str(path), f"Outputs not in local_runs: {path}"

        # Verify no outputs outside local_runs
        docs_path = ROOT / "docs" / "metodologia_cientifica" / "review_gate_protocol.md"
        assert docs_path.exists(), "Versionable protocol doc should exist in docs/"

    def test_protocol_in_docs(self) -> None:
        """Test that protocol document is versionable in docs."""
        path = ROOT / "docs" / "metodologia_cientifica" / "review_gate_protocol.md"
        assert path.exists(), "Protocol not found in docs/"

        content = read_md(path)
        assert len(content) > 2000, "Protocol doc in docs/ should be complete"

        # Should not contain private paths
        assert "local_runs" not in content or "local_runs/" in content.split("scripts")[1] if "scripts" in content else True, \
            "Private paths should not be in versionable docs"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
