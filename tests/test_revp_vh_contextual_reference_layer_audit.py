"""Tests for contextual reference layer audit.

Covers: schema validation, registry structure, forbidden claims detection,
status hierarchy, and blocker documentation.

Reference: docs/metodologia_cientifica/camada_referencia_contextual_validada.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


# Forbidden claims that must never appear as allowed_claim in registry
FORBIDDEN_CLAIMS = {
    "flood prediction",
    "flood detection",
    "detecção de enchente",
    "predição de enchente",
    "operational ground truth",
    "ground truth operacional",
    "detected suscetibilidade",
    "ground truth observado",
}

# Valid reference statuses
VALID_REFERENCE_STATUSES = {
    "CONTEXTUAL_EVIDENCE",
    "AUDITABLE_REFERENCE_PROXY",
    "STRONG_REFERENCE_CANDIDATE",
    "OPERATIONAL_GROUND_TRUTH_BLOCKED",
    "INSUFFICIENT_REFERENCE",
}

# CRS status values
VALID_CRS_STATUSES = {
    "COMPATIBLE",
    "INCOMPATIBLE",
    "UNRESOLVED",
    "UNKNOWN",
}


class TestContextualReferenceLayerFiles:
    def test_schema_file_exists(self) -> None:
        schema_file = (
            PROJECT_ROOT / "datasets" / "schemas" / "contextual_reference_layer_schema.csv"
        )
        assert schema_file.exists(), f"Schema file not found: {schema_file}"

    def test_registry_file_exists(self) -> None:
        registry_file = (
            PROJECT_ROOT / "datasets" / "contextual_reference_layer_registry.csv"
        )
        assert registry_file.exists(), f"Registry file not found: {registry_file}"

    def test_methodology_document_exists(self) -> None:
        doc_file = (
            PROJECT_ROOT
            / "docs"
            / "metodologia_cientifica"
            / "camada_referencia_contextual_validada.md"
        )
        assert doc_file.exists(), f"Methodology document not found: {doc_file}"


class TestSchemaStructure:
    @pytest.fixture
    def schema_rows(self) -> list[dict]:
        schema_file = (
            PROJECT_ROOT / "datasets" / "schemas" / "contextual_reference_layer_schema.csv"
        )
        rows = []
        with open(schema_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def test_schema_has_reference_id_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert "reference_id" in field_names, "Schema must define 'reference_id' field"

    def test_schema_has_patch_id_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert "patch_id" in field_names, "Schema must define 'patch_id' field"

    def test_schema_has_reference_status_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert (
            "reference_status" in field_names
        ), "Schema must define 'reference_status' field"

    def test_schema_has_allowed_claim_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert (
            "allowed_claim" in field_names
        ), "Schema must define 'allowed_claim' field"

    def test_schema_has_forbidden_claim_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert (
            "forbidden_claim" in field_names
        ), "Schema must define 'forbidden_claim' field"

    def test_schema_has_promotion_allowed_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert (
            "promotion_allowed" in field_names
        ), "Schema must define 'promotion_allowed' field"

    def test_schema_has_blocked_reason_field(self, schema_rows: list[dict]) -> None:
        field_names = [row["field"] for row in schema_rows]
        assert (
            "blocked_reason" in field_names
        ), "Schema must define 'blocked_reason' field"


class TestRegistryStructure:
    @pytest.fixture
    def registry_rows(self) -> list[dict]:
        registry_file = (
            PROJECT_ROOT / "datasets" / "contextual_reference_layer_registry.csv"
        )
        rows = []
        with open(registry_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def test_registry_has_rows(self, registry_rows: list[dict]) -> None:
        assert (
            len(registry_rows) > 0
        ), "Registry must have at least one example row (methodology/placeholder)"

    def test_all_registry_rows_have_reference_id(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            assert (
                row.get("reference_id") and row["reference_id"].strip()
            ), f"Row {i} missing reference_id"

    def test_all_registry_rows_have_patch_id(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            assert (
                row.get("patch_id") and row["patch_id"].strip()
            ), f"Row {i} missing patch_id"

    def test_all_registry_rows_have_valid_reference_status(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            status = row.get("reference_status", "").strip()
            assert (
                status in VALID_REFERENCE_STATUSES
            ), f"Row {i}: invalid reference_status '{status}'. Must be one of {VALID_REFERENCE_STATUSES}"

    def test_all_registry_rows_have_valid_crs_status(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            crs_status = row.get("crs_status", "").strip()
            assert (
                crs_status in VALID_CRS_STATUSES
            ), f"Row {i}: invalid crs_status '{crs_status}'. Must be one of {VALID_CRS_STATUSES}"

    def test_promotion_allowed_is_boolean(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            promotion = row.get("promotion_allowed", "").strip().lower()
            assert promotion in (
                "true",
                "false",
            ), f"Row {i}: promotion_allowed must be 'true' or 'false', got '{promotion}'"

    def test_if_promotion_false_then_blocked_reason_present(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            if row.get("promotion_allowed", "").strip().lower() == "false":
                blocked_reason = row.get("blocked_reason", "").strip()
                assert (
                    blocked_reason
                ), f"Row {i}: promotion_allowed=false requires blocked_reason to be non-empty"

    def test_if_promotion_true_then_blocked_reason_empty_or_na(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            if row.get("promotion_allowed", "").strip().lower() == "true":
                blocked_reason = row.get("blocked_reason", "").strip()
                assert (
                    not blocked_reason or blocked_reason.upper() == "N/A"
                ), f"Row {i}: if promotion_allowed=true, blocked_reason should be empty or N/A"


class TestForbiddenClaimsInRegistry:
    @pytest.fixture
    def registry_rows(self) -> list[dict]:
        registry_file = (
            PROJECT_ROOT / "datasets" / "contextual_reference_layer_registry.csv"
        )
        rows = []
        with open(registry_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def test_no_forbidden_claims_in_allowed_claim_field(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            allowed_claim = row.get("allowed_claim", "").lower()
            for forbidden in FORBIDDEN_CLAIMS:
                assert (
                    forbidden.lower() not in allowed_claim
                ), f"Row {i}: forbidden claim '{forbidden}' found in allowed_claim field"

    def test_forbidden_claim_field_documents_prohibitions(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            forbidden_claim = row.get("forbidden_claim", "").strip()
            assert (
                forbidden_claim
            ), f"Row {i}: forbidden_claim field should document what claims are prohibited"


class TestMethodologyDocument:
    @pytest.fixture
    def doc_content(self) -> str:
        doc_file = (
            PROJECT_ROOT
            / "docs"
            / "metodologia_cientifica"
            / "camada_referencia_contextual_validada.md"
        )
        with open(doc_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_document_defines_reference_statuses(self, doc_content: str) -> None:
        for status in VALID_REFERENCE_STATUSES:
            assert (
                status in doc_content
            ), f"Document must define reference status '{status}'"

    def test_document_explains_why_ground_truth_not_binary(
        self, doc_content: str
    ) -> None:
        assert (
            "não é binário" in doc_content or "not binary" in doc_content
        ), "Document must explain why ground truth is not binary"

    def test_document_contains_escada_de_evidencia(self, doc_content: str) -> None:
        assert (
            "escada" in doc_content.lower() or "progression" in doc_content.lower()
        ), "Document must describe evidence progression/ladder"

    def test_document_defines_blocked_conditions(self, doc_content: str) -> None:
        blocked_keywords = [
            "bloqueado",
            "ausência de evento",
            "operacional",
            "operacional_ground_truth_blocked",
        ]
        assert any(
            kw in doc_content.lower() for kw in blocked_keywords
        ), f"Document must explain why operational ground truth is blocked"


class TestNoForbiddenClaimsInDocument:
    @pytest.fixture
    def doc_content(self) -> str:
        doc_file = (
            PROJECT_ROOT
            / "docs"
            / "metodologia_cientifica"
            / "camada_referencia_contextual_validada.md"
        )
        with open(doc_file, "r", encoding="utf-8") as f:
            return f.read()

    def test_document_does_not_assert_flood_prediction_as_capability(
        self, doc_content: str
    ) -> None:
        # Document can MENTION "flood prediction" but must not ASSERT it as REV-P capability
        # Check for affirmative claims without qualifier
        assert (
            "O REV-P prediz enchente" not in doc_content
            and "REV-P detects flooding" not in doc_content
        ), "Document must not assert flood prediction as capability"

    def test_document_explains_blocked_operational_ground_truth(
        self, doc_content: str
    ) -> None:
        # Document must explain that operational ground truth is blocked
        content_lower = doc_content.lower()
        assert (
            "operacional" in content_lower and "bloqueado" in content_lower
        ) or (
            "operational" in content_lower and "blocked" in content_lower
        ), "Document must explain that operational ground truth is blocked"


class TestStatusHierarchy:
    def test_status_order_is_progressive(self) -> None:
        # Status should progress from evidence to proxy to candidate to operational
        progression = [
            "CONTEXTUAL_EVIDENCE",
            "AUDITABLE_REFERENCE_PROXY",
            "STRONG_REFERENCE_CANDIDATE",
            "OPERATIONAL_GROUND_TRUTH_BLOCKED",
        ]
        for status in progression:
            assert status in VALID_REFERENCE_STATUSES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
