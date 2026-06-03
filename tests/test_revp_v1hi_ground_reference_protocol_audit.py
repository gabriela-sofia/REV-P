"""Tests for the Protocolo C ground reference audit.

Covers: ground reference source schema, registry structure, controlled values,
forbidden source-family/use combinations, non-acquired source handling,
and documentation content validation.

Reference: docs/metodologia_cientifica/protocolo_c_construcao_referencia_operacional.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ────────── controlled values ──────────

VALID_SOURCE_FAMILIES = {
    "FIELD_OBSERVATION",
    "OFFICIAL_OBSERVED_FLOOD_MAP",
    "EXPERT_ANNOTATED_HIGH_RES_IMAGE",
    "OPERATIONAL_FLOOD_PRODUCT",
    "MODELLED_SUSCEPTIBILITY_LAYER",
    "HYDROGEOMORPHOLOGICAL_CONTEXT",
    "SENTINEL_STRUCTURAL_EVIDENCE",
    "REVIEW_GATE_EVIDENCE",
}

VALID_OBSERVED_OR_MODELED = {
    "OBSERVED",
    "EXPERT_INTERPRETED",
    "OPERATIONAL_ALGORITHMIC",
    "MODELLED",
    "CONTEXTUAL",
    "UNKNOWN",
}

VALID_LOCAL_ACQUISITION_STATUS = {
    "EXISTS_COMPLETE",
    "EXISTS_PARTIAL",
    "NOT_ACQUIRED",
    "INDEXED_ONLY",
    "METHODOLOGICAL_REFERENCE_ONLY",
}

VALID_COMPATIBLE_REFERENCE_STATUS = {
    "CONTEXTUAL_EVIDENCE",
    "AUDITABLE_REFERENCE_PROXY",
    "STRONG_REFERENCE_CANDIDATE",
    "EVENT_OBSERVED_REFERENCE_ELIGIBLE",
}

# Families that cannot support ground truth as allowed_use
CANNOT_PROMOTE_TO_GROUND_TRUTH = {
    "MODELLED_SUSCEPTIBILITY_LAYER",
    "SENTINEL_STRUCTURAL_EVIDENCE",
    "HYDROGEOMORPHOLOGICAL_CONTEXT",
}

# Forbidden allowed_use phrases — must never appear in source registry
FORBIDDEN_ALLOWED_USE_PHRASES = {
    "flood prediction",
    "flood detection",
    "ground truth operacional",
    "ground truth observado",
    "operational ground truth",
    "detecção de enchente",
    "predição de enchente",
    "label de enchente",
    "flood label",
    "training label",
    "supervised class",
}

# Non-acquired statuses that cannot be used as applied reference
NON_ACQUIRED_STATUSES = {
    "NOT_ACQUIRED",
    "METHODOLOGICAL_REFERENCE_ONLY",
}


class TestProtocolCDocumentFiles:
    def test_protocol_c_document_exists(self) -> None:
        doc = PROJECT_ROOT / "docs" / "metodologia_cientifica" / "protocolo_c_construcao_referencia_operacional.md"
        assert doc.exists(), f"Protocolo C document not found: {doc}"

    def test_source_schema_exists(self) -> None:
        schema = PROJECT_ROOT / "datasets" / "schemas" / "ground_reference_evidence_source_schema.csv"
        assert schema.exists(), f"Schema not found: {schema}"

    def test_source_registry_exists(self) -> None:
        registry = PROJECT_ROOT / "datasets" / "ground_reference_evidence_source_registry.csv"
        assert registry.exists(), f"Registry not found: {registry}"

    def test_contextual_reference_layer_registry_still_exists(self) -> None:
        registry = PROJECT_ROOT / "datasets" / "contextual_reference_layer_registry.csv"
        assert registry.exists(), "Existing contextual reference layer registry must not have been removed"


class TestSourceSchemaFields:
    @pytest.fixture
    def schema_fields(self) -> set[str]:
        schema = PROJECT_ROOT / "datasets" / "schemas" / "ground_reference_evidence_source_schema.csv"
        with open(schema, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["field"] for row in reader}

    REQUIRED_FIELDS = {
        "source_id",
        "source_name",
        "source_family",
        "source_type",
        "region",
        "event_specific",
        "event_name",
        "event_date_start",
        "event_date_end",
        "acquisition_date",
        "spatial_resolution_m",
        "geometry_type",
        "observed_or_modeled",
        "human_interpreted",
        "expert_reviewed",
        "uncertainty_documented",
        "temporal_alignment_required",
        "spatial_alignment_required",
        "compatible_reference_status",
        "allowed_use",
        "forbidden_use",
        "limitations",
        "notes",
    }

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_required_field_present(self, schema_fields: set[str], field: str) -> None:
        assert field in schema_fields, f"Required field '{field}' missing from schema"


class TestSourceRegistryStructure:
    @pytest.fixture
    def registry_rows(self) -> list[dict]:
        registry = PROJECT_ROOT / "datasets" / "ground_reference_evidence_source_registry.csv"
        with open(registry, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_registry_has_rows(self, registry_rows: list[dict]) -> None:
        assert len(registry_rows) > 0, "Registry must have at least one row"

    def test_all_rows_have_source_id(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            assert row.get("source_id", "").strip(), f"Row {i}: source_id must not be empty"

    def test_all_rows_have_source_name(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            assert row.get("source_name", "").strip(), f"Row {i}: source_name must not be empty"

    def test_all_source_families_valid(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            family = row.get("source_family", "").strip()
            assert family in VALID_SOURCE_FAMILIES, (
                f"Row {i} ({row.get('source_id')}): invalid source_family '{family}'"
            )

    def test_all_observed_or_modeled_valid(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            obs = row.get("observed_or_modeled", "").strip()
            assert obs in VALID_OBSERVED_OR_MODELED, (
                f"Row {i} ({row.get('source_id')}): invalid observed_or_modeled '{obs}'"
            )

    def test_all_local_acquisition_statuses_valid(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            status = row.get("local_acquisition_status", "").strip()
            assert status in VALID_LOCAL_ACQUISITION_STATUS, (
                f"Row {i} ({row.get('source_id')}): invalid local_acquisition_status '{status}'"
            )

    def test_all_compatible_reference_statuses_valid(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            compat = row.get("compatible_reference_status", "").strip()
            assert compat in VALID_COMPATIBLE_REFERENCE_STATUS, (
                f"Row {i} ({row.get('source_id')}): invalid compatible_reference_status '{compat}'"
            )

    def test_boolean_fields_are_valid(self, registry_rows: list[dict]) -> None:
        boolean_fields = [
            "event_specific",
            "human_interpreted",
            "expert_reviewed",
            "uncertainty_documented",
            "temporal_alignment_required",
            "spatial_alignment_required",
        ]
        for i, row in enumerate(registry_rows):
            for field in boolean_fields:
                val = row.get(field, "").strip().lower()
                assert val in ("true", "false"), (
                    f"Row {i} ({row.get('source_id')}): field '{field}' must be 'true' or 'false', got '{val}'"
                )

    def test_all_rows_have_limitations(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            assert row.get("limitations", "").strip(), (
                f"Row {i} ({row.get('source_id')}): limitations field must not be empty"
            )


class TestForbiddenAllowedUse:
    @pytest.fixture
    def registry_rows(self) -> list[dict]:
        registry = PROJECT_ROOT / "datasets" / "ground_reference_evidence_source_registry.csv"
        with open(registry, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_no_forbidden_phrases_in_allowed_use(self, registry_rows: list[dict]) -> None:
        for i, row in enumerate(registry_rows):
            allowed = row.get("allowed_use", "").lower()
            for phrase in FORBIDDEN_ALLOWED_USE_PHRASES:
                assert phrase.lower() not in allowed, (
                    f"Row {i} ({row.get('source_id')}): forbidden phrase '{phrase}' found in allowed_use"
                )

    def test_modelled_family_cannot_have_ground_truth_allowed_use(
        self, registry_rows: list[dict]
    ) -> None:
        gt_terms = {"ground truth", "verdade de campo", "label operacional"}
        for i, row in enumerate(registry_rows):
            if row.get("source_family", "") in CANNOT_PROMOTE_TO_GROUND_TRUTH:
                allowed = row.get("allowed_use", "").lower()
                for term in gt_terms:
                    assert term not in allowed, (
                        f"Row {i} ({row.get('source_id')}): source_family "
                        f"'{row.get('source_family')}' cannot have allowed_use claiming ground truth"
                    )

    def test_sentinel_dino_family_cannot_promote_ground_truth(
        self, registry_rows: list[dict]
    ) -> None:
        gt_terms = {"ground truth", "flood label", "label de enchente", "training label"}
        for i, row in enumerate(registry_rows):
            if row.get("source_family") == "SENTINEL_STRUCTURAL_EVIDENCE":
                allowed = row.get("allowed_use", "").lower()
                for term in gt_terms:
                    assert term not in allowed, (
                        f"Row {i} ({row.get('source_id')}): SENTINEL_STRUCTURAL_EVIDENCE "
                        f"cannot claim ground truth in allowed_use"
                    )


class TestNonAcquiredSourceRules:
    @pytest.fixture
    def registry_rows(self) -> list[dict]:
        registry = PROJECT_ROOT / "datasets" / "ground_reference_evidence_source_registry.csv"
        with open(registry, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_non_acquired_source_cannot_be_strong_candidate_or_above(
        self, registry_rows: list[dict]
    ) -> None:
        disallowed_for_non_acquired = {"EVENT_OBSERVED_REFERENCE_ELIGIBLE"}
        for i, row in enumerate(registry_rows):
            if row.get("local_acquisition_status") in NON_ACQUIRED_STATUSES:
                compat = row.get("compatible_reference_status", "")
                assert compat not in disallowed_for_non_acquired, (
                    f"Row {i} ({row.get('source_id')}): non-acquired source cannot claim "
                    f"EVENT_OBSERVED_REFERENCE_ELIGIBLE as compatible_reference_status"
                )

    def test_methodological_only_source_has_forbidden_use_field(
        self, registry_rows: list[dict]
    ) -> None:
        for i, row in enumerate(registry_rows):
            if row.get("local_acquisition_status") == "METHODOLOGICAL_REFERENCE_ONLY":
                assert row.get("forbidden_use", "").strip(), (
                    f"Row {i} ({row.get('source_id')}): METHODOLOGICAL_REFERENCE_ONLY source "
                    f"must document forbidden_use"
                )


class TestProtocolCDocumentContent:
    @pytest.fixture
    def doc_content(self) -> str:
        doc = PROJECT_ROOT / "docs" / "metodologia_cientifica" / "protocolo_c_construcao_referencia_operacional.md"
        with open(doc, "r", encoding="utf-8") as f:
            return f.read()

    def test_document_mentions_protocol_c_origin(self, doc_content: str) -> None:
        assert "Protocolo C" in doc_content, "Document must mention Protocolo C"

    def test_document_defines_evidence_reference_distinction(self, doc_content: str) -> None:
        assert "evidência contextual" in doc_content.lower() or "evidência" in doc_content.lower()
        assert "referência" in doc_content.lower()
        assert "ground truth" in doc_content.lower()

    def test_document_states_ground_truth_is_blocked(self, doc_content: str) -> None:
        assert (
            "bloqueado" in doc_content.lower() or "blocked" in doc_content.lower()
        ), "Document must state that operational ground truth is blocked"

    def test_document_lists_blockers(self, doc_content: str) -> None:
        assert "Bloqueadores" in doc_content or "bloqueadores" in doc_content, (
            "Document must list blockers to ground truth promotion"
        )

    def test_document_mentions_comparable_sources(self, doc_content: str) -> None:
        mentions = ["Sen1Floods11", "Copernicus", "Kuro Siwo"]
        found = [m for m in mentions if m in doc_content]
        assert len(found) >= 2, (
            f"Document must mention at least 2 comparable methodological sources; found: {found}"
        )

    def test_document_does_not_assert_flood_prediction(self, doc_content: str) -> None:
        assert "O REV-P prediz enchente" not in doc_content
        assert "REV-P detecta enchente" not in doc_content

    def test_document_does_not_promote_dino_as_classifier(self, doc_content: str) -> None:
        content_lower = doc_content.lower()
        assert "dino classifica inundação" not in content_lower
        assert "dino prediz" not in content_lower

    def test_document_explains_protocol_abc_relation(self, doc_content: str) -> None:
        assert "Protocolo A" in doc_content or "protocolo a" in doc_content.lower(), (
            "Document must explain relation to Protocolo A"
        )
        assert "Protocolo B" in doc_content or "protocolo b" in doc_content.lower(), (
            "Document must explain relation to Protocolo B"
        )

    def test_document_blocks_flood_prediction_claims(self, doc_content: str) -> None:
        blocked_section = "bloqueado" in doc_content.lower() or "blocked" in doc_content.lower()
        forbidden_section = "não permite" in doc_content.lower() or "proibido" in doc_content.lower()
        assert blocked_section and forbidden_section, (
            "Document must both define what is blocked and what is forbidden"
        )

    def test_contextual_evidence_status_present(self, doc_content: str) -> None:
        assert "CONTEXTUAL_EVIDENCE" in doc_content

    def test_auditable_reference_proxy_status_present(self, doc_content: str) -> None:
        assert "AUDITABLE_REFERENCE_PROXY" in doc_content

    def test_strong_reference_candidate_status_present(self, doc_content: str) -> None:
        assert "STRONG_REFERENCE_CANDIDATE" in doc_content

    def test_operational_ground_truth_blocked_status_present(self, doc_content: str) -> None:
        assert "OPERATIONAL_GROUND_TRUTH_BLOCKED" in doc_content

    def test_event_observed_eligible_is_future_only(self, doc_content: str) -> None:
        assert "EVENT_OBSERVED_REFERENCE_ELIGIBLE" in doc_content
        content_lower = doc_content.lower()
        # The document must qualify this as future / not currently applicable
        assert (
            "elegibilidade futura" in content_lower
            or "futuro" in content_lower
            or "não é um status aplicado" in content_lower
        ), "EVENT_OBSERVED_REFERENCE_ELIGIBLE must be described as future eligibility, not current status"


class TestCamadaReferenciaDocumentContent:
    @pytest.fixture
    def doc_content(self) -> str:
        doc = PROJECT_ROOT / "docs" / "metodologia_cientifica" / "camada_referencia_contextual_validada.md"
        with open(doc, "r", encoding="utf-8") as f:
            return f.read()

    def test_linhagem_protocolo_c_section_present(self, doc_content: str) -> None:
        assert "Linhagem metodológica do Protocolo C" in doc_content, (
            "camada_referencia_contextual_validada.md must contain the Protocolo C lineage section"
        )

    def test_section_mentions_ground_truth_as_conditional(self, doc_content: str) -> None:
        assert "estado final condicionado" in doc_content or "condicionado" in doc_content, (
            "Lineage section must describe ground truth as conditional final state"
        )

    def test_section_links_to_protocol_c_document(self, doc_content: str) -> None:
        assert "protocolo_c_construcao_referencia_operacional" in doc_content, (
            "camada_referencia_contextual_validada.md must link to protocolo_c document"
        )


class TestNoForbiddenClaimsInNewDocuments:
    @pytest.fixture
    def protocol_doc(self) -> str:
        doc = PROJECT_ROOT / "docs" / "metodologia_cientifica" / "protocolo_c_construcao_referencia_operacional.md"
        with open(doc, "r", encoding="utf-8") as f:
            return f.read()

    def test_protocol_doc_does_not_assert_flood_detection_as_revp_capability(
        self, protocol_doc: str
    ) -> None:
        content_lower = protocol_doc.lower()
        assert "o rev-p detecta enchentes" not in content_lower
        assert "o rev-p prediz inundação" not in content_lower
        assert "o rev-p produz ground truth" not in content_lower

    def test_protocol_doc_does_not_use_label_de_enchente_as_positive_claim(
        self, protocol_doc: str
    ) -> None:
        # The phrase can appear in blockers/forbidden sections, but not as a positive claim
        lines = protocol_doc.split("\n")
        for i, line in enumerate(lines):
            if "label de enchente" in line.lower():
                context = "\n".join(lines[max(0, i - 3) : i + 4])
                is_in_forbidden_context = any(
                    kw in context.lower()
                    for kw in ["não permite", "bloqueado", "proibido", "forbidden", "vedado"]
                )
                assert is_in_forbidden_context, (
                    f"'label de enchente' at line {i} appears outside a blocking/forbidden context"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
