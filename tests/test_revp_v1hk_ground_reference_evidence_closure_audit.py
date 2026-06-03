"""Tests for Protocolo C evidence closure and promotion decision stage.

Covers: gap matrix schema/registry, review gate schema/registry, promotion
decision schema/registry, document content validation, and promotion guardrails.

Reference: docs/metodologia_cientifica/protocolo_c_fechamento_evidencias_ground_reference.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ────────── controlled value sets ──────────

VALID_CURRENT_CLOSURE_LEVEL = {
    "EVIDENCE_OPEN",
    "EVIDENCE_PARTIALLY_CLOSED",
    "REFERENCE_CANDIDATE_READY_FOR_REVIEW",
    "STRONG_REFERENCE_READY_FOR_EXTERNAL_VALIDATION",
    "OPERATIONAL_GROUND_TRUTH_NOT_ESTABLISHED",
}

VALID_MISSING_GATE = {
    "G0_PATCH_LINEAGE",
    "G1_EVENT_CONFIRMATION",
    "G2_SOURCE_AVAILABILITY",
    "G3_TEMPORAL_ALIGNMENT",
    "G4_SPATIAL_ALIGNMENT",
    "G5_SOURCE_STRENGTH",
    "G6_UNCERTAINTY_AND_LIMITATIONS",
    "G7_REVIEW_GATE",
    "G8_INDEPENDENT_CORROBORATION",
    "G9_PROMOTION_DECISION",
    "MULTIPLE_GATES",
    "NOT_ASSESSED",
}

VALID_EVIDENCE_PRIORITY = {
    "HIGH",
    "MEDIUM",
    "LOW",
    "METHOD_REFERENCE_ONLY",
}

VALID_METHODOLOGICAL_RISK = {
    "LOW",
    "MODERATE",
    "HIGH",
    "CRITICAL",
    "METHOD_REFERENCE_ONLY",
}

VALID_CURRENT_REFERENCE_STATUS = {
    "CONTEXTUAL_EVIDENCE",
    "AUDITABLE_REFERENCE_PROXY",
    "STRONG_REFERENCE_CANDIDATE",
    "OPERATIONAL_GROUND_TRUTH_NOT_ESTABLISHED",
    "METHOD_REFERENCE_ONLY",
}

VALID_REVIEWER_ROLE = {
    "METHODOLOGICAL_REVIEWER",
    "DOMAIN_REVIEWER",
    "GIS_REVIEWER",
    "REMOTE_SENSING_REVIEWER",
    "FUTURE_EXTERNAL_REVIEWER",
    "NOT_EXECUTED",
}

VALID_CONSISTENCY_STATUS = {
    "CONSISTENT",
    "PARTIAL",
    "INCONSISTENT",
    "NOT_ASSESSED",
    "METHOD_REFERENCE_ONLY",
}

VALID_REVIEW_DECISION = {
    "ACCEPT_AS_CONTEXTUAL_REFERENCE",
    "ACCEPT_AS_AUDITABLE_PROXY",
    "MARK_AS_STRONG_REFERENCE_CANDIDATE",
    "BLOCK_OPERATIONAL_PROMOTION",
    "REQUEST_ADDITIONAL_EVIDENCE",
    "REJECT_AS_INSUFFICIENT",
    "METHOD_REFERENCE_ONLY",
}

VALID_CONFIDENCE_LEVEL = {
    "LOW",
    "MODERATE",
    "HIGH",
    "NOT_APPLICABLE",
    "METHOD_REFERENCE_ONLY",
}

VALID_FINAL_REFERENCE_STATUS = {
    "CONTEXTUAL_EVIDENCE",
    "AUDITABLE_REFERENCE_PROXY",
    "STRONG_REFERENCE_CANDIDATE",
    "OPERATIONAL_GROUND_TRUTH_NOT_ESTABLISHED",
    "INSUFFICIENT_REFERENCE",
    "METHOD_REFERENCE_ONLY",
}

# Critical gates — their failure blocks any promotion
CRITICAL_GATES = {
    "G1_EVENT_CONFIRMATION",
    "G3_TEMPORAL_ALIGNMENT",
    "G4_SPATIAL_ALIGNMENT",
    "G7_REVIEW_GATE",
    "G9_PROMOTION_DECISION",
}

# Review decisions that are permitted only without promotion
NON_PROMOTING_DECISIONS = {
    "ACCEPT_AS_CONTEXTUAL_REFERENCE",
    "ACCEPT_AS_AUDITABLE_PROXY",
    "BLOCK_OPERATIONAL_PROMOTION",
    "REQUEST_ADDITIONAL_EVIDENCE",
    "REJECT_AS_INSUFFICIENT",
    "METHOD_REFERENCE_ONLY",
}


# ────────── file path constants ──────────

GAP_MATRIX_SCHEMA = PROJECT_ROOT / "datasets" / "schemas" / "ground_reference_gap_matrix_schema.csv"
GAP_MATRIX_REGISTRY = PROJECT_ROOT / "datasets" / "ground_reference_gap_matrix.csv"
REVIEW_GATE_SCHEMA = PROJECT_ROOT / "datasets" / "schemas" / "review_gate_reference_schema.csv"
REVIEW_GATE_REGISTRY = PROJECT_ROOT / "datasets" / "review_gate_reference_registry.csv"
PROMOTION_SCHEMA = PROJECT_ROOT / "datasets" / "schemas" / "reference_promotion_decision_schema.csv"
PROMOTION_REGISTRY = PROJECT_ROOT / "datasets" / "reference_promotion_decision_registry.csv"
CLOSURE_DOC = (
    PROJECT_ROOT
    / "docs"
    / "metodologia_cientifica"
    / "protocolo_c_fechamento_evidencias_ground_reference.md"
)
REVIEW_GATE_DOC = (
    PROJECT_ROOT
    / "docs"
    / "metodologia_cientifica"
    / "protocolo_c_revisao_supervisora_referencia.md"
)


# ────────── helpers ──────────

def _read_schema_fields(path: Path) -> set[str]:
    with open(path, "r", encoding="utf-8") as f:
        return {row["field"] for row in csv.DictReader(f)}


def _read_registry(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ════════════════════════════════════════
# 1. File existence
# ════════════════════════════════════════

class TestClosureFilesExist:
    def test_gap_matrix_schema_exists(self) -> None:
        assert GAP_MATRIX_SCHEMA.exists(), f"Missing: {GAP_MATRIX_SCHEMA}"

    def test_gap_matrix_registry_exists(self) -> None:
        assert GAP_MATRIX_REGISTRY.exists(), f"Missing: {GAP_MATRIX_REGISTRY}"

    def test_review_gate_schema_exists(self) -> None:
        assert REVIEW_GATE_SCHEMA.exists(), f"Missing: {REVIEW_GATE_SCHEMA}"

    def test_review_gate_registry_exists(self) -> None:
        assert REVIEW_GATE_REGISTRY.exists(), f"Missing: {REVIEW_GATE_REGISTRY}"

    def test_promotion_schema_exists(self) -> None:
        assert PROMOTION_SCHEMA.exists(), f"Missing: {PROMOTION_SCHEMA}"

    def test_promotion_registry_exists(self) -> None:
        assert PROMOTION_REGISTRY.exists(), f"Missing: {PROMOTION_REGISTRY}"

    def test_closure_document_exists(self) -> None:
        assert CLOSURE_DOC.exists(), f"Missing: {CLOSURE_DOC}"

    def test_review_gate_document_exists(self) -> None:
        assert REVIEW_GATE_DOC.exists(), f"Missing: {REVIEW_GATE_DOC}"

    def test_prior_registries_still_exist(self) -> None:
        for name in [
            "flood_event_candidate_registry.csv",
            "patch_event_reference_link_registry.csv",
            "contextual_reference_layer_registry.csv",
            "ground_reference_evidence_source_registry.csv",
        ]:
            p = PROJECT_ROOT / "datasets" / name
            assert p.exists(), f"Prior registry must not have been removed: {name}"


# ════════════════════════════════════════
# 2. Schema fields — gap matrix
# ════════════════════════════════════════

class TestGapMatrixSchemaFields:
    REQUIRED_FIELDS = {
        "gap_id",
        "region",
        "patch_scope",
        "patch_id",
        "event_id",
        "source_id",
        "current_reference_status",
        "current_closure_level",
        "missing_gate",
        "missing_evidence",
        "required_action",
        "evidence_priority",
        "methodological_risk",
        "expected_artifact",
        "allowed_next_step",
        "forbidden_next_step",
        "promotion_blocked",
        "blocked_reason",
        "notes",
    }

    @pytest.fixture
    def schema_fields(self) -> set[str]:
        return _read_schema_fields(GAP_MATRIX_SCHEMA)

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_required_field_present(self, schema_fields: set[str], field: str) -> None:
        assert field in schema_fields, f"Required field '{field}' missing from gap matrix schema"


# ════════════════════════════════════════
# 3. Schema fields — review gate
# ════════════════════════════════════════

class TestHumanReviewSchemaFields:
    REQUIRED_FIELDS = {
        "review_id",
        "link_id",
        "patch_id",
        "region",
        "event_id",
        "source_id",
        "reviewer_role",
        "review_date",
        "reviewed_materials",
        "temporal_consistency",
        "spatial_consistency",
        "source_consistency",
        "visual_consistency",
        "dino_support_used",
        "dino_support_limitation",
        "review_decision",
        "confidence_level",
        "promotion_allowed",
        "blocked_reason",
        "allowed_claim",
        "forbidden_claim",
        "notes",
    }

    @pytest.fixture
    def schema_fields(self) -> set[str]:
        return _read_schema_fields(REVIEW_GATE_SCHEMA)

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_required_field_present(self, schema_fields: set[str], field: str) -> None:
        assert field in schema_fields, f"Required field '{field}' missing from review gate schema"


# ════════════════════════════════════════
# 4. Schema fields — promotion decision
# ════════════════════════════════════════

class TestPromotionDecisionSchemaFields:
    REQUIRED_FIELDS = {
        "decision_id",
        "link_id",
        "review_id",
        "patch_id",
        "region",
        "event_id",
        "source_id",
        "gate_summary",
        "passed_gates",
        "failed_gates",
        "final_reference_status",
        "promotion_allowed",
        "decision_reason",
        "minimum_missing_evidence",
        "allowed_claim",
        "forbidden_claim",
        "protocol_b_reassessment_allowed",
        "notes",
    }

    @pytest.fixture
    def schema_fields(self) -> set[str]:
        return _read_schema_fields(PROMOTION_SCHEMA)

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_required_field_present(self, schema_fields: set[str], field: str) -> None:
        assert field in schema_fields, f"Required field '{field}' missing from promotion decision schema"


# ════════════════════════════════════════
# 5. Gap matrix registry — structure & controlled values
# ════════════════════════════════════════

class TestGapMatrixRegistryStructure:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(GAP_MATRIX_REGISTRY)

    def test_registry_has_rows(self, rows: list[dict]) -> None:
        assert len(rows) > 0, "Gap matrix registry must have at least one row"

    def test_gap_ids_unique(self, rows: list[dict]) -> None:
        ids = [row["gap_id"] for row in rows]
        assert len(ids) == len(set(ids)), "gap_id values must be unique"

    def test_current_closure_level_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("current_closure_level", "").strip()
            assert val in VALID_CURRENT_CLOSURE_LEVEL, (
                f"Row {i} ({row.get('gap_id')}): invalid current_closure_level '{val}'"
            )

    def test_missing_gate_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("missing_gate", "").strip()
            assert val in VALID_MISSING_GATE, (
                f"Row {i} ({row.get('gap_id')}): invalid missing_gate '{val}'"
            )

    def test_evidence_priority_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("evidence_priority", "").strip()
            assert val in VALID_EVIDENCE_PRIORITY, (
                f"Row {i} ({row.get('gap_id')}): invalid evidence_priority '{val}'"
            )

    def test_methodological_risk_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("methodological_risk", "").strip()
            assert val in VALID_METHODOLOGICAL_RISK, (
                f"Row {i} ({row.get('gap_id')}): invalid methodological_risk '{val}'"
            )

    def test_current_reference_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("current_reference_status", "").strip()
            assert val in VALID_CURRENT_REFERENCE_STATUS, (
                f"Row {i} ({row.get('gap_id')}): invalid current_reference_status '{val}'"
            )

    def test_promotion_blocked_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("promotion_blocked", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('gap_id')}): promotion_blocked must be true/false, got '{val}'"
            )


# ════════════════════════════════════════
# 6. Gap matrix registry — promotion guardrails
# ════════════════════════════════════════

class TestGapMatrixGuardrails:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(GAP_MATRIX_REGISTRY)

    def test_all_current_gaps_have_promotion_blocked(self, rows: list[dict]) -> None:
        """In the current REV-P state, all gaps must have promotion_blocked=true."""
        unblocked = [
            row["gap_id"]
            for row in rows
            if row.get("promotion_blocked", "").strip().lower() != "true"
        ]
        assert unblocked == [], (
            f"Unexpected promotion_blocked=false in gap matrix: {unblocked}. "
            "No gap has sufficient confirmed evidence in the current REV-P state."
        )

    def test_blocked_gaps_have_blocked_reason(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("promotion_blocked", "").strip().lower() == "true":
                reason = row.get("blocked_reason", "").strip()
                assert reason, (
                    f"Gap '{row['gap_id']}' has promotion_blocked=true but blocked_reason is empty"
                )

    def test_no_forbidden_terms_in_allowed_next_step(self, rows: list[dict]) -> None:
        forbidden = [
            "flood prediction",
            "flood detection",
            "ground truth operacional",
            "operational ground truth",
            "treino supervisionado",
            "training label",
            "flood label",
        ]
        for row in rows:
            step = row.get("allowed_next_step", "").lower()
            for term in forbidden:
                assert term not in step, (
                    f"Gap '{row['gap_id']}': allowed_next_step contains forbidden term '{term}'"
                )

    def test_three_revp_regions_present(self, rows: list[dict]) -> None:
        regions = {row.get("region", "") for row in rows}
        for expected in ["Recife", "Petrópolis", "Curitiba"]:
            assert expected in regions, f"Region '{expected}' missing from gap matrix registry"

    def test_no_evidence_open_row_claims_ground_truth(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("current_closure_level") == "EVIDENCE_OPEN":
                ref_status = row.get("current_reference_status", "")
                assert ref_status not in {"STRONG_REFERENCE_CANDIDATE", "OPERATIONAL_GROUND_TRUTH_NOT_ESTABLISHED"}, (
                    f"Gap '{row['gap_id']}': EVIDENCE_OPEN cannot have reference status '{ref_status}'"
                )

    def test_method_reference_rows_have_method_reference_only_status(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("patch_scope", "").strip() == "METHOD_REFERENCE_ONLY":
                ref_status = row.get("current_reference_status", "").strip()
                assert ref_status == "METHOD_REFERENCE_ONLY", (
                    f"Gap '{row['gap_id']}': patch_scope=METHOD_REFERENCE_ONLY "
                    f"but current_reference_status='{ref_status}'"
                )


# ════════════════════════════════════════
# 7. Review gate registry — structure & controlled values
# ════════════════════════════════════════

class TestHumanReviewRegistryStructure:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(REVIEW_GATE_REGISTRY)

    def test_registry_has_rows(self, rows: list[dict]) -> None:
        assert len(rows) > 0, "Review gate registry must have at least one row"

    def test_review_ids_unique(self, rows: list[dict]) -> None:
        ids = [row["review_id"] for row in rows]
        assert len(ids) == len(set(ids)), "review_id values must be unique"

    def test_reviewer_role_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("reviewer_role", "").strip()
            assert val in VALID_REVIEWER_ROLE, (
                f"Row {i} ({row.get('review_id')}): invalid reviewer_role '{val}'"
            )

    def test_temporal_consistency_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("temporal_consistency", "").strip()
            assert val in VALID_CONSISTENCY_STATUS, (
                f"Row {i} ({row.get('review_id')}): invalid temporal_consistency '{val}'"
            )

    def test_spatial_consistency_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("spatial_consistency", "").strip()
            assert val in VALID_CONSISTENCY_STATUS, (
                f"Row {i} ({row.get('review_id')}): invalid spatial_consistency '{val}'"
            )

    def test_source_consistency_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("source_consistency", "").strip()
            assert val in VALID_CONSISTENCY_STATUS, (
                f"Row {i} ({row.get('review_id')}): invalid source_consistency '{val}'"
            )

    def test_visual_consistency_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("visual_consistency", "").strip()
            assert val in VALID_CONSISTENCY_STATUS, (
                f"Row {i} ({row.get('review_id')}): invalid visual_consistency '{val}'"
            )

    def test_review_decision_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("review_decision", "").strip()
            assert val in VALID_REVIEW_DECISION, (
                f"Row {i} ({row.get('review_id')}): invalid review_decision '{val}'"
            )

    def test_confidence_level_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("confidence_level", "").strip()
            assert val in VALID_CONFIDENCE_LEVEL, (
                f"Row {i} ({row.get('review_id')}): invalid confidence_level '{val}'"
            )

    def test_dino_support_used_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("dino_support_used", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('review_id')}): dino_support_used must be true/false, got '{val}'"
            )

    def test_promotion_allowed_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("promotion_allowed", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('review_id')}): promotion_allowed must be true/false, got '{val}'"
            )


# ════════════════════════════════════════
# 8. Review gate registry — guardrails
# ════════════════════════════════════════

class TestHumanReviewGuardrails:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(REVIEW_GATE_REGISTRY)

    def test_all_current_reviews_have_promotion_blocked(self, rows: list[dict]) -> None:
        """All reviews in the current REV-P state must have promotion_allowed=false."""
        promoted = [
            row["review_id"]
            for row in rows
            if row.get("promotion_allowed", "").strip().lower() == "true"
        ]
        assert promoted == [], (
            f"Unexpected promotion_allowed=true in reviews: {promoted}. "
            "No review has sufficient confirmed evidence in the current REV-P state."
        )

    def test_blocked_reviews_have_blocked_reason(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "false":
                reason = row.get("blocked_reason", "").strip()
                assert reason, (
                    f"Review '{row['review_id']}' has promotion_allowed=false but blocked_reason is empty"
                )

    def test_not_executed_reviews_have_not_executed_decision(self, rows: list[dict]) -> None:
        """Placeholder reviews (NOT_EXECUTED reviewer_role) should block promotion."""
        for row in rows:
            if row.get("reviewer_role", "").strip() == "NOT_EXECUTED":
                decision = row.get("review_decision", "").strip()
                assert decision in {
                    "BLOCK_OPERATIONAL_PROMOTION",
                    "METHOD_REFERENCE_ONLY",
                }, (
                    f"Review '{row['review_id']}': NOT_EXECUTED reviewer "
                    f"has unexpected decision '{decision}'"
                )

    def test_dino_support_true_has_limitation_documented(self, rows: list[dict]) -> None:
        """When dino_support_used=true, dino_support_limitation must be non-trivial."""
        for row in rows:
            if row.get("dino_support_used", "").strip().lower() == "true":
                limitation = row.get("dino_support_limitation", "").strip()
                assert limitation and limitation.lower() != "n/a", (
                    f"Review '{row['review_id']}': dino_support_used=true "
                    f"but dino_support_limitation is trivial or empty"
                )

    def test_forbidden_claims_not_in_allowed_claim_field(self, rows: list[dict]) -> None:
        forbidden_patterns = [
            "flood prediction",
            "flood detection",
            "ground truth operacional",
            "operational ground truth",
            "flood label",
            "training label",
            "predição de enchente",
            "detecção de enchente",
        ]
        for row in rows:
            allowed = row.get("allowed_claim", "").lower()
            for pattern in forbidden_patterns:
                assert pattern not in allowed, (
                    f"Review '{row['review_id']}': allowed_claim contains forbidden pattern '{pattern}'"
                )

    def test_method_reference_reviews_not_promoted(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("review_decision", "").strip() == "METHOD_REFERENCE_ONLY":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Review '{row['review_id']}': METHOD_REFERENCE_ONLY decision "
                    f"but promotion_allowed='{promoted}'"
                )

    def test_placeholder_reviews_have_three_revp_regions(self, rows: list[dict]) -> None:
        regions = {
            row.get("region", "")
            for row in rows
            if row.get("reviewer_role", "") == "NOT_EXECUTED"
        }
        for expected in ["Recife", "Petrópolis", "Curitiba"]:
            assert expected in regions, (
                f"Region '{expected}' missing from placeholder review gate entries"
            )


# ════════════════════════════════════════
# 9. Promotion decision registry — structure & guardrails
# ════════════════════════════════════════

class TestPromotionDecisionRegistry:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(PROMOTION_REGISTRY)

    def test_registry_has_rows(self, rows: list[dict]) -> None:
        assert len(rows) > 0, "Promotion decision registry must have at least one row"

    def test_decision_ids_unique(self, rows: list[dict]) -> None:
        ids = [row["decision_id"] for row in rows]
        assert len(ids) == len(set(ids)), "decision_id values must be unique"

    def test_final_reference_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("final_reference_status", "").strip()
            assert val in VALID_FINAL_REFERENCE_STATUS, (
                f"Row {i} ({row.get('decision_id')}): invalid final_reference_status '{val}'"
            )

    def test_promotion_allowed_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("promotion_allowed", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('decision_id')}): promotion_allowed must be true/false, got '{val}'"
            )

    def test_protocol_b_reassessment_allowed_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("protocol_b_reassessment_allowed", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('decision_id')}): protocol_b_reassessment_allowed must be true/false, got '{val}'"
            )

    def test_all_current_decisions_block_promotion(self, rows: list[dict]) -> None:
        """In the current REV-P state, all decisions must have promotion_allowed=false."""
        promoted = [
            row["decision_id"]
            for row in rows
            if row.get("promotion_allowed", "").strip().lower() == "true"
        ]
        assert promoted == [], (
            f"Unexpected promotion_allowed=true in decisions: {promoted}. "
            "No decision has sufficient confirmed evidence in the current REV-P state."
        )

    def test_all_current_decisions_block_protocol_b(self, rows: list[dict]) -> None:
        """In the current REV-P state, all decisions must have protocol_b_reassessment_allowed=false."""
        allowed = [
            row["decision_id"]
            for row in rows
            if row.get("protocol_b_reassessment_allowed", "").strip().lower() == "true"
        ]
        assert allowed == [], (
            f"Unexpected protocol_b_reassessment_allowed=true in decisions: {allowed}. "
            "Protocol B remains blocked in the current REV-P state."
        )

    def test_blocked_decisions_have_decision_reason(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "false":
                reason = row.get("decision_reason", "").strip()
                assert reason, (
                    f"Decision '{row['decision_id']}' has promotion_allowed=false but decision_reason is empty"
                )

    def test_no_forbidden_terms_in_allowed_claim(self, rows: list[dict]) -> None:
        forbidden_patterns = [
            "flood prediction",
            "flood detection",
            "ground truth operacional",
            "operational ground truth",
            "flood label",
            "training label",
            "predição de enchente",
            "detecção de enchente",
        ]
        for row in rows:
            allowed = row.get("allowed_claim", "").lower()
            for pattern in forbidden_patterns:
                assert pattern not in allowed, (
                    f"Decision '{row['decision_id']}': allowed_claim contains forbidden pattern '{pattern}'"
                )

    def test_method_reference_decisions_have_not_assessed_gates(self, rows: list[dict]) -> None:
        for row in rows:
            if row.get("final_reference_status") == "METHOD_REFERENCE_ONLY":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Decision '{row['decision_id']}': METHOD_REFERENCE_ONLY "
                    f"but promotion_allowed='{promoted}'"
                )

    def test_three_revp_regions_have_decisions(self, rows: list[dict]) -> None:
        regions = {row.get("region", "") for row in rows}
        for expected in ["Recife", "Petrópolis", "Curitiba"]:
            assert expected in regions, (
                f"Region '{expected}' missing from promotion decision registry"
            )

    def test_decisions_without_g1_cannot_promote(self, rows: list[dict]) -> None:
        """Decisions listing G1_EVENT_CONFIRMATION as failed cannot have promotion_allowed=true."""
        for row in rows:
            failed = row.get("failed_gates", "")
            if "G1_EVENT_CONFIRMATION" in failed:
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Decision '{row['decision_id']}': G1_EVENT_CONFIRMATION failed "
                    f"but promotion_allowed='{promoted}'"
                )

    def test_decisions_without_g7_cannot_promote(self, rows: list[dict]) -> None:
        """Decisions listing G7_REVIEW_GATE as failed cannot have promotion_allowed=true."""
        for row in rows:
            failed = row.get("failed_gates", "")
            if "G7_REVIEW_GATE" in failed:
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Decision '{row['decision_id']}': G7_REVIEW_GATE failed "
                    f"but promotion_allowed='{promoted}'"
                )

    def test_decisions_without_minimum_missing_evidence_blocked(self, rows: list[dict]) -> None:
        """All blocked decisions must describe what minimum evidence is missing."""
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "false":
                missing = row.get("minimum_missing_evidence", "").strip()
                assert missing, (
                    f"Decision '{row['decision_id']}' blocked but minimum_missing_evidence is empty"
                )


# ════════════════════════════════════════
# 10. Document content — closure document
# ════════════════════════════════════════

class TestClosureDocumentContent:
    @pytest.fixture
    def doc(self) -> str:
        return _read_doc(CLOSURE_DOC)

    def test_document_mentions_gates(self, doc: str) -> None:
        assert "gate" in doc.lower() or "g1_event" in doc.lower(), (
            "Closure document must define promotion gates"
        )

    def test_document_mentions_g1_event_confirmation(self, doc: str) -> None:
        assert "g1_event_confirmation" in doc.lower() or "event_confirmation" in doc.lower(), (
            "Document must address G1 event confirmation gate"
        )

    def test_document_mentions_g7_review_gate(self, doc: str) -> None:
        assert "g7_review_gate" in doc.lower() or "review_gate" in doc.lower(), (
            "Document must address G7 review gate gate"
        )

    def test_document_mentions_closure_levels(self, doc: str) -> None:
        assert "evidence_open" in doc.lower() or "fechamento" in doc.lower(), (
            "Document must define evidence closure levels"
        )

    def test_document_mentions_ground_truth_not_established(self, doc: str) -> None:
        assert "not established" in doc.lower() or "não estabelecido" in doc.lower() or "bloqueado" in doc.lower(), (
            "Document must explicitly state ground truth is not established"
        )

    def test_document_mentions_dino_limitations(self, doc: str) -> None:
        assert "dino" in doc.lower(), (
            "Document must address DINOv2 limitations in the closure context"
        )

    def test_document_states_dino_cannot_close_g1(self, doc: str) -> None:
        assert "g1" in doc.lower(), (
            "Document must explicitly state DINO cannot close G1 (event confirmation)"
        )

    def test_document_mentions_protocol_b_blocked(self, doc: str) -> None:
        assert "protocolo b" in doc.lower() or "protocol b" in doc.lower(), (
            "Document must explain that Protocol B remains blocked"
        )

    def test_document_does_not_assert_flood_prediction(self, doc: str) -> None:
        forbidden_assertions = [
            "o rev-p prediz enchente",
            "o rev-p detecta enchente",
            "ground truth operacional disponível",
            "flood prediction disponível",
            "flood detection disponível",
            "rev-p has ground truth",
        ]
        doc_lower = doc.lower()
        for claim in forbidden_assertions:
            assert claim not in doc_lower, f"Document must not assert: '{claim}'"

    def test_document_metadata_only_statement(self, doc: str) -> None:
        assert "metadata" in doc.lower() or "metadata-only" in doc.lower(), (
            "Document must state this stage is metadata-only"
        )

    def test_document_mentions_sen1floods11_as_reference(self, doc: str) -> None:
        assert "sen1floods11" in doc.lower(), (
            "Closure document must reference Sen1Floods11 methodology"
        )


# ════════════════════════════════════════
# 11. Document content — review gate document
# ════════════════════════════════════════

class TestHumanReviewDocumentContent:
    @pytest.fixture
    def doc(self) -> str:
        return _read_doc(REVIEW_GATE_DOC)

    def test_document_defines_review_role(self, doc: str) -> None:
        assert "auditoria" in doc.lower() or "curadoria" in doc.lower() or "audit" in doc.lower(), (
            "Review gate document must define review as auditing/curation, not arbitrary labeling"
        )

    def test_document_lists_review_inputs(self, doc: str) -> None:
        assert "sentinel" in doc.lower() and "evento" in doc.lower(), (
            "Document must list required review inputs (patch Sentinel + event candidate)"
        )

    def test_document_defines_block_decision(self, doc: str) -> None:
        assert "block_operational_promotion" in doc.lower() or "bloqueio" in doc.lower(), (
            "Document must define BLOCK_OPERATIONAL_PROMOTION decision"
        )

    def test_document_states_dino_is_support_only(self, doc: str) -> None:
        assert "dino" in doc.lower() and "suporte" in doc.lower(), (
            "Document must state DINO is support only in reviews"
        )

    def test_document_states_dino_cannot_create_label(self, doc: str) -> None:
        assert "dino" in doc.lower(), (
            "Document must address DINOv2 limitations in review gate context"
        )

    def test_document_defines_g7_gate(self, doc: str) -> None:
        assert "g7" in doc.lower(), (
            "Review gate document must reference G7 gate"
        )

    def test_document_mentions_protocol_b_dependency(self, doc: str) -> None:
        assert "protocolo b" in doc.lower() or "protocol b" in doc.lower(), (
            "Document must explain relationship with Protocol B"
        )

    def test_document_does_not_assert_flood_prediction(self, doc: str) -> None:
        forbidden = [
            "o rev-p prediz enchente",
            "o rev-p detecta enchente",
            "ground truth operacional disponível",
        ]
        doc_lower = doc.lower()
        for claim in forbidden:
            assert claim not in doc_lower, f"Review gate document must not assert: '{claim}'"

    def test_document_mentions_future_annotation_as_separate_step(self, doc: str) -> None:
        assert "anotação" in doc.lower() or "annotation" in doc.lower(), (
            "Document must address future annotation as a distinct future step"
        )

    def test_document_explains_blocking_criteria(self, doc: str) -> None:
        assert "bloqueio" in doc.lower() or "block" in doc.lower(), (
            "Document must list review blocking criteria"
        )
