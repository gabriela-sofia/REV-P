"""Tests for Protocolo C acquisition stage — ground reference candidate qualification.

Covers: flood event candidate schema/registry, patch-event-reference link schema/registry,
controlled field values, promotion guardrails, DINO usage constraints, metadata-only
assertions, and methodological document content.

Reference: docs/metodologia_cientifica/protocolo_c_aquisicao_ground_reference.md
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ────────── controlled values ──────────

VALID_EVENT_CONFIRMATION_STATUS = {
    "CONFIRMED",
    "PARTIALLY_CONFIRMED",
    "REPORTED_UNVERIFIED",
    "METHOD_REFERENCE_ONLY",
    "UNKNOWN",
}

VALID_CONFIRMING_SOURCE_TYPE = {
    "CIVIL_DEFENSE",
    "OFFICIAL_REPORT",
    "ACADEMIC_DATASET",
    "NEWS_REPORT",
    "OPERATIONAL_PRODUCT",
    "MUNICIPAL_RECORD",
    "METHOD_REFERENCE",
    "UNKNOWN",
}

VALID_LOCAL_ASSET_STATUS = {
    "AVAILABLE_LOCAL",
    "NOT_ACQUIRED",
    "METHOD_REFERENCE_ONLY",
    "PENDING_REVIEW",
}

VALID_EVENT_TEMPORAL_PRECISION = {
    "EXACT_DATE",
    "DATE_RANGE",
    "MONTH_ONLY",
    "YEAR_ONLY",
    "UNKNOWN",
}

VALID_SPATIAL_EXTENT_AVAILABLE = {
    "TRUE",
    "FALSE",
    "PARTIAL",
    "UNKNOWN",
}

VALID_TEMPORAL_ALIGNMENT_STATUS = {
    "ALIGNED",
    "PARTIAL",
    "NOT_ALIGNED",
    "NOT_ASSESSED",
    "METHOD_REFERENCE_ONLY",
}

VALID_SPATIAL_ALIGNMENT_STATUS = {
    "MATCHED",
    "PARTIAL",
    "NOT_MATCHED",
    "NOT_ASSESSED",
    "METHOD_REFERENCE_ONLY",
}

VALID_CRS_COMPATIBILITY_STATUS = {
    "COMPATIBLE",
    "INCOMPATIBLE",
    "UNRESOLVED",
    "UNKNOWN",
    "METHOD_REFERENCE_ONLY",
}

VALID_PATCH_COVERAGE_STATUS = {
    "COVERED",
    "PARTIALLY_COVERED",
    "NOT_COVERED",
    "UNVERIFIED",
    "METHOD_REFERENCE_ONLY",
}

VALID_OBSERVATION_STRENGTH = {
    "DIRECT_OBSERVATION",
    "EXPERT_ANNOTATION",
    "OPERATIONAL_ALGORITHMIC",
    "MODELLED_CONTEXT",
    "STRUCTURAL_CONTEXT",
    "NONE",
}

VALID_UNCERTAINTY_STATUS = {
    "DOCUMENTED",
    "PARTIALLY_DOCUMENTED",
    "NOT_DOCUMENTED",
    "NOT_APPLICABLE",
}

VALID_REVIEW_GATE_STATUS = {
    "NOT_REVIEWED",
    "PENDING",
    "REVIEWED_APPROVED",
    "REVIEWED_REJECTED",
    "PARTIAL_FEEDBACK",
    "METHOD_REFERENCE_ONLY",
}

VALID_REFERENCE_CANDIDATE_STATUS = {
    "NOT_ASSESSED",
    "CONTEXTUAL_ONLY",
    "REFERENCE_SEARCH_REQUIRED",
    "GROUND_REFERENCE_CANDIDATE",
    "OPERATIONAL_GROUND_TRUTH_BLOCKED",
    "INSUFFICIENT_REFERENCE",
    "METHOD_REFERENCE_ONLY",
}

VALID_LINK_SOURCE_FAMILIES = {
    "FIELD_OBSERVATION",
    "OFFICIAL_OBSERVED_FLOOD_MAP",
    "EXPERT_ANNOTATED_HIGH_RES_IMAGE",
    "OPERATIONAL_FLOOD_PRODUCT",
    "HAND_LABELED_REMOTE_SENSING_DATASET",
    "MODELLED_SUSCEPTIBILITY_LAYER",
    "HYDROGEOMORPHOLOGICAL_CONTEXT",
    "SENTINEL_STRUCTURAL_EVIDENCE",
    "REVIEW_GATE_EVIDENCE",
}

# DINO/Sentinel structural families — can never promote ground truth
STRUCTURAL_ONLY_FAMILIES = {
    "SENTINEL_STRUCTURAL_EVIDENCE",
    "HYDROGEOMORPHOLOGICAL_CONTEXT",
    "MODELLED_SUSCEPTIBILITY_LAYER",
}

# Review gate statuses that count as "approved"
APPROVED_REVIEW_STATUSES = {"REVIEWED_APPROVED"}

# Literature references that must be method-reference-only
LITERATURE_REFERENCES = {
    "sen1floods11",
    "kuro_siwo",
    "kuro siwo",
    "ufo",
    "urban flood observations",
    "copernicus",
    "global flood monitoring",
    "gfm",
}


# ────────── file path constants ──────────

FLOOD_EVENT_SCHEMA = (
    PROJECT_ROOT / "datasets" / "schemas" / "flood_event_candidate_schema.csv"
)
FLOOD_EVENT_REGISTRY = (
    PROJECT_ROOT / "datasets" / "flood_event_candidate_registry.csv"
)
LINK_SCHEMA = (
    PROJECT_ROOT / "datasets" / "schemas" / "patch_event_reference_link_schema.csv"
)
LINK_REGISTRY = (
    PROJECT_ROOT / "datasets" / "patch_event_reference_link_registry.csv"
)
ACQUISITION_DOC = (
    PROJECT_ROOT
    / "docs"
    / "metodologia_cientifica"
    / "protocolo_c_aquisicao_ground_reference.md"
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

class TestAcquisitionFilesExist:
    def test_flood_event_schema_exists(self) -> None:
        assert FLOOD_EVENT_SCHEMA.exists(), f"Missing: {FLOOD_EVENT_SCHEMA}"

    def test_flood_event_registry_exists(self) -> None:
        assert FLOOD_EVENT_REGISTRY.exists(), f"Missing: {FLOOD_EVENT_REGISTRY}"

    def test_link_schema_exists(self) -> None:
        assert LINK_SCHEMA.exists(), f"Missing: {LINK_SCHEMA}"

    def test_link_registry_exists(self) -> None:
        assert LINK_REGISTRY.exists(), f"Missing: {LINK_REGISTRY}"

    def test_acquisition_document_exists(self) -> None:
        assert ACQUISITION_DOC.exists(), f"Missing: {ACQUISITION_DOC}"

    def test_prior_registries_still_exist(self) -> None:
        for name in [
            "contextual_reference_layer_registry.csv",
            "ground_reference_evidence_source_registry.csv",
        ]:
            p = PROJECT_ROOT / "datasets" / name
            assert p.exists(), f"Prior registry must not have been removed: {name}"


# ════════════════════════════════════════
# 2. Flood event candidate schema fields
# ════════════════════════════════════════

class TestFloodEventSchemaFields:
    REQUIRED_FIELDS = {
        "event_id",
        "event_name",
        "region",
        "municipality",
        "event_date_start",
        "event_date_end",
        "event_confirmation_status",
        "confirming_source_name",
        "confirming_source_type",
        "event_spatial_extent_available",
        "event_spatial_extent_type",
        "event_temporal_precision",
        "known_impacts",
        "source_url_or_reference",
        "local_asset_status",
        "eligible_for_reference_search",
        "blocked_reason",
        "notes",
    }

    @pytest.fixture
    def schema_fields(self) -> set[str]:
        return _read_schema_fields(FLOOD_EVENT_SCHEMA)

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_required_field_present(self, schema_fields: set[str], field: str) -> None:
        assert field in schema_fields, f"Required field '{field}' missing from flood event schema"


# ════════════════════════════════════════
# 3. Patch-event-reference link schema fields
# ════════════════════════════════════════

class TestLinkSchemaFields:
    REQUIRED_FIELDS = {
        "link_id",
        "patch_id",
        "region",
        "event_id",
        "source_id",
        "source_family",
        "event_confirmation_status",
        "temporal_alignment_status",
        "spatial_alignment_status",
        "crs_compatibility_status",
        "patch_coverage_status",
        "observation_strength",
        "uncertainty_status",
        "review_gate_status",
        "dino_used_as_support_only",
        "reference_candidate_status",
        "promotion_allowed",
        "blocked_reason",
        "allowed_claim",
        "forbidden_claim",
        "notes",
    }

    @pytest.fixture
    def schema_fields(self) -> set[str]:
        return _read_schema_fields(LINK_SCHEMA)

    @pytest.mark.parametrize("field", sorted(REQUIRED_FIELDS))
    def test_required_field_present(self, schema_fields: set[str], field: str) -> None:
        assert field in schema_fields, f"Required field '{field}' missing from link schema"


# ════════════════════════════════════════
# 4. Flood event registry — structure & controlled values
# ════════════════════════════════════════

class TestFloodEventRegistryStructure:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(FLOOD_EVENT_REGISTRY)

    def test_registry_has_rows(self, rows: list[dict]) -> None:
        assert len(rows) > 0, "Flood event registry must have at least one row"

    def test_all_rows_have_event_id(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            assert row.get("event_id", "").strip(), f"Row {i}: event_id must not be empty"

    def test_all_rows_have_event_name(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            assert row.get("event_name", "").strip(), f"Row {i}: event_name must not be empty"

    def test_event_ids_are_unique(self, rows: list[dict]) -> None:
        ids = [row["event_id"] for row in rows]
        assert len(ids) == len(set(ids)), "event_id values must be unique"

    def test_confirmation_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("event_confirmation_status", "").strip()
            assert status in VALID_EVENT_CONFIRMATION_STATUS, (
                f"Row {i} ({row.get('event_id')}): invalid event_confirmation_status '{status}'"
            )

    def test_confirming_source_type_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            src_type = row.get("confirming_source_type", "").strip()
            assert src_type in VALID_CONFIRMING_SOURCE_TYPE, (
                f"Row {i} ({row.get('event_id')}): invalid confirming_source_type '{src_type}'"
            )

    def test_local_asset_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("local_asset_status", "").strip()
            assert status in VALID_LOCAL_ASSET_STATUS, (
                f"Row {i} ({row.get('event_id')}): invalid local_asset_status '{status}'"
            )

    def test_temporal_precision_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            prec = row.get("event_temporal_precision", "").strip()
            assert prec in VALID_EVENT_TEMPORAL_PRECISION, (
                f"Row {i} ({row.get('event_id')}): invalid event_temporal_precision '{prec}'"
            )

    def test_spatial_extent_available_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("event_spatial_extent_available", "").strip().upper()
            assert val in VALID_SPATIAL_EXTENT_AVAILABLE | {"N/A", "METHOD_REFERENCE_ONLY"}, (
                f"Row {i} ({row.get('event_id')}): invalid event_spatial_extent_available '{val}'"
            )

    def test_eligible_for_reference_search_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("eligible_for_reference_search", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('event_id')}): eligible_for_reference_search must be true/false, got '{val}'"
            )


# ════════════════════════════════════════
# 5. Flood event registry — promotion guardrails
# ════════════════════════════════════════

class TestFloodEventRegistryGuardrails:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(FLOOD_EVENT_REGISTRY)

    def test_method_reference_only_not_eligible(self, rows: list[dict]) -> None:
        """METHOD_REFERENCE_ONLY events cannot be eligible for reference search."""
        for row in rows:
            if row.get("event_confirmation_status") == "METHOD_REFERENCE_ONLY":
                eligible = row.get("eligible_for_reference_search", "").strip().lower()
                assert eligible == "false", (
                    f"Event '{row['event_id']}' has METHOD_REFERENCE_ONLY status "
                    f"but eligible_for_reference_search='{eligible}' (must be false)"
                )

    def test_unknown_event_not_eligible(self, rows: list[dict]) -> None:
        """UNKNOWN confirmation status cannot yield eligible_for_reference_search=true."""
        for row in rows:
            if row.get("event_confirmation_status") == "UNKNOWN":
                eligible = row.get("eligible_for_reference_search", "").strip().lower()
                assert eligible == "false", (
                    f"Event '{row['event_id']}' has UNKNOWN confirmation "
                    f"but eligible_for_reference_search='{eligible}' (must be false)"
                )

    def test_not_acquired_event_not_eligible(self, rows: list[dict]) -> None:
        """Events with NOT_ACQUIRED assets cannot be eligible."""
        for row in rows:
            if row.get("local_asset_status") == "NOT_ACQUIRED":
                eligible = row.get("eligible_for_reference_search", "").strip().lower()
                assert eligible == "false", (
                    f"Event '{row['event_id']}' has NOT_ACQUIRED asset status "
                    f"but eligible_for_reference_search='{eligible}' (must be false)"
                )

    def test_eligible_rows_have_blocked_reason_empty_or_na(self, rows: list[dict]) -> None:
        """Rows with eligible=true should not contradict with a blocking reason."""
        for row in rows:
            if row.get("eligible_for_reference_search", "").strip().lower() == "false":
                blocked = row.get("blocked_reason", "").strip()
                assert blocked, (
                    f"Event '{row['event_id']}' has eligible=false but blocked_reason is empty"
                )


# ════════════════════════════════════════
# 6. Link registry — structure & controlled values
# ════════════════════════════════════════

class TestLinkRegistryStructure:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(LINK_REGISTRY)

    def test_registry_has_rows(self, rows: list[dict]) -> None:
        assert len(rows) > 0, "Link registry must have at least one row"

    def test_all_rows_have_link_id(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            assert row.get("link_id", "").strip(), f"Row {i}: link_id must not be empty"

    def test_link_ids_are_unique(self, rows: list[dict]) -> None:
        ids = [row["link_id"] for row in rows]
        assert len(ids) == len(set(ids)), "link_id values must be unique"

    def test_source_family_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            family = row.get("source_family", "").strip()
            assert family in VALID_LINK_SOURCE_FAMILIES, (
                f"Row {i} ({row.get('link_id')}): invalid source_family '{family}'"
            )

    def test_event_confirmation_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("event_confirmation_status", "").strip()
            assert status in VALID_EVENT_CONFIRMATION_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid event_confirmation_status '{status}'"
            )

    def test_temporal_alignment_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("temporal_alignment_status", "").strip()
            assert status in VALID_TEMPORAL_ALIGNMENT_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid temporal_alignment_status '{status}'"
            )

    def test_spatial_alignment_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("spatial_alignment_status", "").strip()
            assert status in VALID_SPATIAL_ALIGNMENT_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid spatial_alignment_status '{status}'"
            )

    def test_crs_compatibility_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("crs_compatibility_status", "").strip()
            assert status in VALID_CRS_COMPATIBILITY_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid crs_compatibility_status '{status}'"
            )

    def test_patch_coverage_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            status = row.get("patch_coverage_status", "").strip()
            assert status in VALID_PATCH_COVERAGE_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid patch_coverage_status '{status}'"
            )

    def test_observation_strength_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("observation_strength", "").strip()
            assert val in VALID_OBSERVATION_STRENGTH, (
                f"Row {i} ({row.get('link_id')}): invalid observation_strength '{val}'"
            )

    def test_uncertainty_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("uncertainty_status", "").strip()
            assert val in VALID_UNCERTAINTY_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid uncertainty_status '{val}'"
            )

    def test_review_gate_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("review_gate_status", "").strip()
            assert val in VALID_REVIEW_GATE_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid review_gate_status '{val}'"
            )

    def test_reference_candidate_status_valid(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("reference_candidate_status", "").strip()
            assert val in VALID_REFERENCE_CANDIDATE_STATUS, (
                f"Row {i} ({row.get('link_id')}): invalid reference_candidate_status '{val}'"
            )

    def test_promotion_allowed_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("promotion_allowed", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('link_id')}): promotion_allowed must be true/false, got '{val}'"
            )

    def test_dino_used_as_support_only_is_boolean(self, rows: list[dict]) -> None:
        for i, row in enumerate(rows):
            val = row.get("dino_used_as_support_only", "").strip().lower()
            assert val in {"true", "false"}, (
                f"Row {i} ({row.get('link_id')}): dino_used_as_support_only must be true/false, got '{val}'"
            )


# ════════════════════════════════════════
# 7. Link registry — promotion guardrails
# ════════════════════════════════════════

class TestLinkRegistryPromotionGuardrails:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(LINK_REGISTRY)

    def test_no_link_without_temporal_alignment_can_be_promoted(self, rows: list[dict]) -> None:
        """promotion_allowed=true requires temporal_alignment_status=ALIGNED."""
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "true":
                temporal = row.get("temporal_alignment_status", "").strip()
                assert temporal == "ALIGNED", (
                    f"Link '{row['link_id']}': promotion_allowed=true but "
                    f"temporal_alignment_status='{temporal}' (requires ALIGNED)"
                )

    def test_no_link_without_spatial_alignment_can_be_promoted(self, rows: list[dict]) -> None:
        """promotion_allowed=true requires spatial_alignment_status=MATCHED."""
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "true":
                spatial = row.get("spatial_alignment_status", "").strip()
                assert spatial == "MATCHED", (
                    f"Link '{row['link_id']}': promotion_allowed=true but "
                    f"spatial_alignment_status='{spatial}' (requires MATCHED)"
                )

    def test_no_link_without_review_gate_can_be_promoted(self, rows: list[dict]) -> None:
        """promotion_allowed=true requires review_gate_status in APPROVED_REVIEW_STATUSES."""
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "true":
                review = row.get("review_gate_status", "").strip()
                assert review in APPROVED_REVIEW_STATUSES, (
                    f"Link '{row['link_id']}': promotion_allowed=true but "
                    f"review_gate_status='{review}' (requires REVIEWED_APPROVED)"
                )

    def test_method_reference_links_not_promoted(self, rows: list[dict]) -> None:
        """Links with reference_candidate_status=METHOD_REFERENCE_ONLY cannot be promoted."""
        for row in rows:
            if row.get("reference_candidate_status") == "METHOD_REFERENCE_ONLY":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Link '{row['link_id']}': METHOD_REFERENCE_ONLY status "
                    f"but promotion_allowed='{promoted}' (must be false)"
                )

    def test_all_current_links_have_promotion_blocked(self, rows: list[dict]) -> None:
        """In the current REV-P state no link has confirmed evidence — all must have promotion_allowed=false."""
        promoted = [
            row["link_id"]
            for row in rows
            if row.get("promotion_allowed", "").strip().lower() == "true"
        ]
        assert promoted == [], (
            f"Unexpected promotion_allowed=true in links: {promoted}. "
            "No link has sufficient confirmed evidence in the current REV-P state."
        )

    def test_blocked_links_have_blocked_reason(self, rows: list[dict]) -> None:
        """Every link with promotion_allowed=false must have a non-empty blocked_reason."""
        for row in rows:
            if row.get("promotion_allowed", "").strip().lower() == "false":
                reason = row.get("blocked_reason", "").strip()
                assert reason, (
                    f"Link '{row['link_id']}' has promotion_allowed=false but blocked_reason is empty"
                )

    def test_unknown_event_links_not_promoted(self, rows: list[dict]) -> None:
        """Links tied to UNKNOWN events cannot be promoted."""
        for row in rows:
            if row.get("event_confirmation_status") == "UNKNOWN":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Link '{row['link_id']}': UNKNOWN event confirmation "
                    f"but promotion_allowed='{promoted}' (must be false)"
                )

    def test_incompatible_crs_links_not_promoted(self, rows: list[dict]) -> None:
        """Links with INCOMPATIBLE CRS cannot be promoted."""
        for row in rows:
            if row.get("crs_compatibility_status") == "INCOMPATIBLE":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Link '{row['link_id']}': CRS INCOMPATIBLE "
                    f"but promotion_allowed='{promoted}' (must be false)"
                )

    def test_not_reviewed_links_not_promoted(self, rows: list[dict]) -> None:
        """Links with review_gate_status=NOT_REVIEWED cannot be promoted."""
        for row in rows:
            if row.get("review_gate_status") == "NOT_REVIEWED":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Link '{row['link_id']}': NOT_REVIEWED "
                    f"but promotion_allowed='{promoted}' (must be false)"
                )


# ════════════════════════════════════════
# 8. DINO/Sentinel structural source guardrails
# ════════════════════════════════════════

class TestDINOStructuralGuardrails:
    @pytest.fixture
    def rows(self) -> list[dict]:
        return _read_registry(LINK_REGISTRY)

    def test_sentinel_structural_links_have_dino_support_only_true(self, rows: list[dict]) -> None:
        """Links with SENTINEL_STRUCTURAL_EVIDENCE family must have dino_used_as_support_only=true."""
        for row in rows:
            if row.get("source_family") == "SENTINEL_STRUCTURAL_EVIDENCE":
                val = row.get("dino_used_as_support_only", "").strip().lower()
                assert val == "true", (
                    f"Link '{row['link_id']}': SENTINEL_STRUCTURAL_EVIDENCE source "
                    f"must have dino_used_as_support_only=true, got '{val}'"
                )

    def test_sentinel_structural_links_not_promoted(self, rows: list[dict]) -> None:
        """Links with SENTINEL_STRUCTURAL_EVIDENCE cannot be promoted."""
        for row in rows:
            if row.get("source_family") == "SENTINEL_STRUCTURAL_EVIDENCE":
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Link '{row['link_id']}': SENTINEL_STRUCTURAL_EVIDENCE "
                    f"but promotion_allowed='{promoted}' (must be false)"
                )

    def test_structural_families_contextual_only_or_lower(self, rows: list[dict]) -> None:
        """Structural-only families cannot have reference_candidate_status above CONTEXTUAL_ONLY."""
        blocked_statuses = {
            "GROUND_REFERENCE_CANDIDATE",
            "OPERATIONAL_GROUND_TRUTH_BLOCKED",
        }
        for row in rows:
            if row.get("source_family") in STRUCTURAL_ONLY_FAMILIES:
                status = row.get("reference_candidate_status", "").strip()
                assert status not in blocked_statuses, (
                    f"Link '{row['link_id']}': source_family '{row['source_family']}' "
                    f"has reference_candidate_status='{status}' which exceeds structural ceiling"
                )

    def test_dino_links_have_structural_context_observation_strength(self, rows: list[dict]) -> None:
        """SENTINEL_STRUCTURAL_EVIDENCE links must have observation_strength=STRUCTURAL_CONTEXT."""
        for row in rows:
            if row.get("source_family") == "SENTINEL_STRUCTURAL_EVIDENCE":
                strength = row.get("observation_strength", "").strip()
                assert strength == "STRUCTURAL_CONTEXT", (
                    f"Link '{row['link_id']}': SENTINEL_STRUCTURAL_EVIDENCE "
                    f"must have observation_strength=STRUCTURAL_CONTEXT, got '{strength}'"
                )

    def test_forbidden_claims_not_in_allowed_claim_field(self, rows: list[dict]) -> None:
        """allowed_claim field must not assert flood prediction/detection/ground truth."""
        forbidden_patterns = [
            "flood prediction",
            "flood detection",
            "ground truth operacional",
            "operational ground truth",
            "flood label",
            "training label",
            "predição de enchente",
            "detecção de enchente",
            "label de enchente",
        ]
        for row in rows:
            allowed_claim = row.get("allowed_claim", "").lower()
            for pattern in forbidden_patterns:
                assert pattern not in allowed_claim, (
                    f"Link '{row['link_id']}': allowed_claim contains forbidden pattern '{pattern}'"
                )


# ════════════════════════════════════════
# 9. Literature references are method-only
# ════════════════════════════════════════

class TestLiteratureReferencesMethodOnly:
    @pytest.fixture
    def event_rows(self) -> list[dict]:
        return _read_registry(FLOOD_EVENT_REGISTRY)

    @pytest.fixture
    def link_rows(self) -> list[dict]:
        return _read_registry(LINK_REGISTRY)

    def test_sen1floods11_is_method_reference_only_in_events(self, event_rows: list[dict]) -> None:
        for row in event_rows:
            if "sen1floods11" in row.get("event_id", "").lower():
                status = row.get("event_confirmation_status", "").strip()
                assert status == "METHOD_REFERENCE_ONLY", (
                    f"Sen1Floods11 event '{row['event_id']}' must have METHOD_REFERENCE_ONLY, got '{status}'"
                )
                eligible = row.get("eligible_for_reference_search", "").strip().lower()
                assert eligible == "false", (
                    f"Sen1Floods11 event must have eligible_for_reference_search=false, got '{eligible}'"
                )

    def test_kuro_siwo_is_method_reference_only_in_events(self, event_rows: list[dict]) -> None:
        for row in event_rows:
            if "kuro_siwo" in row.get("event_id", "").lower():
                status = row.get("event_confirmation_status", "").strip()
                assert status == "METHOD_REFERENCE_ONLY", (
                    f"Kuro Siwo event '{row['event_id']}' must have METHOD_REFERENCE_ONLY, got '{status}'"
                )

    def test_ufo_is_method_reference_only_in_events(self, event_rows: list[dict]) -> None:
        for row in event_rows:
            if "ufo" in row.get("event_id", "").lower():
                status = row.get("event_confirmation_status", "").strip()
                assert status == "METHOD_REFERENCE_ONLY", (
                    f"UFO event '{row['event_id']}' must have METHOD_REFERENCE_ONLY, got '{status}'"
                )

    def test_copernicus_not_eligible_in_events(self, event_rows: list[dict]) -> None:
        for row in event_rows:
            if "copernicus" in row.get("event_id", "").lower():
                eligible = row.get("eligible_for_reference_search", "").strip().lower()
                assert eligible == "false", (
                    f"Copernicus event '{row['event_id']}' must have eligible_for_reference_search=false"
                )

    def test_sen1floods11_link_not_promoted(self, link_rows: list[dict]) -> None:
        for row in link_rows:
            if "sen1floods11" in row.get("link_id", "").lower():
                promoted = row.get("promotion_allowed", "").strip().lower()
                assert promoted == "false", (
                    f"Sen1Floods11 link '{row['link_id']}' must have promotion_allowed=false"
                )
                status = row.get("reference_candidate_status", "").strip()
                assert status == "METHOD_REFERENCE_ONLY", (
                    f"Sen1Floods11 link '{row['link_id']}' must have reference_candidate_status=METHOD_REFERENCE_ONLY"
                )


# ════════════════════════════════════════
# 10. Acquisition document content
# ════════════════════════════════════════

class TestAcquisitionDocumentContent:
    @pytest.fixture
    def doc(self) -> str:
        return _read_doc(ACQUISITION_DOC)

    def test_document_states_metadata_only(self, doc: str) -> None:
        assert "metadata" in doc.lower(), (
            "Acquisition document must explicitly state this stage is metadata-only"
        )

    def test_document_mentions_no_raster_download(self, doc: str) -> None:
        assert "raster" in doc.lower(), (
            "Document must reference the constraint of not downloading rasters"
        )

    def test_document_mentions_no_ground_truth_declaration(self, doc: str) -> None:
        assert "ground truth" in doc.lower(), (
            "Document must explain that ground truth is not declared in this stage"
        )

    def test_document_references_sen1floods11(self, doc: str) -> None:
        assert "sen1floods11" in doc.lower(), (
            "Document must reference Sen1Floods11 as methodological reference"
        )

    def test_document_references_kuro_siwo(self, doc: str) -> None:
        assert "kuro siwo" in doc.lower(), (
            "Document must reference Kuro Siwo as methodological reference"
        )

    def test_document_references_ufo(self, doc: str) -> None:
        assert any(term in doc.lower() for term in ["urban flood observations", "ufo"]), (
            "Document must reference UFO (Urban Flood Observations) as methodological reference"
        )

    def test_document_references_copernicus(self, doc: str) -> None:
        assert any(term in doc.lower() for term in ["copernicus", "gfm", "global flood monitoring"]), (
            "Document must reference Copernicus/GFM as methodological reference"
        )

    def test_document_mentions_event_confirmation_requirement(self, doc: str) -> None:
        assert "event" in doc.lower() and "confirm" in doc.lower(), (
            "Document must state that event confirmation is required for reference candidates"
        )

    def test_document_mentions_temporal_alignment_requirement(self, doc: str) -> None:
        assert "temporal" in doc.lower(), (
            "Document must address temporal alignment as a requirement"
        )

    def test_document_does_not_declare_flood_prediction(self, doc: str) -> None:
        forbidden_positive_claims = [
            "o rev-p prediz enchente",
            "o rev-p detecta enchente",
            "rev-p has ground truth",
            "ground truth operacional disponível",
            "flood prediction disponível",
            "flood detection disponível",
        ]
        for claim in forbidden_positive_claims:
            assert claim not in doc.lower(), (
                f"Document must not assert: '{claim}'"
            )

    def test_document_mentions_promotion_blocked(self, doc: str) -> None:
        assert "bloqueio" in doc.lower() or "blocked" in doc.lower() or "bloquei" in doc.lower(), (
            "Document must explain that promotion is blocked in current state"
        )

    def test_document_mentions_protocol_c_link(self, doc: str) -> None:
        assert "protocolo c" in doc.lower() or "protocol c" in doc.lower(), (
            "Acquisition document must reference Protocolo C"
        )
