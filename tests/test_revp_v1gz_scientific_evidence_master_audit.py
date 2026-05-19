"""Tests for REV-P v1gz: Scientific Evidence Master Audit."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
V1GZ_DIR = ROOT / "local_runs" / "dino_embeddings" / "v1gz"

# Expected output files
EXPECTED_OUTPUTS = {
    "claim_to_evidence_crosswalk_v1gz.csv": {
        "min_rows": 15,
        "required_cols": ["claim_id", "claim_description", "status", "blocking_reason"],
    },
    "evidence_strength_by_claim_v1gz.csv": {
        "min_rows": 8,
        "required_cols": ["claim_id", "evidence_status", "tcc_ready"],
    },
    "remaining_scientific_gaps_v1gz.csv": {
        "min_rows": 4,
        "required_cols": ["gap_id", "category", "description"],
    },
    "tcc_result_readiness_matrix_v1gz.csv": {
        "min_rows": 10,
        "required_cols": ["tcc_section", "readiness", "dependencies"],
    },
    "scientific_evidence_master_summary_v1gz.json": {
        "required_keys": [
            "timestamp",
            "phase",
            "allowed_claims_count",
            "figures_ready",
            "tables_ready",
            "methodological_guardrails",
        ]
    },
}

FORBIDDEN_CLAIM_IDS = {
    "vulnerability_prediction",
    "flood_susceptibility_classification",
    "flood_detection",
    "ground_truth_validation",
    "supervised_model_performance",
    "dino_as_classifier",
    "multimodal_execution",
    "risk_classification",
    "target_variable",
    "causal_inference",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV file."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    """Read JSON file."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestV1GZOutputs:
    """Test v1gz output files existence and structure."""

    @pytest.mark.parametrize("filename", EXPECTED_OUTPUTS.keys())
    def test_output_files_exist(self, filename: str) -> None:
        """Test that all expected output files exist."""
        filepath = V1GZ_DIR / filename
        assert filepath.exists(), f"Output file not found: {filename}"

    def test_claim_crosswalk_valid(self) -> None:
        """Test claim-evidence crosswalk has valid structure."""
        path = V1GZ_DIR / "claim_to_evidence_crosswalk_v1gz.csv"
        rows = read_csv(path)
        assert len(rows) >= EXPECTED_OUTPUTS[path.name]["min_rows"]

        # Check required columns
        if rows:
            for col in EXPECTED_OUTPUTS[path.name]["required_cols"]:
                assert col in rows[0].keys(), f"Missing column: {col}"

        # Check that allowed claims have READY or PARTIAL status
        allowed_rows = [r for r in rows if r.get("status") in ["READY", "PARTIAL"]]
        assert len(allowed_rows) > 0, "No allowed claims with READY/PARTIAL status"

    def test_forbidden_claims_blocked(self) -> None:
        """Test that forbidden claims are explicitly BLOCKED."""
        path = V1GZ_DIR / "claim_to_evidence_crosswalk_v1gz.csv"
        rows = read_csv(path)

        blocked_claims = {r["claim_id"]: r for r in rows if r.get("status") == "BLOCKED"}

        # Check that all forbidden claims are present and blocked
        for claim_id in FORBIDDEN_CLAIM_IDS:
            assert claim_id in blocked_claims, f"Forbidden claim not found/blocked: {claim_id}"
            assert blocked_claims[claim_id]["blocking_reason"], f"No reason for blocking {claim_id}"

    def test_evidence_strength_claims_have_status(self) -> None:
        """Test that evidence strength matrix has valid statuses."""
        path = V1GZ_DIR / "evidence_strength_by_claim_v1gz.csv"
        rows = read_csv(path)

        valid_statuses = {"READY", "PARTIAL", "BLOCKED"}
        for row in rows:
            status = row.get("evidence_status", "").strip()
            assert status in valid_statuses, f"Invalid status '{status}' for {row.get('claim_id')}"

    def test_scientific_gaps_documented(self) -> None:
        """Test that scientific gaps are documented with justifications."""
        path = V1GZ_DIR / "remaining_scientific_gaps_v1gz.csv"
        rows = read_csv(path)

        assert len(rows) >= 3, "Expected at least 3 documented gaps"

        # Each gap should have category, description, and impact
        for row in rows:
            assert row.get("category"), f"Gap {row.get('gap_id')} missing category"
            assert row.get("description"), f"Gap {row.get('gap_id')} missing description"
            assert row.get("impact"), f"Gap {row.get('gap_id')} missing impact"
            assert row.get("mitigation"), f"Gap {row.get('gap_id')} missing mitigation"

    def test_tcc_readiness_sections(self) -> None:
        """Test TCC readiness matrix covers key sections."""
        path = V1GZ_DIR / "tcc_result_readiness_matrix_v1gz.csv"
        rows = read_csv(path)

        assert len(rows) >= 10, "Expected at least 10 TCC sections"

        # Check that sections have valid readiness values
        valid_readiness = {"READY", "PARTIAL", "BLOCKED", "INDEPENDENT", "OPTIONAL"}
        for row in rows:
            readiness = row.get("readiness", "").strip()
            assert readiness in valid_readiness, f"Invalid readiness '{readiness}' for {row.get('tcc_section')}"

    def test_summary_json_structure(self) -> None:
        """Test that summary JSON has required structure."""
        path = V1GZ_DIR / "scientific_evidence_master_summary_v1gz.json"
        data = read_json(path)

        # Check required top-level keys
        for key in EXPECTED_OUTPUTS[path.name]["required_keys"]:
            assert key in data, f"Missing key in summary: {key}"

        # Check phase
        assert data.get("phase") == "v1gz", "Invalid phase in summary"

        # Check counts
        assert data.get("figures_ready", 0) == 5, "Expected 5 ready figures"
        assert data.get("tables_ready", 0) == 6, "Expected 6 ready tables"
        assert data.get("corpus_size") == 12, "Expected corpus size of 12"
        assert data.get("n_regions") == 3, "Expected 3 regions"

    def test_methodological_guardrails_enforced(self) -> None:
        """Test that methodological guardrails are enforced."""
        path = V1GZ_DIR / "scientific_evidence_master_summary_v1gz.json"
        data = read_json(path)

        guardrails = data.get("methodological_guardrails", {})

        # All guardrails should be True
        required_guardrails = {
            "review_only": True,
            "no_labels": True,
            "no_targets": True,
            "no_predictions": True,
            "no_ground_truth_claim": True,
            "no_cluster_as_class": True,
            "gis_contextual_only": True,
            "multimodal_disabled": True,
            "no_heavy_files_in_git": True,
        }

        for key, expected in required_guardrails.items():
            assert guardrails.get(key) == expected, f"Guardrail {key} not properly enforced"


class TestV1GYCaptions:
    """Test v1gy figure/table captions for forbidden terms."""

    def test_no_forbidden_caption_terms(self) -> None:
        """Test that no forbidden caption terms appear in manifest."""
        v1gy_manifest = ROOT / "local_runs" / "tcc_figures" / "v1gy" / "tcc_visual_evidence_manifest_v1gy.csv"

        if not v1gy_manifest.exists():
            pytest.skip("v1gy manifest not found")

        rows = read_csv(v1gy_manifest)

        forbidden_terms = {
            "prediction", "predictive", "detection", "ground truth", "ground-truth",
            "classification", "classify", "risk classification", "risk assessment",
            "class", "label", "labeling", "supervised", "target", "causal",
            "validation", "validates", "proven", "proves",
        }

        violations = []
        for row in rows:
            caption = row.get("caption_draft_pt", "").lower()
            for term in forbidden_terms:
                if term in caption:
                    violations.append(
                        f"Caption {row.get('artifact_id')}: contains '{term}'"
                    )
                    break  # One violation per artifact is enough

        assert not violations, f"Caption violations found: {violations}"

    def test_figure_table_status_ready(self) -> None:
        """Test that figures and tables have READY status."""
        v1gy_manifest = ROOT / "local_runs" / "tcc_figures" / "v1gy" / "tcc_visual_evidence_manifest_v1gy.csv"

        if not v1gy_manifest.exists():
            pytest.skip("v1gy manifest not found")

        rows = read_csv(v1gy_manifest)

        for row in rows:
            status = row.get("status", "").strip()
            assert status == "READY", f"{row.get('artifact_id')} has status {status}, expected READY"


class TestV1GWHumanReview:
    """Test v1gw human review formalization."""

    def test_review_candidates_not_labels(self) -> None:
        """Test that review candidates do not create labels/classes."""
        v1gw_metadata = ROOT / "local_runs" / "dino_embeddings" / "v1gw" / "review_candidates_metadata_v1gw.json"

        if not v1gw_metadata.exists():
            pytest.skip("v1gw metadata not found")

        data = read_json(v1gw_metadata)

        # Check that review is formalized as methodological stage, not labeling
        guardrails = data.get("methodological_guardrails", {})
        assert guardrails.get("review_only") is True, "Review should be review-only"
        assert guardrails.get("review_does_not_create_labels") is True, "Review should not create labels"
        assert guardrails.get("candidates_are_not_classified") is True, "Candidates should not be classified"

    def test_review_protocol_exists(self) -> None:
        """Test that review protocol documentation exists."""
        v1gw_protocol = ROOT / "local_runs" / "dino_embeddings" / "v1gw" / "review_protocol_v1gw.md"

        assert v1gw_protocol.exists(), "Review protocol documentation missing"

        content = v1gw_protocol.read_text(encoding="utf-8")
        assert len(content) > 100, "Review protocol is too minimal"
        assert "review" in content.lower(), "Protocol should mention review"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
