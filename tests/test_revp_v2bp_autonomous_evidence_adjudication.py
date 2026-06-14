"""Tests for revp_v2bp_autonomous_evidence_adjudication.py.

Covers: guardrails, the autonomous decision state machine (accept, region/patch
mismatch, circularity, insufficient evidence, candidate-positive held for
overlay), parsimonious NEEDS_USER_DECISION, the no-label/no-training invariants,
fail-closed behavior, output generation, safe report language and no private
paths.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bp_autonomous_evidence_adjudication import (  # noqa: E402
    ADJUDICATION_FIELDS,
    METHODOLOGICAL_GUARDRAILS,
    adjudicate_package,
    build_artifacts,
    build_source_independence_audit,
    normalize_region,
    region_from_prefix,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic package helpers
# --------------------------------------------------------------------------- #

def _pkg(**over) -> dict:
    base = {
        "package_id": "PKG_test",
        "event_id": "REC_2022_05_24_30",
        "patch_id": "REC_13",
        "region": "Recife",
        "city": "Recife",
        "hazard_type": "urban_flood",
        "sentinel_observation_date": "2022-05-26",
        "event_window_start": "2022-05-24",
        "event_window_end": "2022-05-30",
        "time_delta_days": "2",
        "has_temporal_anchor": "true",
        "has_spatial_support": "true",
        "has_official_source": "true",
        "has_vhr_support": "false",
        "has_only_contextual_sources": "false",
        "has_geometry": "true",
        "has_patch_overlay": "false",
        "intersection_ratio": "UNKNOWN",
        "valid_data_fraction": "UNKNOWN",
        "evidence_count": "3",
        "strong_evidence_count": "2",
        "weak_evidence_count": "1",
        "conflict_count": "0",
        "evidence_score": "0.70",
        "uncertainty_score": "0.17",
        "promotion_candidate_level": "C3",
        "promotion_decision": "C3_CANDIDATE_REFERENCE_HOLD_FOR_OVERLAY",
        "blocking_reason": "NO_PATCH_EVENT_OVERLAY_GEOMETRY",
        "allowed_use": "candidate_reference",
        "notes": "",
    }
    base.update(over)
    return base


def _adj(**over):
    return adjudicate_package(_pkg(**over), {})


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_autonomous_enabled(self):
        assert METHODOLOGICAL_GUARDRAILS["autonomous_adjudication_enabled"] is True

    def test_human_review_is_autonomous(self):
        assert METHODOLOGICAL_GUARDRAILS["human_review_interpreted_as_autonomous_audit"] is True

    def test_labels_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_negative_from_absence_false(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False

    def test_candidate_positive_is_not_label(self):
        assert METHODOLOGICAL_GUARDRAILS["candidate_positive_is_not_label"] is True

    def test_multimodal_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["multimodal_execution_enabled"] is False
        assert METHODOLOGICAL_GUARDRAILS["multimodal_training_enabled"] is False


# --------------------------------------------------------------------------- #
# Normalization primitives
# --------------------------------------------------------------------------- #

class TestNormalization:
    def test_accent_stripped(self):
        assert normalize_region("Petrópolis") == "petropolis"

    def test_ascii_matches_accented(self):
        assert normalize_region("Petropolis") == normalize_region("Petrópolis")

    def test_region_from_patch_prefix(self):
        assert region_from_prefix("REC_13") == "recife"
        assert region_from_prefix("PET_00291") == "petropolis"
        assert region_from_prefix("CUR_00038") == "curitiba"

    def test_region_from_unknown_prefix(self):
        assert region_from_prefix("UNKNOWN_PATCH") == ""


# --------------------------------------------------------------------------- #
# Decision state machine
# --------------------------------------------------------------------------- #

class TestDecisionMachine:
    def test_accept_when_overlay_and_consistent(self):
        row = _adj(has_patch_overlay="true", intersection_ratio="0.5")
        assert row["auto_decision"] == "AUTO_ACCEPT_EVIDENCE_CONSISTENT"
        assert row["candidate_positive_status"] == "AUTO_VALIDATED_CANDIDATE_POSITIVE"

    def test_candidate_positive_held_when_no_overlay(self):
        row = _adj()  # candidate_reference, no overlay
        assert row["auto_decision"] == "READY_FOR_GT_PROTOCOL_REVIEW"
        assert row["candidate_positive_status"] == "AUTO_VALIDATED_CANDIDATE_POSITIVE"

    def test_secondary_blocked_when_no_overlay(self):
        row = _adj(allowed_use="secondary_evaluation_candidate", strong_evidence_count="0")
        assert row["auto_decision"] == "BLOCKED_NO_EVENT_BINDING"
        assert row["candidate_positive_status"] == "SECONDARY_EVALUATION_CANDIDATE_HELD"

    def test_region_mismatch(self):
        # patch prefix REC but declared region Curitiba
        row = _adj(region="Curitiba")
        assert row["auto_decision"] == "AUTO_REJECT_REGION_MISMATCH"

    def test_patch_event_region_mismatch(self):
        # patch REC, event CUR, declared Recife -> patch/event token mismatch
        row = _adj(event_id="CUR_2022_01_01", region="Recife", patch_id="REC_13")
        assert row["auto_decision"] == "AUTO_REJECT_PATCH_ID_MISMATCH"

    def test_event_missing_rejected(self):
        row = _adj(event_id="REC_EVENT_REGISTRY_MISSING", patch_id="UNKNOWN_PATCH")
        assert row["auto_decision"] == "AUTO_REJECT_EVIDENCE_CONTRADICTORY"

    def test_conflict_rejected(self):
        row = _adj(conflict_count="2")
        assert row["auto_decision"] == "AUTO_REJECT_EVIDENCE_CONTRADICTORY"

    def test_circularity_rejected(self):
        row = _adj(source_used_as_feature_and_label="true")
        assert row["auto_decision"] == "AUTO_REJECT_SOURCE_CIRCULARITY"

    def test_insufficient_when_context_only(self):
        row = _adj(has_only_contextual_sources="true", has_official_source="false")
        assert row["auto_decision"] == "AUTO_REVIEW_INSUFFICIENT_EVIDENCE"

    def test_insufficient_when_no_temporal(self):
        row = _adj(has_temporal_anchor="false")
        assert row["auto_decision"] == "AUTO_REVIEW_INSUFFICIENT_EVIDENCE"

    def test_needs_user_only_when_ambiguous(self):
        row = _adj(ambiguous_event_assignment="true")
        assert row["auto_decision"] == "NEEDS_USER_DECISION"
        assert row["needs_user_decision"] == "True"
        assert row["auto_decision_confidence"] == "NOT_APPLICABLE"

    def test_needs_user_false_by_default(self):
        assert _adj()["needs_user_decision"] == "False"


# --------------------------------------------------------------------------- #
# Invariants: no label, no training, no negative from absence
# --------------------------------------------------------------------------- #

class TestInvariants:
    def test_never_allows_training(self):
        for over in ({}, {"has_patch_overlay": "true"}, {"region": "Curitiba"}, {"conflict_count": "3"}):
            assert _adj(**over)["allowed_for_training"] == "False"

    def test_never_sets_flood_label(self):
        for over in ({}, {"has_patch_overlay": "true", "intersection_ratio": "0.9"}):
            assert _adj(**over)["gt_patch_flood_observed"] == ""

    def test_candidate_positive_has_no_label(self):
        row = _adj(has_patch_overlay="true", intersection_ratio="0.9")
        assert row["candidate_positive_status"] == "AUTO_VALIDATED_CANDIDATE_POSITIVE"
        assert row["gt_patch_flood_observed"] == ""
        assert row["allowed_for_training"] == "False"


# --------------------------------------------------------------------------- #
# Source independence / circularity audit
# --------------------------------------------------------------------------- #

class TestSourceIndependence:
    def test_context_low_cannot_promote_alone(self):
        rows = build_source_independence_audit([{"source_id": "MEDIA", "source_class": "context_low"}])
        assert rows[0]["decision"] == "CONTEXT_ONLY_CANNOT_PROMOTE_ALONE"

    def test_official_is_independent_candidate(self):
        rows = build_source_independence_audit([{"source_id": "ANA", "source_class": "official_hydromet"}])
        assert rows[0]["decision"] == "INDEPENDENT_LABEL_CANDIDATE_SOURCE"
        assert rows[0]["circularity_risk"] == "NONE"

    def test_operational_mapping_flags_feature_overlap(self):
        rows = build_source_independence_audit([{"source_id": "EMS", "source_class": "operational_mapping"}])
        assert rows[0]["used_as_feature"] == "yes"
        assert rows[0]["independent_from_gis_proxy"] == "False"


# --------------------------------------------------------------------------- #
# Build artifacts on real-shaped synthetic registry + fail-closed
# --------------------------------------------------------------------------- #

class TestBuildArtifacts:
    def _registry(self, tmp_path, packages):
        path = tmp_path / "pkg.csv"
        fields = list(packages[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(packages)
        return path

    def test_mixed_registry(self, tmp_path):
        pkgs = [
            _pkg(package_id="P1"),  # candidate positive held
            _pkg(package_id="P2", region="Curitiba"),  # region mismatch reject
            _pkg(package_id="P3", allowed_use="secondary_evaluation_candidate", strong_evidence_count="0"),  # blocked
        ]
        reg = self._registry(tmp_path, pkgs)
        art = build_artifacts(reg, tmp_path / "nosrc.csv", tmp_path / "noft.csv")
        s = art["summary"]
        assert s["package_count"] == 3
        assert s["candidate_positive_count"] == 1
        assert s["auto_rejected"] == 1
        assert s["blocked"] == 1
        assert s["needs_user_decision"] == 0
        assert art["guardrails"]["overall"] == "PASS"

    def test_empty_registry_fail_closed(self, tmp_path):
        art = build_artifacts(tmp_path / "none.csv", tmp_path / "none2.csv", tmp_path / "none3.csv")
        assert art["summary"]["package_count"] == 0
        assert art["guardrails"]["overall"] == "PASS"

    def test_gate_blocked(self, tmp_path):
        reg = self._registry(tmp_path, [_pkg()])
        art = build_artifacts(reg, tmp_path / "nosrc.csv", tmp_path / "noft.csv")
        gate = art["gate"]
        assert gate["labels_created"] is False
        assert gate["formal_negatives_created"] is False
        assert gate["allowed_for_training_count"] == 0
        assert gate["promotion_to_operational_gt"] is False


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "autonomous_evidence_adjudication_v2bp.csv",
        "patch_event_consistency_matrix_v2bp.csv",
        "autonomous_candidate_positive_registry_v2bp.csv",
        "autonomous_rejection_registry_v2bp.csv",
        "autonomous_ambiguity_registry_v2bp.csv",
        "source_independence_audit_v2bp.csv",
        "temporal_spatial_alignment_audit_v2bp.csv",
        "gt_promotion_gate_v2bp.json",
        "autonomous_adjudication_guardrails_v2bp.json",
        "autonomous_adjudication_summary_v2bp.json",
        "autonomous_adjudication_report_v2bp.md",
    ]

    def _run(self, tmp_path):
        path = tmp_path / "pkg.csv"
        pkgs = [_pkg(package_id="P1"), _pkg(package_id="P2", region="Curitiba")]
        fields = list(pkgs[0].keys())
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(pkgs)
        src = tmp_path / "src.csv"
        with src.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["source_id", "source_class"])
            w.writeheader()
            w.writerow({"source_id": "ANA", "source_class": "official_hydromet"})
        art = build_artifacts(path, src, tmp_path / "noft.csv")
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        return out

    def test_all_files_created(self, tmp_path):
        out = self._run(tmp_path)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_forbidden_extensions(self, tmp_path):
        out = self._run(tmp_path)
        forbidden = {".npz", ".npy", ".parquet", ".tif", ".tiff", ".pt", ".pth", ".ckpt", ".safetensors"}
        for p in out.glob("*"):
            assert p.suffix.lower() not in forbidden

    def test_adjudication_csv_schema(self, tmp_path):
        out = self._run(tmp_path)
        with (out / "autonomous_evidence_adjudication_v2bp.csv").open(encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        assert fields == ADJUDICATION_FIELDS

    def test_gate_json_invariants(self, tmp_path):
        out = self._run(tmp_path)
        gate = json.loads((out / "gt_promotion_gate_v2bp.json").read_text(encoding="utf-8"))
        assert gate["labels_created"] is False
        assert gate["supervised_training_enabled"] is False
        assert gate["allowed_for_training_count"] == 0

    def test_report_safe_language(self, tmp_path):
        out = self._run(tmp_path)
        text = (out / "autonomous_adjudication_report_v2bp.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            # phrases may only appear inside an explicit negation ("no ...")
            assert f"no {phrase}" in text or phrase not in text


# --------------------------------------------------------------------------- #
# No private paths
# --------------------------------------------------------------------------- #

class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bp_autonomous_evidence_adjudication.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
