"""Tests for revp_v2bx_formal_gt_protocol_dry_run.py.

Covers: dry-run positive candidate for the robust patch (REC_00276, not a label),
method-dependent held (REC_00299, not promoted), comparable negatives as dry-run
only (never formal negatives; too-far excluded; non-compatibility without protocol
is not a negative), positive/negative conflict detection, the NA/no-label/
no-training invariants, label and training gates blocked, anti-leakage split plan
(no random split; blocked with a single positive), output generation, no heavy
outputs, no private paths, safe report language and guardrails.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bx_formal_gt_protocol_dry_run import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    build_conflict_audit,
    build_negative_protocol,
    build_positive_protocol,
    write_artifacts,
)

EVENT = "REC_2022_05_24_30"


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

def _dossier():
    return [{
        "canonical_patch_id": "REC_00276", "candidate_event_id": EVENT, "region": "Recife",
        "qa_compatibility_status": "QA_COMPATIBLE_ROBUST", "robustness_status": "robust_multi_method_with_tight_geometry",
        "intersecting_methods": "buffer_union_250;buffer_union_500;cluster_envelope_c0;convex_hull",
        "alternatives_tested": "5", "alternatives_intersecting": "4",
        "max_intersection_ratio_patch": "0.88", "mean_intersection_ratio_patch": "0.53",
    }]


def _method():
    return [{
        "canonical_patch_id": "REC_00299", "candidate_event_id": EVENT,
        "qa_compatibility_status": "QA_COMPATIBLE_METHOD_DEPENDENT",
        "intersecting_methods": "buffer_union_500;convex_hull", "max_intersection_ratio_patch": "0.33",
    }]


def _neg(pid, status, dist, boundary="True"):
    return {
        "negative_candidate_id": f"NEG_{pid}", "canonical_patch_id": pid, "candidate_event_id": EVENT, "region": "Recife",
        "source_pool": "v2bu_noncompatible_recovered_boundaries", "qa_compatibility_status": "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES",
        "boundary_available": boundary, "same_event_context": "True", "same_region": "True", "comparable_source_family": "True",
        "distance_to_qa_footprint": str(dist), "distance_units": "km", "exposure_matching_status": "PLAUSIBLE_SAME_REGION_RECOVERED_BOUNDARY",
        "spatial_block_group": "Recife_event_" + EVENT, "negative_comparability_status": status,
        "formal_negative_label_created": "false", "gt_patch_flood_observed": "", "allowed_for_training": "false",
        "blocked_reason": "x", "notes": "x",
    }


def _negatives():
    return [
        _neg("REC_00183", "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY", 3.76),
        _neg("REC_00204", "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY", 4.33),
        _neg("REC_00260", "NOT_COMPARABLE_NEGATIVE_CANDIDATE_DISTANCE_TOO_FAR", 16.43),
        _neg("REC_00279", "NOT_COMPARABLE_NEGATIVE_CANDIDATE_DISTANCE_TOO_FAR", 20.39),
    ]


def _features():
    return [
        {"canonical_patch_id": "REC_00276", "split_group": "Recife__sentinel_tif__patch_recife_00276", "region": "Recife"},
        {"canonical_patch_id": "REC_00299", "split_group": "Recife__sentinel_tif__patch_recife_00299", "region": "Recife"},
        {"canonical_patch_id": "REC_00183", "split_group": "Recife__sentinel_tif__patch_recife_00183", "region": "Recife"},
    ]


def _art(dossier=None, method=None, negatives=None, features=None, footprint="OFFICIAL_FOOTPRINT_NOT_FOUND"):
    return build_artifacts(
        Path("x"), Path("x"), Path("x"), Path("x"), Path("x"), Path("x"),
        dossier_override=_dossier() if dossier is None else dossier,
        method_override=_method() if method is None else method,
        negative_override=_negatives() if negatives is None else negatives,
        feature_override=_features() if features is None else features,
        footprint_decision_override=footprint,
    )


# --------------------------------------------------------------------------- #
# 1. Smoke
# --------------------------------------------------------------------------- #

class TestSmoke:
    def test_runs_minimal(self):
        art = _art()
        assert art["summary"]["phase"] == "v2bx"
        assert art["summary"]["event_id"] == EVENT

    def test_fail_closed_empty(self):
        art = _art(dossier=[], method=[], negatives=[], features=[])
        assert art["guardrails"]["overall"] == "PASS"
        assert art["summary"]["dry_run_positive_candidates"] == 0
        assert art["summary"]["dry_run_negative_candidates"] == 0


# --------------------------------------------------------------------------- #
# 2-3. Positives
# --------------------------------------------------------------------------- #

class TestPositives:
    def test_rec276_dry_run_positive_not_label(self):
        art = _art()
        pos = {p["canonical_patch_id"]: p for p in art["positives"]}
        r = pos["REC_00276"]
        assert r["would_be_positive_if_protocol_approved"] == "true"
        assert r["gt_patch_flood_observed"] == "NA"
        assert r["allowed_for_training"] == "false"
        assert r["blocked_reason"] == "PROTOCOL_DRY_RUN_ONLY_OFFICIAL_FOOTPRINT_NOT_FOUND"

    def test_rec299_method_dependent_not_positive(self):
        art = _art()
        pos = {p["canonical_patch_id"]: p for p in art["positives"]}
        r = pos["REC_00299"]
        assert r["would_be_positive_if_protocol_approved"] == "false"
        assert r["source_case"] == "method_dependent_registry_v2bv"

    def test_method_dependent_not_in_candidate_positives(self):
        art = _art()
        cand = {c["canonical_patch_id"]: c for c in art["candidates"]}
        assert cand["REC_00299"]["dry_run_role"] == "METHOD_DEPENDENT_HELD"
        assert cand["REC_00276"]["dry_run_role"] == "POSITIVE_CANDIDATE"


# --------------------------------------------------------------------------- #
# 4-6. Negatives
# --------------------------------------------------------------------------- #

class TestNegatives:
    def test_comparable_dry_run_negative_not_label(self):
        art = _art()
        neg = {n["canonical_patch_id"]: n for n in art["negatives"]}
        r = neg["REC_00183"]
        assert r["would_be_negative_if_protocol_approved"] == "true"
        assert r["formal_negative_label_created"] == "false"
        assert r["gt_patch_flood_observed"] == "NA"
        assert r["allowed_for_training"] == "false"
        assert r["blocked_reason"] == "PROTOCOL_DRY_RUN_ONLY_NO_FORMAL_NEGATIVE_APPROVAL"

    def test_too_far_not_negative(self):
        art = _art()
        neg = {n["canonical_patch_id"]: n for n in art["negatives"]}
        r = neg["REC_00260"]
        assert r["would_be_negative_if_protocol_approved"] == "false"
        assert r["blocked_reason"] == "DISTANCE_TOO_FAR_FOR_COMPARABLE_NEGATIVE"

    def test_noncompatible_without_protocol_not_formal_negative(self):
        negs = build_negative_protocol(_negatives())
        assert all(n["formal_negative_label_created"] == "false" for n in negs)


# --------------------------------------------------------------------------- #
# 7. Conflict detection
# --------------------------------------------------------------------------- #

class TestConflicts:
    def test_no_conflict_in_real_data(self):
        art = _art()
        decisions = {c["decision"] for c in art["conflicts"]}
        assert "NO_CONFLICT_DETECTED" in decisions

    def test_positive_negative_conflict_detected(self):
        # Force the same patch to be both a robust positive and a comparable negative.
        positives = build_positive_protocol(_dossier(), [], "OFFICIAL_FOOTPRINT_NOT_FOUND")
        negatives = build_negative_protocol([_neg("REC_00276", "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY", 3.0)])
        conflicts = build_conflict_audit(positives, negatives)
        types = {c["conflict_type"] for c in conflicts}
        assert "POSITIVE_AND_NEGATIVE" in types
        row = next(c for c in conflicts if c["conflict_type"] == "POSITIVE_AND_NEGATIVE")
        assert row["decision"] == "EXCLUDE_FROM_DRY_RUN"


# --------------------------------------------------------------------------- #
# 8-12. Invariants and gates
# --------------------------------------------------------------------------- #

class TestInvariants:
    def test_gt_na_everywhere(self):
        art = _art()
        for r in art["positives"] + art["negatives"] + art["candidates"]:
            assert r["gt_patch_flood_observed"] == "NA"

    def test_label_created_false_everywhere(self):
        art = _art()
        for c in art["candidates"]:
            assert c["label_created"] == "false"

    def test_allowed_for_training_false_everywhere(self):
        art = _art()
        for r in art["positives"] + art["negatives"] + art["candidates"]:
            assert r["allowed_for_training"] == "false"

    def test_label_gate_blocked(self):
        art = _art()
        g = art["label_gate"]
        assert g["label_creation_allowed"] is False
        assert g["formal_positive_labels_created"] is False
        assert g["formal_negative_labels_created"] is False
        assert g["gt_patch_flood_observed_created"] is False

    def test_training_gate_blocked(self):
        art = _art()
        g = art["training_gate"]
        assert g["can_train_supervised_model"] is False
        assert g["can_train_dry_run_model"] is False
        assert g["formal_labels_available"] is False


# --------------------------------------------------------------------------- #
# 13-15. Anti-leakage split
# --------------------------------------------------------------------------- #

class TestSplit:
    def test_split_plan_created(self):
        art = _art()
        assert len(art["split_rows"]) > 0
        assert len(art["group_audit"]) > 0

    def test_split_blocked_single_positive(self):
        art = _art()
        assert all(s["split_status"] == "SPLIT_BLOCKED_TOO_FEW_POSITIVES" for s in art["split_rows"])

    def test_no_random_split(self):
        art = _art()
        # No row is assigned to a concrete train/test fold; all grouped and held.
        for s in art["split_rows"]:
            assert s["recommended_split_role"] == "HELD_NOT_ASSIGNED"
            assert "no_random_split" in s["notes"]
        # Groups carry real grouping keys (event/region/spatial block/source family).
        kinds = {g["group_kind"] for g in art["group_audit"]}
        assert {"event_group", "region", "spatial_block_group", "source_family"} <= kinds


# --------------------------------------------------------------------------- #
# 16-18. Outputs
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "formal_gt_protocol_dry_run_summary_v2bx.json",
        "positive_protocol_dry_run_v2bx.csv",
        "negative_protocol_dry_run_v2bx.csv",
        "dry_run_label_candidate_registry_v2bx.csv",
        "dry_run_label_conflict_audit_v2bx.csv",
        "anti_leakage_split_plan_v2bx.csv",
        "anti_leakage_group_audit_v2bx.csv",
        "label_readiness_gate_v2bx.json",
        "training_readiness_gate_v2bx.json",
        "protocol_dry_run_guardrails_v2bx.json",
        "protocol_dry_run_report_v2bx.md",
    ]

    def test_all_outputs(self, tmp_path):
        art = _art()
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_heavy_outputs(self, tmp_path):
        art = _art()
        out = tmp_path / "out2"
        out.mkdir()
        write_artifacts(out, art)
        forbidden = {".tif", ".tiff", ".shp", ".npz", ".npy", ".pt", ".pth", ".parquet", ".ckpt", ".safetensors"}
        for p in out.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in forbidden

    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bx_formal_gt_protocol_dry_run.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text


# --------------------------------------------------------------------------- #
# 19. Report language
# --------------------------------------------------------------------------- #

class TestReportLanguage:
    def test_no_forbidden_claims(self, tmp_path):
        art = _art()
        out = tmp_path / "out3"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "protocol_dry_run_report_v2bx.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


# --------------------------------------------------------------------------- #
# 20. Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_guardrails_pass(self):
        art = _art()
        assert art["guardrails"]["overall"] == "PASS"

    def test_guardrail_invariants(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["dry_run_candidate_is_label"] is False
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False
        assert METHODOLOGICAL_GUARDRAILS["method_dependent_promoted"] is False
        assert METHODOLOGICAL_GUARDRAILS["qa_geometry_promoted_to_gt"] is False
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False
