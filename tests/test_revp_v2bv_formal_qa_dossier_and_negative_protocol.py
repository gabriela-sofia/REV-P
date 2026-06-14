"""Tests for revp_v2bv_formal_qa_dossier_and_negative_protocol.py.

Covers: the positive QA dossier for a robust patch (held, not a label), the
separate method-dependent register, the comparable-negative scaffold (never a
formal negative; absence/non-compatibility are not negatives), the formal GT
gap analysis and gate, the no-label/no-training invariants, output generation,
no heavy outputs, no private paths, safe report language and guardrails.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bv_formal_qa_dossier_and_negative_protocol import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    build_gap_analysis,
    build_negative_audit,
    build_negative_scaffold,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic matrix helper
# --------------------------------------------------------------------------- #

def _row(pid, status, *, methods="", ratio="0.5", dist="3.0", tested="5"):
    return {
        "canonical_patch_id": pid, "candidate_event_id": "REC_2022_05_24_30", "alternatives_tested": tested,
        "alternatives_intersecting": str(len(methods.split(";")) if methods else 0), "intersecting_methods": methods,
        "non_intersecting_methods": "", "max_intersection_ratio_patch": ratio, "mean_intersection_ratio_patch": ratio,
        "median_intersection_ratio_patch": ratio, "max_intersection_area": "0.0001", "min_centroid_distance": dist,
        "robustness_status": "robust_multi_method_with_tight_geometry" if status == "QA_COMPATIBLE_ROBUST" else "method_or_scale_dependent",
        "qa_compatibility_status": status, "ready_for_formal_gt_review": "True" if status == "QA_COMPATIBLE_ROBUST" else "False",
        "gt_patch_flood_observed": "", "allowed_for_training": "False", "promotion_blocker": "x", "notes": "x",
    }


def _matrix():
    return [
        _row("REC_00276", "QA_COMPATIBLE_ROBUST", methods="buffer_union_250;buffer_union_500;cluster_envelope_c0;convex_hull", ratio="0.88"),
        _row("REC_00299", "QA_COMPATIBLE_METHOD_DEPENDENT", methods="buffer_union_500;convex_hull", ratio="0.33", dist="1.0"),
        _row("REC_00100", "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES", dist="3.5"),   # comparable
        _row("REC_00101", "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES", dist="11.0"),  # too far
        _row("REC_00102", "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES", dist="4.0"),   # comparable
    ]


def _art(matrix=None):
    return build_artifacts(Path("x"), Path("x"), Path("x"), matrix_override=matrix if matrix is not None else _matrix())


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_labels_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_positive_not_label(self):
        assert METHODOLOGICAL_GUARDRAILS["positive_candidate_promoted_to_label"] is False

    def test_negative_from_absence_false(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False

    def test_negative_from_noncompat_false(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_noncompatibility"] is False


# --------------------------------------------------------------------------- #
# Front A — positive dossier
# --------------------------------------------------------------------------- #

class TestPositiveDossier:
    def test_robust_dossier_created(self):
        art = _art()
        assert len(art["dossier"]) == 1
        d = art["dossier"][0]
        assert d["canonical_patch_id"] == "REC_00276"
        assert d["formal_positive_candidate_status"] == "STRONG_QA_POSITIVE_CANDIDATE_HELD_FOR_FORMAL_FOOTPRINT_VALIDATION"

    def test_robust_not_label(self):
        d = _art()["dossier"][0]
        assert d["gt_patch_flood_observed"] == ""
        assert d["formal_gt_ready"] == "false"
        assert d["allowed_for_training"] == "false"

    def test_gt_not_ready_even_strong(self):
        art = _art()
        assert art["gate"]["formal_gt_ready"] is False


# --------------------------------------------------------------------------- #
# Front B — method-dependent
# --------------------------------------------------------------------------- #

class TestMethodDependent:
    def test_method_dependent_separate(self):
        art = _art()
        assert len(art["method_dependent"]) == 1
        m = art["method_dependent"][0]
        assert m["canonical_patch_id"] == "REC_00299"
        assert m["candidate_status"] == "METHOD_DEPENDENT_HELD_FOR_TIGHTER_EVENT_GEOMETRY"

    def test_method_dependent_not_in_dossier(self):
        art = _art()
        dossier_ids = {d["canonical_patch_id"] for d in art["dossier"]}
        assert "REC_00299" not in dossier_ids

    def test_method_dependent_no_training(self):
        m = _art()["method_dependent"][0]
        assert m["gt_patch_flood_observed"] == ""
        assert m["allowed_for_training"] == "false"


# --------------------------------------------------------------------------- #
# Front C — comparable-negative scaffold
# --------------------------------------------------------------------------- #

class TestNegativeScaffold:
    def test_noncompatible_not_negative(self):
        scaffold = build_negative_scaffold(_matrix())
        for s in scaffold:
            assert s["formal_negative_label_created"] == "false"
            assert s["gt_patch_flood_observed"] == ""
            assert s["allowed_for_training"] == "false"

    def test_comparable_within_band(self):
        scaffold = build_negative_scaffold(_matrix())
        by_id = {s["canonical_patch_id"]: s for s in scaffold}
        assert by_id["REC_00100"]["negative_comparability_status"] == "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY"
        assert by_id["REC_00102"]["negative_comparability_status"] == "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY"

    def test_too_far_blocked(self):
        scaffold = build_negative_scaffold(_matrix())
        by_id = {s["canonical_patch_id"]: s for s in scaffold}
        assert by_id["REC_00101"]["negative_comparability_status"] == "NOT_COMPARABLE_NEGATIVE_CANDIDATE_DISTANCE_TOO_FAR"

    def test_only_noncompatible_in_scaffold(self):
        scaffold = build_negative_scaffold(_matrix())
        pids = {s["canonical_patch_id"] for s in scaffold}
        assert "REC_00276" not in pids  # robust positive excluded
        assert "REC_00299" not in pids  # method-dependent excluded

    def test_audit_formal_protocol_blocks(self):
        scaffold = build_negative_scaffold(_matrix())
        audit = build_negative_audit(scaffold)
        fp = next(a for a in audit if a["criterion"] == "formal_protocol_exists")
        assert fp["can_create_formal_negative"] == "false"
        assert fp["passed_count"] == 0

    def test_audit_all_cannot_create_negative(self):
        scaffold = build_negative_scaffold(_matrix())
        audit = build_negative_audit(scaffold)
        assert all(a["can_create_formal_negative"] == "false" for a in audit)


# --------------------------------------------------------------------------- #
# Front D — gaps / gate / readiness
# --------------------------------------------------------------------------- #

class TestGapsAndGate:
    def test_gaps_present(self):
        gaps = build_gap_analysis()
        types = {g["gap_type"] for g in gaps}
        assert any("footprint" in t.lower() for t in types)
        assert any("negative" in t.lower() for t in types)
        assert any("leakage" in t.lower() for t in types)

    def test_gate_invariants(self):
        art = _art()
        gate = art["gate"]
        assert gate["formal_positive_labels_created"] is False
        assert gate["formal_negative_labels_created"] is False
        assert gate["gt_patch_flood_observed_created"] is False
        assert gate["formal_gt_ready"] is False
        assert gate["allowed_for_training_count"] == 0

    def test_training_readiness_blocked(self):
        art = _art()
        tr = art["training_readiness"]
        assert tr["can_train_supervised_model"] is False
        assert tr["allowed_models_now"] == []
        assert tr["formal_labels_available"] is False

    def test_empty_fail_closed(self):
        art = _art(matrix=[])
        assert art["guardrails"]["overall"] == "PASS"
        assert art["gate"]["positive_qa_dossier_count"] == 0


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "formal_qa_positive_dossier_v2bv.csv",
        "formal_qa_positive_dossier_REC_00276_v2bv.md",
        "method_dependent_candidate_registry_v2bv.csv",
        "comparable_negative_candidate_scaffold_v2bv.csv",
        "negative_comparability_audit_v2bv.csv",
        "gt_protocol_gap_analysis_v2bv.csv",
        "formal_gt_gate_v2bv.json",
        "training_readiness_after_qa_dossier_v2bv.json",
        "qa_dossier_guardrails_v2bv.json",
        "qa_dossier_summary_v2bv.json",
        "qa_dossier_report_v2bv.md",
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

    def test_guardrails_pass(self):
        art = _art()
        assert art["guardrails"]["overall"] == "PASS"

    def test_report_safe_language(self, tmp_path):
        art = _art()
        out = tmp_path / "out3"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "qa_dossier_report_v2bv.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bv_formal_qa_dossier_and_negative_protocol.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
