"""Tests for revp_v2by_cohort_expansion_candidate_discovery.py.

Covers: detecting the v2bx single-positive training block, event/patch inventory
classification (already-processed, point evidence, polygon geometry, context-only
blocked, patch with/without boundary), HIGH prioritisation when boundary +
evidence + embedding are present, the processing queue, yield projection that uses
NOT_ESTIMABLE without a basis, the no-label/no-negative/no-training invariants,
no-absence/no-random-background guardrails, output generation, no heavy outputs,
no private paths, safe report language and guardrails.
"""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2by_cohort_expansion_candidate_discovery import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

def _adj(pid, event, region, *, decision="READY_FOR_GT_PROTOCOL_REVIEW", evidence_status="present", source_family="defesa_civil", cps="AUTO_VALIDATED_CANDIDATE_POSITIVE", geometry="EVENT_GEOMETRY_PRESENT_NO_PATCH_OVERLAY", evidence_type="urban_flood"):
    return {
        "canonical_patch_id": pid, "candidate_event_id": event, "region": region, "source_family": source_family,
        "evidence_type": evidence_type, "evidence_status": evidence_status, "geometry_status": geometry,
        "auto_decision": decision, "candidate_positive_status": cps,
        "gt_patch_flood_observed": "", "allowed_for_training": "false", "needs_user_decision": "false",
    }


def _adjudication():
    return [
        # Already-processed event (Recife) — full signals.
        _adj("REC_0001", "REC_EVENT", "Recife"),
        _adj("REC_0002", "REC_EVENT", "Recife"),
        # Ready event — points + boundary + embedding.
        _adj("RDY_0001", "READY_EVENT", "Readyland"),
        # Point-evidence event, no boundary.
        _adj("PTS_0001", "PTS_EVENT", "Pointsland"),
        # Polygon-geometry event, no boundary.
        _adj("POLY_0001", "POLY_EVENT", "Polyland"),
        # Context-only event — has context, no geometry/points.
        _adj("CTX_0001", "CTX_EVENT", "Ctxland", evidence_type="mass_movement"),
        # Blocked: no binding, no context.
        _adj("BLK_0001", "BLK_EVENT", "Blkland", decision="BLOCKED_NO_EVENT_BINDING", evidence_status="", source_family="", cps="SECONDARY_EVALUATION_CANDIDATE_HELD"),
    ]


def _features():
    return [
        {"canonical_patch_id": "REC_0001", "region": "Recife", "dino_embedding_available": "true", "gis_feature_available": "false"},
        {"canonical_patch_id": "RDY_0001", "region": "Readyland", "dino_embedding_available": "true", "gis_feature_available": "false"},
    ]


def _recovery():
    return [
        {"canonical_patch_id": "REC_0001", "boundary_recovery_status": "PATCH_BOUNDARY_RECOVERED", "boundary_source_type": "recovered_v2br"},
        {"canonical_patch_id": "RDY_0001", "boundary_recovery_status": "PATCH_BOUNDARY_RECOVERED", "boundary_source_type": "recovered_v2br"},
        {"canonical_patch_id": "POLY_0001", "boundary_recovery_status": "PATCH_BOUNDARY_NOT_FOUND"},
    ]


def _geo_scan():
    return {
        "recife": {"point": 400, "polygon": 4, "files": 10},
        "readyland": {"point": 10, "polygon": 0, "files": 1},
        "pointsland": {"point": 12, "polygon": 0, "files": 1},
        "polyland": {"point": 0, "polygon": 3, "files": 1},
        "ctxland": {"point": 0, "polygon": 0, "files": 0},
        "blkland": {"point": 0, "polygon": 0, "files": 0},
    }


def _summary():
    return {"dry_run_positive_candidates": 1, "dry_run_negative_candidates": 14, "event_id": "REC_EVENT"}


def _art(adjudication=None, features=None, recovery=None, geo=None, summary=None):
    return build_artifacts(
        Path("x"), Path("x"), Path("x"), Path("x"), Path("x"), Path("x"), Path("x"),
        adjudication_override=_adjudication() if adjudication is None else adjudication,
        feature_override=_features() if features is None else features,
        recovery_override=_recovery() if recovery is None else recovery,
        geo_scan_override=_geo_scan() if geo is None else geo,
        vbx_summary_override=_summary() if summary is None else summary,
    )


def _events_by_id(art):
    return {e["event_id"]: e for e in art["events"]}


# --------------------------------------------------------------------------- #
# 1-2. Smoke / training block detection
# --------------------------------------------------------------------------- #

class TestSmoke:
    def test_runs_minimal(self):
        art = _art()
        assert art["summary"]["phase"] == "v2by"
        assert art["summary"]["events_inventoried"] == 6

    def test_detects_single_positive_training_blocked(self):
        art = _art()
        assert art["summary"]["current_dry_run_positive_count"] == 1
        assert art["training_gate"]["can_train_supervised_model"] is False
        assert art["training_gate"]["can_train_dry_run_model"] is False
        assert art["summary"]["training_blocked_reason"] == "TOO_FEW_POSITIVES_FOR_ANY_TRAINING_OR_EVALUATION"

    def test_fail_closed_empty(self):
        art = _art(adjudication=[], features=[], recovery=[], geo={})
        assert art["guardrails"]["overall"] == "PASS"
        assert art["summary"]["events_inventoried"] == 0


# --------------------------------------------------------------------------- #
# 3-6. Event classification
# --------------------------------------------------------------------------- #

class TestEventClassification:
    def test_already_processed(self):
        e = _events_by_id(_art())["REC_EVENT"]
        assert e["expansion_status"] == "EXPANSION_EVENT_ALREADY_PROCESSED"
        assert e["priority"] == "ALREADY_PROCESSED"

    def test_point_evidence(self):
        e = _events_by_id(_art())["PTS_EVENT"]
        assert e["expansion_status"] == "EXPANSION_EVENT_HAS_POINT_EVIDENCE"
        assert e["has_point_evidence"] == "true"
        assert e["priority"] == "MEDIUM"

    def test_polygon_geometry(self):
        e = _events_by_id(_art())["POLY_EVENT"]
        assert e["expansion_status"] == "EXPANSION_EVENT_HAS_POLYGON_GEOMETRY"
        assert e["has_polygon_geometry"] == "true"
        assert e["priority"] == "MEDIUM"

    def test_context_only_blocked(self):
        e = _events_by_id(_art())["CTX_EVENT"]
        assert e["expansion_status"] == "EXPANSION_EVENT_CONTEXT_ONLY"
        assert e["priority"] == "LOW"
        assert e["blocked_reason"] == "NO_LOCAL_GEOMETRY_OR_POINTS"

    def test_ready_event_high_priority(self):
        e = _events_by_id(_art())["READY_EVENT"]
        assert e["expansion_status"] == "EXPANSION_EVENT_READY_FOR_QA_GEOMETRY"
        assert e["priority"] == "HIGH"


# --------------------------------------------------------------------------- #
# 7-8. Patch classification / prioritisation
# --------------------------------------------------------------------------- #

class TestPatchClassification:
    def test_patch_without_boundary_blocked(self):
        patches = {p["canonical_patch_id"]: p for p in _art()["patches"]}
        # POLY_0001 has binding + evidence but no recovered boundary.
        assert patches["POLY_0001"]["has_boundary"] == "false"
        assert patches["POLY_0001"]["expansion_patch_status"] == "EXPANSION_PATCH_BLOCKED_NO_BOUNDARY"

    def test_patch_ready_when_boundary_evidence_embedding(self):
        patches = {p["canonical_patch_id"]: p for p in _art()["patches"]}
        assert patches["RDY_0001"]["expansion_patch_status"] == "EXPANSION_PATCH_READY_FOR_OVERLAY"
        assert patches["RDY_0001"]["priority"] == "HIGH"

    def test_patch_no_binding_blocked(self):
        patches = {p["canonical_patch_id"]: p for p in _art()["patches"]}
        assert patches["BLK_0001"]["expansion_patch_status"] == "EXPANSION_PATCH_BLOCKED_NO_EVENT_BINDING"


# --------------------------------------------------------------------------- #
# 9. Queue
# --------------------------------------------------------------------------- #

class TestQueue:
    def test_queue_generated_excludes_processed(self):
        art = _art()
        eids = {q["event_id"] for q in art["queue"]}
        assert "REC_EVENT" not in eids  # already processed excluded
        assert "PTS_EVENT" in eids
        assert len(art["queue"]) >= 1

    def test_queue_needs_user_false_for_technical(self):
        art = _art()
        assert all(q["needs_user_decision"] == "false" for q in art["queue"])


# --------------------------------------------------------------------------- #
# 10-12. No label / negative / training
# --------------------------------------------------------------------------- #

class TestNoLabelNoTraining:
    def test_no_label_created(self):
        art = _art()
        assert art["summary"]["labels_created"] is False
        assert art["training_gate"]["formal_labels_created"] is False

    def test_no_formal_negative(self):
        art = _art()
        assert art["summary"]["formal_negatives_created"] is False
        assert art["training_gate"]["formal_negatives_created"] is False

    def test_no_training(self):
        art = _art()
        assert art["summary"]["allowed_for_training_count"] == 0
        assert art["training_gate"]["allowed_for_training_count"] == 0
        assert art["training_gate"]["can_train_supervised_model"] is False


# --------------------------------------------------------------------------- #
# 13. Yield projection
# --------------------------------------------------------------------------- #

class TestYield:
    def test_not_estimable_without_basis(self):
        art = _art()
        y = {r["event_id"]: r for r in art["yield"]}
        assert y["CTX_EVENT"]["estimated_dry_run_positive_max"] == "NOT_ESTIMABLE"
        assert y["CTX_EVENT"]["trainability_impact"] == "BLOCKED"

    def test_processed_event_no_change(self):
        art = _art()
        y = {r["event_id"]: r for r in art["yield"]}
        assert y["REC_EVENT"]["trainability_impact"] == "NO_CHANGE"
        assert y["REC_EVENT"]["estimated_dry_run_positive_max"] == "0"


# --------------------------------------------------------------------------- #
# 14-15. No absence / no random background
# --------------------------------------------------------------------------- #

class TestNegativeSafety:
    def test_no_negative_from_absence(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False
        assert _art()["guardrails"]["checks"]["no_negative_from_absence"] == "PASS"

    def test_no_random_background_negative(self):
        assert METHODOLOGICAL_GUARDRAILS["random_background_negative"] is False
        assert _art()["guardrails"]["checks"]["no_random_background_negative"] == "PASS"


# --------------------------------------------------------------------------- #
# 16-18. Outputs
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "cohort_expansion_summary_v2by.json",
        "event_expansion_candidate_inventory_v2by.csv",
        "patch_expansion_candidate_inventory_v2by.csv",
        "evidence_source_expansion_audit_v2by.csv",
        "geometry_readiness_expansion_audit_v2by.csv",
        "dino_gis_feature_readiness_expansion_audit_v2by.csv",
        "dry_run_yield_projection_v2by.csv",
        "next_event_processing_queue_v2by.csv",
        "cohort_expansion_antileakage_plan_v2by.csv",
        "cohort_expansion_training_gate_v2by.json",
        "cohort_expansion_guardrails_v2by.json",
        "cohort_expansion_report_v2by.md",
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
        text = (SCRIPTS_DIR / "revp_v2by_cohort_expansion_candidate_discovery.py").read_text(encoding="utf-8", errors="replace")
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
        text = (out / "cohort_expansion_report_v2by.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


# --------------------------------------------------------------------------- #
# 20. Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_guardrails_pass(self):
        assert _art()["guardrails"]["overall"] == "PASS"

    def test_guardrail_invariants(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["dry_run_projection_is_label"] is False
        assert METHODOLOGICAL_GUARDRAILS["method_dependent_promoted"] is False
        assert METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False
