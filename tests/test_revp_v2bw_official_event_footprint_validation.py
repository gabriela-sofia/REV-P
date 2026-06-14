"""Tests for revp_v2bw_official_event_footprint_validation.py.

Covers: source classification (official context without geometry, official point
evidence, media-derived charter polygon, QA-derived geometry), the
OFFICIAL_FOOTPRINT_NOT_FOUND decision when no reviewed official polygon exists,
REC_00276 held as QA-aligned but blocked by the missing formal footprint, the
comparable negatives kept as scaffold without a formal label, the
no-label/no-negative/no-training invariants, offline fail-closed behaviour
(EXTERNAL_WEB_SEARCH_NOT_PERFORMED), output generation, no heavy outputs, no
private paths, safe report language and guardrails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bw_official_event_footprint_validation import (  # noqa: E402
    EVENT_ID,
    FP_NOT_FOUND,
    METHODOLOGICAL_GUARDRAILS,
    SRC_MEDIA,
    SRC_OFFICIAL_CONTEXT,
    SRC_POINT,
    SRC_QA,
    build_artifacts,
    classify_source,
    discover_sources,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic input builders
# --------------------------------------------------------------------------- #

def _polygon(lon: float = -34.92, lat: float = -8.05, d: float = 0.02) -> dict:
    ring = [[lon - d, lat - d], [lon + d, lat - d], [lon + d, lat + d], [lon - d, lat + d], [lon - d, lat - d]]
    return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}]}


def _points(lon: float = -34.92, lat: float = -8.05, n: int = 4) -> dict:
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [lon + 0.001 * i, lat + 0.001 * i]}} for i in range(n)]
    return {"type": "FeatureCollection", "features": feats}


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(header)] + [",".join(r) for r in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sources_override() -> list[dict]:
    """Minimal deterministic source set: official context + point evidence."""
    return [
        {
            "source_id": "SRC_ctx", "event_id": EVENT_ID, "source_name": "charter758_activation.html",
            "source_family": SRC_OFFICIAL_CONTEXT, "source_type": "html", "source_path_or_url": "datasets/x/charter758/activation.html",
            "is_local": "true", "is_external": "false", "is_official": "true", "is_geometry_source": "false",
            "is_context_source": "true", "is_point_source": "false", "is_media_derived": "false", "is_qa_derived": "false",
            "date_or_period": "2022-05-24/2022-05-30", "temporal_alignment_status": "EVENT_WINDOW_MATCH",
            "source_independence_status": "INDEPENDENT_OFFICIAL", "source_status": "ACTIVE", "notes": "",
        },
        {
            "source_id": "SRC_pt", "event_id": EVENT_ID, "source_name": "recife_defesa_civil_risk_locations.geojson",
            "source_family": SRC_POINT, "source_type": "geojson", "source_path_or_url": "datasets/x/defesa/points.geojson",
            "is_local": "true", "is_external": "false", "is_official": "true", "is_geometry_source": "false",
            "is_context_source": "false", "is_point_source": "true", "is_media_derived": "false", "is_qa_derived": "false",
            "date_or_period": "2022-05-24/2022-05-30", "temporal_alignment_status": "EVENT_WINDOW_MATCH",
            "source_independence_status": "INDEPENDENT_OFFICIAL", "source_status": "ACTIVE", "notes": "",
        },
    ]


def _build_inputs(tmp_path: Path) -> dict[str, Path]:
    charter = tmp_path / "charter758" / "event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"
    _write_json(charter, _polygon(lon=-34.95, lat=-8.20))  # offset to conflict with points
    dcivil = tmp_path / "defesa" / "recife_defesa_civil_risk_locations.geojson"
    _write_json(dcivil, _points())
    alt_dir = tmp_path / "alts"
    _write_json(alt_dir / "alt_event_geometry_convex_hull.geojson", _polygon())
    _write_json(alt_dir / "alt_event_geometry_buffer_union_500.geojson", _polygon(d=0.03))
    charter_decision = tmp_path / "charter_decision.csv"
    _write_csv(charter_decision, ["reliability_decision"], [["CHARTER_POLYGON_REJECTED_FOR_EVENT_QA"]])
    recovered = tmp_path / "recovered"
    _write_json(recovered / "patch_boundary_REC_00276_recovered_v2br.geojson", _polygon())
    _write_json(recovered / "patch_boundary_REC_00299_recovered_v2br.geojson", _polygon(d=0.01))
    dossier = tmp_path / "dossier.csv"
    _write_csv(dossier, ["canonical_patch_id", "formal_positive_candidate_status"],
               [["REC_00276", "STRONG_QA_POSITIVE_CANDIDATE_HELD_FOR_FORMAL_FOOTPRINT_VALIDATION"]])
    neg = tmp_path / "neg.csv"
    _write_csv(neg, ["negative_candidate_id", "canonical_patch_id", "negative_comparability_status"],
               [["NEG_1", "REC_00100", "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY"],
                ["NEG_2", "REC_00102", "COMPARABLE_NEGATIVE_CANDIDATE_QA_ONLY"]])
    return {
        "charter": charter, "dcivil": dcivil, "alt_dir": alt_dir, "charter_decision": charter_decision,
        "recovered": recovered, "dossier": dossier, "neg": neg,
    }


def _art(tmp_path: Path):
    p = _build_inputs(tmp_path)
    return build_artifacts(
        p["charter"], p["dcivil"], p["alt_dir"], p["charter_decision"],
        p["recovered"], p["dossier"], p["neg"], sources_override=_sources_override(),
    )


# --------------------------------------------------------------------------- #
# 1. Runs with minimal synthetic inputs
# --------------------------------------------------------------------------- #

class TestSmoke:
    def test_runs_with_synthetic_inputs(self, tmp_path):
        art = _art(tmp_path)
        assert art["summary"]["phase"] == "v2bw"
        assert art["summary"]["event_id"] == EVENT_ID


# --------------------------------------------------------------------------- #
# 2-5. Source classification
# --------------------------------------------------------------------------- #

class TestClassification:
    def test_official_context_no_geometry(self):
        info = classify_source(Path("datasets/x/charter758/activation.html"))
        assert info is not None
        assert info["source_family"] == SRC_OFFICIAL_CONTEXT
        assert info["is_official"] == "true"
        assert info["is_geometry_source"] == "false"
        assert info["is_context_source"] == "true"

    def test_official_point_evidence(self):
        info = classify_source(Path("datasets/x/defesa_civil/recife_defesa_civil_risk_locations.geojson"))
        assert info is not None
        assert info["source_family"] == SRC_POINT
        assert info["is_point_source"] == "true"
        assert info["is_geometry_source"] == "false"

    def test_charter_polygon_media_derived(self):
        info = classify_source(Path("datasets/x/charter758/derived/event_polygon_REC_2022_05_24_30_charter758_digitized_candidate.geojson"))
        assert info is not None
        assert info["source_family"] == SRC_MEDIA
        assert info["is_media_derived"] == "true"
        assert info["is_official"] == "false"

    def test_qa_derived_geometry(self):
        info = classify_source(Path("local_runs/ground_truth/v2bt/alternative_event_geometries/alt_event_geometry_convex_hull.geojson"))
        assert info is not None
        assert info["source_family"] == SRC_QA
        assert info["is_qa_derived"] == "true"
        assert info["is_official"] == "false"

    def test_irrelevant_source_skipped(self):
        assert classify_source(Path("datasets/random/unrelated_file.geojson")) is None


# --------------------------------------------------------------------------- #
# 6. OFFICIAL_FOOTPRINT_NOT_FOUND when no official polygon
# --------------------------------------------------------------------------- #

class TestFootprintDecision:
    def test_not_found_with_only_points_and_qa(self, tmp_path):
        art = _art(tmp_path)
        assert art["decision"][0]["footprint_decision"] == FP_NOT_FOUND
        assert art["summary"]["official_geometry_sources_discovered"] == 0
        assert art["gate"]["official_footprint_validated_for_gt_protocol"] is False

    def test_charter_rejected_status(self, tmp_path):
        art = _art(tmp_path)
        assert art["decision"][0]["charter_polygon_status"] == "CHARTER_POLYGON_REJECTED_FOR_EVENT_QA"


# --------------------------------------------------------------------------- #
# 7. REC_00276 held QA-aligned but blocked
# --------------------------------------------------------------------------- #

class TestRec276:
    def test_rec276_qa_aligned_blocked(self, tmp_path):
        art = _art(tmp_path)
        rec = art["rec276"]
        assert len(rec) == 1
        r = rec[0]
        assert r["canonical_patch_id"] == "REC_00276"
        assert r["official_footprint_status"] == FP_NOT_FOUND
        assert r["formal_positive_protocol_ready"] == "false"
        assert r["allowed_for_training"] == "false"
        assert r["gt_patch_flood_observed"] == ""


# --------------------------------------------------------------------------- #
# 8. Comparable negatives without formal label
# --------------------------------------------------------------------------- #

class TestNegatives:
    def test_negatives_no_formal_label(self, tmp_path):
        art = _art(tmp_path)
        neg = art["negative_alignment"]
        assert len(neg) == 2
        for n in neg:
            assert n["formal_negative_protocol_ready"] == "false"
            assert n["formal_negative_label_created"] == "false"
            assert n["allowed_for_training"] == "false"
            assert n["gt_patch_flood_observed"] == ""


# --------------------------------------------------------------------------- #
# 9-12. No GT, no training, training gate blocked
# --------------------------------------------------------------------------- #

class TestNoGroundTruth:
    def test_no_gt_flood_observed_one(self, tmp_path):
        art = _art(tmp_path)
        for r in art["rec276"] + art["negative_alignment"]:
            assert r["gt_patch_flood_observed"] != "1"

    def test_no_gt_flood_observed_zero(self, tmp_path):
        art = _art(tmp_path)
        for r in art["rec276"] + art["negative_alignment"]:
            assert r["gt_patch_flood_observed"] != "0"

    def test_no_allowed_for_training_true(self, tmp_path):
        art = _art(tmp_path)
        for r in art["rec276"] + art["negative_alignment"]:
            assert r["allowed_for_training"] != "True"
            assert r["allowed_for_training"] != "true"
        assert art["gate"]["allowed_for_training_count"] == 0

    def test_training_gate_blocked(self, tmp_path):
        art = _art(tmp_path)
        readiness = art["readiness"]
        assert readiness["can_train_supervised_model"] is False
        assert readiness["training_target_available"] is False
        assert art["gate"]["supervised_training_enabled"] is False


# --------------------------------------------------------------------------- #
# 13. Fail-closed without internet
# --------------------------------------------------------------------------- #

class TestOffline:
    def test_web_search_not_performed(self, tmp_path):
        art = _art(tmp_path)
        assert art["summary"]["external_web_search"] == "EXTERNAL_WEB_SEARCH_NOT_PERFORMED"

    def test_discover_sources_marks_web(self):
        sources = discover_sources()
        markers = [s for s in sources if s["source_path_or_url"] == "EXTERNAL_WEB_SEARCH_NOT_PERFORMED"]
        assert len(markers) == 1


# --------------------------------------------------------------------------- #
# 14-16. Outputs / no heavy outputs / no private paths
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "official_event_footprint_source_inventory_v2bw.csv",
        "official_event_footprint_geometry_inventory_v2bw.csv",
        "event_source_reconciliation_matrix_v2bw.csv",
        "official_footprint_candidate_scoring_v2bw.csv",
        "charter_vs_official_vs_qa_decision_v2bw.csv",
        "rec00276_formal_footprint_alignment_v2bw.csv",
        "comparable_negative_footprint_alignment_v2bw.csv",
        "formal_footprint_validation_gate_v2bw.json",
        "gt_protocol_readiness_after_footprint_v2bw.json",
        "footprint_validation_guardrails_v2bw.json",
        "footprint_validation_summary_v2bw.json",
        "footprint_validation_report_v2bw.md",
    ]

    def test_all_outputs(self, tmp_path):
        art = _art(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_heavy_outputs(self, tmp_path):
        art = _art(tmp_path)
        out = tmp_path / "out2"
        out.mkdir()
        write_artifacts(out, art)
        forbidden = {".tif", ".tiff", ".shp", ".npz", ".npy", ".pt", ".pth", ".parquet", ".ckpt", ".safetensors"}
        for p in out.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in forbidden

    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bw_official_event_footprint_validation.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text


# --------------------------------------------------------------------------- #
# 17. Report has no forbidden claims
# --------------------------------------------------------------------------- #

class TestReportLanguage:
    def test_no_forbidden_claims(self, tmp_path):
        art = _art(tmp_path)
        out = tmp_path / "out3"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "footprint_validation_report_v2bw.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


# --------------------------------------------------------------------------- #
# 18. Guardrails PASS
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_guardrails_pass(self, tmp_path):
        art = _art(tmp_path)
        assert art["guardrails"]["overall"] == "PASS"

    def test_guardrail_invariants(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["formal_positive_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["formal_negative_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False
        assert METHODOLOGICAL_GUARDRAILS["qa_geometry_promoted_to_gt"] is False
        assert METHODOLOGICAL_GUARDRAILS["charter_polygon_repromoted"] is False
        assert METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False
