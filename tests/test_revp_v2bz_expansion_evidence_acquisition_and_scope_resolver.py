"""Tests for revp_v2bz_expansion_evidence_acquisition_and_scope_resolver.py.

Covers: hazard scope resolution (Petrópolis mass-movement kept as a separate
cohort, never flood; Curitiba flood-compatible after repair), source/geometry
classification (point evidence, polygon geometry, risk-area not a formal
footprint), context-only blocked events, the Curitiba registry repair scaffold
that references a real candidate without inventing an event, the processing queue
with no spurious NEEDS_USER, the no-label/no-negative/no-training invariants,
no-event/no-geometry-invented guardrails, output generation, no heavy outputs, no
private paths, safe report language and guardrails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bz_expansion_evidence_acquisition_and_scope_resolver import (  # noqa: E402
    HS_FLOOD,
    HS_MASS,
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    build_curitiba_repair,
    classify_local_source,
    resolve_hazard_scope,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic inputs
# --------------------------------------------------------------------------- #

def _v2by_events():
    return [
        {"event_id": "REC_2022_05_24_30", "region": "Recife", "evidence_type": "urban_flood", "priority": "ALREADY_PROCESSED"},
        {"event_id": "PET_2022_02_15", "region": "Petropolis", "evidence_type": "mass_movement", "priority": "LOW"},
        {"event_id": "PET_2024_03_21_28", "region": "Petropolis", "evidence_type": "mass_movement", "priority": "LOW"},
        {"event_id": "CUR_EVENT_REGISTRY_MISSING", "region": "Curitiba", "evidence_type": "unknown_hazard", "priority": "BLOCKED"},
    ]


def _src(region, family, *, point=False, polygon=False, risk=False, name="ctx.csv"):
    return {
        "source_id": f"SRC_{region}_{name}", "event_id": region[:3].upper() + "_REGION_CONTEXT", "region": region,
        "hazard_type": "mass_movement" if region == "Petropolis" else "flood", "source_name": name, "source_family": family,
        "source_type": name.rsplit(".", 1)[-1], "source_path_or_url": f"datasets/{name}", "is_local": "true", "is_external": "false",
        "is_official": "true", "is_context_source": str(not (point or polygon)).lower(), "is_point_source": str(point).lower(),
        "is_polygon_source": str(polygon).lower(), "is_risk_area_source": str(risk and polygon).lower(),
        "date_or_period": "", "temporal_alignment_status": "NOT_ASSESSED", "source_independence_status": "LOCAL",
        "source_status": "INVENTORIED", "recommended_use": "review", "notes": "",
    }


def _curitiba_candidates():
    return [
        {"event_id_candidate": "CUR_2022_01_15", "start_date": "2022-01-15", "end_date": "2022-01-15",
         "hazard_scope": "urban_flooding|intense_rain", "official_source_status": "OFFICIAL_PUBLIC_SOURCE", "confidence_score": "90"},
        {"event_id_candidate": "CUR_2022_01_05", "start_date": "2022-01-05", "end_date": "2022-01-05",
         "hazard_scope": "urban_flooding|intense_rain", "official_source_status": "OFFICIAL_PUBLIC_SOURCE", "confidence_score": "90"},
    ]


def _art(events=None, sources=None, candidates=None):
    return build_artifacts(
        Path("x"), Path("x"), Path("x"), Path("x"),
        v2by_events_override=_v2by_events() if events is None else events,
        sources_override=[_src("Petropolis", "OFFICIAL_CONTEXT_SOURCE")] if sources is None else sources,
        curitiba_candidates_override=_curitiba_candidates() if candidates is None else candidates,
    )


def _scope_by_event(art):
    return {s["event_id"]: s for s in art["scope"]}


# --------------------------------------------------------------------------- #
# 1. Smoke
# --------------------------------------------------------------------------- #

class TestSmoke:
    def test_runs_minimal(self):
        art = _art()
        assert art["summary"]["phase"] == "v2bz"
        assert art["summary"]["target_events_audited"] == 3  # 2 PET + 1 CUR (REC excluded)

    def test_fail_closed_empty(self):
        art = _art(events=[], sources=[], candidates=[])
        assert art["guardrails"]["overall"] == "PASS"
        assert art["summary"]["target_events_audited"] == 0


# --------------------------------------------------------------------------- #
# 2,7. Hazard scope — mass-movement separate, not flood
# --------------------------------------------------------------------------- #

class TestHazardScope:
    def test_petropolis_mass_movement_separate(self):
        s = _scope_by_event(_art())["PET_2022_02_15"]
        assert s["scope_decision"] == HS_MASS
        assert s["can_join_flood_cohort"] == "false"
        assert s["requires_separate_target"] == "true"

    def test_mass_movement_not_forced_into_flood(self):
        art = _art()
        assert art["guardrails"]["checks"]["mass_movement_not_forced_into_flood"] == "PASS"
        for s in art["scope"]:
            if s["detected_hazard_type"] == "mass_movement":
                assert s["can_join_flood_cohort"] == "false"

    def test_curitiba_flood_compatible_after_repair(self):
        s = _scope_by_event(_art())["CUR_EVENT_REGISTRY_MISSING"]
        assert s["detected_hazard_type"] == "flood"
        assert s["scope_decision"] == HS_FLOOD

    def test_resolve_scope_direct(self):
        targets = [{"event_id": "PET_X", "region": "Petropolis", "hazard_type": "mass_movement"}]
        rows = resolve_hazard_scope(targets, [], {})
        assert rows[0]["scope_decision"] == HS_MASS


# --------------------------------------------------------------------------- #
# 3-4,8. Source / geometry classification
# --------------------------------------------------------------------------- #

class TestSourceClassification:
    def _write(self, path, doc):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_point_source(self, tmp_path):
        p = tmp_path / "petropolis_defesa_civil_points.geojson"
        self._write(p, {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [-43.1, -22.5]}}]})
        info = classify_local_source(p, {"Petropolis"})
        assert info is not None
        assert info["source_family"] == "OFFICIAL_POINT_EVIDENCE_SOURCE"
        assert info["is_point_source"] == "true"

    def test_polygon_source(self, tmp_path):
        p = tmp_path / "petropolis_footprint.geojson"
        self._write(p, {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-43.1, -22.5], [-43.0, -22.5], [-43.0, -22.4], [-43.1, -22.5]]]}}]})
        info = classify_local_source(p, {"Petropolis"})
        assert info is not None
        assert info["source_family"] == "OFFICIAL_POLYGON_GEOMETRY_SOURCE"
        assert info["is_polygon_source"] == "true"

    def test_risk_area_not_formal_footprint(self, tmp_path):
        p = tmp_path / "petropolis_risco_areas.geojson"
        self._write(p, {"type": "FeatureCollection", "features": [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-43.1, -22.5], [-43.0, -22.5], [-43.0, -22.4], [-43.1, -22.5]]]}}]})
        info = classify_local_source(p, {"Petropolis"})
        assert info is not None
        assert info["source_family"] == "RISK_AREA_GEOMETRY_SOURCE"
        assert info["is_risk_area_source"] == "true"

    def test_risk_polygon_geometry_not_formal_gt(self):
        sources = [_src("Petropolis", "RISK_AREA_GEOMETRY_SOURCE", polygon=True, risk=True, name="risco.geojson")]
        art = _art(sources=sources)
        for g in art["geometry"]:
            assert g["can_support_formal_gt"] == "false"

    def test_geojson_without_geometry_skipped(self, tmp_path):
        p = tmp_path / "curitiba_empty.geojson"
        self._write(p, {"type": "FeatureCollection", "features": []})
        info = classify_local_source(p, {"Curitiba"})
        # File matched but has no real geometry: classified as context, not point/polygon.
        assert info is not None
        assert info["is_point_source"] == "false"
        assert info["is_polygon_source"] == "false"


# --------------------------------------------------------------------------- #
# 5. Context-only blocked
# --------------------------------------------------------------------------- #

class TestContextOnly:
    def test_context_only_geometry_blocked(self):
        art = _art()  # only context sources
        for g in art["geometry"]:
            assert g["geometry_quality_status"] == "NO_GEOMETRY"
            assert g["blocked_reason"] == "NO_LOCAL_GEOMETRY_OR_POINTS"

    def test_petropolis_readiness_blocked(self):
        art = _art()
        assert len(art["pet_readiness"]) == 2
        for r in art["pet_readiness"]:
            assert r["readiness_status"] == "PET_EVIDENCE_CONTEXT_ONLY_NO_GEOMETRY"
            assert r["is_event_specific_geometry"] == "false"


# --------------------------------------------------------------------------- #
# 6,16. Curitiba repair without inventing event
# --------------------------------------------------------------------------- #

class TestCuritibaRepair:
    def test_repair_references_real_candidate(self):
        art = _art()
        row = art["curitiba"][0]
        assert row["candidate_event_found"] == "true"
        assert row["candidate_event_id"] == "CUR_2022_01_15"
        assert row["repair_status"] == "CURITIBA_EVENT_REGISTRY_REPAIR_SCAFFOLD_READY"

    def test_no_candidate_means_still_missing_no_invention(self):
        targets = [{"event_id": "CUR_EVENT_REGISTRY_MISSING", "region": "Curitiba", "hazard_type": "unknown"}]
        repair, rows = build_curitiba_repair(targets, [])
        assert repair["candidate_event_found"] == "false"
        assert repair["candidate_event_id"] == ""
        assert repair["repair_status"] == "CURITIBA_EVENT_REGISTRY_STILL_MISSING"

    def test_no_event_invented_guardrail(self):
        art = _art(candidates=[])
        assert art["guardrails"]["checks"]["no_event_invented"] == "PASS"


# --------------------------------------------------------------------------- #
# 9-11. Queue / needs_user / training gate
# --------------------------------------------------------------------------- #

class TestQueueAndGate:
    def test_queue_generated(self):
        art = _art()
        assert len(art["queue"]) == 3

    def test_no_spurious_needs_user(self):
        art = _art()
        assert all(q["needs_user_decision"] == "false" for q in art["queue"])
        assert art["summary"]["needs_user_decision_count"] == 0

    def test_training_blocked(self):
        art = _art()
        g = art["gate"]
        assert g["can_train_supervised_model"] is False
        assert g["can_train_dry_run_model"] is False
        assert g["allowed_for_training_count"] == 0
        assert g["blocked_reason"] == "COHORT_EXPANSION_DATA_NOT_READY"


# --------------------------------------------------------------------------- #
# 12-15. No label / negative / absence / invented geometry
# --------------------------------------------------------------------------- #

class TestSafety:
    def test_no_label(self):
        art = _art()
        assert art["summary"]["labels_created"] is False
        assert art["gate"]["formal_labels_created"] is False

    def test_no_formal_negative(self):
        art = _art()
        assert art["summary"]["formal_negatives_created"] is False
        assert art["guardrails"]["checks"]["formal_negative_not_created"] == "PASS"

    def test_no_negative_from_absence(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_absence"] is False
        assert _art()["guardrails"]["checks"]["no_negative_from_absence"] == "PASS"

    def test_no_geometry_invented(self):
        art = _art()
        assert art["guardrails"]["checks"]["no_geometry_invented"] == "PASS"
        assert all(g["can_support_formal_gt"] == "false" for g in art["geometry"])


# --------------------------------------------------------------------------- #
# 17-19. Outputs
# --------------------------------------------------------------------------- #

class TestOutputs:
    EXPECTED = [
        "expansion_evidence_acquisition_summary_v2bz.json",
        "target_event_source_inventory_v2bz.csv",
        "target_event_geometry_inventory_v2bz.csv",
        "hazard_scope_resolution_v2bz.csv",
        "petropolis_evidence_readiness_v2bz.csv",
        "curitiba_event_registry_repair_scaffold_v2bz.csv",
        "external_source_search_log_v2bz.csv",
        "expansion_event_processing_queue_v2bz.csv",
        "acquisition_gap_analysis_v2bz.csv",
        "cohort_growth_readiness_gate_v2bz.json",
        "expansion_acquisition_guardrails_v2bz.json",
        "expansion_acquisition_report_v2bz.md",
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
        text = (SCRIPTS_DIR / "revp_v2bz_expansion_evidence_acquisition_and_scope_resolver.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text


# --------------------------------------------------------------------------- #
# 20. Report language
# --------------------------------------------------------------------------- #

class TestReportLanguage:
    def test_no_forbidden_claims(self, tmp_path):
        art = _art()
        out = tmp_path / "out3"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "expansion_acquisition_report_v2bz.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text

    def test_web_search_not_performed(self):
        art = _art()
        assert art["summary"]["external_web_search"] == "EXTERNAL_WEB_SEARCH_NOT_PERFORMED"


# --------------------------------------------------------------------------- #
# 21. Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_guardrails_pass(self):
        assert _art()["guardrails"]["overall"] == "PASS"

    def test_guardrail_invariants(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["event_invented"] is False
        assert METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False
        assert METHODOLOGICAL_GUARDRAILS["hazard_scope_collapsed"] is False
        assert METHODOLOGICAL_GUARDRAILS["mass_movement_forced_into_flood"] is False
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False
