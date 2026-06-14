"""Tests for revp_v2bs_recovered_boundary_overlay_retry.py.

Covers: reading the v2br queue and recovered boundaries, real intersection /
non-intersection, the no-positive-label / no-negative / no-training invariants,
event reliability gating on can_be_ground_truth / provided_unreviewed,
Defesa Civil alignment (aligned vs conflicting) as contextual-only, blocked vs
NEEDS_USER, multiple-event-geometry ambiguity, output generation, no heavy
outputs, no private paths, safe report language and guardrails.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bs_recovered_boundary_overlay_retry import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    compute_retry_overlay,
    defense_civil_alignment,
    event_reliability,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #

def _poly(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _recovered_sidecar(dir_path: Path, pid, geom):
    dir_path.mkdir(parents=True, exist_ok=True)
    p = dir_path / f"patch_boundary_{pid}_recovered_v2bs.geojson"
    p.write_text(json.dumps({"type": "Feature", "properties": {"patch_id": pid, "crs": "EPSG:4326", "boundary_quality": "DERIVED_BBOX"}, "geometry": geom}), encoding="utf-8")
    return p


def _event_file(tmp_path, geom, review="provided_unreviewed", can_gt="false"):
    p = tmp_path / "event.geojson"
    p.write_text(json.dumps({"type": "Feature", "properties": {"event_id": "REC_2022_05_24_30", "review_status": review, "can_be_ground_truth": can_gt, "crs": "EPSG:4326"}, "geometry": geom}), encoding="utf-8")
    return p


def _dcivil_file(tmp_path, pts):
    p = tmp_path / "dcivil.geojson"
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [x, y]}} for x, y in pts]
    p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    return p


def _queue(tmp_path, ids_priorities):
    p = tmp_path / "queue.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["queue_id", "canonical_patch_id", "candidate_event_id", "patch_boundary_available_after_v2br", "event_geometry_available_after_v2br", "can_retry_overlay", "retry_reason", "priority"])
        w.writeheader()
        for pid, prio, can_retry in ids_priorities:
            w.writerow({"queue_id": "Q_" + pid, "canonical_patch_id": pid, "candidate_event_id": "REC_2022_05_24_30", "patch_boundary_available_after_v2br": "True", "event_geometry_available_after_v2br": "True", "can_retry_overlay": can_retry, "retry_reason": "x", "priority": prio})
    return p


def _props(review="provided_unreviewed", can_gt="false"):
    return {"event_id": "REC_2022_05_24_30", "review_status": review, "can_be_ground_truth": can_gt}


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_labels_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_no_positive_from_overlay(self):
        assert METHODOLOGICAL_GUARDRAILS["positive_label_from_overlay"] is False

    def test_no_negative_from_non_intersection(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_label_from_non_intersection"] is False

    def test_event_not_gt(self):
        assert METHODOLOGICAL_GUARDRAILS["event_polygon_promoted_to_gt"] is False


# --------------------------------------------------------------------------- #
# Overlay computation
# --------------------------------------------------------------------------- #

class TestComputeOverlay:
    def test_intersects_held(self):
        comp = compute_retry_overlay(_poly(-34.99, -8.23, -34.97, -8.21), _poly(-34.98, -8.22, -34.96, -8.20))
        assert comp["status"] == "OVERLAY_INTERSECTS_HELD_EVENT_GEOMETRY_UNREVIEWED"
        assert comp["intersection_area"] > 0

    def test_no_intersection_held(self):
        comp = compute_retry_overlay(_poly(-34.99, -8.23, -34.98, -8.22), _poly(-34.94, -8.00, -34.92, -7.98))
        assert comp["status"] == "OVERLAY_NO_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED"

    def test_invalid_patch(self):
        comp = compute_retry_overlay({"type": "Polygon", "coordinates": []}, _poly(-34.99, -8.23, -34.98, -8.22))
        assert comp["status"] == "OVERLAY_REJECT_INVALID_PATCH_GEOMETRY"

    def test_invalid_event(self):
        comp = compute_retry_overlay(_poly(-34.99, -8.23, -34.98, -8.22), {"type": "Polygon", "coordinates": []})
        assert comp["status"] == "OVERLAY_REJECT_INVALID_EVENT_GEOMETRY"


# --------------------------------------------------------------------------- #
# Event reliability
# --------------------------------------------------------------------------- #

class TestEventReliability:
    def test_can_be_gt_false_blocks(self):
        dc = {"point_alignment_status": "POINT_SUPPORT_ALIGNED", "point_count": 5}
        r = event_reliability(_poly(-34.94, -8.0, -34.92, -7.98), _props(can_gt="false"), "src", dc, {"intersect": 1, "nonintersect": 0, "blocked": 0})
        assert r["gt_promotion_allowed"] == "False"
        assert r["recommended_use"] == "USE_FOR_OVERLAY_QA_ONLY"

    def test_unreviewed_low_reliability(self):
        dc = {"point_alignment_status": "POINT_SUPPORT_WEAK", "point_count": 5}
        r = event_reliability(_poly(-34.94, -8.0, -34.92, -7.98), _props(review="provided_unreviewed", can_gt="false"), "src", dc, {"intersect": 0, "nonintersect": 1, "blocked": 0})
        assert r["event_reliability_status"] == "EVENT_GEOMETRY_RELIABILITY_LOW_UNREVIEWED_MEDIA_POLYGON"

    def test_conflicting_points_block(self):
        dc = {"point_alignment_status": "POINT_SUPPORT_CONFLICTING", "point_count": 400}
        r = event_reliability(_poly(-34.94, -8.0, -34.92, -7.98), _props(can_gt="false"), "src", dc, {"intersect": 0, "nonintersect": 37, "blocked": 0})
        assert r["event_reliability_status"] == "EVENT_GEOMETRY_RELIABILITY_BLOCKED_CONFLICTS_WITH_DEFENSE_CIVIL_POINTS"
        assert r["gt_promotion_allowed"] == "False"


# --------------------------------------------------------------------------- #
# Defesa Civil alignment (contextual only)
# --------------------------------------------------------------------------- #

class TestDefenseCivil:
    def test_aligned_support(self, tmp_path):
        # points inside the event polygon
        ev = _poly(-34.95, -8.05, -34.90, -8.00)
        dc = _dcivil_file(tmp_path, [(-34.93, -8.02), (-34.92, -8.03)])
        row = defense_civil_alignment(dc, {"type": "Polygon", "coordinates": ev["coordinates"]}, "src")
        assert row["point_alignment_status"] == "POINT_SUPPORT_ALIGNED"
        assert row["can_define_gt"] == "False"
        assert row["can_define_overlay"] == "False"

    def test_conflicting_support(self, tmp_path):
        ev = _poly(-34.94, -8.00, -34.92, -7.98)
        dc = _dcivil_file(tmp_path, [(-34.50, -8.50), (-34.55, -8.55)])  # far away
        row = defense_civil_alignment(dc, {"type": "Polygon", "coordinates": ev["coordinates"]}, "src")
        assert row["point_alignment_status"] == "POINT_SUPPORT_CONFLICTING"
        assert row["can_define_gt"] == "False"


# --------------------------------------------------------------------------- #
# build_artifacts on synthetic inputs
# --------------------------------------------------------------------------- #

class TestBuildArtifacts:
    def _setup(self, tmp_path, *, patch_geom=None, event_geom=None, dcivil_pts=None, ids=None):
        rec_dir = tmp_path / "recovered"
        _recovered_sidecar(rec_dir, "REC_00099", patch_geom or _poly(-34.96, -8.16, -34.95, -8.15))
        ev = _event_file(tmp_path, event_geom or _poly(-34.94, -8.00, -34.92, -7.98))
        dc = _dcivil_file(tmp_path, dcivil_pts or [(-34.50, -8.50)])
        queue = _queue(tmp_path, ids or [("REC_00099", "HIGH", "True")])
        # event quality audit empty -> use default event path
        eqa = tmp_path / "eqa.csv"
        eqa.write_text("geometry_source\n" + str(ev) + "\n", encoding="utf-8")
        return queue, rec_dir, eqa, tmp_path / "patch19.geojson", ev, dc

    def test_reads_queue_and_boundary(self, tmp_path):
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path)
        art = build_artifacts(q, rec, eqa, pg, ev, dc)
        assert art["summary"]["retry_candidate_count"] == 1
        assert art["guardrails"]["overall"] == "PASS"

    def test_non_intersection_not_negative(self, tmp_path):
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path)  # patch south, event north
        art = build_artifacts(q, rec, eqa, pg, ev, dc)
        r = art["retry"][0]
        assert r["overlay_retry_status"] == "OVERLAY_NO_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED"
        assert r["gt_patch_flood_observed"] == ""

    def test_intersection_not_positive_label(self, tmp_path):
        # make patch overlap event
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path, patch_geom=_poly(-34.94, -8.00, -34.92, -7.98))
        art = build_artifacts(q, rec, eqa, pg, ev, dc)
        r = art["retry"][0]
        assert r["overlay_retry_status"] == "OVERLAY_INTERSECTS_HELD_EVENT_GEOMETRY_UNREVIEWED"
        assert r["gt_patch_flood_observed"] == ""
        assert r["allowed_for_training"] == "False"
        assert r["gt_protocol_status"] == "BLOCKED_EVENT_GEOMETRY_NOT_GT"

    def test_ready_for_gt_false(self, tmp_path):
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path)
        art = build_artifacts(q, rec, eqa, pg, ev, dc)
        assert all(x["ready_for_formal_gt_protocol"] == "False" for x in art["queue"])

    def test_missing_boundary_blocked_not_user(self, tmp_path):
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path, ids=[("REC_77777", "HIGH", "True")])
        art = build_artifacts(q, rec, eqa, pg, ev, dc)
        r = next(x for x in art["retry"] if x["canonical_patch_id"] == "REC_77777")
        assert r["overlay_retry_status"] == "OVERLAY_BLOCKED_PATCH_BOUNDARY_MISSING"
        assert r["needs_user_decision"] == "False"

    def test_multiple_event_geometries_ambiguous(self, tmp_path):
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path)
        eg = [
            (_poly(-34.94, -8.00, -34.92, -7.98), _props(), "ev_a"),
            (_poly(-34.50, -8.50, -34.48, -8.48), _props(), "ev_b"),
        ]
        art = build_artifacts(q, rec, eqa, pg, ev, dc, event_geoms=eg)
        r = art["retry"][0]
        assert r["overlay_retry_status"] == "OVERLAY_AMBIGUOUS_MULTIPLE_EVENT_GEOMETRIES"
        assert r["needs_user_decision"] == "True"

    def test_never_flood_label(self, tmp_path):
        q, rec, eqa, pg, ev, dc = self._setup(tmp_path)
        art = build_artifacts(q, rec, eqa, pg, ev, dc)
        for r in art["retry"]:
            assert r["gt_patch_flood_observed"] == ""
            assert r["allowed_for_training"] == "False"

    def test_empty_fail_closed(self, tmp_path):
        art = build_artifacts(tmp_path / "no.csv", tmp_path / "nodir", tmp_path / "noeqa.csv", tmp_path / "nopatch.geojson", tmp_path / "noevent.geojson", tmp_path / "nodc.geojson")
        assert art["guardrails"]["overall"] == "PASS"


class TestOutputs:
    EXPECTED = [
        "recovered_boundary_overlay_retry_v2bs.csv",
        "overlay_intersection_registry_v2bs.csv",
        "overlay_non_intersection_registry_v2bs.csv",
        "overlay_blocked_registry_v2bs.csv",
        "event_geometry_reliability_audit_v2bs.csv",
        "defense_civil_alignment_audit_v2bs.csv",
        "overlay_distance_matrix_v2bs.csv",
        "gt_protocol_candidate_queue_v2bs.csv",
        "gt_readiness_after_overlay_retry_v2bs.json",
        "overlay_retry_guardrails_v2bs.json",
        "overlay_retry_summary_v2bs.json",
        "overlay_retry_report_v2bs.md",
    ]

    def _build(self, tmp_path):
        rec_dir = tmp_path / "recovered"
        _recovered_sidecar(rec_dir, "REC_00099", _poly(-34.96, -8.16, -34.95, -8.15))
        ev = _event_file(tmp_path, _poly(-34.94, -8.00, -34.92, -7.98))
        dc = _dcivil_file(tmp_path, [(-34.50, -8.50)])
        q = _queue(tmp_path, [("REC_00099", "HIGH", "True")])
        eqa = tmp_path / "eqa.csv"
        eqa.write_text("geometry_source\n" + str(ev) + "\n", encoding="utf-8")
        return build_artifacts(q, rec_dir, eqa, tmp_path / "p19.geojson", ev, dc)

    def test_all_outputs(self, tmp_path):
        art = self._build(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_heavy_outputs(self, tmp_path):
        art = self._build(tmp_path)
        out = tmp_path / "out2"
        out.mkdir()
        write_artifacts(out, art)
        forbidden = {".tif", ".tiff", ".shp", ".npz", ".npy", ".pt", ".pth", ".parquet", ".ckpt", ".safetensors"}
        for p in out.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in forbidden

    def test_gate_invariants(self, tmp_path):
        art = self._build(tmp_path)
        gate = art["gate"]
        assert gate["labels_created"] is False
        assert gate["allowed_for_training_count"] == 0
        assert gate["promotion_to_operational_gt"] is False

    def test_report_safe_language(self, tmp_path):
        art = self._build(tmp_path)
        out = tmp_path / "out3"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "overlay_retry_report_v2bs.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bs_recovered_boundary_overlay_retry.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
