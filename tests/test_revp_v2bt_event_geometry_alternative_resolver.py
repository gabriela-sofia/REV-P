"""Tests for revp_v2bt_event_geometry_alternative_resolver.py.

Covers: point-cloud audit, charter rejection/downgrade vs point conflict,
QA-only construction of convex hull / buffered union / cluster envelopes, the
no-GT / no-label / no-training invariants for every point-derived geometry, the
retry queue, insufficient-points and CRS-unknown blocks, conflicting-cluster
ambiguity, output generation, no heavy outputs, no private paths, safe report
language and guardrails.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bt_event_geometry_alternative_resolver import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    build_buffer_union,
    build_cluster_envelopes,
    build_convex_hull,
    charter_reliability_decision,
    load_points,
    point_cloud_audit,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #

def _grid_points(cx, cy, n=60, span=0.01):
    pts = []
    import math
    side = int(math.sqrt(n))
    for i in range(side):
        for j in range(side):
            pts.append((cx + (i / side) * span, cy + (j / side) * span))
    return pts


def _charter_poly(x0=-34.94, y0=-8.00, x1=-34.92, y1=-7.98):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _write_points(path: Path, pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [x, y]}} for x, y in pts]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def _write_charter(path: Path, geom, review="provided_unreviewed", can_gt="false"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "Feature", "properties": {"event_id": "REC_2022_05_24_30", "review_status": review, "can_be_ground_truth": can_gt}, "geometry": geom}), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_labels_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_points_not_gt(self):
        assert METHODOLOGICAL_GUARDRAILS["points_promoted_to_gt"] is False

    def test_hull_not_gt(self):
        assert METHODOLOGICAL_GUARDRAILS["point_hull_promoted_to_gt"] is False

    def test_charter_not_gt(self):
        assert METHODOLOGICAL_GUARDRAILS["charter_polygon_promoted_to_gt"] is False


# --------------------------------------------------------------------------- #
# Point cloud + charter decision
# --------------------------------------------------------------------------- #

class TestCloudAndCharter:
    def test_reads_points(self, tmp_path):
        p = tmp_path / "dc.geojson"
        _write_points(p, _grid_points(-34.95, -8.13))
        pts = load_points(p)
        assert len(pts) >= 40

    def test_audit_detects_conflict(self):
        pts = _grid_points(-34.95, -8.13)  # south
        cloud = point_cloud_audit(pts, "src", _charter_poly(), "charter")  # charter north
        assert cloud["points_inside_charter_polygon"] == 0
        assert cloud["can_define_gt"] == "False"

    def test_charter_rejected_when_conflict(self):
        pts = _grid_points(-34.95, -8.13)
        cloud = point_cloud_audit(pts, "src", _charter_poly(), "charter")
        dec = charter_reliability_decision(_charter_poly(), {"review_status": "provided_unreviewed", "can_be_ground_truth": "false"}, cloud, "charter")
        assert dec["reliability_decision"] == "CHARTER_POLYGON_REJECTED_FOR_EVENT_QA"
        assert dec["can_use_for_formal_gt"] == "False"

    def test_charter_consistent_when_points_inside(self):
        # points inside the charter polygon
        pts = _grid_points(-34.935, -7.99, span=0.005)
        cloud = point_cloud_audit(pts, "src", _charter_poly(), "charter")
        dec = charter_reliability_decision(_charter_poly(), {"review_status": "provided_unreviewed", "can_be_ground_truth": "false"}, cloud, "charter")
        assert dec["reliability_decision"] == "CHARTER_POLYGON_CONSISTENT_WITH_POINT_EVIDENCE"


# --------------------------------------------------------------------------- #
# Alternative geometry construction
# --------------------------------------------------------------------------- #

class TestConstruction:
    def test_convex_hull_created(self):
        hull = build_convex_hull(_grid_points(-34.95, -8.13))
        assert hull is not None
        assert hull["type"] == "Polygon"

    def test_convex_hull_insufficient_points(self):
        assert build_convex_hull([(-34.95, -8.13), (-34.94, -8.12)]) is None

    def test_buffer_union_created(self):
        geom = build_buffer_union(_grid_points(-34.95, -8.13), 250)
        assert geom is not None
        assert geom["type"] in {"Polygon", "MultiPolygon"}

    def test_cluster_envelopes(self):
        audit, geoms, conflict = build_cluster_envelopes(_grid_points(-34.95, -8.13))
        assert isinstance(audit, list)
        assert conflict is False  # one compact cloud

    def test_clusters_conflict_when_far(self):
        far = _grid_points(-34.95, -8.13) + _grid_points(-34.50, -8.60)
        audit, geoms, conflict = build_cluster_envelopes(far)
        assert conflict is True


# --------------------------------------------------------------------------- #
# build_artifacts invariants
# --------------------------------------------------------------------------- #

class TestBuildArtifacts:
    def _build(self, tmp_path, *, points=None, charter=None, out_name="out"):
        out = tmp_path / out_name
        dc = tmp_path / "dc.geojson"
        _write_points(dc, points if points is not None else _grid_points(-34.95, -8.13))
        ch = tmp_path / "charter.geojson"
        _write_charter(ch, charter or _charter_poly())
        return build_artifacts(dc, ch, out), out

    def test_creates_alternatives(self, tmp_path):
        art, _ = self._build(tmp_path)
        assert art["summary"]["alternative_geometries_created"] >= 3
        assert art["guardrails"]["overall"] == "PASS"

    def test_charter_rejected(self, tmp_path):
        art, _ = self._build(tmp_path)
        assert art["summary"]["charter_reliability_decision"] == "CHARTER_POLYGON_REJECTED_FOR_EVENT_QA"

    def test_no_geometry_promoted_to_gt(self, tmp_path):
        art, _ = self._build(tmp_path)
        for r in art["registry"]:
            assert r["can_use_for_formal_gt"] == "False"
            assert r["can_create_label"] == "False"

    def test_no_label_no_training(self, tmp_path):
        art, _ = self._build(tmp_path)
        gate = art["gate"]
        assert gate["labels_created"] is False
        assert gate["allowed_for_training_count"] == 0
        assert gate["event_geometry_ready_for_formal_gt"] is False

    def test_retry_queue_built(self, tmp_path):
        art, _ = self._build(tmp_path)
        assert len(art["queue"]) >= 1
        for q in art["queue"]:
            assert q["gt_promotion_allowed"] == "False"
            assert q["training_allowed"] == "False"
            assert q["candidate_patch_scope"] in {"37_RETRIED_PATCHES", "36_RECOVERED_BOUNDARIES"}

    def test_insufficient_points_blocked(self, tmp_path):
        art, _ = self._build(tmp_path, points=[(-34.95, -8.13), (-34.94, -8.12)])
        statuses = {r["status"] for r in art["registry"]}
        assert "ALTERNATIVE_EVENT_GEOMETRY_BLOCKED_INSUFFICIENT_POINTS" in statuses

    def test_crs_unknown_blocked(self, tmp_path):
        # projected coordinates (metres) -> not degrees -> CRS unknown
        proj = [(280000 + i * 10, 9090000 + j * 10) for i in range(8) for j in range(8)]
        art, _ = self._build(tmp_path, points=proj)
        statuses = {r["status"] for r in art["registry"]}
        assert "ALTERNATIVE_EVENT_GEOMETRY_BLOCKED_CRS_UNKNOWN" in statuses

    def test_conflicting_clusters_ambiguous(self, tmp_path):
        far = _grid_points(-34.95, -8.13) + _grid_points(-34.50, -8.60)
        art, _ = self._build(tmp_path, points=far)
        statuses = [r["status"] for r in art["registry"] if r["geometry_method"] == "cluster_envelope"]
        assert "ALTERNATIVE_EVENT_GEOMETRY_AMBIGUOUS_MULTIPLE_CLUSTERS" in statuses

    def test_empty_fail_closed(self, tmp_path):
        out = tmp_path / "out"
        art = build_artifacts(tmp_path / "no.geojson", tmp_path / "noc.geojson", out)
        assert art["guardrails"]["overall"] == "PASS"


class TestOutputs:
    EXPECTED = [
        "event_geometry_alternative_summary_v2bt.json",
        "event_geometry_source_audit_v2bt.csv",
        "defense_civil_point_cloud_audit_v2bt.csv",
        "charter_polygon_reliability_decision_v2bt.csv",
        "alternative_event_geometry_registry_v2bt.csv",
        "alternative_event_geometry_scoring_v2bt.csv",
        "point_cluster_audit_v2bt.csv",
        "alternative_overlay_retry_queue_v2bt.csv",
        "event_geometry_reliability_gate_v2bt.json",
        "event_geometry_alternative_guardrails_v2bt.json",
        "event_geometry_alternative_report_v2bt.md",
    ]

    def _build(self, tmp_path, out_name="out"):
        out = tmp_path / out_name
        dc = tmp_path / "dc.geojson"
        _write_points(dc, _grid_points(-34.95, -8.13))
        ch = tmp_path / "charter.geojson"
        _write_charter(ch, _charter_poly())
        return build_artifacts(dc, ch, out), out

    def test_all_outputs(self, tmp_path):
        art, out = self._build(tmp_path, "out2")
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_sidecars_geojson(self, tmp_path):
        _, out = self._build(tmp_path, "out3")  # build writes sidecars into out3
        sidecars = list((out / "alternative_event_geometries").glob("*"))
        assert sidecars
        for p in sidecars:
            assert p.suffix == ".geojson"

    def test_no_heavy_outputs(self, tmp_path):
        art, out = self._build(tmp_path, "out4")
        write_artifacts(out, art)
        forbidden = {".tif", ".tiff", ".shp", ".npz", ".npy", ".pt", ".pth", ".parquet", ".ckpt", ".safetensors"}
        for p in out.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in forbidden

    def test_report_safe_language(self, tmp_path):
        art, out = self._build(tmp_path, "out5")
        write_artifacts(out, art)
        text = (out / "event_geometry_alternative_report_v2bt.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bt_event_geometry_alternative_resolver.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
