"""Tests for revp_v2bq_patch_event_overlay_geometry_resolver.py.

Covers: geometry discovery/classification, overlay resolution when polygons
intersect, rejection on no-intersection and invalid geometry, blocking on
missing patch/event/both geometry, CRS fail-closed, centroid-not-promoted,
parsimonious NEEDS_USER for genuine multi-geometry ambiguity, the
no-label/no-training invariants, output generation, no heavy outputs, no private
paths and safe report language.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bq_patch_event_overlay_geometry_resolver import (  # noqa: E402
    RESOLUTION_FIELDS,
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    classify_geojson,
    compute_overlay,
    conflicting_ids,
    resolve_candidate,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic geometry helpers
# --------------------------------------------------------------------------- #

def _square(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _src(*, role, geom=None, crs="EPSG:4326", patch_hint="", event_hint="", status="REAL_POLYGON", path="datasets/x.geojson", notes=""):
    bbox = None
    if geom and geom.get("coordinates"):
        from revp_v2bq_patch_event_overlay_geometry_resolver import geometry_bbox
        bbox = geometry_bbox(geom)
    return {
        "source_path": path, "format": "geojson", "geometry_role": role,
        "geometry_type": (geom or {}).get("type", ""), "feature_count": 1 if geom else 0,
        "has_real_coordinates": "True" if geom else "False", "crs": crs,
        "patch_id_hint": patch_hint, "event_id_hint": event_hint, "status": status,
        "notes": notes, "_path": Path(path), "_geom": geom or {}, "_polys": [],
        "_bbox": bbox, "_props": {},
    }


def _patch(geom, **kw):
    return _src(role="patch_polygon", geom=geom, patch_hint="REC_00019", **kw)


def _event(geom, **kw):
    return _src(role="event_polygon", geom=geom, event_hint="REC_2022_05_24_30", **kw)


def _cand(patch="REC_00019", event="REC_2022_05_24_30", region="Recife"):
    return {"candidate_id": "CPOS_1", "canonical_patch_id": patch, "candidate_event_id": event, "region": region}


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_review_only(self):
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True

    def test_labels_created_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_geometry_not_invented(self):
        assert METHODOLOGICAL_GUARDRAILS["geometry_invented"] is False

    def test_centroid_not_promoted(self):
        assert METHODOLOGICAL_GUARDRAILS["centroid_promoted_to_overlay"] is False

    def test_overlay_not_label(self):
        assert METHODOLOGICAL_GUARDRAILS["overlay_equals_label"] is False

    def test_training_disabled(self):
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False


# --------------------------------------------------------------------------- #
# Discovery / classification
# --------------------------------------------------------------------------- #

class TestClassification:
    def test_classify_patch_polygon(self, tmp_path):
        p = tmp_path / "patch_boundary_REC_00019.geojson"
        p.write_text(json.dumps({"type": "Feature", "properties": {"patch_id": "REC_00019"}, "geometry": _square(-34.99, -8.23, -34.98, -8.22)}), encoding="utf-8")
        info = classify_geojson(p)
        assert info["geometry_role"] == "patch_polygon"
        assert info["status"] == "REAL_POLYGON"
        assert info["crs"] == "EPSG:4326"  # inferred from degrees

    def test_classify_event_polygon(self, tmp_path):
        p = tmp_path / "event_rec-2022-05-24-30.geojson"
        p.write_text(json.dumps({"type": "Feature", "properties": {"event_id": "REC_2022_05_24_30"}, "geometry": _square(-34.94, -8.0, -34.92, -7.98)}), encoding="utf-8")
        info = classify_geojson(p)
        assert info["geometry_role"] == "event_polygon"
        assert info["event_id_hint"] == "REC_2022_05_24_30"

    def test_classify_points_weak(self, tmp_path):
        p = tmp_path / "risk_points.geojson"
        p.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [-34.9, -8.0]}}]}), encoding="utf-8")
        info = classify_geojson(p)
        assert info["geometry_role"] == "context_points"
        assert info["status"] == "POINTS_WEAK_SUPPORT"

    def test_classify_empty(self, tmp_path):
        p = tmp_path / "empty.geojson"
        p.write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
        info = classify_geojson(p)
        assert info["status"] == "EMPTY_OR_PLACEHOLDER"


# --------------------------------------------------------------------------- #
# Overlay computation
# --------------------------------------------------------------------------- #

class TestComputeOverlay:
    def test_resolved_when_intersect(self):
        patch = _patch(_square(-34.99, -8.23, -34.97, -8.21))
        event = _event(_square(-34.98, -8.22, -34.96, -8.20))
        comp = compute_overlay(patch, event)
        assert comp["status"] == "OVERLAY_RESOLVED"
        assert comp["intersection_area"] > 0

    def test_reject_no_intersection(self):
        patch = _patch(_square(-34.99, -8.23, -34.98, -8.22))
        event = _event(_square(-34.94, -8.00, -34.92, -7.98))
        comp = compute_overlay(patch, event)
        assert comp["status"] == "OVERLAY_REJECT_NO_INTERSECTION"

    def test_crs_unknown_blocks(self):
        patch = _patch(_square(-34.99, -8.23, -34.98, -8.22), crs="UNKNOWN")
        event = _event(_square(-34.99, -8.23, -34.98, -8.22))
        comp = compute_overlay(patch, event)
        assert comp["status"] == "OVERLAY_BLOCKED_CRS_UNKNOWN"

    def test_invalid_geometry_rejected(self):
        patch = _patch({"type": "Polygon", "coordinates": []})
        event = _event(_square(-34.99, -8.23, -34.98, -8.22))
        comp = compute_overlay(patch, event)
        assert comp["status"] in {"OVERLAY_REJECT_INVALID_GEOMETRY", "OVERLAY_BLOCKED_CRS_UNKNOWN"}


# --------------------------------------------------------------------------- #
# Per-candidate resolution
# --------------------------------------------------------------------------- #

class TestResolveCandidate:
    def test_resolved(self):
        pidx = {"REC_00019": _patch(_square(-34.99, -8.23, -34.97, -8.21))}
        eidx = {"REC_2022_05_24_30": _event(_square(-34.98, -8.22, -34.96, -8.20))}
        row, comp = resolve_candidate(_cand(), pidx, eidx, {})
        assert row["overlay_status"] == "OVERLAY_RESOLVED"
        assert row["gt_protocol_status_after_overlay"] == "READY_FOR_FORMAL_GT_PROTOCOL"
        assert row["allowed_for_training"] == "False"
        assert row["gt_patch_flood_observed"] == ""

    def test_reject_no_intersection(self):
        pidx = {"REC_00019": _patch(_square(-34.99, -8.23, -34.98, -8.22))}
        eidx = {"REC_2022_05_24_30": _event(_square(-34.94, -8.00, -34.92, -7.98))}
        row, _ = resolve_candidate(_cand(), pidx, eidx, {})
        assert row["overlay_status"] == "OVERLAY_REJECT_NO_INTERSECTION"

    def test_block_patch_missing(self):
        eidx = {"REC_2022_05_24_30": _event(_square(-34.94, -8.0, -34.92, -7.98))}
        row, _ = resolve_candidate(_cand(), {}, eidx, {})
        assert row["overlay_status"] == "OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING"

    def test_block_event_missing(self):
        pidx = {"REC_00019": _patch(_square(-34.99, -8.23, -34.98, -8.22))}
        row, _ = resolve_candidate(_cand(), pidx, {}, {})
        assert row["overlay_status"] == "OVERLAY_BLOCKED_EVENT_GEOMETRY_MISSING"

    def test_block_both_missing(self):
        row, _ = resolve_candidate(_cand(), {}, {}, {})
        assert row["overlay_status"] == "OVERLAY_BLOCKED_BOTH_GEOMETRIES_MISSING"

    def test_centroid_not_promoted(self):
        # only point support discovered -> not indexed as a patch/event polygon
        sources = [_src(role="context_points", geom={"type": "Point", "coordinates": [-34.9, -8.0]}, status="POINTS_WEAK_SUPPORT", event_hint="REC_2022_05_24_30")]
        art = build_artifacts(Path("none.csv"), Path("none.csv"), sources=sources)
        # with no real polygons, nothing resolves
        assert art["summary"]["overlay_resolved_count"] == 0

    def test_needs_user_on_ambiguous_geometry(self):
        pidx = {"REC_00019": _patch(_square(-34.99, -8.23, -34.98, -8.22))}
        eidx = {"REC_2022_05_24_30": _event(_square(-34.98, -8.22, -34.96, -8.20))}
        row, _ = resolve_candidate(_cand(), pidx, eidx, {}, ambiguous_patches={"REC_00019"})
        assert row["overlay_status"] == "OVERLAY_REVIEW_AMBIGUOUS_MULTIPLE_GEOMETRIES"
        assert row["needs_user_decision"] == "True"

    def test_needs_user_false_normally(self):
        pidx = {"REC_00019": _patch(_square(-34.99, -8.23, -34.97, -8.21))}
        eidx = {"REC_2022_05_24_30": _event(_square(-34.98, -8.22, -34.96, -8.20))}
        row, _ = resolve_candidate(_cand(), pidx, eidx, {})
        assert row["needs_user_decision"] == "False"


class TestConflictDetection:
    def test_distinct_bboxes_conflict(self):
        s = [
            _src(role="patch_polygon", geom=_square(0, 0, 1, 1), patch_hint="REC_00019", path="a.geojson"),
            _src(role="patch_polygon", geom=_square(5, 5, 6, 6), patch_hint="REC_00019", path="b.geojson"),
        ]
        assert "REC_00019" in conflicting_ids(s, "patch_polygon", "patch_id_hint")

    def test_same_bbox_no_conflict(self):
        s = [
            _src(role="patch_polygon", geom=_square(0, 0, 1, 1), patch_hint="REC_00019", path="a.geojson"),
            _src(role="patch_polygon", geom=_square(0, 0, 1, 1), patch_hint="REC_00019", path="b.geojson"),
        ]
        assert conflicting_ids(s, "patch_polygon", "patch_id_hint") == set()


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

class TestInvariants:
    def _all_rows(self):
        pidx = {"REC_00019": _patch(_square(-34.99, -8.23, -34.97, -8.21))}
        eidx = {"REC_2022_05_24_30": _event(_square(-34.98, -8.22, -34.96, -8.20))}
        cases = [resolve_candidate(_cand(), pidx, eidx, {})[0], resolve_candidate(_cand(), {}, {}, {})[0]]
        return cases

    def test_never_flood_label(self):
        for r in self._all_rows():
            assert r["gt_patch_flood_observed"] == ""

    def test_never_training(self):
        for r in self._all_rows():
            assert r["allowed_for_training"] == "False"

    def test_no_negative_from_absence(self):
        # a fully-blocked (missing) candidate is not turned into a negative
        row, _ = resolve_candidate(_cand(), {}, {}, {})
        assert "negative" not in row["overlay_status"].lower()
        assert row["gt_patch_flood_observed"] == ""


# --------------------------------------------------------------------------- #
# Build artifacts on synthetic sources + outputs
# --------------------------------------------------------------------------- #

class TestBuildAndOutputs:
    EXPECTED = [
        "patch_event_overlay_resolution_v2bq.csv",
        "patch_geometry_inventory_v2bq.csv",
        "event_geometry_inventory_v2bq.csv",
        "geometry_source_discovery_v2bq.csv",
        "overlay_computation_audit_v2bq.csv",
        "overlay_resolved_registry_v2bq.csv",
        "overlay_blocked_registry_v2bq.csv",
        "overlay_rejection_registry_v2bq.csv",
        "overlay_geometry_sidecar_index_v2bq.csv",
        "gt_protocol_readiness_after_overlay_v2bq.json",
        "overlay_guardrails_v2bq.json",
        "overlay_resolution_summary_v2bq.json",
        "overlay_resolution_report_v2bq.md",
    ]

    def _registry(self, tmp_path):
        reg = tmp_path / "cand.csv"
        with reg.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["candidate_id", "canonical_patch_id", "candidate_event_id", "region"])
            w.writeheader()
            w.writerow({"candidate_id": "CPOS_1", "canonical_patch_id": "REC_00019", "candidate_event_id": "REC_2022_05_24_30", "region": "Recife"})
            w.writerow({"candidate_id": "CPOS_2", "canonical_patch_id": "REC_00099", "candidate_event_id": "REC_2022_05_24_30", "region": "Recife"})
        return reg

    def _art(self, tmp_path):
        sources = [
            _patch(_square(-34.99, -8.23, -34.97, -8.21)),
            _event(_square(-34.98, -8.22, -34.96, -8.20)),
        ]
        return build_artifacts(self._registry(tmp_path), tmp_path / "noft.csv", sources=sources)

    def test_resolved_and_blocked_counts(self, tmp_path):
        art = self._art(tmp_path)
        s = art["summary"]
        assert s["candidate_positive_input_count"] == 2
        assert s["overlay_resolved_count"] == 1  # REC_00019 intersects
        assert s["blocked_patch_geometry"] == 1  # REC_00099 has no patch geometry
        assert art["guardrails"]["overall"] == "PASS"

    def test_empty_fail_closed(self, tmp_path):
        art = build_artifacts(tmp_path / "none.csv", tmp_path / "none2.csv", sources=[])
        assert art["summary"]["candidate_positive_input_count"] == 0
        assert art["guardrails"]["overall"] == "PASS"

    def test_all_outputs(self, tmp_path):
        art = self._art(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_heavy_outputs(self, tmp_path):
        art = self._art(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        forbidden = {".npz", ".npy", ".parquet", ".tif", ".tiff", ".pt", ".pth", ".ckpt", ".safetensors", ".shp"}
        for p in out.glob("*"):
            assert p.suffix.lower() not in forbidden

    def test_resolution_schema(self, tmp_path):
        art = self._art(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        with (out / "patch_event_overlay_resolution_v2bq.csv").open(encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        assert fields == RESOLUTION_FIELDS

    def test_gate_json_invariants(self, tmp_path):
        art = self._art(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        gate = json.loads((out / "gt_protocol_readiness_after_overlay_v2bq.json").read_text(encoding="utf-8"))
        assert gate["labels_created"] is False
        assert gate["allowed_for_training_count"] == 0
        assert gate["promotion_to_operational_gt"] is False

    def test_report_safe_language(self, tmp_path):
        art = self._art(tmp_path)
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "overlay_resolution_report_v2bq.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bq_patch_event_overlay_geometry_resolver.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
