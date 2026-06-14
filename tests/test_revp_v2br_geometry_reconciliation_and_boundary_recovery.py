"""Tests for revp_v2br_geometry_reconciliation_and_boundary_recovery.py.

Covers: REC_00019 non-intersection reconciliation (confirmed vs held for
unreviewed event), the no-negative/no-label/no-training invariants, patch
boundary recovery from recorded bounds (WGS84 and reprojected UTM), centroid /
CRS-unknown / conflicting-bounds / not-found blocks, points-not-promoted,
event-not-promoted-to-GT, output generation, no heavy outputs, no private paths,
safe report language and guardrails.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2br_geometry_reconciliation_and_boundary_recovery import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    build_v1fs_bounds_index,
    reconcile_rec00019,
    recover_boundary,
)


# --------------------------------------------------------------------------- #
# Synthetic geometry / input helpers
# --------------------------------------------------------------------------- #

def _poly(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _write_feature(path: Path, geom, props):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"type": "Feature", "properties": props, "geometry": geom}), encoding="utf-8")


def _write_points(path: Path, pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = [{"type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [x, y]}} for x, y in pts]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")


def _patch_geom(tmp_path, *, x0=-34.99, y0=-8.23, x1=-34.98, y1=-8.22):
    p = tmp_path / "patch.geojson"
    _write_feature(p, _poly(x0, y0, x1, y1), {"patch_id": "REC_00019", "source_method": "preserved_raster_header_bounds_reprojected", "source_crs": "EPSG:32725", "crs": "EPSG:4326"})
    return p


def _event_geom(tmp_path, *, x0=-34.94, y0=-8.00, x1=-34.92, y1=-7.98, review="provided_unreviewed", can_gt="false"):
    p = tmp_path / "event.geojson"
    _write_feature(p, _poly(x0, y0, x1, y1), {"event_id": "REC_2022_05_24_30", "source_crs": "EPSG:32725", "crs": "EPSG:4326", "review_status": review, "can_be_ground_truth": can_gt})
    return p


def _dcivil(tmp_path, pts=None):
    p = tmp_path / "dcivil.geojson"
    _write_points(p, pts or [(-34.95, -8.12), (-34.94, -8.13)])
    return p


def _v1fs(tmp_path, rows):
    p = tmp_path / "v1fs.csv"
    fields = ["candidate_id", "region", "crs_if_header_available", "bounds_if_header_available", "asset_path", "exists"]
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return p


def _candidates(tmp_path, ids):
    p = tmp_path / "cand.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["candidate_id", "canonical_patch_id", "candidate_event_id", "region"])
        w.writeheader()
        for i in ids:
            w.writerow({"candidate_id": "CPOS_" + i, "canonical_patch_id": i, "candidate_event_id": "REC_2022_05_24_30", "region": "Recife"})
    return p


def _blocked(tmp_path, ids):
    p = tmp_path / "blocked.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["canonical_patch_id", "candidate_event_id", "overlay_status"])
        w.writeheader()
        for i in ids:
            w.writerow({"canonical_patch_id": i, "candidate_event_id": "REC_2022_05_24_30", "overlay_status": "OVERLAY_BLOCKED_PATCH_GEOMETRY_MISSING"})
    return p


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_labels_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_no_negative_from_non_intersection(self):
        assert METHODOLOGICAL_GUARDRAILS["negative_from_non_intersection"] is False

    def test_centroid_not_boundary(self):
        assert METHODOLOGICAL_GUARDRAILS["centroid_promoted_to_boundary"] is False

    def test_event_not_gt(self):
        assert METHODOLOGICAL_GUARDRAILS["event_polygon_promoted_to_gt"] is False


# --------------------------------------------------------------------------- #
# Front A — reconciliation
# --------------------------------------------------------------------------- #

class TestReconciliation:
    def test_held_when_event_unreviewed(self, tmp_path):
        recon, _, _, eq = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        assert recon["non_intersection_decision"] == "NON_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED"
        assert recon["candidate_positive_status"] == "HELD_FOR_GEOMETRY_RECONCILIATION"

    def test_confirmed_when_event_reviewed(self, tmp_path):
        ev = _event_geom(tmp_path, review="reviewed", can_gt="true")
        recon, _, _, _ = reconcile_rec00019(_patch_geom(tmp_path), ev, _dcivil(tmp_path))
        assert recon["non_intersection_decision"] == "NON_INTERSECTION_CONFIRMED_GEOMETRICALLY"

    def test_non_intersection_not_negative(self, tmp_path):
        recon, _, _, _ = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        assert recon["gt_patch_flood_observed"] == ""
        assert "0" != recon["gt_patch_flood_observed"]

    def test_never_training(self, tmp_path):
        recon, _, _, _ = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        assert recon["allowed_for_training"] == "False"

    def test_centroid_distance_measured(self, tmp_path):
        recon, _, _, _ = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        assert float(recon["centroid_distance"]) > 10  # ~26 km separation

    def test_event_quality_qa_only(self, tmp_path):
        _, _, _, eq = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        assert eq["recommended_use"] == "USE_FOR_OVERLAY_QA_ONLY"
        assert eq["can_be_ground_truth"] == "false"

    def test_dcivil_points_not_overlay(self, tmp_path):
        _, _, dc, _ = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        assert dc["can_define_overlay"] == "False"
        assert dc["can_define_gt"] == "False"

    def test_hypotheses_present(self, tmp_path):
        _, hyps, _, _ = reconcile_rec00019(_patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path))
        names = {h["hypothesis"] for h in hyps}
        assert "CRS_SWAPPED" in names
        assert "AXIS_ORDER_INVERTED" in names


# --------------------------------------------------------------------------- #
# Front B — boundary recovery
# --------------------------------------------------------------------------- #

class TestBoundaryRecovery:
    def _index(self, tmp_path, rows):
        return build_v1fs_bounds_index(_v1fs(tmp_path, rows))

    def test_recover_wgs84_bounds(self, tmp_path):
        idx = self._index(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:4326", "bounds_if_header_available": "-34.99,-8.23,-34.98,-8.22", "asset_path": "data/sentinel/x.tif", "exists": "yes"}])
        row, feat = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovery_status"] == "PATCH_BOUNDARY_RECOVERED"
        assert feat is not None
        assert (tmp_path / "rec" / "patch_boundary_REC_00099_recovered_v2br.geojson").exists()

    def test_recover_utm_reprojected(self, tmp_path):
        idx = self._index(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:32725", "bounds_if_header_available": "280830.0,9090040.0,281860.0,9091050.0", "asset_path": "data/sentinel/x.tif", "exists": "yes"}])
        row, feat = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovery_status"] == "PATCH_BOUNDARY_RECOVERED"
        # reprojected ring must land in Recife degrees
        ring = feat["geometry"]["coordinates"][0]
        assert all(-35.5 < x < -34.5 and -8.5 < y < -7.8 for x, y in ring)

    def test_recovery_does_not_need_live_raster(self, tmp_path):
        idx = self._index(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:4326", "bounds_if_header_available": "-34.99,-8.23,-34.98,-8.22", "asset_path": "data/sentinel/missing.tif", "exists": "no"}])
        row, _ = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovered"] == "True"
        assert row["raster_header_read"] == "False"  # bounds came from recorded metadata

    def test_centroid_only_blocked(self, tmp_path):
        idx = self._index(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:4326", "bounds_if_header_available": "-34.99,-8.23,-34.99,-8.23", "asset_path": "", "exists": "no"}])
        row, feat = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovery_status"] == "PATCH_BOUNDARY_NOT_RECOVERED_CENTROID_ONLY"
        assert feat is None

    def test_crs_unknown_blocked(self, tmp_path):
        idx = self._index(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "UNKNOWN", "bounds_if_header_available": "100,200,300,400", "asset_path": "", "exists": "no"}])
        row, feat = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovery_status"] == "PATCH_BOUNDARY_BLOCKED_CRS_UNKNOWN"

    def test_conflicting_bounds_ambiguous(self, tmp_path):
        idx = self._index(tmp_path, [
            {"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:4326", "bounds_if_header_available": "-34.99,-8.23,-34.98,-8.22", "asset_path": "", "exists": "no"},
            {"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:4326", "bounds_if_header_available": "-34.80,-8.10,-34.79,-8.09", "asset_path": "", "exists": "no"},
        ])
        row, feat = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovery_status"] == "PATCH_BOUNDARY_AMBIGUOUS_MULTIPLE_CONFLICTING_SOURCES"
        assert row["needs_user_decision"] == "True"

    def test_not_found_no_user_decision(self, tmp_path):
        idx = self._index(tmp_path, [])
        row, _ = recover_boundary("REC_99999", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert row["boundary_recovery_status"] == "PATCH_BOUNDARY_NOT_FOUND"
        assert row["needs_user_decision"] == "False"  # technical missing is not a user decision

    def test_recovered_never_training(self, tmp_path):
        idx = self._index(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:4326", "bounds_if_header_available": "-34.99,-8.23,-34.98,-8.22", "asset_path": "", "exists": "no"}])
        row, _ = recover_boundary("REC_00099", "Recife", "EV", "BLOCKED", idx, tmp_path / "rec")
        assert "allowed_for_training" not in row or row.get("allowed_for_training", "False") == "False"


# --------------------------------------------------------------------------- #
# Build artifacts + outputs
# --------------------------------------------------------------------------- #

class TestBuildAndOutputs:
    EXPECTED = [
        "geometry_reconciliation_summary_v2br.json",
        "rec00019_non_intersection_reconciliation_v2br.csv",
        "patch_boundary_recovery_audit_v2br.csv",
        "recovered_patch_boundary_registry_v2br.csv",
        "unresolved_patch_boundary_registry_v2br.csv",
        "event_geometry_quality_audit_v2br.csv",
        "defense_civil_point_support_audit_v2br.csv",
        "geometry_hypothesis_test_audit_v2br.csv",
        "next_overlay_candidate_queue_v2br.csv",
        "gt_readiness_after_reconciliation_v2br.json",
        "geometry_reconciliation_guardrails_v2br.json",
        "geometry_reconciliation_report_v2br.md",
    ]

    def _build(self, tmp_path, out_name="out"):
        out = tmp_path / out_name
        cand = _candidates(tmp_path, ["REC_00019", "REC_00099", "REC_55555"])
        blocked = _blocked(tmp_path, ["REC_00099", "REC_55555"])
        v1fs = _v1fs(tmp_path, [{"candidate_id": "REC_00099", "region": "Recife", "crs_if_header_available": "EPSG:32725", "bounds_if_header_available": "280830.0,9090040.0,281860.0,9091050.0", "asset_path": "data/sentinel/x.tif", "exists": "yes"}])
        art = build_artifacts(cand, blocked, v1fs, _patch_geom(tmp_path), _event_geom(tmp_path), _dcivil(tmp_path), out)
        return art, out

    def test_counts(self, tmp_path):
        art, _ = self._build(tmp_path)
        s = art["summary"]
        assert s["rec00019_decision"] == "NON_INTERSECTION_HELD_EVENT_GEOMETRY_UNREVIEWED"
        assert s["patch_boundaries_recovered"] == 1   # REC_00099
        assert s["patch_boundaries_blocked"] == 1      # REC_55555 not found
        assert s["needs_user_decision_count"] == 0
        assert art["guardrails"]["overall"] == "PASS"

    def test_empty_fail_closed(self, tmp_path):
        out = tmp_path / "out"
        art = build_artifacts(tmp_path / "a.csv", tmp_path / "b.csv", tmp_path / "c.csv", tmp_path / "d.geojson", tmp_path / "e.geojson", tmp_path / "f.geojson", out)
        assert art["guardrails"]["overall"] == "PASS"

    def test_all_outputs(self, tmp_path):
        from revp_v2br_geometry_reconciliation_and_boundary_recovery import write_artifacts
        art, out = self._build(tmp_path, "out2")
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_heavy_outputs(self, tmp_path):
        from revp_v2br_geometry_reconciliation_and_boundary_recovery import write_artifacts
        art, out = self._build(tmp_path, "out3")
        write_artifacts(out, art)
        forbidden = {".tif", ".tiff", ".shp", ".npz", ".npy", ".pt", ".pth", ".parquet", ".ckpt", ".safetensors"}
        for p in out.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in forbidden

    def test_sidecars_are_geojson(self, tmp_path):
        # build_artifacts writes recovered sidecars into its output_dir
        _, out = self._build(tmp_path, "out4")
        sidecars = list((out / "recovered_patch_boundaries").glob("*"))
        assert sidecars
        for p in sidecars:
            assert p.suffix == ".geojson"

    def test_gate_invariants(self, tmp_path):
        art, _ = self._build(tmp_path)
        gate = art["gate"]
        assert gate["labels_created"] is False
        assert gate["allowed_for_training_count"] == 0
        assert gate["promotion_to_operational_gt"] is False

    def test_report_safe_language(self, tmp_path):
        from revp_v2br_geometry_reconciliation_and_boundary_recovery import write_artifacts
        art, out = self._build(tmp_path, "out5")
        write_artifacts(out, art)
        text = (out / "geometry_reconciliation_report_v2br.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2br_geometry_reconciliation_and_boundary_recovery.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
