"""Tests for revp_v2bu_alternative_event_overlay_sensitivity.py.

Covers: pairwise overlay against QA-only alternatives, the robust /
method-dependent / buffer-only / noncompatible classification, the
no-label / no-negative / no-training invariants (including for
ready_for_formal_gt_review), output generation, no heavy outputs, no private
paths, safe report language and guardrails.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "multimodal"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v2bu_alternative_event_overlay_sensitivity import (  # noqa: E402
    METHODOLOGICAL_GUARDRAILS,
    build_artifacts,
    classify_patch,
    compute_pairwise,
    write_artifacts,
)


# --------------------------------------------------------------------------- #
# Synthetic helpers
# --------------------------------------------------------------------------- #

def _poly(x0, y0, x1, y1):
    return {"type": "Polygon", "coordinates": [[[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]]}


def _alt(method, geom, *, buffer_m="", cluster_id="", quality="HIGH"):
    label = method + (f"_{buffer_m}" if buffer_m else "") + (f"_c{cluster_id}" if cluster_id != "" else "")
    is_tight = method == "cluster_envelope" or (method == "buffer_union" and buffer_m and int(buffer_m) <= 250)
    family = "hull" if method == "convex_hull" else ("buffer" if method == "buffer_union" else ("cluster" if method == "cluster_envelope" else "other"))
    return {
        "alternative_geometry_id": "ALT_" + label, "geometry_method": method, "method_label": label,
        "buffer_m": str(buffer_m), "cluster_id": str(cluster_id), "geom": geom, "source": f"local_runs/x/{label}.geojson",
        "crs": "EPSG:4326", "quality": quality, "is_tight": is_tight, "family": family,
    }


def _patch(geom, pid="REC_00276"):
    return {pid: {"geom": geom, "source": f"local_runs/x/{pid}.geojson", "quality": "DERIVED_BBOX", "crs": "EPSG:4326"}}


def _five_alts_overlapping(box):
    # all five alternatives overlap `box`
    x0, y0, x1, y1 = box
    g = _poly(x0 - 0.001, y0 - 0.001, x1 + 0.001, y1 + 0.001)
    return [
        _alt("convex_hull", g),
        _alt("buffer_union", g, buffer_m="250"),
        _alt("buffer_union", g, buffer_m="500"),
        _alt("cluster_envelope", g, cluster_id="0"),
        _alt("cluster_envelope", g, cluster_id="1", quality="MEDIUM"),
    ]


# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

class TestGuardrails:
    def test_labels_false(self):
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False

    def test_no_positive_from_qa_overlay(self):
        assert METHODOLOGICAL_GUARDRAILS["positive_label_from_qa_overlay"] is False

    def test_alt_not_gt(self):
        assert METHODOLOGICAL_GUARDRAILS["alternative_geometry_promoted_to_gt"] is False

    def test_review_not_training(self):
        assert METHODOLOGICAL_GUARDRAILS["ready_for_formal_review_is_training_ready"] is False


# --------------------------------------------------------------------------- #
# Pairwise overlay
# --------------------------------------------------------------------------- #

class TestPairwise:
    def test_intersects(self):
        p = {"geom": _poly(-34.95, -8.13, -34.94, -8.12)}
        a = {"geom": _poly(-34.945, -8.125, -34.93, -8.11)}
        comp = compute_pairwise(p, a)
        assert comp["status"] == "QA_OVERLAY_INTERSECTS"
        assert comp["intersects"] == "True"

    def test_no_intersection(self):
        p = {"geom": _poly(-34.95, -8.13, -34.94, -8.12)}
        a = {"geom": _poly(-34.80, -8.00, -34.79, -7.99)}
        comp = compute_pairwise(p, a)
        assert comp["status"] == "QA_OVERLAY_NO_INTERSECTION"

    def test_invalid_patch(self):
        comp = compute_pairwise({"geom": {"type": "Polygon", "coordinates": []}}, {"geom": _poly(-34.95, -8.13, -34.94, -8.12)})
        assert comp["status"] == "QA_OVERLAY_BLOCKED_INVALID_PATCH_GEOMETRY"


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

class TestClassify:
    def test_robust(self):
        alts = [
            {"family": "hull", "is_tight": False}, {"family": "buffer", "is_tight": True},
            {"family": "cluster", "is_tight": True},
        ]
        status, _ = classify_patch(alts, True)
        assert status == "QA_COMPATIBLE_ROBUST"

    def test_method_dependent(self):
        alts = [{"family": "hull", "is_tight": False}, {"family": "buffer", "is_tight": False}]
        status, _ = classify_patch(alts, True)
        assert status == "QA_COMPATIBLE_METHOD_DEPENDENT"

    def test_buffer_only(self):
        alts = [{"family": "buffer", "is_tight": True}, {"family": "buffer", "is_tight": False}]
        status, _ = classify_patch(alts, True)
        assert status == "QA_COMPATIBLE_BUFFER_ONLY"

    def test_not_compatible(self):
        status, _ = classify_patch([], True)
        assert status == "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES"

    def test_blocked_no_geometry(self):
        status, _ = classify_patch([], False)
        assert status == "QA_BLOCKED_PATCH_GEOMETRY"


# --------------------------------------------------------------------------- #
# build_artifacts on synthetic inputs
# --------------------------------------------------------------------------- #

class TestBuildArtifacts:
    def _run(self, patch_geom, alts):
        return build_artifacts(Path("x"), Path("x"), Path("x"), Path("x"),
                               alternatives_override=alts, patches_override=_patch(patch_geom))

    def test_robust_patch(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        art = self._run(_poly(*box), _five_alts_overlapping(box))
        m = art["matrix"][0]
        assert m["qa_compatibility_status"] == "QA_COMPATIBLE_ROBUST"
        assert m["ready_for_formal_gt_review"] == "True"

    def test_method_dependent_patch(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        g = _poly(-34.951, -8.131, -34.939, -8.119)
        # only hull and buffer_500 overlap; buffer_250 and clusters are far
        far = _poly(-34.80, -8.00, -34.79, -7.99)
        alts = [_alt("convex_hull", g), _alt("buffer_union", g, buffer_m="500"),
                _alt("buffer_union", far, buffer_m="250"), _alt("cluster_envelope", far, cluster_id="0")]
        art = self._run(_poly(*box), alts)
        assert art["matrix"][0]["qa_compatibility_status"] == "QA_COMPATIBLE_METHOD_DEPENDENT"

    def test_buffer_only_patch(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        g = _poly(-34.951, -8.131, -34.939, -8.119)
        far = _poly(-34.80, -8.00, -34.79, -7.99)
        alts = [_alt("buffer_union", g, buffer_m="250"), _alt("buffer_union", g, buffer_m="500"),
                _alt("convex_hull", far), _alt("cluster_envelope", far, cluster_id="0")]
        art = self._run(_poly(*box), alts)
        assert art["matrix"][0]["qa_compatibility_status"] == "QA_COMPATIBLE_BUFFER_ONLY"

    def test_noncompatible_patch(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        far = _poly(-34.80, -8.00, -34.79, -7.99)
        alts = [_alt("convex_hull", far), _alt("buffer_union", far, buffer_m="500")]
        art = self._run(_poly(*box), alts)
        assert art["matrix"][0]["qa_compatibility_status"] == "QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES"

    def test_no_intersection_not_negative(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        far = _poly(-34.80, -8.00, -34.79, -7.99)
        art = self._run(_poly(*box), [_alt("convex_hull", far)])
        for r in art["pairwise"]:
            assert r["gt_patch_flood_observed"] == ""

    def test_qa_intersection_not_label(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        art = self._run(_poly(*box), _five_alts_overlapping(box))
        for r in art["pairwise"]:
            assert r["gt_patch_flood_observed"] == ""
            assert r["allowed_for_training"] == "False"

    def test_ready_not_training(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        art = self._run(_poly(*box), _five_alts_overlapping(box))
        for m in art["matrix"]:
            if m["ready_for_formal_gt_review"] == "True":
                assert m["allowed_for_training"] == "False"
                assert m["gt_patch_flood_observed"] == ""

    def test_pairwise_count(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        art = self._run(_poly(*box), _five_alts_overlapping(box))
        assert art["summary"]["pairwise_overlay_count"] == 5  # 1 patch x 5 alternatives

    def test_gate_invariants(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        art = self._run(_poly(*box), _five_alts_overlapping(box))
        gate = art["gate"]
        assert gate["labels_created"] is False
        assert gate["allowed_for_training_count"] == 0
        assert gate["promotion_to_operational_gt"] is False
        assert art["guardrails"]["overall"] == "PASS"

    def test_empty_fail_closed(self):
        art = build_artifacts(Path("none"), Path("none"), Path("none"), Path("none"), alternatives_override=[], patches_override={})
        assert art["guardrails"]["overall"] == "PASS"
        assert art["summary"]["pairwise_overlay_count"] == 0


class TestOutputs:
    EXPECTED = [
        "alternative_overlay_pairwise_results_v2bu.csv",
        "alternative_overlay_patch_sensitivity_matrix_v2bu.csv",
        "qa_compatible_patch_registry_v2bu.csv",
        "qa_noncompatible_patch_registry_v2bu.csv",
        "method_dependent_patch_registry_v2bu.csv",
        "alternative_geometry_method_summary_v2bu.csv",
        "formal_gt_review_queue_qa_only_v2bu.csv",
        "overlay_sensitivity_gate_v2bu.json",
        "overlay_sensitivity_guardrails_v2bu.json",
        "overlay_sensitivity_summary_v2bu.json",
        "overlay_sensitivity_report_v2bu.md",
    ]

    def _art(self):
        box = (-34.95, -8.13, -34.94, -8.12)
        return build_artifacts(Path("x"), Path("x"), Path("x"), Path("x"),
                               alternatives_override=_five_alts_overlapping(box), patches_override=_patch(_poly(*box)))

    def test_all_outputs(self, tmp_path):
        art = self._art()
        out = tmp_path / "out"
        out.mkdir()
        write_artifacts(out, art)
        for fname in self.EXPECTED:
            assert (out / fname).exists(), f"Missing {fname}"

    def test_no_heavy_outputs(self, tmp_path):
        art = self._art()
        out = tmp_path / "out2"
        out.mkdir()
        write_artifacts(out, art)
        forbidden = {".tif", ".tiff", ".shp", ".npz", ".npy", ".pt", ".pth", ".parquet", ".ckpt", ".safetensors"}
        for p in out.rglob("*"):
            if p.is_file():
                assert p.suffix.lower() not in forbidden

    def test_matrix_schema(self, tmp_path):
        from revp_v2bu_alternative_event_overlay_sensitivity import MATRIX_FIELDS
        art = self._art()
        out = tmp_path / "out3"
        out.mkdir()
        write_artifacts(out, art)
        with (out / "alternative_overlay_patch_sensitivity_matrix_v2bu.csv").open(encoding="utf-8") as f:
            fields = csv.DictReader(f).fieldnames
        assert fields == MATRIX_FIELDS

    def test_report_safe_language(self, tmp_path):
        art = self._art()
        out = tmp_path / "out4"
        out.mkdir()
        write_artifacts(out, art)
        text = (out / "overlay_sensitivity_report_v2bu.md").read_text(encoding="utf-8").lower()
        for phrase in ("operational flood detection", "validated prediction", "flood accuracy", "operational model"):
            assert f"no {phrase}" in text or phrase not in text


class TestNoPrivatePaths:
    def test_no_private_path_in_script(self):
        user_prefix = "Users" + "\\" + "gabriela"
        text = (SCRIPTS_DIR / "revp_v2bu_alternative_event_overlay_sensitivity.py").read_text(encoding="utf-8", errors="replace")
        assert user_prefix not in text
        assert "/home/" not in text
