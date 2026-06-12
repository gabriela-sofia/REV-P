"""v2au - methodological guardrail tests (no label, no training, max C4 candidate)."""

from __future__ import annotations

import csv
import json


def _run(engine, ds, tmp_path):
    code, summary = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                               config_dir=str(tmp_path / "cfg"))
    assert code == 0
    return summary, tmp_path / "out"


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _overlap(make_geom, event_id, patch_id):
    return [
        make_geom("patch_boundary", "bbox", "0,0,10,10", linked_patch_id=patch_id),
        make_geom("event_observed_geometry", "bbox", "5,5,15,15", linked_event_id=event_id),
    ]


def test_summary_blocks_training(v2au_engine, v2au_dataset, tmp_path):
    summary, out = _run(v2au_engine, v2au_dataset(), tmp_path)
    data = json.load(open(out / "execution_reports" /
                          "v2au_patch_event_overlay_geometry_summary.json", encoding="utf-8"))
    assert data["can_train_model"] is False
    assert data["can_create_operational_labels"] is False
    assert data["methodological_status"] == "GEOMETRY_OVERLAY_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING"
    assert summary["can_train_model"] is False


def test_update_never_creates_label(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    # Even with a confirmed overlay, the decision can only be a C4 candidate.
    pkgs = [v2au_make_package("PKG_lbl", "E1", "P1")]
    ds = v2au_dataset(packages=pkgs, geometry_sources=_overlap(v2au_make_geom, "E1", "P1"))
    _run(v2au_engine, ds, tmp_path)
    upd = _read(ds / "v2au_event_patch_package_overlay_update.csv")
    for r in upd:
        assert r["can_create_operational_label"] == "false"
        decision = r["new_promotion_decision"].upper()
        assert "OPERATIONAL_LABEL" not in decision
        assert "TRAINING_LABEL" not in decision
        assert "GROUND_TRUTH_FINAL" not in decision
        assert r["new_promotion_candidate_level"] != "C4"  # only C4_CANDIDATE allowed


def test_max_decision_is_c4_candidate(v2au_engine, v2au_dataset, tmp_path, v2au_make_package, v2au_make_geom):
    pkgs = [v2au_make_package("PKG_max", "E1", "P1")]
    ds = v2au_dataset(packages=pkgs, geometry_sources=_overlap(v2au_make_geom, "E1", "P1"))
    _run(v2au_engine, ds, tmp_path)
    upd = _read(ds / "v2au_event_patch_package_overlay_update.csv")[0]
    assert upd["new_promotion_decision"] == "C4_CANDIDATE_REQUIRES_HUMAN_REVIEW"


def test_no_model_artifact_created(v2au_engine, v2au_dataset, tmp_path):
    ds = v2au_dataset()
    _, out = _run(v2au_engine, ds, tmp_path)
    bad = (".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors")
    for base in (ds, out):
        for path in base.rglob("*"):
            assert path.suffix.lower() not in bad, f"unexpected model artifact: {path}"


def test_absence_of_geometry_not_negative(v2au_engine, v2au_dataset, tmp_path):
    # Missing geometry must produce a blocker, never a negative/label.
    ds = v2au_dataset()
    _run(v2au_engine, ds, tmp_path)
    ov = _read(ds / "v2au_patch_event_overlay_registry.csv")
    for r in ov:
        assert r["overlay_status"].startswith("BLOCKED") or r["overlay_status"] == "NO_INTERSECTION"
        assert r["allowed_use"] in {"blocked_missing_geometry", "geometry_review_only",
                                    "blocked_invalid_geometry", "blocked_context_only"}


def test_exit_zero_with_no_packages(v2au_engine, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, summary = v2au_engine.run(dataset_dir=str(empty), output_dir=str(tmp_path / "o"),
                                    config_dir=str(tmp_path / "c"))
    assert code == 0
    assert summary["can_train_model"] is False
    assert summary["total_packages"] >= 1
