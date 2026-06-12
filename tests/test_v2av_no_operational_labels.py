"""v2av - methodological guardrail tests (no label, no ground truth, no training)."""

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


def test_summary_blocks_training(v2av_engine, v2av_dataset, tmp_path):
    summary, out = _run(v2av_engine, v2av_dataset(), tmp_path)
    data = json.load(open(out / "execution_reports" /
                          "v2av_patch_boundary_geometry_builder_summary.json", encoding="utf-8"))
    assert data["can_train_model"] is False
    assert data["can_create_operational_labels"] is False
    assert data["methodological_status"] == "PATCH_BOUNDARY_RECOVERY_READY_FOR_OVERLAY_NOT_FOR_TRAINING"
    assert summary["can_train_model"] is False


def test_built_boundary_is_not_a_label(v2av_engine, v2av_dataset, tmp_path, v2av_make_geom_source):
    # A built boundary is geometry for v2au overlay, never an event label/ground truth.
    geoms = [v2av_make_geom_source("REC_00205", "bbox", "0,0,10,10", crs="EPSG:3857")]
    ds = v2av_dataset(geometry_sources=geoms)
    _run(v2av_engine, ds, tmp_path)
    reg = _read(ds / "v2av_patch_boundary_geometry_registry.csv")
    for r in reg:
        notes = r["notes"].lower()
        assert "label" not in notes or "never" in notes
        assert "ground truth" not in notes or "never" in notes


def test_absence_not_negative(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _run(v2av_engine, ds, tmp_path)
    reg = _read(ds / "v2av_patch_boundary_geometry_registry.csv")
    for r in reg:
        # Missing metadata is a blocker, never a negative/label.
        assert r["is_valid_geometry"] == "false"
        assert r["blocking_reason"] != ""


def test_no_model_artifact_created(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()
    _, out = _run(v2av_engine, ds, tmp_path)
    bad = (".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors")
    for base in (ds, out):
        for path in base.rglob("*"):
            assert path.suffix.lower() not in bad, f"unexpected model artifact: {path}"


def test_geojson_only_written_when_built(v2av_engine, v2av_dataset, tmp_path):
    ds = v2av_dataset()  # no metadata -> nothing built
    summary, _ = _run(v2av_engine, ds, tmp_path)
    assert summary["geojson_files_written"] == 0
    geo_dir = ds / "geometries" / "patch_boundaries"
    assert not geo_dir.exists() or not list(geo_dir.glob("*.geojson"))


def test_exit_zero_with_no_inputs(v2av_engine, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    code, summary = v2av_engine.run(dataset_dir=str(empty), output_dir=str(tmp_path / "o"),
                                    config_dir=str(tmp_path / "c"))
    assert code == 0
    assert summary["can_train_model"] is False
    assert summary["total_unique_patches"] >= 1
