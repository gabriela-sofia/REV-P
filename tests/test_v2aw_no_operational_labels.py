"""v2aw - methodological guardrails: no labels, models, training or final ground truth."""

from __future__ import annotations

import json


def _run(engine, ds, tmp_path):
    code, summary = engine.run(dataset_dir=str(ds), output_dir=str(tmp_path / "out"),
                               config_dir=str(tmp_path / "cfg"))
    assert code == 0
    return summary, tmp_path / "out"


def test_summary_explicitly_blocks_model_and_labels(v2aw_engine, v2aw_dataset, tmp_path):
    summary, out = _run(v2aw_engine, v2aw_dataset(), tmp_path)
    written = json.loads((out / "execution_reports" /
                          "v2aw_geometry_source_intake_summary.json").read_text(encoding="utf-8"))
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
    assert written["can_train_model"] is False
    assert written["can_create_operational_labels"] is False


def test_no_model_or_label_artifacts_created(v2aw_engine, v2aw_dataset, tmp_path):
    ds = v2aw_dataset()
    _, out = _run(v2aw_engine, ds, tmp_path)
    model_suffixes = {".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors"}
    forbidden_names = ("operational_label", "training_label", "ground_truth_final", "trained_model")
    for base in (ds, out):
        for path in base.rglob("*"):
            assert path.suffix.lower() not in model_suffixes
            assert not any(term in path.name.lower() for term in forbidden_names)


def test_valid_geometry_is_still_not_a_label(v2aw_engine, v2aw_dataset, tmp_path,
                                             v2aw_make_patch_source, v2aw_make_event_source):
    ds = v2aw_dataset(provided_patch=[v2aw_make_patch_source()],
                      provided_event=[v2aw_make_event_source()])
    summary, _ = _run(v2aw_engine, ds, tmp_path)
    assert summary["ready_for_v2av_count"] == 1
    assert summary["ready_for_v2au_count"] == 1
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False
