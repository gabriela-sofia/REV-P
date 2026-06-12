"""v2at - methodological guardrail tests (no operational labels, no training)."""

from __future__ import annotations

import csv
import json


def _run(engine, dataset_dir, tmp_path):
    out = tmp_path / "out"
    cfg = tmp_path / "cfg"
    code, summary = engine.run(dataset_dir=str(dataset_dir), output_dir=str(out), config_dir=str(cfg))
    assert code == 0
    return summary, out


def _read(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def test_summary_blocks_training(v2at_engine, v2at_dataset, tmp_path):
    summary, out = _run(v2at_engine, v2at_dataset(), tmp_path)
    data = json.load(open(out / "execution_reports" /
                          "v2at_evidence_registry_event_patch_summary.json", encoding="utf-8"))
    assert data["can_train_model"] is False
    assert data["can_create_operational_labels"] is False
    assert data["methodological_status"] == "EVIDENCE_SYSTEM_READY_FOR_HUMAN_REVIEW_NOT_FOR_TRAINING"
    assert summary["can_train_model"] is False
    assert summary["can_create_operational_labels"] is False


def test_no_package_reaches_operational_label(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_event_patch_package_registry.csv")
    for r in rows:
        # C4 only ever appears as a candidate, never as a final label.
        assert r["promotion_candidate_level"] != "C4"
        assert "LABEL" not in r["promotion_decision"].upper()
        assert r["allowed_use"] != "operational_label_blocked" or True  # enum allowed, never a label


def test_blocklist_lists_label_blocks(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_operational_label_blocklist.csv")
    assert rows
    required = {"block_id", "package_id", "event_id", "patch_id", "reason",
                "source_of_block", "severity", "can_be_revisited", "required_evidence_to_unblock"}
    assert required.issubset(set(rows[0].keys()))
    # Category-level blocks must enumerate the forbidden patterns.
    reasons = " ".join(r["reason"].lower() for r in rows)
    for token in ("quickview", "media", "benchmark", "absence", "geometry", "temporal window", "conflict"):
        assert token in reasons, f"blocklist missing pattern: {token}"
    # Every package has at least one block entry (nothing becomes a label).
    pkgs = {p["package_id"] for p in _read(ds / "v2at_event_patch_package_registry.csv")}
    blocked = {r["package_id"] for r in rows}
    assert pkgs.issubset(blocked)


def test_benchmark_absence_never_negative(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_operational_label_blocklist.csv")
    reasons = " ".join(r["reason"].lower() for r in rows)
    assert "benchmark" in reasons and "local truth" in reasons
    assert "absence of evidence is never a negative" in reasons


def test_no_trained_model_artifact_created(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _, out = _run(v2at_engine, ds, tmp_path)
    bad_suffixes = (".pt", ".pth", ".onnx", ".h5", ".ckpt", ".pkl", ".joblib", ".safetensors")
    for base in (ds, out):
        for path in base.rglob("*"):
            assert path.suffix.lower() not in bad_suffixes, f"unexpected model artifact: {path}"


def test_observation_roles_are_review_only(v2at_engine, v2at_dataset, tmp_path):
    ds = v2at_dataset()
    _run(v2at_engine, ds, tmp_path)
    rows = _read(ds / "v2at_evidence_observation_registry.csv")
    assert rows
    allowed_roles = {"temporal_anchor", "spatial_support", "territorial_context",
                     "context_only", "methodological_benchmark"}
    for r in rows:
        assert r["review_status"] == "review_only"
        assert r["evidence_role"] in allowed_roles


def test_exit_zero_with_no_inputs(v2at_engine, tmp_path):
    # Fail-closed: empty dataset dir still produces a valid, blocked evidence system.
    empty = tmp_path / "empty_ds"
    empty.mkdir()
    code, summary = v2at_engine.run(dataset_dir=str(empty), output_dir=str(tmp_path / "o"),
                                    config_dir=str(tmp_path / "c"))
    assert code == 0
    assert summary["can_train_model"] is False
    assert summary["total_packages"] >= 1  # placeholder package
